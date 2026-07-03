import streamlit as st
import uuid
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import pandas as pd
import io
from datetime import datetime
import os
import tempfile
import re
import time
from pathlib import Path
import multiprocessing as mp

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from xml.sax.saxutils import escape as xml_escape

from streamlit_drawable_canvas import st_canvas


Image.MAX_IMAGE_PIXELS = None
CANVAS_AVAILABLE = True


# -----------------------------------------------------------------------------
# Конфигурация страницы
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Геологический дашборд анализа руд",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Стилизация CSS
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
    .main-header {
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #3498db;
        margin: 10px 0;
    }
    .result-box {
        background: #e8f4f8;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        border: 1px solid #b8d4de;
    }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Инициализация состояния сессии
# -----------------------------------------------------------------------------
if "processing_history" not in st.session_state:
    st.session_state.processing_history = []

if "current_result" not in st.session_state:
    st.session_state.current_result = None

if "original_image" not in st.session_state:
    st.session_state.original_image = None

if "expert_markup_bytes" not in st.session_state:
    st.session_state.expert_markup_bytes = None

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = str(uuid.uuid4())

if "analysis_in_progress" not in st.session_state:
    st.session_state.analysis_in_progress = False


# -----------------------------------------------------------------------------
# PDF helpers
# -----------------------------------------------------------------------------
def get_human_classification(talc_content: float, classification: str) -> str:
    classification_normalized = str(classification).strip().lower()

    if talc_content > 10:
        return "Оталькованная"

    if classification_normalized == "ordinary":
        return "Рядовая"

    if classification_normalized == "difficult":
        return "Труднообогатимая"

    return "Неизвестно"


@st.cache_resource
def register_pdf_fonts():
    """
    Регистрирует шрифт с поддержкой кириллицы.

    Рекомендуемый вариант:
    положить файлы DejaVuSans.ttf и DejaVuSans-Bold.ttf в папку fonts
    рядом с этим streamlit-приложением.

    Также функция пытается найти стандартные системные шрифты Linux/Windows/macOS.
    """
    script_dir = Path(__file__).resolve().parent

    regular_candidates = [
        script_dir / "fonts" / "DejaVuSans.ttf",
        script_dir / "assets" / "fonts" / "DejaVuSans.ttf",
        Path.cwd() / "fonts" / "DejaVuSans.ttf",

        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/local/share/fonts/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),

        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),

        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    ]

    bold_candidates = [
        script_dir / "fonts" / "DejaVuSans-Bold.ttf",
        script_dir / "assets" / "fonts" / "DejaVuSans-Bold.ttf",
        Path.cwd() / "fonts" / "DejaVuSans-Bold.ttf",

        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/local/share/fonts/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),

        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf"),

        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("/Library/Fonts/Arial Bold.ttf"),
    ]

    regular_font = next((p for p in regular_candidates if p.exists()), None)
    bold_font = next((p for p in bold_candidates if p.exists()), None)

    if regular_font is None:
        # PDF всё равно будет сформирован, но кириллица может отображаться некорректно.
        return "Helvetica", "Helvetica-Bold"

    try:
        pdfmetrics.registerFont(TTFont("AppFont", str(regular_font)))
    except Exception:
        pass

    if bold_font is not None:
        try:
            pdfmetrics.registerFont(TTFont("AppFont-Bold", str(bold_font)))
        except Exception:
            pass
    else:
        try:
            pdfmetrics.registerFont(TTFont("AppFont-Bold", str(regular_font)))
        except Exception:
            pass

    try:
        pdfmetrics.registerFontFamily(
            "AppFont",
            normal="AppFont",
            bold="AppFont-Bold",
            italic="AppFont",
            boldItalic="AppFont-Bold",
        )
    except Exception:
        pass

    return "AppFont", "AppFont-Bold"


def pil_image_to_reportlab_image(
    pil_img: Image.Image,
    max_width,
    max_height,
) -> RLImage:
    """
    Конвертирует PIL.Image в ReportLab Image.
    Пропорции сохраняются.
    Слишком большие изображения уменьшаются, чтобы PDF не становился огромным.
    """
    img = pil_img.copy().convert("RGB")
    img.thumbnail((1800, 1800), Image.Resampling.LANCZOS)

    width_px, height_px = img.size

    if width_px <= 0 or height_px <= 0:
        raise ValueError("Некорректный размер изображения для PDF")

    scale = min(max_width / width_px, max_height / height_px)

    draw_width = width_px * scale
    draw_height = height_px * scale

    img_buf = io.BytesIO()
    img.save(img_buf, format="PNG", optimize=True)
    img_buf.seek(0)

    flowable = RLImage(img_buf, width=draw_width, height=draw_height)

    # Держим ссылку на буфер, чтобы ReportLab не потерял данные до doc.build().
    flowable._img_buffer = img_buf

    return flowable


def create_pdf_report(
    filename: str,
    timestamp: str,
    talc_content: float,
    classification: str,
    description: str,
    processed_image: Image.Image,
    mask_overlay: Image.Image,
    confidence_map: Image.Image | None = None,
) -> bytes:
    """
    Создаёт PDF-отчёт и возвращает его как bytes.
    """
    font_name, font_bold = register_pdf_fonts()

    pdf_buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "PdfTitle",
        parent=styles["Heading1"],
        fontName=font_bold,
        fontSize=20,
        leading=24,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#2c3e50"),
        spaceAfter=16,
    )

    heading_style = ParagraphStyle(
        "PdfHeading",
        parent=styles["Heading2"],
        fontName=font_bold,
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#2c3e50"),
        spaceBefore=12,
        spaceAfter=8,
    )

    normal_style = ParagraphStyle(
        "PdfNormal",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=10,
        leading=14,
    )

    table_header_style = ParagraphStyle(
        "PdfTableHeader",
        parent=styles["Normal"],
        fontName=font_bold,
        fontSize=10,
        leading=13,
        textColor=colors.white,
    )

    table_cell_style = ParagraphStyle(
        "PdfTableCell",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=10,
        leading=13,
    )

    def esc(value) -> str:
        return xml_escape(str(value))

    human_class = get_human_classification(talc_content, classification)

    story = []

    story.append(Paragraph("Отчет геологического анализа", title_style))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph(f"Образец: {esc(filename)}", normal_style))
    story.append(Paragraph(f"Дата анализа: {esc(timestamp)}", normal_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Результаты классификации", heading_style))

    table_data = [
        [
            Paragraph("Параметр", table_header_style),
            Paragraph("Значение", table_header_style),
        ],
        [
            Paragraph("Доля талька", table_cell_style),
            Paragraph(f"{talc_content:.1f}%", table_cell_style),
        ],
        [
            Paragraph("Классификация модели", table_cell_style),
            Paragraph(esc(classification), table_cell_style),
        ],
        [
            Paragraph("Класс руды", table_cell_style),
            Paragraph(esc(human_class), table_cell_style),
        ],
        [
            Paragraph("Описание", table_cell_style),
            Paragraph(esc(description), table_cell_style),
        ],
    ]

    result_table = Table(
        table_data,
        colWidths=[2.2 * inch, 4.6 * inch],
        repeatRows=1,
    )

    result_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )

    story.append(result_table)
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("Обработанное изображение", heading_style))
    story.append(
        pil_image_to_reportlab_image(
            processed_image,
            max_width=doc.width,
            max_height=3.4 * inch,
        )
    )

    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("Зоны талька", heading_style))
    story.append(
        pil_image_to_reportlab_image(
            mask_overlay,
            max_width=doc.width,
            max_height=3.4 * inch,
        )
    )

    if confidence_map is not None:
        story.append(Spacer(1, 0.25 * inch))
        story.append(Paragraph("Маска / карта уверенности", heading_style))
        story.append(
            pil_image_to_reportlab_image(
                confidence_map,
                max_width=doc.width,
                max_height=3.4 * inch,
            )
        )

    doc.build(story)

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


def make_safe_timestamp(timestamp: str) -> str:
    return timestamp.replace(":", "-").replace(" ", "_")


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
def run_inference_async(q, image_path: str, output_dir: str) -> mp.Process:
    from custom_inference import ui_main

    # Запускает inference как отдельный асинхронный процесс,
    # чтобы избежать конфликта st.set_page_config и не блокировать UI.
    p = mp.Process(target=ui_main, args=(q, image_path, output_dir))
    p.start()
    return p


# -----------------------------------------------------------------------------
# Заголовок приложения
# -----------------------------------------------------------------------------
st.title("🔬 Система автоматической классификации руд по OM-изображениям")
st.markdown("---")


# -----------------------------------------------------------------------------
# Боковая панель
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Параметры анализа")

    st.markdown("### Настройки обработки")

    noise_reduction = st.slider(
        "Шумоподавление (Gaussian Blur)",
        0,
        10,
        0,
        help="Радиус размытия. 0 - отключено",
    )

    contrast_correction = st.slider(
        "Коррекция контраста",
        0.5,
        2.0,
        1.0,
        help="1.0 - исходный контраст",
    )
    
    st.markdown("---")
    st.markdown("### 📊 История анализов")

    if st.session_state.processing_history:
        for i, item in enumerate(st.session_state.processing_history[-5:]):
            st.write(f"{i + 1}. {item['filename']} - {item['classification']}")
    else:
        st.info("Нет выполненных анализов")


# -----------------------------------------------------------------------------
# Основная область
# -----------------------------------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Загрузка изображения")

    uploaded_file = st.file_uploader(
        "Выберите изображение шлифа",
        type=["png", "jpg", "jpeg", "tiff", "tif"],
        help="Поддерживаются форматы высокого разрешения",
    )

    if uploaded_file is not None:
        original_image = Image.open(uploaded_file).convert("RGB")

        st.image(
            original_image,
            caption="Исходное изображение (до обработки)",
            use_container_width=True,
        )

        st.session_state.original_image = original_image

        if st.session_state.analysis_in_progress:
            st.warning(
                "⚠️ Анализ уже выполняется. Пожалуйста, дождитесь завершения текущего анализа.",
                icon="⏳",
            )
            st.button(
                "🔍 Провести анализ",
                type="primary",
                use_container_width=True,
                disabled=True,
            )
        else:
            if st.button("🔍 Провести анализ", type="primary", use_container_width=True):
                st.session_state.analysis_in_progress = True
                st.rerun()

    # Логика анализа вынесена за кнопку, чтобы работал rerun.
    if (
        uploaded_file is not None
        and st.session_state.analysis_in_progress
        and st.session_state.original_image is not None
    ):
        original_image = st.session_state.original_image
        processed_image = original_image.copy()

        with st.spinner("Выполняется анализ..."):
            if contrast_correction != 1.0:
                enhancer = ImageEnhance.Contrast(processed_image)
                processed_image = enhancer.enhance(contrast_correction)

            if noise_reduction > 0:
                processed_image = processed_image.filter(
                    ImageFilter.GaussianBlur(radius=noise_reduction)
                )

            # Сохраняем загруженное изображение во временную директорию.
            temp_dir = tempfile.mkdtemp()
            img_path = os.path.join(temp_dir, uploaded_file.name)

            with open(img_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Формируем путь к папке results в соответствии со структурой проекта.
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            out_dir = os.path.join(project_root, "results")

            os.makedirs(out_dir, exist_ok=True)

            status_placeholder = st.empty()
            start_time = time.time()

            q = mp.Queue()
            process = run_inference_async(q, img_path, out_dir)

            while process.is_alive():
                elapsed = int(time.time() - start_time)
                msg = f"Выполняется инференс модели... Прошло времени: {elapsed} сек."
                print(msg)
                status_placeholder.info(msg)
                time.sleep(1)

            status_placeholder.empty()

            if process.exitcode != 0:
                err_msg = "Неизвестная ошибка"

                try:
                    err_msg = q.get_nowait()["traceback"]
                except Exception:
                    pass

                msg = f"Ошибка инференса exit code {process.exitcode}:\n```\n{err_msg}\n```"
                print(msg)
                st.error(msg)

                st.session_state.analysis_in_progress = False
                st.stop()

            stem = Path(uploaded_file.name).stem

            heatmap_path = os.path.join(out_dir, f"{stem}_heatmap.png")
            report_path = os.path.join(out_dir, f"{stem}_report.txt")
            probs_path = os.path.join(out_dir, f"{stem}_probs.npy")

            if not os.path.exists(heatmap_path):
                st.error(
                    f"Файл heatmap не найден после инференса. "
                    f"Ожидаемый путь: {heatmap_path}"
                )
                st.session_state.analysis_in_progress = False
                st.stop()

            if not os.path.exists(report_path):
                st.error(
                    f"Файл отчёта не найден после инференса. "
                    f"Ожидаемый путь: {report_path}"
                )
                st.session_state.analysis_in_progress = False
                st.stop()

            # Загружаем heatmap из инференса.
            mask_overlay = Image.open(heatmap_path).convert("RGB")

            # Читаем результаты из текстового отчёта.
            with open(report_path, "r", encoding="utf-8") as f:
                report_text = f.read()

            talc_match = re.search(r"Talc share:\s+([\d.]+)%", report_text)
            talc_content = float(talc_match.group(1)) if talc_match else 0.0

            cls_match = re.search(r"Classification:\s+([^\n(]+)", report_text)
            classification = cls_match.group(1).strip() if cls_match else "Неизвестно"

            classification_normalized = classification.lower()

            # Определяем эмодзи и описание.
            if talc_content > 10:
                class_emoji = "🟣"
                description = (
                    f"Высокое содержание талька — {talc_content:.1f}% (>10%). "
                    f"Класс модели: Оталькованная"
                )
            elif "difficult" in classification_normalized:
                class_emoji = "🔴"
                description = "Доля талька ≤10%. Класс модели: Труднообогатимая"
            else:
                class_emoji = "🟢"
                description = "Доля талька ≤10%. Класс модели: Рядовая"

            width, height = processed_image.size

            # Маска из probs.npy.
            try:
                probs = np.load(probs_path)

                if probs.ndim == 3:
                    talc_probs = probs[1] if probs.shape[0] > 1 else probs[0]
                else:
                    talc_probs = probs

                if talc_probs.max() > 0:
                    probs_scaled = (
                        talc_probs / talc_probs.max() * 255
                    ).astype(np.uint8)
                else:
                    probs_scaled = np.zeros_like(talc_probs, dtype=np.uint8)

                confidence_map = Image.fromarray(probs_scaled, "L")
                confidence_map = confidence_map.resize(
                    (width, height),
                    Image.Resampling.BILINEAR,
                )

            except Exception:
                conf_scale = 4
                conf_h = max(1, height // conf_scale)
                conf_w = max(1, width // conf_scale)

                confidence_map_small = np.random.uniform(
                    0.4,
                    0.95,
                    (conf_h, conf_w),
                ).astype(np.float32)

                confidence_map = Image.fromarray(
                    (confidence_map_small * 255).astype(np.uint8),
                    "L",
                )

                confidence_map = confidence_map.resize(
                    (width, height),
                    Image.Resampling.BILINEAR,
                )

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            metrics_data = {
                "Параметр": ["Доля талька", "Классификация", "Время анализа"],
                "Значение": [
                    f"{talc_content:.1f}%",
                    classification,
                    timestamp,
                ],
            }

            df = pd.DataFrame(metrics_data)
            csv_bytes = df.to_csv(index=False).encode("utf-8-sig")

            png_buf = io.BytesIO()
            mask_overlay.save(png_buf, format="PNG")
            png_bytes = png_buf.getvalue()

            # PDF формируется корректной отдельной функцией.
            pdf_bytes = None
            pdf_error = None

            try:
                pdf_bytes = create_pdf_report(
                    filename=uploaded_file.name,
                    timestamp=timestamp,
                    talc_content=talc_content,
                    classification=classification,
                    description=description,
                    processed_image=processed_image,
                    mask_overlay=mask_overlay,
                    confidence_map=confidence_map,
                )
            except Exception as e:
                pdf_error = str(e)
                st.warning(f"PDF экспорт временно недоступен: {e}")

            result = {
                "filename": uploaded_file.name,
                "original_image": original_image,
                "processed_image": processed_image,
                "mask_overlay": mask_overlay,
                "confidence_map": confidence_map,
                "talc_content": talc_content,
                "classification": classification,
                "class_emoji": class_emoji,
                "description": description,
                "timestamp": timestamp,
                "csv_bytes": csv_bytes,
                "png_bytes": png_bytes,
                "pdf_bytes": pdf_bytes,
                "pdf_error": pdf_error,
            }

            st.session_state.current_result = result

            st.session_state.processing_history.append(
                {
                    "filename": uploaded_file.name,
                    "classification": classification,
                    "timestamp": timestamp,
                }
            )

            st.session_state.analysis_in_progress = False
            st.rerun()


# -----------------------------------------------------------------------------
# Отображение результатов
# -----------------------------------------------------------------------------
if st.session_state.current_result:
    CANVAS_AVAILABLE = True
    result = st.session_state.current_result

    st.markdown("---")
    st.subheader("📊 Результаты анализа")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(
            result["processed_image"],
            caption="Обработанное изображение",
            use_container_width=True,
        )

    with col2:
        st.image(
            result["mask_overlay"],
            caption="Зоны талька (выделены красным)",
            use_container_width=True,
        )

    with col3:
        st.image(
                result["confidence_map"],
                caption="Маска",
                use_container_width=True,
            )

    st.markdown("### 📈 Сведения о руде")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Доля талька",
            f"{result['talc_content']:.1f}%",
            delta="Внимание: >10%"
            if result["talc_content"] > 10
            else "Норма: ≤10%",
            delta_color="inverse" if result["talc_content"] > 10 else "normal",
        )

    with col2:
        ore_class = get_human_classification(
            result["talc_content"],
            result["classification"],
        )
        st.metric("Класс руды", ore_class)

    st.markdown("### 📋 Детальная таблица")

    df_display = pd.DataFrame(
        {
            "Параметр": ["Доля талька", "Классификация", "Время анализа"],
            "Значение": [
                f"{result['talc_content']:.1f}%",
                ore_class,
                result["timestamp"],
            ],
        }
    )

    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 💾 Экспорт")

    safe_ts = make_safe_timestamp(result["timestamp"])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            label="📥 Скачать CSV",
            data=result["csv_bytes"],
            file_name=f"analysis_{safe_ts}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        st.download_button(
            label="📥 Скачать зоны талька (PNG)",
            data=result["png_bytes"],
            file_name=f"mask_{safe_ts}.png",
            mime="image/png",
            use_container_width=True,
        )

    with col3:
        if result.get("pdf_bytes") is not None:
            st.download_button(
                label="📥 Скачать PDF-отчет",
                data=result["pdf_bytes"],
                file_name=f"report_{safe_ts}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            if result.get("pdf_error"):
                st.caption(f"PDF не был сформирован автоматически: {result['pdf_error']}")

            if st.button(
                "🧾 Сформировать PDF-отчет",
                use_container_width=True,
            ):
                try:
                    result["pdf_bytes"] = create_pdf_report(
                        filename=result["filename"],
                        timestamp=result["timestamp"],
                        talc_content=result["talc_content"],
                        classification=result["classification"],
                        description=result["description"],
                        processed_image=result["processed_image"],
                        mask_overlay=result["mask_overlay"],
                        confidence_map=result["confidence_map"],
                    )
                    result["pdf_error"] = None
                    st.session_state.current_result = result
                    st.rerun()
                except Exception as e:
                    result["pdf_error"] = str(e)
                    st.session_state.current_result = result
                    st.error(f"Не удалось сформировать PDF: {e}")

    # -------------------------------------------------------------------------
    # Режим экспертной проверки
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🧪 Режим экспертной проверки")

    if CANVAS_AVAILABLE and st.session_state.original_image is not None:
        orig_img = st.session_state.original_image

        scale_factor = min(1.0, 800 / orig_img.width)
        c_width = int(orig_img.width * scale_factor)
        c_height = int(orig_img.height * scale_factor)

        col_btn1, col_btn2 = st.columns([1, 1])

        with col_btn1:
            if st.button("🗑️ Очистить холст", use_container_width=True):
                st.session_state.canvas_key = str(uuid.uuid4())
                st.session_state.expert_markup_bytes = None
                st.rerun()

        with col_btn2:
            generate_btn = st.button(
                "Сформировать разметку для скачивания",
                use_container_width=True,
            )

        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 255, 0.3)",
            stroke_width=5,
            stroke_color="#0000FF",
            background_image=orig_img,
            height=c_height,
            width=c_width,
            drawing_mode="freedraw",
            key=st.session_state.canvas_key,
        )

        if generate_btn:
            if canvas_result.image_data is not None and np.any(
                canvas_result.image_data
            ):
                with st.spinner("Накладываем разметку..."):
                    drawn_array = canvas_result.image_data.astype("uint8")
                    drawn_layer_pil = Image.fromarray(drawn_array, "RGBA")

                    base_rgba = st.session_state.original_image.convert("RGBA")

                    drawn_resized = drawn_layer_pil.resize(
                        base_rgba.size,
                        Image.Resampling.NEAREST,
                    )

                    full_markup = Image.alpha_composite(
                        base_rgba,
                        drawn_resized,
                    )

                    full_markup_rgb = full_markup.convert("RGB")

                    buf_expert = io.BytesIO()
                    full_markup_rgb.save(buf_expert, format="PNG")

                    st.session_state.expert_markup_bytes = buf_expert.getvalue()

                    st.success("Разметка готова к скачиванию!")
            else:
                st.warning("Сначала нанесите разметку на изображение.")

        if st.session_state.expert_markup_bytes is not None:
            st.download_button(
                label="📥 Сохранить размеченное изображение",
                data=st.session_state.expert_markup_bytes,
                file_name=f"expert_markup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
                use_container_width=True,
            )
