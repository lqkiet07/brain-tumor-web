from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"
INDEX_HTML = WEB_DIR / "index.html"
DEMO_JS = WEB_DIR / "js" / "demo.js"


def test_model_options_are_not_hardcoded_in_html():
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert '<select class="model-select" id="model-select"' in html
    assert 'option value="vgg16_b4b5"' not in html, "index.html van con hardcode option vgg16_b4b5"
    assert 'option value="resnet50"' not in html, "index.html van con hardcode option resnet50"
    assert 'option value="vgg16_block5"' not in html, "index.html van con hardcode option vgg16_block5"


def test_demo_script_fetches_models_from_api():
    js = DEMO_JS.read_text(encoding="utf-8")
    assert 'fetch("/api/models")' in js or "fetch('/api/models')" in js, "demo.js phai goi /api/models"


def test_no_external_font_links_in_html():
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert "fonts.googleapis.com" not in html, "index.html van con link toi fonts.googleapis.com"
    assert "fonts.gstatic.com" not in html, "index.html van con preconnect toi fonts.gstatic.com"
    assert 'href="/css/fonts.css"' in html, "index.html phai link toi /css/fonts.css noi bo"


def test_fonts_css_and_referenced_files_exist():
    fonts_css = WEB_DIR / "css" / "fonts.css"
    assert fonts_css.exists(), f"File {fonts_css} chua ton tai"
    content = fonts_css.read_text(encoding="utf-8")
    assert "fonts.googleapis.com" not in content
    assert "fonts.gstatic.com" not in content

    import re
    urls = re.findall(r'url\((?:["\']?)(/assets/fonts/[^)"\']+)', content)
    assert len(urls) >= 5, f"Can it nhat 5 @font-face woff2, tim thay {len(urls)}"
    for url in urls:
        rel = url.lstrip("/")
        target = WEB_DIR / rel
        assert target.exists(), f"Thieu file font: {target}"
        assert target.stat().st_size > 0, f"File font rong: {target}"


def test_slider_and_file_input_have_accessible_labels():
    import re

    html = INDEX_HTML.read_text(encoding="utf-8")
    file_match = re.search(r"<input[^>]*id=[\"']file-input[\"'][^>]*>", html)
    assert file_match is not None, "Khong tim thay input #file-input"
    assert "aria-label=" in file_match.group(0), "input #file-input thieu aria-label"

    slider_match = re.search(r"<input[^>]*id=[\"']heatmap-slider[\"'][^>]*>", html)
    assert slider_match is not None, "Khong tim thay input #heatmap-slider"
    assert "aria-label=" in slider_match.group(0), "input #heatmap-slider thieu aria-label"


def test_leaderboard_handles_null_metrics_in_app_js():
    js = (WEB_DIR / "js" / "app.js").read_text(encoding="utf-8")
    assert "null-val" in js or "—" in js, "app.js phai xu ly cac gia tri null/undefined bang dau gach ngang —"
    assert "evaluate_only" in js or "macro_f1" in js, "app.js phai nhan dien evaluation_type hoac macro_f1 null"


def test_index_html_has_academic_integrity_footnote():
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert "evaluate()" in html or "model.evaluate" in html or "đo qua evaluate" in html.lower(), (
        "index.html phai co chu thich hoc thuat giai thich ro cac mo hinh chi do qua evaluate()"
    )


def test_no_student_or_personal_metadata_in_html_and_js():
    html = INDEX_HTML.read_text(encoding="utf-8")
    js = (WEB_DIR / "js" / "app.js").read_text(encoding="utf-8")
    for forbidden in ["Lê Quốc Kiệt", "B2303878", "CT223H", "Cần Thơ", "Công Thương"]:
        assert forbidden not in html, f"index.html con chua thong tin ca nhan: {forbidden}"
        assert forbidden not in js, f"app.js con chua thong tin ca nhan: {forbidden}"


def test_style_css_has_accessible_contrast_and_focus_visible():
    css = (WEB_DIR / "css" / "style.css").read_text(encoding="utf-8")
    assert "--text-muted: #6b7280" in css, "style.css phai dung #6b7280 de dat WCAG AA >= 4.5:1"
    assert ":focus-visible" in css, "style.css phai co quy tac :focus-visible"
    assert "outline: none" not in css, "style.css khong duoc triet tieu outline tren form control"


def test_table_has_min_width_for_mobile_horizontal_scroll():
    css = (WEB_DIR / "css" / "style.css").read_text(encoding="utf-8")
    assert "min-width: 720px" in css or "min-width: 680px" in css, "data-table phai co min-width de cuon ngang"


def test_empty_state_has_clean_non_dashed_styling():
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert 'border: 1px dashed' not in html, "index.html khong duoc chua inline border: 1px dashed tren empty-state"
    css = (WEB_DIR / "css" / "style.css").read_text(encoding="utf-8")
    assert ".empty-state" in css, "style.css phai co class .empty-state"


def test_no_redundant_filler_text_in_html_and_js():
    html = INDEX_HTML.read_text(encoding="utf-8")
    js = (WEB_DIR / "js" / "app.js").read_text(encoding="utf-8")
    unwanted = [
        "Mỗi thành viên phụ trách",
        "Top 1 Toàn Nhóm",
        "thực nghiệm gốc của nhóm",
        "nhóm tác giả chưa trích xuất",
        "giữ kín tuyệt đối để đánh giá công bằng",
        "Tải báo cáo (.docx)",
        "Họ Dense",
        "Họ ConvNeXt",
        '<p class="eyebrow">Thực nghiệm Học máy & Giải thuật</p>',
        '<p class="eyebrow">Đối sánh tổng thể</p>',
        '<p class="eyebrow">Khảo sát chuyên sâu theo nhánh</p>',
        '<p class="eyebrow">Tập dữ liệu chuẩn hóa</p>',
        '<p class="eyebrow">Phòng thử nghiệm trực tiếp</p>',
    ]
    for text in unwanted:
        assert text not in html, f"index.html van con chua text thua: {text}"
        assert text not in js, f"app.js van con chua text thua: {text}"






