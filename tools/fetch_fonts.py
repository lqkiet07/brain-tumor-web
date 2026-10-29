import hashlib
import re
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FONTS_DIR = BASE_DIR / "web" / "assets" / "fonts"
FONTS_CSS = BASE_DIR / "web" / "css" / "fonts.css"

CSS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Be+Vietnam+Pro:wght@400;500;600;700&"
    "family=IBM+Plex+Mono:wght@400;500;600&"
    "display=swap"
)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def fetch_and_save():
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    FONTS_CSS.parent.mkdir(parents=True, exist_ok=True)

    print(f"1. Fetching font CSS from Google Fonts ...")
    req = urllib.request.Request(CSS_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        css = resp.read().decode("utf-8")

    urls = re.findall(r"url\((https://fonts\.gstatic\.com/[^)]+)\)", css)
    print(f"2. Found {len(urls)} woff2 font files to download.")

    replacements = {}
    for i, url in enumerate(urls, 1):
        url_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
        family = "be-vietnam-pro" if "vietnam" in url.lower() or "bevietnampro" in url.lower() else "ibm-plex-mono"
        filename = f"{family}-{url_hash}.woff2"
        target_path = FONTS_DIR / filename

        if not target_path.exists() or target_path.stat().st_size == 0:
            print(f"   [{i}/{len(urls)}] Downloading {filename} ...")
            font_req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(font_req, timeout=30) as f_resp:
                target_path.write_bytes(f_resp.read())
        else:
            print(f"   [{i}/{len(urls)}] Already exists: {filename}")

        replacements[url] = f"/assets/fonts/{filename}"

    for orig_url, local_url in replacements.items():
        css = css.replace(orig_url, local_url)

    # Clean multi-line comments
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    # Normalize extra blank lines
    css = re.sub(r"\n\s*\n+", "\n\n", css).strip() + "\n"

    FONTS_CSS.write_text(css, encoding="utf-8")
    print(f"3. Saved {FONTS_CSS} successfully.")


if __name__ == "__main__":
    fetch_and_save()
