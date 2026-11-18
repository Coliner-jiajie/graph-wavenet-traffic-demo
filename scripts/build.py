"""Inline the data, Leaflet and styles into src/template.html and write docs/index.html.

Run:  python3 scripts/build.py          (standard library only)

docs/index.html is a single self-contained file (map tiles load from OpenStreetMap when online).
GitHub Pages serves it directly from the /docs folder.
build/artifact_page.html is the same page without the HTML shell, for tools that wrap the page themselves.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
template = (ROOT / "src" / "template.html").read_text()
data = (ROOT / "data" / "demo_data.json").read_text()
json.loads(data)  # fail early if the JSON is broken
leaflet_css = (ROOT / "vendor" / "leaflet.css").read_text()
leaflet_js = (ROOT / "vendor" / "leaflet.js").read_text()

body = (template.replace("__DATA__", data)
                .replace("/*LEAFLET_CSS*/", leaflet_css)
                .replace("/*LEAFLET_JS*/", leaflet_js.replace("</script>", "<\\/script>")))
shell = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
         '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')

(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "build").mkdir(exist_ok=True)
page = shell + body + "\n</body></html>\n"
(ROOT / "docs" / "index.html").write_text(page)
(ROOT / "build" / "artifact_page.html").write_text(body)
print("built docs/index.html", f"({len(page) / 1e6:.1f} MB)")
