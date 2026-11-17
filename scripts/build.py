"""Inline data/demo_data.json into src/template.html and write docs/index.html.

Run:  python scripts/build.py          (standard library only)

docs/index.html is a single self-contained file. GitHub Pages can serve it directly from the /docs folder.
build/artifact_page.html is the same page without the HTML shell, for pasting into tools that wrap the page themselves.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
template = (ROOT / "src" / "template.html").read_text()
data = (ROOT / "data" / "demo_data.json").read_text()
json.loads(data)  # fail early if the JSON is broken

body = template.replace("__DATA__", data)
shell = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
         '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')

(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "build").mkdir(exist_ok=True)
page = shell + body + "\n</body></html>\n"
(ROOT / "docs" / "index.html").write_text(page)
(ROOT / "build" / "artifact_page.html").write_text(body)
print("built docs/index.html", f"({len(page) / 1024:.0f} KB)")
