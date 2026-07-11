"""Render reports/informe.md -> reports/informe.html -> reports/informe.pdf.

Keeps the CSS wrapper from the previous informe.html.
"""
from __future__ import annotations

from pathlib import Path

import markdown

REPO = Path(__file__).resolve().parents[1]
MD_PATH = REPO / "reports" / "informe.md"
HTML_PATH = REPO / "reports" / "informe.html"
PDF_PATH = REPO / "reports" / "informe.pdf"

CSS = """
@page { size: A4; margin: 22mm 18mm; @bottom-right { content: counter(page) "/" counter(pages); font-size: 9pt; color: #555; } }
body { font-family: -apple-system, "Segoe UI", "Liberation Sans", sans-serif; font-size: 10.5pt; line-height: 1.45; color: #111; }
h1 { font-size: 22pt; border-bottom: 2px solid #2563eb; padding-bottom: 4px; }
h2 { font-size: 15pt; margin-top: 18pt; color: #1e3a8a; border-bottom: 1px solid #cbd5e1; padding-bottom: 2px; }
h3 { font-size: 12pt; margin-top: 12pt; color: #1e293b; }
h4 { font-size: 11pt; color: #334155; }
p, li { text-align: justify; }
code { font-family: "JetBrains Mono", "DejaVu Sans Mono", monospace; font-size: 9pt; background: #f1f5f9; padding: 1px 4px; border-radius: 3px; }
pre { background: #0f172a; color: #e2e8f0; padding: 8px 10px; border-radius: 4px; font-size: 8.5pt; line-height: 1.35; page-break-inside: avoid; }
pre code { background: transparent; color: inherit; padding: 0; font-size: inherit; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 9.5pt; page-break-inside: avoid; }
th, td { border: 1px solid #cbd5e1; padding: 4px 6px; text-align: left; }
th { background: #e2e8f0; }
img { max-width: 100%; height: auto; display: block; margin: 6pt auto; page-break-inside: avoid; }
blockquote { border-left: 3px solid #94a3b8; padding: 4pt 10pt; color: #475569; background: #f8fafc; margin: 8pt 0; }
hr { border: 0; border-top: 1px solid #cbd5e1; margin: 14pt 0; }
"""

TITLE_HTML = """
<h1>Hipica-ML — Clasificador de Trifecta de Maroñas</h1>
<p><strong>Obligatorio · Machine Learning en Producción · Universidad ORT Uruguay</strong></p>
<table>
<tbody>
<tr><td>Autores</td><td>Mathias Gili · Bruno Bellizzi</td></tr>
<tr><td>Curso</td><td>Machine Learning en Producción</td></tr>
<tr><td>Fecha</td><td>Julio 2026</td></tr>
<tr><td>Repositorio</td><td><a href="https://github.com/MathiasGili/hipica-ml">https://github.com/MathiasGili/hipica-ml</a></td></tr>
<tr><td>Licencia</td><td>MIT</td></tr>
</tbody>
</table>
<hr />
"""


def _strip_image_widths(md_text: str) -> str:
    # Pandoc-style {width=95%} attributes aren't rendered by python-markdown; drop them.
    import re
    return re.sub(r"\{\s*width=[^}]+\}", "", md_text)


def _strip_leading_title(md_text: str) -> str:
    # The md file has its own front matter / title block before "## 1. Resumen ejecutivo";
    # replace everything before the first "## " with our injected title.
    idx = md_text.find("\n## ")
    return md_text[idx + 1 :] if idx > 0 else md_text


def render() -> None:
    md_text = MD_PATH.read_text(encoding="utf-8")
    md_text = _strip_image_widths(md_text)
    md_text = _strip_leading_title(md_text)
    body_html = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )
    html_doc = (
        f'<!doctype html><html lang="es"><head><meta charset="utf-8">'
        f'<title>Hipica-ML — Informe</title><style>{CSS}</style></head>'
        f"<body>{TITLE_HTML}{body_html}</body></html>"
    )
    HTML_PATH.write_text(html_doc, encoding="utf-8")
    print(f"HTML  -> {HTML_PATH.relative_to(REPO)}  ({len(html_doc):,} bytes)")

    from weasyprint import HTML

    HTML(string=html_doc, base_url=str(REPO / "reports")).write_pdf(str(PDF_PATH))
    print(f"PDF   -> {PDF_PATH.relative_to(REPO)}  ({PDF_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    render()
