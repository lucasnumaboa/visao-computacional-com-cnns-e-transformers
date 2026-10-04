"""Build the technical report: relatorio/relatorio.md -> HTML (with CSS) -> PDF via headless Microsoft Edge.

Usage: python tools/build_report.py
"""
import re
import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "relatorio" / "relatorio.md"
HTML = ROOT / "relatorio" / "relatorio.html"
PDF = ROOT / "relatorio" / "lucas_ferreira_deep-learning-and-vision_computer-vision.pdf"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
html { font-family: 'Segoe UI', Calibri, Arial, sans-serif; font-size: 10.2pt; color: #1b1b1b; line-height: 1.42; }
body { margin: 0; }
h1 { font-size: 19pt; color: #0b3d63; border-bottom: 2px solid #0b3d63; padding-bottom: 4px; margin: 0 0 10px; page-break-before: always; }
h1.nobreak, .cover + h1 { page-break-before: avoid; }
h2 { font-size: 13.5pt; color: #0b3d63; margin: 18px 0 6px; }
h3 { font-size: 11.2pt; color: #24577f; margin: 14px 0 4px; }
h4 { font-size: 10.4pt; margin: 10px 0 3px; }
p { margin: 4px 0 7px; text-align: justify; }
ul, ol { margin: 3px 0 7px 0; padding-left: 20px; }
li { margin: 2px 0; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 10px; font-size: 8.8pt; page-break-inside: avoid; }
th, td { border: 1px solid #b9c6d2; padding: 3px 5px; vertical-align: top; }
th { background: #e6eef5; text-align: left; }
tr:nth-child(even) td { background: #f7f9fb; }
code { font-family: Consolas, monospace; font-size: 8.8pt; background: #f1f3f5; padding: 0 2px; border-radius: 2px; }
pre { background: #f1f3f5; padding: 6px 8px; font-size: 8.4pt; overflow: hidden; white-space: pre-wrap; }
img { display: block; margin: 6px auto 2px; max-width: 100%; page-break-inside: avoid; }
figure { margin: 8px 0 10px; page-break-inside: avoid; }
figcaption, .caption { font-size: 8.6pt; color: #444; text-align: center; margin: 2px 0 8px; font-style: italic; }
blockquote { border-left: 3px solid #0b3d63; margin: 6px 0; padding: 2px 10px; background: #f3f7fa; }
.cover { page-break-after: always; height: 250mm; display: flex; flex-direction: column; justify-content: center; text-align: center; }
.cover h1 { border: none; font-size: 24pt; page-break-before: avoid; }
.cover .sub { font-size: 13pt; color: #24577f; margin: 6px 0 40px; }
.cover .meta { font-size: 11pt; line-height: 1.8; }
.toc ul { list-style: none; padding-left: 0; } .toc li { margin: 3px 0; }
.small { font-size: 8.8pt; }
.repo { border: 2px solid #0b3d63; border-radius: 6px; background: #eef4f9; padding: 10px 14px; margin: 0 0 30px; font-size: 10.5pt; text-align: left; line-height: 1.6; }
.repo code { font-size: 9.5pt; background: #fff; }
.kpi { display: inline-block; border: 1px solid #b9c6d2; border-radius: 4px; padding: 3px 8px; margin: 2px 4px 2px 0; background: #f3f7fa; font-size: 9pt; }
"""


def compress(src):
    """Copia a figura como JPEG (largura máx. 1200 px, qualidade 75) para o PDF não ficar pesado."""
    from PIL import Image
    path = (ROOT / "relatorio" / src).resolve()
    out = ROOT / "relatorio" / "_img" / (path.parent.name + "_" + path.stem + ".jpg")
    out.parent.mkdir(exist_ok=True)
    if not out.exists() or out.stat().st_mtime < path.stat().st_mtime:
        im = Image.open(path).convert("RGB")
        if im.width > 1200:
            im = im.resize((1200, round(im.height * 1200 / im.width)), Image.LANCZOS)
        im.save(out, quality=75, optimize=True)
    return "_img/" + out.name


def figure(match):
    alt, src = match.group(1), compress(match.group(2))
    width = ""
    if "|" in alt:
        alt, w = alt.split("|", 1)
        width = f' style="width:{w.strip()}"'
    return f'<figure><img src="{src}" alt="{alt}"{width}><figcaption>{alt}</figcaption></figure>'


SECTIONS = ["00_capa_intro", "a1_vit", "a2_clip", "a3_cnn", "a4_raiox", "a42_trafego", "conclusoes", "99_ia_referencias"]


def main():
    parts = [(ROOT / "relatorio" / "secoes" / f"{s}.md").read_text(encoding="utf-8") for s in SECTIONS]
    SRC.write_text("\n\n".join(parts), encoding="utf-8")
    text = SRC.read_text(encoding="utf-8")
    text = re.sub(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", figure, text, flags=re.M)
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "attr_list", "md_in_html", "sane_lists"])
    HTML.write_text(f"<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'><title>Relatório técnico</title>"
                    f"<style>{CSS}</style></head><body>{body}</body></html>", encoding="utf-8")
    subprocess.run([str(EDGE), "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={PDF}",
                    HTML.as_uri()], check=True, timeout=180)
    # O Chromium reencoda as imagens em alta qualidade; recomprimimos para caber no limite de upload (20 MB)
    import pymupdf
    doc = pymupdf.open(PDF)
    doc.rewrite_images(dpi_threshold=160, dpi_target=150, quality=72)
    tmp = PDF.with_suffix(".tmp.pdf"); doc.save(tmp, garbage=4, deflate=True); doc.close()
    tmp.replace(PDF)
    print("PDF:", PDF, f"{PDF.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
