"""
========================================================================================
ACADEMIC DOCUMENT & DISSERTATION HTML/PDF COMPILER
Converts Markdown Manuscripts into Print-Ready Academic HTML/PDF Documents
========================================================================================
Usage:
    python export_manuscripts_html.py
========================================================================================
"""

import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MANUSCRIPT_DIR = os.path.join(BASE_DIR, "manuscript")

ACADEMIC_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:ital,wght@0,300;0,400;0,700;1,300&family=Open+Sans:wght@400;600;700&display=swap');
    
    * { box-sizing: border-box; }
    body {
        font-family: 'Merriweather', Georgia, serif;
        font-size: 11pt;
        line-height: 1.7;
        color: #222222;
        background: #f8fafc;
        margin: 0;
        padding: 40px 20px;
    }
    .page-container {
        max-width: 900px;
        margin: 0 auto;
        background: #ffffff;
        padding: 60px 75px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-radius: 4px;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Open Sans', -apple-system, sans-serif;
        color: #0f172a;
        font-weight: 700;
        margin-top: 1.8em;
        margin-bottom: 0.6em;
        line-height: 1.3;
    }
    h1 { font-size: 20pt; border-bottom: 2px solid #0f172a; padding-bottom: 8px; margin-top: 0; }
    h2 { font-size: 15pt; border-bottom: 1px solid #cbd5e1; padding-bottom: 4px; }
    h3 { font-size: 12.5pt; color: #1e3a8a; }
    h4 { font-size: 11pt; color: #334155; }
    p { margin-top: 0; margin-bottom: 1.2em; text-align: justify; }
    ul, ol { margin-top: 0; margin-bottom: 1.2em; padding-left: 28px; }
    li { margin-bottom: 0.4em; }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 25px 0;
        font-family: 'Open Sans', sans-serif;
        font-size: 9.5pt;
    }
    th, td {
        border: 1px solid #cbd5e1;
        padding: 8px 12px;
        text-align: left;
    }
    th {
        background-color: #f1f5f9;
        font-weight: 700;
        color: #0f172a;
    }
    blockquote {
        margin: 20px 0;
        padding: 12px 20px;
        background-color: #eff6ff;
        border-left: 4px solid #2563eb;
        color: #1e3a8a;
        font-size: 10.5pt;
    }
    pre, code {
        font-family: Consolas, Monaco, 'Courier New', monospace;
    }
    code {
        background: #f1f5f9;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 9pt;
        color: #b91c1c;
    }
    pre code {
        display: block;
        padding: 14px 18px;
        background: #0f172a;
        color: #f8fafc;
        border-radius: 6px;
        overflow-x: auto;
        font-size: 8.5pt;
        line-height: 1.5;
    }
    hr {
        border: 0;
        height: 1px;
        background: #e2e8f0;
        margin: 35px 0;
    }
    .print-bar {
        position: fixed;
        top: 15px;
        right: 25px;
        background: #1e3a8a;
        color: white;
        padding: 8px 18px;
        border-radius: 20px;
        font-family: 'Open Sans', sans-serif;
        font-size: 12px;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        cursor: pointer;
        z-index: 1000;
        transition: transform 0.2s;
    }
    .print-bar:hover {
        transform: translateY(-2px);
        background: #1d4ed8;
    }
    @media print {
        body { background: white; padding: 0; font-size: 10pt; }
        .page-container { box-shadow: none; padding: 0; max-width: 100%; }
        .print-bar { display: none !important; }
        @page { margin: 20mm; size: A4 portrait; }
        h1, h2, h3 { page-break-after: avoid; }
        table, pre, blockquote { page-break-inside: avoid; }
    }
</style>
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
"""


def markdown_to_html(md_text: str) -> str:
    """Simple parser converting markdown headers, bold, code, lists, and tables to HTML."""
    html_lines = []
    lines = md_text.splitlines()
    in_code = False
    code_block = []
    in_table = False
    table_lines = []

    for line in lines:
        # Code fence
        if line.strip().startswith("```"):
            if in_code:
                in_code = False
                escaped_code = "\n".join(code_block).replace("<", "&lt;").replace(">", "&gt;")
                html_lines.append(f"<pre><code>{escaped_code}</code></pre>")
                code_block = []
            else:
                in_code = True
            continue

        if in_code:
            code_block.append(line)
            continue

        # Table detection
        if "|" in line and (line.strip().startswith("|") or line.strip().endswith("|")):
            in_table = True
            table_lines.append(line)
            continue
        elif in_table:
            in_table = False
            # Parse table
            if len(table_lines) >= 2:
                t_html = "<table>"
                headers = [c.strip() for c in table_lines[0].split("|")[1:-1]]
                t_html += "<thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead><tbody>"
                for row in table_lines[2:]:
                    cells = [c.strip() for c in row.split("|")[1:-1]]
                    t_html += "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
                t_html += "</tbody></table>"
                html_lines.append(t_html)
            table_lines = []

        # Empty line
        if not line.strip():
            continue

        # Headings
        if line.startswith("# "):
            html_lines.append(f"<h1>{line[2:].strip()}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{line[3:].strip()}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{line[4:].strip()}</h3>")
        elif line.startswith("#### "):
            html_lines.append(f"<h4>{line[5:].strip()}</h4>")
        elif line.startswith("##### "):
            html_lines.append(f"<h5>{line[6:].strip()}</h5>")
        elif line.strip() == "---":
            html_lines.append("<hr>")
        elif line.startswith("> "):
            html_lines.append(f"<blockquote>{line[2:].strip()}</blockquote>")
        elif line.strip().startswith("- ") or line.strip().startswith("* "):
            content = line.strip()[2:]
            html_lines.append(f"<li>{content}</li>")
        else:
            html_lines.append(f"<p>{line.strip()}</p>")

    # Clean formatting inside lines (bold, italics, code)
    content = "\n".join(html_lines)
    content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
    content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', content)
    content = re.sub(r'`(.*?)`', r'<code>\1</code>', content)
    return content


def compile_all_manuscripts():
    files_to_compile = [
        ("PAPER_DRAFT.md", "PAPER_DRAFT.html", "Journal Paper Manuscript"),
        ("FINAL_YEAR_PROJECT_REPORT.md", "FINAL_YEAR_PROJECT_REPORT.html", "Project Dissertation Report"),
        ("PROJECT_DEFENSE_PRESENTATION.md", "PROJECT_DEFENSE_PRESENTATION.html", "Oral Defense Presentation Deck"),
        ("FUTURE_WORK_ROADMAP.md", "FUTURE_WORK_ROADMAP.html", "Strategic Future Work Roadmap")
    ]

    print("=" * 70)
    print(" COMPILING ACADEMIC MANUSCRIPTS TO PRINT-READY HTML/PDF")
    print("=" * 70)

    for md_name, html_name, title in files_to_compile:
        md_path = os.path.join(MANUSCRIPT_DIR, md_name)
        html_path = os.path.join(MANUSCRIPT_DIR, html_name)

        if not os.path.exists(md_path):
            print(f"Skipping {md_name}: File not found.")
            continue

        with open(md_path, "r", encoding="utf-8") as f:
            md_text = f.read()

        body_html = markdown_to_html(md_text)

        full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    {ACADEMIC_CSS}
</head>
<body>
    <button class="print-bar" onclick="window.print()">🖨️ Print to PDF / A4 Document</button>
    <div class="page-container">
        {body_html}
    </div>
</body>
</html>"""

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(full_html)

        print(f"  [COMPILED] {md_name} -> {html_name} ({len(full_html):,} bytes)")

    print("=" * 70)
    print("SUCCESS: All academic artifacts compiled. Open .html files in any browser to print/save as PDF!\n")


if __name__ == "__main__":
    compile_all_manuscripts()
