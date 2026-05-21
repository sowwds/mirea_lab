from __future__ import annotations

import html
import re
from pathlib import Path


AUTHORS = "Ефремов А.И., Лазарев Г.С., Никитин А.В."
TEACHER = "Холмогоров В.В."


def inline_md(text: str) -> str:
    text = text.replace(" - ", " — ")
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    return text


def parse_table(lines: list[str], start: int) -> tuple[str, int]:
    table_lines = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        table_lines.append(lines[i].strip())
        i += 1

    rows = []
    for idx, line in enumerate(table_lines):
        cells = [c.strip() for c in line.strip("|").split("|")]
        if idx == 1 and all(set(c) <= {"-", ":"} for c in cells):
            continue
        tag = "th" if idx == 0 else "td"
        rows.append("<tr>" + "".join(f"<{tag}>{inline_md(c)}</{tag}>" for c in cells) + "</tr>")

    return "<table>\n" + "\n".join(rows) + "\n</table>", i


def md_to_html(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    in_code = False
    code_lines: list[str] = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            if not in_code:
                in_code = True
                code_lines = []
            else:
                out.append("<pre><code>" + html.escape("\n".join(code_lines)) + "</code></pre>")
                in_code = False
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        if not stripped:
            i += 1
            continue

        if stripped.startswith("|"):
            table_html, i = parse_table(lines, i)
            out.append(table_html)
            continue

        image_match = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
        if image_match:
            caption, src = image_match.groups()
            caption = caption.replace(" - ", " — ")
            out.append(
                '<div class="figure">'
                f'<img src="{html.escape(src)}" alt="{html.escape(caption)}">'
                f'<p class="caption">{inline_md(caption)}</p>'
                "</div>"
            )
            i += 1
            continue

        heading_match = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            text = heading_match.group(2)
            main_heading = (
                text.upper() in {
                    "СОДЕРЖАНИЕ",
                    "ВВЕДЕНИЕ",
                    "ЗАКЛЮЧЕНИЕ",
                    "СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ",
                    "ПРИЛОЖЕНИЕ А",
                }
                or re.match(r"^\d+\.\s+", text) is not None
                or re.match(r"^\d+\s+", text) is not None
            )
            if level == 1 or (level == 2 and main_heading):
                out.append(f"<h1>{inline_md(text)}</h1>")
            elif level == 2 or level == 3:
                out.append(f"<h2>{inline_md(text)}</h2>")
            else:
                out.append(f"<h3>{inline_md(text)}</h3>")
            i += 1
            continue

        if stripped.startswith("- "):
            items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append("<li>" + inline_md(lines[i].strip()[2:]) + "</li>")
                i += 1
            out.append("<ul>\n" + "\n".join(items) + "\n</ul>")
            continue

        if re.match(r"^\d+\.\s+", stripped):
            items = []
            while i < len(lines) and re.match(r"^\d+\.\s+", lines[i].strip()):
                item = re.sub(r"^\d+\.\s+", "", lines[i].strip())
                items.append("<li>" + inline_md(item) + "</li>")
                i += 1
            out.append("<ol>\n" + "\n".join(items) + "\n</ol>")
            continue

        paragraph = [stripped]
        i += 1
        while (
            i < len(lines)
            and lines[i].strip()
            and not lines[i].strip().startswith(("#", "|", "- ", "```", "!["))
            and not re.match(r"^\d+\.\s+", lines[i].strip())
        ):
            paragraph.append(lines[i].strip())
            i += 1
        out.append("<p>" + inline_md(" ".join(paragraph)) + "</p>")

    return "\n".join(out)


def extract_lab_number(md: str, fallback: str) -> str:
    match = re.search(r"Практическая работа №\s*(\d+)", md)
    return match.group(1) if match else fallback


def extract_topic(md: str) -> str:
    match = re.search(r"\*\*Тема:\*\*\s*(.+)", md)
    return match.group(1).strip() if match else "Практическая работа"


def strip_md_header(md: str) -> str:
    lines = md.splitlines()
    start = 0
    for idx, line in enumerate(lines):
        if line.startswith("## Содержание"):
            start = idx
            break
    return "\n".join(lines[start:])


def render(md_path: Path, html_path: Path, fallback_lab_number: str) -> None:
    md = md_path.read_text(encoding="utf-8")
    lab_number = extract_lab_number(md, fallback_lab_number)
    topic = extract_topic(md)
    body = md_to_html(strip_md_header(md))

    page = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Практическая работа №{lab_number}</title>
  <style>
    @page {{ size: A4; margin: 20mm 10mm 20mm 30mm; }}
    body {{
      font-family: "Times New Roman", serif;
      font-size: 14pt;
      line-height: 1.5;
      color: #111;
    }}
    h1, h2, h3, h4 {{
      font-family: "Times New Roman", serif;
      font-weight: bold;
      page-break-after: avoid;
    }}
    h1 {{
      font-size: 18pt;
      text-align: center;
      text-transform: uppercase;
      margin: 0 0 12pt 0;
      page-break-before: always;
    }}
    .title-page h1 {{ page-break-before: auto; text-transform: none; }}
    h2 {{
      font-size: 16pt;
      text-align: justify;
      margin: 24pt 0 12pt 0;
    }}
    h3 {{
      font-size: 14pt;
      text-align: justify;
      margin: 24pt 0 12pt 0;
    }}
    p {{
      text-align: justify;
      text-indent: 1.25cm;
      margin: 0;
    }}
    ul, ol {{
      margin-top: 0;
      margin-bottom: 0;
      padding-left: 2.25cm;
    }}
    li {{
      margin: 0;
      text-align: justify;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 10pt 0;
      font-size: 12pt;
    }}
    th, td {{ border: 1px solid #333; padding: 5px 7px; vertical-align: top; }}
    th {{ font-weight: bold; text-align: center; }}
    code {{
      font-family: "Courier New", monospace;
      font-size: 11pt;
    }}
    pre {{
      font-family: "Courier New", monospace;
      font-size: 9pt;
      white-space: pre-wrap;
      border: 1px solid #aaa;
      padding: 9px;
      background: #f7f7f7;
    }}
    .title-page {{
      text-align: center;
      page-break-after: always;
    }}
    .title-page p {{
      text-align: center;
      text-indent: 0;
      margin: 0;
      line-height: 1.5;
    }}
    .title-spacer-large {{ height: 120pt; }}
    .title-spacer-medium {{ height: 60pt; }}
    .right-block {{
      text-align: right;
      margin-left: 45%;
    }}
    .right-block p {{ text-align: right; text-indent: 0; }}
    .figure {{
      text-align: center;
      margin: 10pt 0;
      page-break-inside: avoid;
    }}
    .figure img {{
      max-width: 100%;
      height: auto;
    }}
    .caption {{
      text-align: center;
      text-indent: 0;
      font-style: italic;
      font-size: 12pt;
    }}
    .toc p {{ text-indent: 0; }}
    .no-indent {{ text-indent: 0; }}
  </style>
</head>
<body>
  <section class="title-page">
    <div>
      <p style="text-align: center; text-indent: 0;">МИНОБРНАУКИ РОССИИ</p>
      <p style="text-align: center; text-indent: 0;">Федеральное государственное бюджетное образовательное учреждение высшего образования «МИРЭА - Российский технологический университет»</p>
      <p style="text-align: center; text-indent: 0;">РТУ МИРЭА</p>
    </div>
    <div class="title-spacer-large"></div>
    <div>
      <p style="text-align: center; text-indent: 0;">Практическая работа {lab_number}</p>
      <p style="text-align: center; text-indent: 0;">Тема: {html.escape(topic)}</p>
    </div>
    <div class="title-spacer-medium"></div>
    <div class="right-block">
      <p style="text-align: right; text-indent: 0;">Выполнили: {AUTHORS}</p>
      <p style="text-align: right; text-indent: 0;">Принял: {TEACHER}</p>
    </div>
    <div class="title-spacer-large"></div>
    <div>
      <p style="text-align: center; text-indent: 0;">Москва</p>
      <p style="text-align: center; text-indent: 0;">2026 г.</p>
    </div>
  </section>
  {body}
</body>
</html>
"""
    html_path.write_text(page, encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    render(root / "reports" / "lab1" / "report_lab1.md", root / "reports" / "lab1" / "report_lab1_final.html", "1")
    render(root / "reports" / "lab2" / "report_lab2.md", root / "reports" / "lab2" / "report_lab2_final.html", "2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
