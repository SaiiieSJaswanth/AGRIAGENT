import os
import re
import zipfile
import shutil

def parse_txt_to_html(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()

    lines = text.split('\n')
    
    html_lines = []
    html_lines.append("""<!DOCTYPE html>
<html xmlns:o='urn:schemas-microsoft-com:office:office' xmlns:w='urn:schemas-microsoft-com:office:word' xmlns='http://www.w3.org/TR/REC-html40'>
<head>
<meta charset="utf-8">
<title>AIML CASESTUDY REPORT</title>
<!--[if gte mso 9]>
<xml>
<w:WordDocument>
<w:View>Print</w:View>
<w:Zoom>100</w:Zoom>
<w:DoNotOptimizeForBrowser/>
</w:WordDocument>
</xml>
<![endif]-->
<style>
@page Section1 {
    size: 8.5in 11.0in;
    margin: 1.0in 1.0in 1.0in 1.0in;
    mso-header-margin: .5in;
    mso-footer-margin: .5in;
    mso-paper-source: 0;
}
div.Section1 {
    page: Section1;
}
body {
    font-family: 'Times New Roman', Times, serif;
    font-size: 12pt;
    line-height: 1.5;
    color: #111111;
    background-color: #ffffff;
    margin: 40px auto;
    max-width: 900px;
}
h1 {
    font-family: 'Times New Roman', Times, serif;
    font-size: 16pt;
    font-weight: bold;
    text-align: center;
    margin-top: 24pt;
    margin-bottom: 6pt;
    text-transform: uppercase;
}
h2 {
    font-family: 'Times New Roman', Times, serif;
    font-size: 13pt;
    font-weight: bold;
    text-align: center;
    margin-top: 0pt;
    margin-bottom: 24pt;
}
h3 {
    font-family: 'Times New Roman', Times, serif;
    font-size: 14pt;
    font-weight: bold;
    margin-top: 18pt;
    margin-bottom: 8pt;
    border-bottom: 1.5px solid #333333;
    padding-bottom: 4px;
    page-break-after: avoid;
}
h4 {
    font-family: 'Times New Roman', Times, serif;
    font-size: 12.5pt;
    font-weight: bold;
    margin-top: 14pt;
    margin-bottom: 6pt;
    page-break-after: avoid;
}
p {
    font-family: 'Times New Roman', Times, serif;
    font-size: 12pt;
    line-height: 1.5;
    margin-top: 0pt;
    margin-bottom: 8pt;
    text-align: justify;
}
ul, ol {
    margin-top: 0pt;
    margin-bottom: 8pt;
    padding-left: 28px;
}
li {
    font-family: 'Times New Roman', Times, serif;
    font-size: 12pt;
    line-height: 1.45;
    margin-bottom: 4pt;
    text-align: justify;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin-top: 10pt;
    margin-bottom: 14pt;
    font-family: 'Times New Roman', Times, serif;
    page-break-inside: avoid;
}
th, td {
    border: 1px solid #333333;
    padding: 6pt 8pt;
    text-align: left;
    vertical-align: top;
    font-size: 11pt;
    line-height: 1.35;
}
th {
    background-color: #f2f2f2;
    font-weight: bold;
}
pre {
    font-family: 'Courier New', Courier, monospace;
    font-size: 9.5pt;
    line-height: 1.25;
    background-color: #fbfbfb;
    border: 1px solid #cccccc;
    padding: 10pt;
    margin-top: 8pt;
    margin-bottom: 12pt;
    white-space: pre;
    overflow-x: auto;
}
hr {
    border: none;
    border-top: 1px solid #bbbbbb;
    margin: 18pt 0;
}
.table-title {
    font-weight: bold;
    font-size: 11pt;
    margin-bottom: 4pt;
    margin-top: 10pt;
}
</style>
</head>
<body>
<div class="Section1">
""")

    in_table = False
    in_pre = False
    pre_content = []
    table_rows = []
    
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].rstrip()
        
        # Check pre/code block
        if line.startswith('```'):
            if in_pre:
                html_lines.append('<pre>' + '\n'.join(pre_content) + '</pre>')
                pre_content = []
                in_pre = False
            else:
                in_pre = True
            i += 1
            continue
            
        if in_pre:
            pre_content.append(line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))
            i += 1
            continue
            
        # Check table
        if line.strip().startswith('|') and line.strip().endswith('|'):
            if not in_table:
                in_table = True
                table_rows = []
            table_rows.append(line)
            i += 1
            continue
        elif in_table:
            in_table = False
            html_table = render_markdown_table(table_rows)
            html_lines.append(html_table)
            table_rows = []
            
        # Header / Title checks
        if line == "AIML CASESTUDY REPORT":
            html_lines.append('<h1>AIML CASESTUDY REPORT</h1>')
            i += 1
            continue
        if line.startswith("TITLE: "):
            html_lines.append(f'<h2>{line}</h2>')
            i += 1
            continue
        if line.startswith("--------------------------------------------------------------------------------") or line.startswith("================================================================================"):
            html_lines.append('<hr>')
            i += 1
            continue
        if line.startswith("---"):
            html_lines.append('<hr style="border-top: 1px dashed #cccccc; margin: 12pt 0;">')
            i += 1
            continue
            
        m_sec = re.match(r'^(\d+\.\s+[A-Z\s&/]+)$', line.strip())
        if m_sec:
            html_lines.append(f'<h3>{m_sec.group(1)}</h3>')
            i += 1
            continue
            
        if line.startswith("Paper ") or line.startswith("Dataset ") or line.startswith("Module ") or line.startswith("Quantitative Machine Learning Evaluation:") or line.startswith("Discussion of Model Results:") or line.startswith("Diagnostic Visualizations:") or line.startswith("Automated Test Suite Verification:"):
            html_lines.append(f'<h4>{line}</h4>')
            i += 1
            continue
            
        if line.startswith("Table "):
            html_lines.append(f'<p class="table-title"><b>{line}</b></p>')
            i += 1
            continue
            
        if line.strip().startswith('- ') or line.strip().startswith('* '):
            item_text = line.strip()[2:]
            item_text = format_inline_html(item_text)
            html_lines.append(f'<ul><li>{item_text}</li></ul>')
            i += 1
            continue
            
        m_ref = re.match(r'^(\[\d+\])\s+(.*)$', line.strip())
        if m_ref:
            ref_id = m_ref.group(1)
            ref_body = format_inline_html(m_ref.group(2))
            html_lines.append(f'<p style="margin-left: 24pt; text-indent: -24pt;"><b>{ref_id}</b> {ref_body}</p>')
            i += 1
            continue
            
        m_num = re.match(r'^(\d+\.)\s+(.*)$', line.strip())
        if m_num and not m_sec:
            num_id = m_num.group(1)
            num_body = format_inline_html(m_num.group(2))
            html_lines.append(f'<p style="margin-left: 20pt; text-indent: -20pt;"><b>{num_id}</b> {num_body}</p>')
            i += 1
            continue
            
        if line.strip() == "":
            i += 1
            continue
            
        p_text = format_inline_html(line)
        html_lines.append(f'<p>{p_text}</p>')
        i += 1

    if in_table and table_rows:
        html_table = render_markdown_table(table_rows)
        html_lines.append(html_table)
        
    html_lines.append("""
</div>
</body>
</html>
""")
    return '\n'.join(html_lines)

def format_inline_html(text):
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
    text = re.sub(r'\$(.*?)\$', r'<i>\1</i>', text)
    return text

def render_markdown_table(rows):
    if not rows:
        return ""
    
    table_html = ['<table>']
    header_cells = [c.strip() for c in rows[0].strip().split('|')[1:-1]]
    table_html.append('<thead><tr>')
    for cell in header_cells:
        cell_fmt = format_inline_html(cell).replace('&lt;br&gt;', '<br>')
        table_html.append(f'<th>{cell_fmt}</th>')
    table_html.append('</tr></thead>')
    
    table_html.append('<tbody>')
    start_idx = 1
    if len(rows) > 1 and '---' in rows[1]:
        start_idx = 2
        
    for r in rows[start_idx:]:
        cells = [c.strip() for c in r.strip().split('|')[1:-1]]
        table_html.append('<tr>')
        for cell in cells:
            cell_fmt = format_inline_html(cell).replace('&lt;br&gt;', '<br>')
            table_html.append(f'<td>{cell_fmt}</td>')
        table_html.append('</tr>')
        
    table_html.append('</tbody></table>')
    return '\n'.join(table_html)

def make_zip(source_dir, zip_dest):
    print(f"Creating ZIP archive at {zip_dest} from {source_dir}...")
    parent_dir = os.path.dirname(source_dir)
    with zipfile.ZipFile(zip_dest, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(source_dir):
            dirs[:] = [d for d in dirs if d not in ('__pycache__', '.pytest_cache', '.git')]
            for file in files:
                if file.endswith('.pyc'):
                    continue
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, parent_dir)
                zf.write(full_path, rel_path)
    size_mb = os.path.getsize(zip_dest) / (1024 * 1024)
    print(f"Archive saved: {zip_dest} ({size_mb:.2f} MB)")

def main():
    root_dir = r"D:\AIML_CASESTUDY\AGRIAGENT_CROP_ML_HYBRID"
    txt_path = os.path.join(root_dir, "AIML_CASESTUDY_REPORT.txt")
    html_path = os.path.join(root_dir, "AIML_CASESTUDY_REPORT.html")
    doc_path = os.path.join(root_dir, "AIML_CASESTUDY_REPORT.doc")
    
    print(f"Reading text report from {txt_path}...")
    html_content = parse_txt_to_html(txt_path)
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Saved HTML report to {html_path}")
    
    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Saved Word-compatible DOC report to {doc_path}")
    
    # Create both ZIPs so any file name the user expects is ready
    make_zip(root_dir, r"D:\AIML_CASESTUDY\AGRIAGENT_FINAL_COMPLETE.zip")
    make_zip(root_dir, r"D:\AIML_CASESTUDY\AGRIAGENT_CROP_ML_HYBRID_WITH_HIDDEN_CROP_ML.zip")

if __name__ == '__main__':
    main()
