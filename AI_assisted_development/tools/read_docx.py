import sys
import zipfile
import xml.etree.ElementTree as ET


def extract_docx_text(path):
    with zipfile.ZipFile(path) as z:
        try:
            xml = z.read('word/document.xml')
        except KeyError:
            print('Error: document.xml not found in the .docx file', file=sys.stderr)
            return None
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    root = ET.fromstring(xml)
    paragraphs = []
    for p in root.findall('.//w:p', ns):
        texts = [t.text for t in p.findall('.//w:t', ns) if t.text]
        if texts:
            paragraphs.append(''.join(texts))
    return '\n\n'.join(paragraphs)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python read_docx.py <path/to/file.docx>', file=sys.stderr)
        sys.exit(2)
    path = sys.argv[1]
    text = extract_docx_text(path)
    if text is None:
        sys.exit(1)
    print(text)
