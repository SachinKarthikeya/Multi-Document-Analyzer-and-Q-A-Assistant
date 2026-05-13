import pymupdf4llm
import re
import unicodedata
from collections import Counter
import streamlit as st

def extract_and_clean_pdf_text(pdf_path):
    text = pymupdf4llm.to_text(pdf_path)
    
    text = unicodedata.normalize("NFC", text)
    
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'[\u00ad\u200b\u200c\u200d\ufeff]', '', text)
    
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    
    text = re.sub(r'==>.*?<==', '', text)
    
    text = re.sub(r'^\s*\d{1,4}\s*$', '', text, flags=re.MULTILINE)
    
    lines = text.split('\n')
    line_counts = Counter(l.strip() for l in lines if len(l.strip()) > 8)
    repeated_lines = {line for line, count in line_counts.items() if count >= 2}
    lines = [l for l in lines if l.strip() not in repeated_lines]
    text = '\n'.join(lines)
    
    result = []
    for line in text.split('\n'):
        if re.match(r'^\s*\+[-+]+\+\s*$', line):
            continue
        elif re.match(r'^\s*\|', line):
            cells = [c.strip() for c in line.split('|') if c.strip()]
            if cells:
                result.append(' | '.join(cells))
            else:
                result.append(line)
        else:
            result.append(line)
    text = '\n'.join(result)
    
    lines = text.split('\n')
    consolidated_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if (i + 1 < len(lines) and 
            re.match(r'^[A-Z][a-z]+$', line.strip()) and  
            re.match(r'^[A-Z][a-z]+.*?:\s*', lines[i+1].strip())):  
            merged = line.strip() + ' ' + lines[i+1].strip()
            consolidated_lines.append(merged)
            i += 2
        else:
            consolidated_lines.append(line)
            i += 1
    text = '\n'.join(consolidated_lines)
    
    text = re.sub(r'(\| [^\n]+)\n(\([^)]+\))\s*\n', lambda m: m.group(1).rstrip() + ' ' + m.group(2) + '\n', text)
    
    text = re.sub(r'\n([A-Z][a-zA-Z ]{3,})\n(?=[A-Z\w])', r'\n\n\1\n', text)
    
    text = re.sub(r'[ ]{2,}', ' ', text)
    
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    lines = text.split('\n')
    processed_lines = []
    
    for line in lines:
        if '|' in line:
            processed_lines.append(line)
            continue
        
        if line.strip() == '' or re.match(r'^[A-Z][a-zA-Z ]{3,}$', line.strip()):
            processed_lines.append(line)
            continue
        
        pairs = re.findall(r'([A-Z][a-zA-Z0-9 ]+?): ([^:]+?)(?=(?:[A-Z][a-zA-Z0-9 ]+?:|$))', line)
        
        if pairs and len(pairs) > 1:
            for label, value in pairs:
                processed_lines.append(f"{label}: {value.strip()}")
        else:
            processed_lines.append(line)
    
    text = '\n'.join(processed_lines)
    
    text = text.strip()
    
    return text

if __name__ == "__main__":
    st.title("PDF Text Extractor")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file is not None:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        extracted_text = extract_and_clean_pdf_text("temp.pdf")
        st.text_area("Extracted Text", extracted_text, height=400)