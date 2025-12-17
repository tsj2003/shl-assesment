import pandas as pd
import pypdf
import os

def read_pdf(path):
    print(f"--- XML/Text content of {path} ---")
    try:
        reader = pypdf.PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        print(text[:2000]) # First 2000 chars
        if "http" in text:
            print("\n--- URLs found in PDF ---")
            words = text.split()
            urls = [w for w in words if "http" in w]
            print(urls)
    except Exception as e:
        print(f"Error reading PDF: {e}")

def read_excel(path):
    print(f"\n--- Content of {path} ---")
    try:
        df = pd.read_excel(path, sheet_name=None)
        for sheet, data in df.items():
            print(f"Sheet: {sheet}")
            print(data.head())
            print(data.columns)
    except Exception as e:
        print(f"Error reading Excel: {e}")

if __name__ == "__main__":
    pdf_path = "/Users/tarandeepsinghjuneja/Downloads/SHL AI Intern RE Generative AI assignment.pdf"
    xlsx_path = "/Users/tarandeepsinghjuneja/Downloads/Gen_AI Dataset.xlsx"
    
    if os.path.exists(pdf_path):
        read_pdf(pdf_path)
    else:
        print(f"File not found: {pdf_path}")
        
    if os.path.exists(xlsx_path):
        read_excel(xlsx_path)
    else:
        print(f"File not found: {xlsx_path}")
