import os
import csv
import re
from pypdf import PdfReader
import pandas as pd


def get_pdf_paths_from_folder(folder_path):
    pdf_paths = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):  # Check if the file is a PDF
            pdf_paths.append(os.path.join(folder_path, filename))
    return pdf_paths


def get_labels_from_csv(csv_path):
    labels = []
    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            labels.append(row[1])  # row[1] is the ModifiedLevel
    return labels


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text


def clean_text(text):
   
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？；：、（）【】《》]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()  # Clean up whitespace
    return text


def segment_text(text):
    sentences = re.split(r'(?<=[。！？]) +', text)
    return sentences


def process_pdf(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    text = clean_text(text)
    text = ' '.join(segment_text(text))
    return text


def create_dataset(pdf_folder_path, csv_path, output_csv_path):
    
    pdf_paths = get_pdf_paths_from_folder(pdf_folder_path)

    
    labels = get_labels_from_csv(csv_path)

    
    if len(pdf_paths) != len(labels):
        raise ValueError("The number of PDF files and labels do not match.")

    
    data = []
    for pdf_path, label in zip(pdf_paths, labels):
        text = process_pdf(pdf_path)  
        data.append({'text': text, 'label': label})

    
    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')  


def main():
    
    pdf_folder_path = '../../dataset/TDBSW'
    csv_path = '../../result/AES_labels.csv'
    output_csv_path = 'dataset.csv'

    
    create_dataset(pdf_folder_path, csv_path, output_csv_path)

if __name__ == "__main__":
    main()
