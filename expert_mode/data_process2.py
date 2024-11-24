import os
from pdfminer.high_level import extract_text
import jieba
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'



def extract_pdf_text(pdf_path):
    return extract_text(pdf_path)



def segment_text(text):
    words = jieba.lcut(text)
    custom_words = set()
    for word in words:
        if any(char.isalpha() for char in word):
            custom_words.add(word)
    for word in custom_words:
        jieba.add_word(word)
    return ' '.join(words)



def extract_keywords_tfidf(texts, top_n=10):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
    sorted_indices = np.argsort(tfidf_scores)[::-1]
    top_keywords = [feature_names[i] for i in sorted_indices[:top_n]]
    return top_keywords



def evaluate_keywords_coverage(paper_keywords, description_keywords, model, tokenizer):
    def get_embedding(text):
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

    paper_vectors = np.array(
        [get_embedding(word) for word in paper_keywords if len(tokenizer.encode(word, add_special_tokens=False)) > 0])
    description_vectors = np.array([get_embedding(word) for word in description_keywords if
                                    len(tokenizer.encode(word, add_special_tokens=False)) > 0])

    if paper_vectors.ndim != 2 or description_vectors.ndim != 2:
        return 0.0

    similarity_matrix = cosine_similarity(paper_vectors, description_vectors)
    coverage = np.mean(np.max(similarity_matrix, axis=1))

    return round(coverage * 10, 2)


def process_folder(folder_path, description_keywords, bert_model, bert_tokenizer):
    results = {}
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    for pdf_file in pdf_files:
        try:
            pdf_path = os.path.join(folder_path, pdf_file)
            paper_text = extract_pdf_text(pdf_path)
            segmented_text = segment_text(paper_text)

            paper_keywords = extract_keywords_tfidf([segmented_text])
            score = evaluate_keywords_coverage(paper_keywords, description_keywords, bert_model,
                                               bert_tokenizer)
            results[pdf_file] = {'score': score, 'paper_keywords': paper_keywords}
        except Exception as e:
            print(f"Failed to process {pdf_file}: {str(e)}")
            results[pdf_file] = {'score': 0, 'paper_keywords': []}
    return results


def process_data(folder_path, text_path):
    with open(text_path, 'r', encoding='utf-8') as file:
        text = file.read()


    description_text_segmented = segment_text(text)
    description_keywords = extract_keywords_tfidf([description_text_segmented])
    print(f"Description Keywords: {description_keywords}")


    bert_model = BertModel.from_pretrained('bert-base-chinese')
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    data = process_folder(folder_path, description_keywords, bert_model, bert_tokenizer)
    return data



folder_path = "./paper_file"
text_path = "./description.txt"


data = process_data(folder_path, text_path)


scores_df = pd.read_csv('./lunwen_scores1.csv')


new_scores = []
for file in scores_df['filename']:
    if file in data:
        new_scores.append(data[file]['score'])
    else:
        new_scores.append(0)

scores_df['new_score'] = new_scores
scores_df['total_score'] = scores_df['total_score'] + scores_df['new_score']


scores_df.to_csv('./lunwen_scores1.csv', index=False)


for file, info in data.items():
    print(f"File: {file}, Score: {info['score']}, Keywords: {info['paper_keywords']}")

print(scores_df.head())
