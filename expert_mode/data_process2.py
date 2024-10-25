import os
from pdfminer.high_level import extract_text
import jieba
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 设置环境变量以避免冲突


# 从PDF文件中提取文本的函数
def extract_pdf_text(pdf_path):
    return extract_text(pdf_path)


# 分词并更新jieba词库的函数
def segment_text(text):
    words = jieba.lcut(text)
    custom_words = set()
    for word in words:
        if any(char.isalpha() for char in word):  # 检查是否包含英文字符
            custom_words.add(word)
    for word in custom_words:
        jieba.add_word(word)
    return ' '.join(words)


# 使用TF-IDF提取关键词的函数
def extract_keywords_tfidf(texts, top_n=10):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
    sorted_indices = np.argsort(tfidf_scores)[::-1]
    top_keywords = [feature_names[i] for i in sorted_indices[:top_n]]
    return top_keywords


# 使用BERT模型计算关键词覆盖率的函数
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

    return round(coverage * 10, 2)  # 扩展评分范围为0-10


# 处理文件夹中的所有PDF文件的函数
def process_folder(folder_path, description_keywords, bert_model, bert_tokenizer):
    results = {}
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]  # 获取所有PDF文件
    for pdf_file in pdf_files:
        try:
            pdf_path = os.path.join(folder_path, pdf_file)
            paper_text = extract_pdf_text(pdf_path)  # 提取PDF文本
            segmented_text = segment_text(paper_text)  # 分词处理

            paper_keywords = extract_keywords_tfidf([segmented_text])
            score = evaluate_keywords_coverage(paper_keywords, description_keywords, bert_model,
                                               bert_tokenizer)  # 评估关键词覆盖率
            results[pdf_file] = {'score': score, 'paper_keywords': paper_keywords}
        except Exception as e:
            print(f"Failed to process {pdf_file}: {str(e)}")
            results[pdf_file] = {'score': 0, 'paper_keywords': []}  # 出现错误时，分数赋值为0
    return results


# 处理数据并以字典形式返回的函数
def process_data(folder_path, text_path):
    with open(text_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # 提取题目描述中的关键词并打印
    description_text_segmented = segment_text(text)
    description_keywords = extract_keywords_tfidf([description_text_segmented])
    print(f"Description Keywords: {description_keywords}")

    # 加载BERT模型和分词器
    bert_model = BertModel.from_pretrained('bert-base-chinese')
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    data = process_folder(folder_path, description_keywords, bert_model, bert_tokenizer)
    return data


# 文件夹路径和题目描述文件路径
folder_path = "E:\\Thesis project\\agnet_example\\paper_file"
text_path = "E:\\Thesis project\\agnet_example\\description.txt"

# 处理数据
data = process_data(folder_path, text_path)

# 读取已有评分数据
scores_df = pd.read_csv('E:\\Thesis project\\agnet_example\\lunwen_scores1.csv')

# 将计算的得分合并到已有的评分数据中
new_scores = []
for file in scores_df['filename']:
    if file in data:
        new_scores.append(data[file]['score'])
    else:
        new_scores.append(0)  # 如果文件未处理成功，默认得分为0

scores_df['new_score'] = new_scores
scores_df['total_score'] = scores_df['total_score'] + scores_df['new_score']

# 保存更新后的评分数据
scores_df.to_csv('E:\\Thesis project\\agnet_example\\lunwen_scores1.csv', index=False)

# 打印处理结果
for file, info in data.items():
    print(f"File: {file}, Score: {info['score']}, Keywords: {info['paper_keywords']}")

# 保存结果示例
print(scores_df.head())
