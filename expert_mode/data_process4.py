import re
import os
from pdfminer.high_level import extract_text

from fuzzywuzzy import fuzz
import spacy
import numpy as np
import pandas as pd
from tqdm import tqdm
from pdf2image import convert_from_path

nlp = spacy.load('zh_core_web_sm')


# Function to extract text from PDF
def extract_pdf_text(pdf_path):
    return extract_text(pdf_path)

def get_pdf_pages(file_path):
    try:
        images = convert_from_path(file_path)
        pages = len(images)
    except Exception as e:
        print(f"Error reading PDF file with pdf2image: {e}")
        pages = 0
    return pages



# Function to recognize paper structure and return matched titles
def recognize_structure(text, titles):
    recognized_titles = []
    for title in titles:
        for line in text.split('\n'):
            if fuzz.partial_ratio(title, line) > 80:  # Matching threshold
                recognized_titles.append(title)
                break
    return list(set(recognized_titles))  # Remove duplicates


# Dynamic weighted scoring mechanism
def evaluate_structure(recognized_titles, expected_titles):
    weights = {
        "摘要": 1.5, "目录": 1, "问题重述": 1, "假设条件": 1,
        "符号说明": 1, "模型建立": 1.5, "模型求解": 1.5,
        "模型检验": 1.5, "结果分析": 1.5, "结论": 2,
        "参考文献": 2, "附录": 1
    }
    total_weight = sum(weights.values())
    initial_score = sum(weights.get(title, 0) for title in recognized_titles[:5])

    dynamic_score = initial_score
    remaining_titles = recognized_titles[5:]
    for idx, title in enumerate(remaining_titles):
        if title in weights:
            adjusted_weight = weights[title] * (1 / (1 + 0.1 * (idx + 1)))
            dynamic_score += adjusted_weight

    final_score = (dynamic_score / total_weight) * 10
    return final_score


# Process all PDF files in a folder and return their scores
def process_pdf_folder(folder_path, expected_titles):
    results = []
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            pdf_path = os.path.join(folder_path, pdf_file)
            text = extract_pdf_text(pdf_path)
            recognized_titles = recognize_structure(text, expected_titles)
            score = evaluate_structure(recognized_titles, expected_titles)
            results.append((pdf_file, text, score))
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
            results.append((pdf_file, "", 0))
    return results


# Function to read papers from folder
def read_papers_from_folder(folder_path):
    paper_texts = []
    for filename in tqdm(os.listdir(folder_path), desc="Reading papers"):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            text = extract_text(file_path)
            text = text.replace('\n', ' ')
            paper_texts.append((filename, text))
    return paper_texts


# Split paper into sections based on keywords
def split_paper_into_sections(text):
    sections = {
        'introduction': '',
        'literature_review': '',
        'method': '',
        'results': '',
        'discussion': '',
        'conclusion': ''
    }
    introduction_pattern = re.compile(r'引言|背景|研究背景|Introduction|Background|目标|目的', re.IGNORECASE)
    literature_review_pattern = re.compile(
        r'文献综述|相关研究|研究现状|Literature Review|Related Work|Previous Work|Review of the Literature|研究贡献|研究空白|研究结果综述|研究方法综述|文献|文献回顾',
        re.IGNORECASE)
    method_pattern = re.compile(r'方法|研究方法|实验方法|Method|Methodology|Experimental Methods|Research Methods|研究问题|研究设计|技术路线|数据来源',
                                re.IGNORECASE)
    results_pattern = re.compile(r'结果|研究结果|实验结果|Results|Findings|研究发现', re.IGNORECASE)
    discussion_pattern = re.compile(r'讨论|结果讨论|分析与讨论|Discussion|Analysis|讨论与分析|讨论与启示', re.IGNORECASE)
    conclusion_pattern = re.compile(r'结论|总结|结语|Conclusion|Summary|Closing Remarks|结论与建议', re.IGNORECASE)

    introduction_start = introduction_pattern.search(text)
    literature_review_start = literature_review_pattern.search(text)
    method_start = method_pattern.search(text)
    results_start = results_pattern.search(text)
    discussion_start = discussion_pattern.search(text)
    conclusion_start = conclusion_pattern.search(text)

    if introduction_start:
        sections['introduction'] = text[
                                   introduction_start.start():literature_review_start.start()] if literature_review_start else text[
                                                                                                                               introduction_start.start():]
    if literature_review_start:
        sections['literature_review'] = text[
                                        literature_review_start.start():method_start.start()] if method_start else text[
                                                                                                                   literature_review_start.start():]
    if method_start:
        sections['method'] = text[method_start.start():results_start.start()] if results_start else text[
                                                                                                    method_start.start():]
    if results_start:
        sections['results'] = text[results_start.start():discussion_start.start()] if discussion_start else text[
                                                                                                            results_start.start():]
    if discussion_start:
        sections['discussion'] = text[discussion_start.start():conclusion_start.start()] if conclusion_start else text[
                                                                                                                  discussion_start.start():]
    if conclusion_start:
        sections['conclusion'] = text[conclusion_start.start():]

    return sections


# Define individual section evaluation functions
def evaluate_introduction(introduction):
    score = 0
    if '背景' in introduction and '研究意义' in introduction:
        score += 2
    if len(introduction) > 500:
        score += 2
    if '研究问题' in introduction:
        score += 2
    if '目标' in introduction or '目的' in introduction:
        score += 2
    if '方法' in introduction:
        score += 2
    if '结果' in introduction:
        score += 2
    return min(10, score)


def evaluate_literature_review(literature_review):
    score = 0
    if '相关研究' in literature_review and '研究现状' in literature_review:
        score += 2
    if len(literature_review) > 1000:
        score += 2
    if '文献综述' in literature_review:
        score += 2
    if '研究方法综述' in literature_review:
        score += 2
    if '研究结果综述' in literature_review:
        score += 2
    if '研究空白' in literature_review:
        score += 2
    if '研究贡献' in literature_review:
        score += 2
    if '文献回顾' in literature_review:
        score += 2
    return min(10, score)


def evaluate_method(method):
    score = 0
    if '研究方法' in method and '数据来源' in method:
        score += 2
    if len(method) > 800:
        score += 2
    if '数据收集' in method:
        score += 2
    if '数据分析' in method:
        score += 2
    if '研究设计' in method:
        score += 2
    if '变量' in method:
        score += 2
    if '技术路线' in method:
        score += 2
    return min(10, score)


def evaluate_results(results):
    score = 0
    if '研究结果' in results and '数据分析' in results:
        score += 2
    if len(results) > 800:
        score += 2
    if '数据展示' in results:
        score += 2
    if '图表' in results:
        score += 2
    if '统计分析' in results:
        score += 2
    if '假设检验' in results:
        score += 2
    if '研究发现' in results:
        score += 2
    return min(10, score)


def evaluate_discussion(discussion):
    score = 0
    if '结果讨论' in discussion or '研究意义' in discussion:
        score += 2
    if len(discussion) > 500:
        score += 2
    if '研究局限' in discussion:
        score += 2
    if '未来研究' in discussion:
        score += 2
    if '实践意义' in discussion:
        score += 2
    if '讨论与分析' in discussion:
        score += 2
    if '讨论与启示' in discussion:
        score += 2
    return min(10, score)


def evaluate_conclusion(conclusion):
    score = 0
    if '研究结论' in conclusion or '未来研究' in conclusion:
        score += 2
    if len(conclusion) > 300:
        score += 2
    if '研究贡献' in conclusion:
        score += 2
    if '研究意义' in conclusion:
        score += 2
    if '政策建议' in conclusion:
        score += 2
    if '结论与建议' in conclusion:
        score += 2
    return min(10, score)


def evaluate_overall_structure(paper):
    score = 0
    sections = ['introduction', 'literature_review', 'method', 'results', 'discussion', 'conclusion']
    for section in sections:
        if paper[section]:
            score += 1
    return score


def evaluate_references(text):
    references_pattern = re.compile(r'\[\d+\][^\[\]\s]+')
    references = references_pattern.findall(text)
    count = len(references)
    references_score = min(10, count)
    return references_score


import re

def evaluate_formulas(text):

    formulas_pattern = re.compile(
        r'\(\d+\)|公式|下式|如公式|如下式|'  
        r'\b\d+\s*[\+\-\*/÷=]\s*\d+\b|'  
        r'\b(?:sin|cos|tan|log|exp|sqrt)\b|'  
        r'\b\d+\s*\^\s*\d+\b|'  
        r'∫|∑|∏'
    )


    formulas = formulas_pattern.findall(text)

    formulas_count = len(formulas)

    formulas_score = min(10, formulas_count)
    return formulas_score

def evaluate_images_and_tables(text):
    images_pattern = re.compile(r'图\s*(\d+|一|二|三|四|五|六|七|八|九|十)')
    tables_pattern = re.compile(r'表\s*(\d+|一|二|三|四|五|六|七|八|九|十)')

    images = images_pattern.findall(text)
    tables = tables_pattern.findall(text)

    images_count = min(7, len(images))
    tables_count = min(3, len(tables))

    images_score = images_count
    tables_score = tables_count

    total_score = min(10, images_score + tables_score)

    return total_score


def evaluate_text_flow(text):
    doc = nlp(text)
    sentences = list(doc.sents)
    num_sentences = len(sentences)

    flow_score = 0
    sentence_lengths = []
    complex_sentences = 0
    grammar_errors = 0

    for sentence in sentences:
        length = len(sentence.text)
        sentence_lengths.append(length)

        if len(list(sentence.root.subtree)) > 1:
            complex_sentences += 1

        for token in sentence:
            if token.dep_ == 'punct':
                continue
            if token.dep_ in ['nsubj', 'nsubjpass'] and token.head.pos_ not in ['VERB', 'AUX']:
                grammar_errors += 1
            if token.dep_ in ['dobj', 'iobj'] and token.head.pos_ not in ['VERB']:
                grammar_errors += 1
            if token.dep_ in ['amod', 'advmod'] and token.head.pos_ not in ['NOUN', 'VERB', 'ADJ']:
                grammar_errors += 1

    avg_sentence_length = np.mean(sentence_lengths)
    if avg_sentence_length > 100:
        flow_score += 2
    else:
        flow_score += 1

    if complex_sentences / num_sentences > 0.5:
        flow_score += 2
    else:
        flow_score += 1

    if grammar_errors / num_sentences < 0.1:
        flow_score += 2
    else:
        flow_score += 1

    simple_sentences = num_sentences - complex_sentences
    if simple_sentences / num_sentences < 0.5:
        flow_score += 2
    else:
        flow_score += 1

    paragraphs = text.split('\n\n')
    num_paragraphs = len(paragraphs)
    if num_paragraphs > 20:
        flow_score += 2
    else:
        flow_score += 1

    flow_score = min(10, flow_score)
    return flow_score


def evaluate_writing_standard(text):
    doc = nlp(text)
    standard_score = 0
    for token in doc:
        if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']:
            standard_score += 1
    sentences = list(doc.sents)
    for sentence in sentences:
        if sentence[-1].is_punct:
            standard_score += 1
    standard_score = min(10, standard_score / len(doc) * 10)
    return standard_score


def evaluate_pdf_length(file_path, text=None):
    if text is None:
        text = extract_pdf_text(file_path)

    if text is None:
        raise ValueError("Extracted text is None")

    references_start = text.find("参考文献")
    if references_start != -1:
        text_before_references = text[:references_start]
    else:
        text_before_references = text


    char_count = len(text_before_references)


    pages_before_references = char_count / 1000.0

    pdf_length_score = 10
    if pages_before_references < 15:
        pdf_length_score -= (15 - pages_before_references)
    elif pages_before_references > 20:
        pdf_length_score -= (pages_before_references - 20) / 3

    pdf_length_score = max(0, min(10, pdf_length_score))
    return pdf_length_score


def evaluate_word_count(text):
    word_count = len(text)
    target_word_count = 15000
    word_count_score = 10
    if word_count < target_word_count:
        word_count_score -= (target_word_count - word_count) // 1000
    word_count_score = max(0, min(10, word_count_score))
    return word_count_score


def evaluate_paper_structure(paper):
    structure_score = 0
    num_sections = 6
    introduction_score = evaluate_introduction(paper['introduction'])
    structure_score += introduction_score
    literature_review_score = evaluate_literature_review(paper['literature_review'])
    structure_score += literature_review_score
    method_score = evaluate_method(paper['method'])
    structure_score += method_score
    results_score = evaluate_results(paper['results'])
    structure_score += results_score
    discussion_score = evaluate_discussion(paper['discussion'])
    structure_score += discussion_score
    conclusion_score = evaluate_conclusion(paper['conclusion'])
    structure_score += conclusion_score
    overall_score = evaluate_overall_structure(paper)
    structure_score += overall_score
    structure_score = min(10, structure_score / (num_sections * 2) * 3)
    return structure_score


def process_and_evaluate_papers(folder_path, expected_titles):
    pdf_results = process_pdf_folder(folder_path, expected_titles)
    paper_texts = read_papers_from_folder(folder_path)
    scores = []

    for filename, paper_text in tqdm(paper_texts, desc="Evaluating papers"):
        file_path = os.path.join(folder_path, filename)
        sections = split_paper_into_sections(paper_text)
        paper = Paper(sections['introduction'], sections['literature_review'], sections['method'], sections['results'],
                      sections['discussion'], sections['conclusion'])
        text_flow_score = evaluate_text_flow(paper_text)
        writing_standard_score = evaluate_writing_standard(paper_text)
        structure_score = evaluate_paper_structure(sections)
        references_score = evaluate_references(paper_text)
        formulas_score = evaluate_formulas(paper_text)
        images_and_tables_score = evaluate_images_and_tables(paper_text)
        pdf_length_score = evaluate_pdf_length(file_path, paper_text)
        word_count_score = evaluate_word_count(paper_text)
        title_score = next((score for pdf, _, score in pdf_results if pdf == filename), 0)
        total_score = text_flow_score + writing_standard_score + structure_score + references_score + formulas_score + images_and_tables_score + pdf_length_score + word_count_score + title_score
        scores.append({
            'filename': filename,
            'text_flow_score': text_flow_score,
            'writing_standard_score': writing_standard_score,
            'structure_score': structure_score,
            'references_score': references_score,
            'formulas_score': formulas_score,
            'images_and_tables_score': images_and_tables_score,
            'pdf_length_score': pdf_length_score,
            'word_count_score': word_count_score,
            'title_score': title_score,
            'total_score': total_score
        })

    return scores

class Paper:
    def __init__(self, introduction, literature_review, method, results, discussion, conclusion):
        self.introduction = introduction
        self.literature_review = literature_review
        self.method = method
        self.results = results
        self.discussion = discussion
        self.conclusion = conclusion


def save_scores_to_csv(scores, output_path):

    directory = os.path.dirname(output_path)


    if not os.path.exists(directory):
        os.makedirs(directory)


    df = pd.DataFrame(scores)
    df.to_csv(output_path, index=False)
    print(f"Scores saved to {output_path}")


folder_path = "../other_pdfs"
output_path = "/output/scores.csv"
expected_titles = ["摘要", "目录", "问题重述", "假设条件", "符号说明", "模型建立", "模型求解", "模型检验", "结果分析", "结论", "参考文献", "附录"]

scores = process_and_evaluate_papers(folder_path, expected_titles)
save_scores_to_csv(scores, output_path)

for i, score in enumerate(scores):
    print(f"Document {i} ({score['filename']}) scores:")
    print(f"  Text Flow Score: {score['text_flow_score']}")
    print(f"  Writing Standard Score: {score['writing_standard_score']}")
    print(f"  Structure Score: {score['structure_score']}")
    print(f"  References Score: {score['references_score']}")
    print(f"  Formulas Score: {score['formulas_score']}")
    print(f"  Images and Tables Score: {score['images_and_tables_score']}")
    print(f"  PDF Length Score: {score['pdf_length_score']}")
    print(f"  Word Count Score: {score['word_count_score']}")
    print(f"  Title Score: {score['title_score']}")
    print(f"  Total Score: {score['total_score']}")
