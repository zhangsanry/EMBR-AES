import re
import os
from pdfminer.high_level import extract_text
from fuzzywuzzy import fuzz



def extract_pdf_text(pdf_path):
    return extract_text(pdf_path)



def recognize_structure(text, titles):
    recognized_titles = []


    for title in titles:
        for line in text.split('\n'):
            if fuzz.partial_ratio(title, line) > 80:
                recognized_titles.append(title)
                break

    return list(set(recognized_titles))


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
    print(final_score)
    return final_score



def process_pdf_folder(folder_path, expected_titles):
    results = {}
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    for pdf_file in pdf_files:
        try:
            pdf_path = os.path.join(folder_path, pdf_file)
            text = extract_pdf_text(pdf_path)
            recognized_titles = recognize_structure(text, expected_titles)
            score = evaluate_structure(recognized_titles, expected_titles)
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
            score = 0
        results[pdf_file] = score
    return results



def process_data(folder_path, expected_titles):
    data = process_pdf_folder(folder_path, expected_titles)
    return data



folder_path = "../paper_file"
expected_titles = ["摘要", "目录", "问题重述", "假设条件", "符号说明", "模型建立", "模型求解", "模型检验", "结果分析",
                   "结论", "参考文献", "附录"]
data = process_data(folder_path, expected_titles)
print(data)
