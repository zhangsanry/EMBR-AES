
import os
import gensim
from gensim.models.doc2vec import Doc2Vec
import numpy as np
import pandas as pd
from tqdm import tqdm


os.environ["OMP_NUM_THREADS"] = "3"

def load_data(file_path):
    print("加载数据...")
    data = pd.read_csv(file_path)
    texts = data['text'].fillna("").tolist() 
    print(f"数据加载完成，共加载 {len(texts)} 条文本数据。")
    return texts


def load_doc2vec_model(model_path="doc2vec_model"):
    print("加载 Doc2Vec 模型...")
    model = Doc2Vec.load(model_path)
    print("Doc2Vec 模型加载完成！")
    return model


def save_doc_vectors(model, texts, save_path):
    print("获取文本向量并保存...")
    vectors = [model.infer_vector(text.split()) for text in tqdm(texts, desc="计算文本向量")]
    np.save(save_path, np.array(vectors))
    print("文本向量保存完成！")

def main():
    print("程序开始执行...")

    
    texts = load_data('dataset.csv')

    
    model = load_doc2vec_model()

    
    save_doc_vectors(model, texts, "text_vectors.npy")

    print("程序执行完毕！")

if __name__ == "__main__":
    main()
