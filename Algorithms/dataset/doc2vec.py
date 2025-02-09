import os
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd 


os.environ["OMP_NUM_THREADS"] = "3"


def load_data(file_path):
    print("加载数据...")
    data = pd.read_csv(file_path)
    texts = data['text'].fillna("").tolist() 
    print(f"数据加载完成，共加载 {len(texts)} 条文本数据。")
    return texts



def clean_text(text):
    if isinstance(text, str):  
        return text
    else:
        return ""  


def train_doc2vec(texts, vector_size=256, window=5, min_count=2, epochs=5000, model_path="doc2vec_model"):
    print("训练 Doc2Vec 模型...")
    
    tagged_data = [TaggedDocument(words=clean_text(text).split(), tags=[str(i)]) for i, text in enumerate(texts)]

    
    model = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, epochs=epochs)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    print("Doc2Vec 模型训练完成！")

   
    model.save(model_path)
    print(f"模型保存到 {model_path}")


def main():
    
    texts = load_data('dataset.csv')

    
    train_doc2vec(texts)


if __name__ == "__main__":
    main()
