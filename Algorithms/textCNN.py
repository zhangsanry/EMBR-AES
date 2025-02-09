import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score



class Config(object):
    def __init__(self):
        self.model_name = 'TextCNN'
        self.text_vectors_path = 'dataset/text_vectors.npy'
        self.labels_path = 'dataset/dataset.csv'
        self.class_list = ['low', 'medium', 'high']  
        self.save_path = 'saved_dict/' + self.model_name + '.ckpt'  
        self.log_path = 'log/' + self.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

        self.dropout = 0.5  
        self.require_improvement = 10000  
        self.num_classes = len(self.class_list) 
        self.num_epochs = 10000 
        self.batch_size = 128 
        self.pad_size = 32  
        self.learning_rate = 1e-3  
        self.embed = 256  
        self.filter_sizes = (1,)  
        self.num_filters = 64 



class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(config.embed, config.embed)  
        self.convs = nn.ModuleList([nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        """卷积和池化操作"""
        x = F.relu(conv(x)).squeeze(3) 
        x = F.max_pool1d(x, x.size(2)).squeeze(2) 
        return x

    def forward(self, x):
        """前向传播"""
        out = x.unsqueeze(1)  
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1) 
        out = self.dropout(out)  
        out = self.fc(out)  
        return out



def load_data(text_vectors_path, labels_path):
    print("加载数据...")

   
    texts = np.load(text_vectors_path) 

    
    data = pd.read_csv(labels_path)
    labels = data['label'].apply(lambda x: {'low': 0, 'medium': 1, 'high': 2}.get(x.strip().lower()))

    
    invalid_labels = labels.isnull().sum()
    if invalid_labels > 0:
        print(f"警告：发现 {invalid_labels} 个无效标签，无法转换为数字。")

    
    labels = labels.dropna().tolist()
    texts = texts[:len(labels)] 

    print(f"数据加载完成，共加载 {len(texts)} 条文本数据，标签处理后共有 {len(labels)} 条有效数据。")
    return texts, labels



def train_model(config):
   
    texts, labels = load_data(config.text_vectors_path, config.labels_path)  

    
    train_texts, dev_texts, train_labels, dev_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=61
    )

    
    train_vectors = torch.tensor(train_texts, dtype=torch.float32)
    dev_vectors = torch.tensor(dev_texts, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    dev_labels = torch.tensor(dev_labels, dtype=torch.long)

    
    train_vectors = train_vectors.unsqueeze(1)  
    dev_vectors = dev_vectors.unsqueeze(1)     

    
    model = TextCNN(config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_f1_score = 0
    best_accuracy = 0
    best_epoch = 0
    best_model_path = "model/best_model_TextCNN.pth"

    
    for epoch in range(config.num_epochs):
        model.train()
        optimizer.zero_grad()

        output = model(train_vectors.to(config.device))
        loss = criterion(output, train_labels.to(config.device))
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{config.num_epochs}, Loss: {loss.item():.4f}")

       
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                output = model(dev_vectors.to(config.device))
                pred = torch.argmax(output, dim=1)
                acc = accuracy_score(dev_labels.cpu(), pred.cpu())
                f1 = f1_score(dev_labels.cpu(), pred.cpu(), average='weighted')
                print(f"Validation Accuracy: {acc:.4f}, F1-score: {f1:.4f}")

                
                if f1 > best_f1_score or acc > best_accuracy:
                    best_f1_score = f1
                    best_accuracy = acc
                    best_epoch = epoch
                   
                    torch.save(model.state_dict(), best_model_path)
                    print(f"Best model saved at epoch {epoch + 1}")

    print("\n====================")
    print("最佳结果：")
    print(f"Epoch: {best_epoch + 1}")
    print(f"Accuracy: {best_accuracy:.4f}")
    print(f"F1-score: {best_f1_score:.4f}")
    print(f"Best model saved at: {best_model_path}")
    print("====================")


def main():
    """主函数"""
    config = Config()  
    train_model(config)


if __name__ == "__main__":
    main()
