# train_and_evaluate.py

import os
import torch
from torch import nn, optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from config_BP import tasks, model_params, training_params

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义自定义BP神经网络模型
class PaperScoringModel(nn.Module):
    def __init__(self, input_dim=768):
        super(PaperScoringModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.5)

        self.fc5 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.squeeze(1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.dropout4(x)
        x = self.fc5(x)
        return x


# 初始化模型
def initialize_model(input_dim):
    return PaperScoringModel(input_dim=input_dim).to(device)


# 加载嵌入向量和标签
def load_embeddings_and_labels(embedding_file, score_file, score_column):
    X = np.load(embedding_file)
    scores_df = pd.read_excel(score_file)
    y = scores_df[score_column].dropna().values
    return X, y


# 训练和评估函数
def train_and_evaluate_model(task):
    # 获取输入文件、输出文件和其他参数
    input_files = task['input_files']
    output_files = task['output_files']
    score_column = task['score_column']

    train_embedding_file, train_score_file = input_files['train_embeddings'], input_files['train_labels']
    val_embedding_file, val_score_file = input_files['val_embeddings'], input_files['val_labels']
    num_epochs = training_params['num_epochs']
    batch_size = training_params['batch_size']
    learning_rate = training_params['learning_rate']
    input_dim = model_params['input_dim']

    # 初始化模型
    model = initialize_model(input_dim)

    # 读取训练集
    X_train, y_train = load_embeddings_and_labels(train_embedding_file, train_score_file, score_column)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        for i in range(0, X_train_tensor.size(0), batch_size):
            batch_X = X_train_tensor[i:i + batch_size]
            batch_y = y_train_tensor[i:i + batch_size]

            if batch_X.size(0) < 2:  # 检查批次大小是否大于1
                continue  # 跳过这个批次

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        if (epoch + 1) % 100 == 0:
            num_batches = X_train_tensor.size(0) // batch_size
            avg_loss = epoch_train_loss / num_batches
            print(
                f"Task {tasks.index(task) + 1} | Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # 保存模型
    torch.save(model.state_dict(), output_files['model'])
    print(f"Task {tasks.index(task) + 1} | 训练好的模型已保存到 {output_files['model']}")

    # 评估模型
    X_val, y_val = load_embeddings_and_labels(val_embedding_file, val_score_file, score_column)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

    model.eval()
    with torch.no_grad():
        total_val_loss = 0
        for i in range(0, X_val_tensor.size(0), batch_size):
            batch_X = X_val_tensor[i:i + batch_size]
            batch_y = y_val_tensor[i:i + batch_size]
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_val_loss += loss.item()
        print(
            f"Task {tasks.index(task) + 1} | Validation Loss: {total_val_loss / (X_val_tensor.size(0) / batch_size):.4f}")

    # 保存验证结果
    results = []
    for i in range(len(X_val_tensor)):
        predicted_score = model(X_val_tensor[i:i + 1]).item()
        actual_score = y_val[i]
        results.append((i, predicted_score, actual_score))

    results_df = pd.DataFrame(results, columns=['Index', 'Predicted Score', 'Actual Score'])
    results_df.to_csv(output_files['validation_results'], index=False)
    print(f"Task {tasks.index(task) + 1} | 验证结果已保存到 {output_files['validation_results']}")


# 主函数，依次处理每个任务
if __name__ == "__main__":
    for task in tasks:
        print(f"开始处理任务 {tasks.index(task) + 1}")
        train_and_evaluate_model(task)
        print(f"任务 {tasks.index(task) + 1} 处理完毕\n")
