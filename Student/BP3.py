import os
import torch
from torch import nn, optim
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_svg import FigureCanvasSVG

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PaperScoringModel(nn.Module):
    def __init__(self):
        super(PaperScoringModel, self).__init__()
        self.fc1 = nn.Linear(768, 512)
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


paper_scoring_model = PaperScoringModel().to(device)


def load_embeddings_and_labels(embedding_file, score_file, score_column):
    X = np.load(embedding_file)
    scores_df = pd.read_excel(score_file)
    y = scores_df[score_column].dropna().values
    return X, y


train_embedding_file = '../embedding/embeddings_bert/train8.npy'
train_score_file = '../dataset/train8.xlsx'
train_score_column = 'rea_score'

X_train, y_train = load_embeddings_and_labels(train_embedding_file, train_score_file, train_score_column)


X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)


criterion = nn.MSELoss()
optimizer = optim.Adam(paper_scoring_model.parameters(), lr=0.00001)


train_losses = []


num_epochs = 50000
for epoch in range(num_epochs):
    paper_scoring_model.train()
    epoch_train_loss = 0
    for i in range(0, X_train_tensor.size(0), 32):
        batch_X = X_train_tensor[i:i+32]
        batch_y = y_train_tensor[i:i+32]

        optimizer.zero_grad()
        outputs = paper_scoring_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    train_losses.append(epoch_train_loss / (X_train_tensor.size(0) / 32))
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_train_loss / (X_train_tensor.size(0) / 32):.4f}")


torch.save(paper_scoring_model.state_dict(), "paper_scoring_model_trained.pth")
print("训练好的模型已保存")


val_embedding_file = '../embedding/embeddings_bert/val8.npy'
val_score_file = '../dataset/val8.xlsx'
val_score_column = 'rea_score'

X_val, y_val = load_embeddings_and_labels(val_embedding_file, val_score_file, val_score_column)


X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)


paper_scoring_model.eval()
val_losses = []
with torch.no_grad():
    total_val_loss = 0
    for i in range(0, X_val_tensor.size(0), 32):
        batch_X = X_val_tensor[i:i+32]
        batch_y = y_val_tensor[i:i+32]
        outputs = paper_scoring_model(batch_X)
        loss = criterion(outputs, batch_y)
        total_val_loss += loss.item()
    val_losses.append(total_val_loss / (X_val_tensor.size(0) / 32))
    print(f"Validation Loss: {total_val_loss / (X_val_tensor.size(0) / 32):.4f}")


results = []
for i in range(len(X_val_tensor)):
    predicted_score = paper_scoring_model(X_val_tensor[i:i+1]).item()
    actual_score = y_val[i]
    results.append((i, predicted_score, actual_score))

results_df = pd.DataFrame(results, columns=['Index', 'Predicted Score', 'Actual Score'])
results_df.to_csv('./output/bert_validation_results8.csv', index=False)

print("验证结果已保存到CSV文件")
