import torch
from torch import nn, optim
import numpy as np
import pandas as pd
from Student import config_LSTM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PaperScoringLSTMModel(nn.Module):
    def __init__(self, input_dim):
        super(PaperScoringLSTMModel, self).__init__()
        self.input_dim = input_dim

        self.seq_len = 16
        self.input_size = input_dim // self.seq_len


        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=128, num_layers=2, batch_first=True)


        self.fc1 = nn.Linear(128, 64)
        self.bn1 = nn.LayerNorm(64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):

        x = x.view(x.size(0), self.seq_len, self.input_size)

        lstm_out, (h_n, c_n) = self.lstm(x)



        h_n_last = h_n[-1]

        x = self.fc1(h_n_last)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        return x


def quadratic_weighted_kappa(y_true, y_pred):
    """
    计算Quadratic Weighted Kappa (QWK)

    参数:
    y_true: 真实的分数数组
    y_pred: 预测的分数数组

    返回:
    qwk: 二次加权卡帕系数
    """
    min_rating = int(min(min(y_true), min(y_pred)))
    max_rating = int(max(max(y_true), max(y_pred)))

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    num_ratings = int(max_rating - min_rating + 1)
    weights = np.zeros((num_ratings, num_ratings))
    for i in range(num_ratings):
        for j in range(num_ratings):
            weights[i, j] = float((i - j) ** 2) / (num_ratings - 1) ** 2


    hist_true = np.zeros(num_ratings)
    hist_pred = np.zeros(num_ratings)
    for i in range(len(y_true)):
        hist_true[y_true[i] - min_rating] += 1
        hist_pred[y_pred[i] - min_rating] += 1

    E = np.outer(hist_true, hist_pred)
    E = E / E.sum()

    O = np.zeros((num_ratings, num_ratings))
    for i in range(len(y_true)):
        O[y_true[i] - min_rating, y_pred[i] - min_rating] += 1
    O = O / O.sum()

    num = np.sum(weights * O)
    den = np.sum(weights * E)

    qwk = 1.0 - num / den
    return qwk


def load_embeddings_and_labels(embedding_file, score_file, score_column):
    X = np.load(embedding_file)
    scores_df = pd.read_excel(score_file)
    y = scores_df[score_column].dropna().values
    return X, y


def train_and_evaluate(task):

    X_train, y_train = load_embeddings_and_labels(
        task['input_files']['train_embeddings'],
        task['input_files']['train_labels'],
        task['score_column']
    )


    X_val, y_val = load_embeddings_and_labels(
        task['input_files']['val_embeddings'],
        task['input_files']['val_labels'],
        task['score_column']
    )


    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)


    model = PaperScoringLSTMModel(input_dim=config_LSTM.model_params['input_dim']).to(device)


    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config_LSTM.training_params['learning_rate'])


    train_losses = []


    best_qwk = -np.inf
    best_model_state = None


    epochs_since_improvement = 0
    max_epochs_since_improvement = 10000

    # 训练模型
    num_epochs = config_LSTM.training_params['num_epochs']
    batch_size = config_LSTM.training_params['batch_size']
    for epoch in range(num_epochs):
        if epochs_since_improvement >= max_epochs_since_improvement:
            print(f"超过 {max_epochs_since_improvement} 个 epoch 没有取得更好的 QWK，停止训练。")
            break

        model.train()
        epoch_train_loss = 0
        permutation = torch.randperm(X_train_tensor.size(0))
        for i in range(0, X_train_tensor.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_X = X_train_tensor[indices]
            batch_y = y_train_tensor[indices]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        average_train_loss = epoch_train_loss / (X_train_tensor.size(0) / batch_size)
        train_losses.append(average_train_loss)
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_train_loss:.4f}")


            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)

                val_predicted_scores = val_outputs.squeeze(1).cpu().numpy()
                val_actual_scores = y_val_tensor.squeeze(1).cpu().numpy()


                val_predicted_scores_rounded = np.round(val_predicted_scores).astype(int)
                val_actual_scores_int = np.round(val_actual_scores).astype(int)


                qwk = quadratic_weighted_kappa(val_actual_scores_int, val_predicted_scores_rounded)
                print(f"Validation QWK: {qwk:.4f}")


                if qwk > best_qwk:
                    best_qwk = qwk
                    best_model_state = model.state_dict()
                    epochs_since_improvement = 0

                    torch.save(model.state_dict(), task['output_files']['model'])
                    print(f"新的最佳模型已保存，QWK: {best_qwk:.4f}")
                else:
                    epochs_since_improvement += 100


    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"加载了最佳模型，QWK: {best_qwk:.4f}")
    else:

        torch.save(model.state_dict(), task['output_files']['model'])
        print(f"训练好的模型已保存到 {task['output_files']['model']}")


    model.eval()
    val_losses = []
    with torch.no_grad():
        total_val_loss = 0
        for i in range(0, X_val_tensor.size(0), batch_size):
            batch_X = X_val_tensor[i:i+batch_size]
            batch_y = y_val_tensor[i:i+batch_size]
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_val_loss += loss.item()
        average_val_loss = total_val_loss / (X_val_tensor.size(0) / batch_size)
        val_losses.append(average_val_loss)
        print(f"Final Validation Loss: {average_val_loss:.4f}")


    results = []
    with torch.no_grad():
        for i in range(len(X_val_tensor)):
            predicted_score = model(X_val_tensor[i:i+1]).item()
            actual_score = y_val[i]
            results.append((i, predicted_score, actual_score))

    results_df = pd.DataFrame(results, columns=['Index', 'Predicted Score', 'Actual Score'])
    results_df.to_csv(task['output_files']['validation_results'], index=False)

    print(f"验证结果已保存到 {task['output_files']['validation_results']}")


if __name__ == "__main__":
    for task in config_LSTM.tasks:
        print(f"开始训练任务，模型将保存到 {task['output_files']['model']}")
        train_and_evaluate(task)
