import pandas as pd


original_file = "result/expert_labels.csv"
modified_file = "result/AES_labels.csv"


original_df = pd.read_csv(original_file)
modified_df = pd.read_csv(modified_file)


required_columns = ["Filename", "OriginalLevel"]
if "ModifiedLevel" not in modified_df.columns:
    raise ValueError(f"预测标签文件中缺少必要的列：ModifiedLevel")
if not all(col in original_df.columns for col in required_columns):
    raise ValueError(f"基准标签文件中缺少必要的列：{', '.join(required_columns)}")


df = pd.merge(original_df, modified_df, on="Filename", how="inner")

categories = ["High", "Medium", "Low"]
accuracy = {}


for category in categories:
    total = df[df["OriginalLevel"] == category].shape[0]
    correct = df[(df["OriginalLevel"] == category) & (df["ModifiedLevel"] == category)].shape[0]
    accuracy[category] = correct / total if total > 0 else 0


for category, acc in accuracy.items():
    print(f"{category} 类别的预测准确率: {acc:.2%}")


overall_total = df.shape[0]
overall_correct = df[df["OriginalLevel"] == df["ModifiedLevel"]].shape[0]
overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0
print(f"总体预测准确率: {overall_accuracy:.2%}")
