import pandas as pd

# 读取CSV文件
data = pd.read_csv('./output/scores.csv')

# 创建一个字典以论文名字为键，总得分为值
dataset = {row['filename']: row['total_score'] for _, row in data.iterrows()}

# 打印结果
for filename, total_score in dataset.items():
    print(f"Filename: {filename}, Total Score: {total_score}")

# 将数据集保存为CSV文件
dataset_df = pd.DataFrame(list(dataset.items()), columns=['Filename', 'Total Score'])
dataset_df.to_csv('./output/dataset.csv', index=False)
