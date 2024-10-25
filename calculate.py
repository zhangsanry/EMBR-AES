import pandas as pd

# 从CSV文件读取数据
input_path = './output/predicted_vs_actual_scores.csv'  # 输入文件路径
output_path = './output/final_scores.csv'  # 输出文件路径

# 读取数据
df = pd.read_csv(input_path)

# 计算最终结果
df['Final Score'] = (df['Predicted Score'] * 0.5) + (df['Actual Score'] * 0.5)

# 保存为CSV文件
df.to_csv(output_path, index=False)

print(f"最终结果已保存到 {output_path}")
