import pandas as pd


input_path = './output/predicted_vs_actual_scores.csv'
output_path = './output/final_scores.csv'


df = pd.read_csv(input_path)


df['Final Score'] = (df['Predicted Score'] * 0.5) + (df['Actual Score'] * 0.5)


df.to_csv(output_path, index=False)

print(f"最终结果已保存到 {output_path}")
