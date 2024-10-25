import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_paper_scores(input_path, bar_output_path, dist_output_path, dpi=100):
    # 读取数据
    df = pd.read_csv(input_path, encoding='ISO-8859-1')

    # 检查数据是否有缺失值，并删除这些行
    df = df.dropna(subset=['Final Score'])

    # 检查数据是否在合理范围内
    if not df['Final Score'].between(0, 100).all():
        print("Warning: Some scores are out of the expected range (0-100). Please check your data.")

    # 设置Seaborn的主题
    sns.set(style="whitegrid")

    # 设置全局字体属性
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 30  # 增大字体大小

    # 绘制柱状图
    plt.figure(figsize=(14, 8), dpi=dpi)
    bin_edges = range(0, 71, 5)  # 设置柱状图的x轴区间
    sns.distplot(df['Final Score'], bins=bin_edges, kde=False)
    plt.title('Distribution of Paper Scores - Bar Chart', fontsize=32)
    plt.xlabel('Scores', fontsize=30)
    plt.ylabel('Frequency', fontsize=30)
    plt.tight_layout()  # 调整布局，避免标签重叠
    plt.savefig(bar_output_path, format='svg', dpi=dpi)
    plt.close()

    # 绘制分布图
    plt.figure(figsize=(14, 8), dpi=dpi)
    sns.distplot(df['Final Score'], bins=bin_edges, kde=True)
    plt.title('Distribution of Paper Scores - Distribution Plot', fontsize=32)
    plt.xlabel('Scores', fontsize=30)
    plt.ylabel('Density', fontsize=30)
    plt.tight_layout()  # 调整布局，避免标签重叠
    plt.savefig(dist_output_path, format='svg', dpi=dpi)
    plt.close()

    print(f"可视化图表已保存到:\n{bar_output_path}\n{dist_output_path}")

# 输入和输出文件路径
input_csv_path = '../output/final_scores.csv'  # 输入文件路径
bar_output_image_path = '../pictures/paper_scores_bar_chart.svg'  # 柱状图输出图片路径
dist_output_image_path = '../pictures/paper_scores_distribution_plot.svg'  # 分布图输出图片路径

# 调用函数生成可视化图表
visualize_paper_scores(input_csv_path, bar_output_image_path, dist_output_image_path, dpi=150)
