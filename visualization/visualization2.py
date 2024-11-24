import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_paper_scores(input_path, bar_output_path, dist_output_path, dpi=100):

    df = pd.read_csv(input_path, encoding='ISO-8859-1')


    df = df.dropna(subset=['Final Score'])


    if not df['Final Score'].between(0, 100).all():
        print("Warning: Some scores are out of the expected range (0-100). Please check your data.")


    sns.set(style="whitegrid")


    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 30


    plt.figure(figsize=(14, 8), dpi=dpi)
    bin_edges = range(0, 71, 5)
    sns.distplot(df['Final Score'], bins=bin_edges, kde=False)
    plt.title('Distribution of Paper Scores - Bar Chart', fontsize=32)
    plt.xlabel('Scores', fontsize=30)
    plt.ylabel('Frequency', fontsize=30)
    plt.tight_layout()
    plt.savefig(bar_output_path, format='svg', dpi=dpi)
    plt.close()


    plt.figure(figsize=(14, 8), dpi=dpi)
    sns.distplot(df['Final Score'], bins=bin_edges, kde=True)
    plt.title('Distribution of Paper Scores - Distribution Plot', fontsize=32)
    plt.xlabel('Scores', fontsize=30)
    plt.ylabel('Density', fontsize=30)
    plt.tight_layout()
    plt.savefig(dist_output_path, format='svg', dpi=dpi)
    plt.close()

    print(f"可视化图表已保存到:\n{bar_output_path}\n{dist_output_path}")


input_csv_path = '../output/final_scores.csv'
bar_output_image_path = '../pictures/paper_scores_bar_chart.svg'
dist_output_image_path = '../pictures/paper_scores_distribution_plot.svg'


visualize_paper_scores(input_csv_path, bar_output_image_path, dist_output_image_path, dpi=150)
