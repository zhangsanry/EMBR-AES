import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


data = pd.read_csv('../output/predicted_vs_actual_scores.csv')


output_dir = ('../pictures')
os.makedirs(output_dir, exist_ok=True)


data['Weighted Score'] = (data['Actual Score'] + data['Predicted Score']) / 2


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20

def plot_scatter(data, output_dir, dpi=100):
    plt.figure(figsize=(10, 6), dpi=dpi)
    sns.scatterplot(x='Actual Score', y='Predicted Score', data=data)
    plt.plot([data['Actual Score'].min(), data['Actual Score'].max()],
             [data['Actual Score'].min(), data['Actual Score'].max()],
             color='red', linestyle='--')
    plt.xlabel('Actual Score', fontsize=22)
    plt.ylabel('Predicted Score', fontsize=22)
    plt.title('Scatter Plot of Actual vs Predicted Scores', fontsize=24)
    plt.savefig(os.path.join(output_dir, 'scatter_plot.svg'), format='svg')
    plt.close()

def plot_residual(data, output_dir, dpi=100):
    plt.figure(figsize=(10, 6), dpi=dpi)
    residuals = data['Predicted Score'] - data['Actual Score']
    sns.scatterplot(x=data['Actual Score'], y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Actual Score', fontsize=22)
    plt.ylabel('Residuals', fontsize=22)
    plt.title('Residual Plot', fontsize=24)
    plt.savefig(os.path.join(output_dir, 'residual_plot.svg'), format='svg')
    plt.close()

def plot_bar(data, output_dir, dpi=100):
    plt.figure(figsize=(14, 8), dpi=dpi)
    data_sorted = data.sort_values('Actual Score').reset_index()
    width = 0.25
    indices = range(len(data_sorted))
    plt.bar(indices, data_sorted['Actual Score'], width=width, label='Actual Score', alpha=0.6)
    plt.bar([i + width for i in indices], data_sorted['Predicted Score'], width=width, label='Predicted Score', alpha=0.6)
    plt.bar([i + 2*width for i in indices], data_sorted['Weighted Score'], width=width, label='Weighted Score', alpha=0.6)
    plt.xlabel('Papers', fontsize=30)
    plt.ylabel('Score', fontsize=30)
    plt.title('Bar Plot of Actual, Predicted and Weighted Scores', fontsize=32)
    plt.legend(fontsize=28)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bar_plot.svg'), format='svg')
    plt.close()

def plot_distribution(data, output_dir, dpi=100):
    plt.figure(figsize=(10, 6), dpi=dpi)
    sns.distplot(data['Actual Score'], hist=True, kde=True, color='blue', label='Actual Score')
    sns.distplot(data['Predicted Score'], hist=True, kde=True, color='orange', label='Predicted Score')
    sns.distplot(data['Weighted Score'], hist=True, kde=True, color='green', label='Weighted Score')
    plt.xlabel('Score', fontsize=22)
    plt.ylabel('Density', fontsize=22)
    plt.title('Distribution Plot of Actual, Predicted and Weighted Scores', fontsize=24)
    plt.legend(fontsize=20)
    plt.savefig(os.path.join(output_dir, 'distribution_plot.svg'), format='svg')
    plt.close()


plot_scatter(data, output_dir, dpi=100)
plot_residual(data, output_dir, dpi=100)
plot_bar(data, output_dir, dpi=100)
plot_distribution(data, output_dir, dpi=100)
