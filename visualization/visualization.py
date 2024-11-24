import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


data = pd.read_csv('../output/predicted_vs_actual_scores.csv')


output_dir = '../pictures'
os.makedirs(output_dir, exist_ok=True)

def plot_scatter(data, output_dir, dpi=100):
    plt.figure(figsize=(10, 6), dpi=dpi)
    sns.scatterplot(x='Actual Score', y='Predicted Score', data=data)
    plt.plot([data['Actual Score'].min(), data['Actual Score'].max()],
             [data['Actual Score'].min(), data['Actual Score'].max()],
             color='red', linestyle='--')
    plt.xlabel('Actual Score')
    plt.ylabel('Predicted Score')
    plt.title('Scatter Plot of Actual vs Predicted Scores')
    plt.savefig(os.path.join(output_dir, 'scatter_plot.svg'), format='svg')
    plt.close()

def plot_residual(data, output_dir, dpi=100):
    plt.figure(figsize=(10, 6), dpi=dpi)
    residuals = data['Predicted Score'] - data['Actual Score']
    sns.scatterplot(x=data['Actual Score'], y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Actual Score')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.savefig(os.path.join(output_dir, 'residual_plot.svg'), format='svg')
    plt.close()

def plot_bar(data, output_dir, dpi=100):
    plt.figure(figsize=(14, 8), dpi=dpi)
    data_sorted = data.sort_values('Actual Score').reset_index()
    plt.bar(data_sorted.index, data_sorted['Actual Score'], label='Actual Score', alpha=0.6)
    plt.bar(data_sorted.index, data_sorted['Predicted Score'], label='Predicted Score', alpha=0.6)
    plt.xlabel('Papers')
    plt.ylabel('Score')
    plt.title('Bar Plot of Actual vs Predicted Scores')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'bar_plot.svg'), format='svg')
    plt.close()

def plot_distribution(data, output_dir, dpi=100):
    plt.figure(figsize=(10, 6), dpi=dpi)
    sns.histplot(data['Actual Score'], kde=True, color='blue', label='Actual Score')
    sns.histplot(data['Predicted Score'], kde=True, color='orange', label='Predicted Score')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.title('Distribution Plot of Actual vs Predicted Scores')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'distribution_plot.svg'), format='svg')
    plt.close()


plot_scatter(data, output_dir, dpi=100)
plot_residual(data, output_dir, dpi=100)
plot_bar(data, output_dir, dpi=100)
plot_distribution(data, output_dir, dpi=100)
