import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import linear_sum_assignment  
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import itertools


def load_data(file_path):
    print("加载数据...")
    data = pd.read_csv(file_path)
    texts = data['text'].fillna("").tolist()  
    labels = data['label'].tolist()
    print(f"数据加载完成，共加载 {len(texts)} 条文本数据。")
    return texts, labels


def load_doc_vectors(vectors_path):
    print("加载文本向量...")
    vectors = np.load(vectors_path)
    print("文本向量加载完成！")
    return vectors


def balanced_gmm(vectors, n_clusters=3, max_iter=250, min_size=150):
    print("开始 Balanced GMM 聚类...")
    gmm = GaussianMixture(n_components=n_clusters, random_state=42, max_iter=max_iter)
    cluster_labels = gmm.fit_predict(vectors)
    centers = gmm.means_

    cluster_counts = Counter(cluster_labels)
    print(f"初始聚类样本数量：{dict(cluster_counts)}")

    distances = euclidean_distances(vectors, centers)

    
    for cluster_id, count in cluster_counts.items():
        if count < min_size:
            deficit_samples = min_size - count
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            farthest_samples = np.argsort(distances[cluster_indices, cluster_id])[-deficit_samples:]
            for idx in farthest_samples:
                
                new_cluster = np.argmin(distances[cluster_indices[idx]])
                cluster_labels[cluster_indices[idx]] = new_cluster

    cluster_counts = Counter(cluster_labels)
    print(f"调整后聚类样本数量：{dict(cluster_counts)}")
    print("Balanced GMM 聚类完成！")
    return cluster_labels


def hungarian_algorithm(cluster_labels, true_labels, n_clusters):
    
    print("开始使用匈牙利算法进行标签匹配...")
    cost_matrix = np.zeros((n_clusters, n_clusters))
    cluster_labels = np.array(cluster_labels)
    true_labels = np.array(true_labels)
    
    label_map = {'Low': 0, 'Medium': 1, 'High': 2,
                 'low': 0, 'medium': 1, 'high': 2}
    true_labels_numeric = np.array([label_map[label] for label in true_labels])

    for i in range(n_clusters):
        for j in range(n_clusters):
            
            cost_matrix[i, j] = np.sum((cluster_labels == i) & (true_labels_numeric == j))

    
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)

    new_cluster_labels = np.copy(cluster_labels)
    for i, j in zip(row_ind, col_ind):
        new_cluster_labels[cluster_labels == i] = j

    print("匈牙利算法标签匹配完成！")
    return new_cluster_labels, true_labels_numeric


def calculate_accuracy_and_f1(mapped_labels, true_labels):
    accuracy = accuracy_score(true_labels, mapped_labels)
    f1 = f1_score(true_labels, mapped_labels, average='weighted')
    return accuracy, f1


def best_mapping_search(cluster_labels, true_labels, n_clusters=3):
    
    best_accuracy = 0
    best_f1 = 0
    best_mapping = None
    for perm in itertools.permutations(range(n_clusters)):
        mapping = {i: perm[i] for i in range(n_clusters)}
        mapped_labels = [mapping[label] for label in cluster_labels]
        acc, f1 = calculate_accuracy_and_f1(mapped_labels, true_labels)
        if acc > best_accuracy:
            best_accuracy = acc
            best_f1 = f1
            best_mapping = mapping
    return best_accuracy, best_f1, best_mapping


def main():
    print("程序开始执行...")

    
    texts, true_labels = load_data('dataset/dataset.csv')

    
    vectors = load_doc_vectors('dataset/text_vectors.npy')

    
    cluster_labels = balanced_gmm(vectors, n_clusters=3, min_size=150)

    
    optimized_labels, true_labels_numeric = hungarian_algorithm(cluster_labels, true_labels, n_clusters=3)

   
    best_accuracy, best_f1, best_mapping = best_mapping_search(optimized_labels, true_labels_numeric, n_clusters=3)

    print(f"最佳标签映射组合：{best_mapping}")
    print(f"最佳聚类结果的准确率：{best_accuracy:.4f}")
    print(f"最佳聚类结果的F1-score：{best_f1:.4f}")

    print("程序执行完毕！")


if __name__ == "__main__":
    main()
