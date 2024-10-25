import numpy as np

# 假设文件名为 example.npy，并且与你的脚本位于同一目录下
file_path1 = 'embedding/embeddings_albert/train2.npy'

# 加载 .npy 文件
data1 = np.load(file_path1)

# 输出数据
print(data1.shape)
