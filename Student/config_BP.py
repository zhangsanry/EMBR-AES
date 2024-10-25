# config_BP.py

# 输入和输出文件的配置，多个任务
tasks = [

{
        'input_files': {
            'train_embeddings': './embeddings_albert/train7.npy',
            'train_labels': './dataset/train7.xlsx',
            'val_embeddings': './embeddings_albert/val7.npy',
            'val_labels': './dataset/val7.xlsx'
        },
        'output_files': {
            'model': './albert_result/paper_scoring_model_trained_7.pth',
            'validation_results': './albert_result/albert_validation_results7.csv'
        },
        'score_column': 'rea_score'  # 标签所在列
    },
]

# 模型参数
model_params = {
    'input_dim': 768  # 输入维度
}

# 训练参数
training_params = {
    'num_epochs': 50000,  # 训练轮数
    'batch_size': 32,  # 批量大小
    'learning_rate': 1e-5  # 学习率
}
