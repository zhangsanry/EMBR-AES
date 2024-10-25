# config_BP.py

# 输入和输出文件的配置，多个任务
tasks = [

    {
        'input_files': {
            'train_embeddings': './embeddings_albert/train2.npy',
            'train_labels': './dataset/train2.xlsx',
            'val_embeddings': './embeddings_albert/val2.npy',
            'val_labels': './dataset/val2.xlsx'
        },
        'output_files': {
            'model': './output/bert_bp_with_math_model_trained_2.pth',
            'validation_results': './BP_optimized_result/al_bp_validation_results2.csv'
        },
        'score_column': 'rea_score'  # 标签所在列
    }
]

# 模型参数
model_params = {
    'input_dim': 768  # 原始输入维度，实际模型中会更新为512
}

# 训练参数
training_params = {
    'num_epochs': 30000,  # 训练轮数
    'batch_size': 32,  # 批量大小
    'learning_rate': 1e-5  # 学习率
}
