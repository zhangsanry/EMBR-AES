# config_BP.py


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
        'score_column': 'rea_score'
    },
]


model_params = {
    'input_dim': 768
}


training_params = {
    'num_epochs': 50000,
    'batch_size': 32,
    'learning_rate': 1e-5
}
