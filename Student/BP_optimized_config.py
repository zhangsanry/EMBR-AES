

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
        'score_column': 'rea_score'
    }
]


model_params = {
    'input_dim': 768
}


training_params = {
    'num_epochs': 30000,
    'batch_size': 32,
    'learning_rate': 1e-5
}
