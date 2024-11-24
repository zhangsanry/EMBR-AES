# config_BP.py

tasks = [

{
        'input_files': {
            'train_embeddings': './embeddings_bert/train2.npy',
            'train_labels': './dataset/train2.xlsx',
            'val_embeddings': './embeddings_bert/val2.npy',
            'val_labels': './dataset/val2.xlsx'
        },
        'output_files': {
            'model': './output/bert_cnn_model_trained_2.pth',
            'validation_results': './bert_result/cnn_validation_results2.csv'
        },
        'score_column': 'rea_score'
    },

]


model_params = {
    'input_dim': 768
}


training_params = {
    'num_epochs': 30000,
    'batch_size': 32,
    'learning_rate': 1e-5
}
