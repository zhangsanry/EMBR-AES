# config_LSTM.py


tasks = [


{
        'input_files': {
            'train_embeddings': './embeddings_albert/train1.npy',
            'train_labels': './dataset/train1.xlsx',
            'val_embeddings': './embeddings_albert/val1.npy',
            'val_labels': './dataset/val1.xlsx'
        },
        'output_files': {
            'model': './output/albert_lstm_model_trained_1.pth',
            'validation_results': './LSTM_result/lstm_validation_alresults1.csv'
        },
        'score_column': 'rea_score'
    },
    {
        'input_files': {
            'train_embeddings': './embeddings_albert/train2.npy',
            'train_labels': './dataset/train2.xlsx',
            'val_embeddings': './embeddings_albert/val2.npy',
            'val_labels': './dataset/val2.xlsx'
        },
        'output_files': {
            'model': './output/albert_lstm_model_trained_2.pth',
            'validation_results': './LSTM_result/lstm_validation_alresults2.csv'
        },
        'score_column': 'rea_score'
    },
    {
        'input_files': {
            'train_embeddings': './embeddings_albert/train3.npy',
            'train_labels': './dataset/train3.xlsx',
            'val_embeddings': './embeddings_albert/val3.npy',
            'val_labels': './dataset/val3.xlsx'
        },
        'output_files': {
            'model': './output/albert_lstm_model_trained_3.pth',
            'validation_results': './LSTM_result/lstm_validation_alresults3.csv'
        },
        'score_column': 'rea_score'
    },
    {
        'input_files': {
            'train_embeddings': './embeddings_albert/train4.npy',
            'train_labels': './dataset/train4.xlsx',
            'val_embeddings': './embeddings_albert/val4.npy',
            'val_labels': './dataset/val4.xlsx'
        },
        'output_files': {
            'model': './output/albert_lstm_model_trained_4.pth',
            'validation_results': './LSTM_result/lstm_validation_alresults4.csv'
        },
        'score_column': 'rea_score'
    },
    {
        'input_files': {
            'train_embeddings': './embeddings_albert/train5.npy',
            'train_labels': './dataset/train5.xlsx',
            'val_embeddings': './embeddings_albert/val5.npy',
            'val_labels': './dataset/val5.xlsx'
        },
        'output_files': {
            'model': './output/albert_lstm_model_trained_5.pth',
            'validation_results': './LSTM_result/lstm_validation_alresults5.csv'
        },
        'score_column': 'rea_score'
    },
    {
        'input_files': {
            'train_embeddings': './embeddings_albert/train6.npy',
            'train_labels': './dataset/train6.xlsx',
            'val_embeddings': './embeddings_albert/val6.npy',
            'val_labels': './dataset/val6.xlsx'
        },
        'output_files': {
            'model': './output/albert_lstm_model_trained_6.pth',
            'validation_results': './LSTM_result/lstm_validation_alresults6.csv'
        },
        'score_column': 'rea_score'
    },
    {
        'input_files': {
            'train_embeddings': './embeddings_albert/train7.npy',
            'train_labels': './dataset/train7.xlsx',
            'val_embeddings': './embeddings_albert/val7.npy',
            'val_labels': './dataset/val7.xlsx'
        },
        'output_files': {
            'model': './output/albert_lstm_model_trained_7.pth',
            'validation_results': './LSTM_result/lstm_validation_alresults7.csv'
        },
        'score_column': 'rea_score'
    },
    {
        'input_files': {
            'train_embeddings': './embeddings_albert/train8.npy',
            'train_labels': './dataset/train8.xlsx',
            'val_embeddings': './embeddings_albert/val8.npy',
            'val_labels': './dataset/val8.xlsx'
        },
        'output_files': {
            'model': './output/albert_lstm_model_trained_8.pth',
            'validation_results': './LSTM_result/lstm_validation_alresults8.csv'
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
