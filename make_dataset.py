import pandas as pd


data = pd.read_csv('./output/scores.csv')


dataset = {row['filename']: row['total_score'] for _, row in data.iterrows()}


for filename, total_score in dataset.items():
    print(f"Filename: {filename}, Total Score: {total_score}")


dataset_df = pd.DataFrame(list(dataset.items()), columns=['Filename', 'Total Score'])
dataset_df.to_csv('./output/dataset.csv', index=False)
