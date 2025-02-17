import pandas as pd

# load data
df = pd.read_parquet('../data/raw/train-00000-of-00002.parquet')
df = df['text'][:10000]

# save data to txt file
txt_file = '../data/vocab_train_corpus.txt'
with open(txt_file, 'a', encoding='utf-8') as file:
    df.to_csv(file, sep='\t', index=False, header=False)
    print('Data saved to file: {}'.format(txt_file))