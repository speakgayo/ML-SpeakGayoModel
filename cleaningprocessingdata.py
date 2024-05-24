import pandas as pd
import re
from unicodedata import normalize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_csv('indogayo10rb.csv', names=['id', 'gy'], usecols=['id', 'gy'], sep=',')
df = df.sample(frac=1, random_state=42)
df = df.reset_index(drop=True)
df = df.dropna()
df.head()

def clean_text(text):
    text = normalize('NFD', text.lower())
    text = re.sub('[^A-Za-z ]+', '', text)
    return text

def clean_and_prepare_text(text):
    text = '[start] ' + clean_text(text) + ' [end]'
    return text

df['id'] = df['id'].apply(lambda row: clean_text(row))
df['gy'] = df['gy'].apply(lambda row: clean_and_prepare_text(row))
df.head()

id = df['id']
gy = df['gy']

id_max_len = max(len(line.split()) for line in id)
gy_max_len = max(len(line.split()) for line in gy)
sequence_len = max(id_max_len, gy_max_len)

print(f'Max phrase length (Indonesia): {id_max_len}')
print(f'Max phrase length (Gayo): {gy_max_len}')
print(f'Sequence length: {sequence_len}')

id_tokenizer = Tokenizer()
id_tokenizer.fit_on_texts(id)
id_sequences = id_tokenizer.texts_to_sequences(id)
id_x = pad_sequences(id_sequences, maxlen=sequence_len, padding='post')

gy_tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@\\^_`{|}~\t\n')
gy_tokenizer.fit_on_texts(gy)
gy_sequences = gy_tokenizer.texts_to_sequences(gy)
gy_y = pad_sequences(gy_sequences, maxlen=sequence_len + 1, padding='post')

id_vocab_size = len(id_tokenizer.word_index) + 1
gy_vocab_size = len(gy_tokenizer.word_index) + 1

print(f'Vocabulary size (Indonesia): {id_vocab_size}')
print(f'Vocabulary size (Gayo): {gy_vocab_size}')

inputs = { 'encoder_input': id_x, 'decoder_input': gy_y[:, :-1] }
outputs = gy_y[:, 1:]

