import os
import pandas as pd
import matplotlib.pyplot as plt
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Specify the path to the Writings directory
directory = '/Users/williamgay/iCloud/Writings'

# Retrieve all the files in the directory
files = os.listdir(directory)

data = []
for file in files:
    if file.endswith('.md'):
        with open(os.path.join(directory, file), 'r') as f:
            text = f.read()
            date = pd.to_datetime(file[:-3], format='%m-%d-%y')  # parse date from filename
            data.append({'date': date, 'text': text})

df = pd.DataFrame(data)
print(df.__len__())

stop_words = set(stopwords.words('english'))

all_words = []

for entry in df['text']:
    tokens = word_tokenize(entry)
    tokens = [word.lower() for word in tokens if word.isalpha()]  # exclude non-alphabetic tokens
    tokens = [word for word in tokens if word not in stop_words]  # exclude stop words
    all_words.extend(tokens)

freq_dist = FreqDist(all_words)
plt.figure(figsize=(10, 6))
freq_dist.plot(30)  # plot the 30 most common words

from nltk import bigrams, trigrams, FreqDist

all_bigrams = list(bigrams(all_words))  # all_words from previous steps
all_trigrams = list(trigrams(all_words))

bigram_freq = FreqDist(all_bigrams)
trigram_freq = FreqDist(all_trigrams)
print(bigram_freq.most_common(10))
print(trigram_freq.most_common(10))

# Filter for bigrams/trigrams that start with 'going'
going_bigrams = {bg: freq for bg, freq in bigram_freq.items() if bg[0] == 'going'}
going_trigrams = {tg: freq for tg, freq in trigram_freq.items() if tg[0] == 'going'}

# Sort the dictionaries by frequency
sorted_going_bigrams = sorted(going_bigrams.items(), key=lambda x: x[1], reverse=True)
sorted_going_trigrams = sorted(going_trigrams.items(), key=lambda x: x[1], reverse=True)

# Print the top 10 most frequent bigrams/trigrams
print(sorted_going_bigrams[:10])
print(sorted_going_trigrams[:10])