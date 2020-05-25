import re
import numpy as np
import pandas as pd
from pprint import pprint

import nltk; nltk.download('stopwords')

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.wrappers import LdaMallet

# spacy for lemmatization
import spacy
import es_core_news_sm

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
%matplotlib inline

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# NLTK Stop words
from nltk.corpus import stopwords

## Carga archivo

data = pd.read_csv(r"C:\Users\LENOVO\Desktop\QUEJAS_20200520.csv", sep = "~", engine = "python")
data.columns

#List unique values in the df['name'] column
#df.name.unique()


# NLTK Stop words

stop_words = stopwords.words('spanish')
#stop_words1 = stopwords.words('spanish')
#stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'Abstract', 'abstract', 'la', 'en', 'el', 'los', 'se', 'del', 'una', 'para', 'por', 'entre', 'e', 'fue', 'su', 'más', 'este'])


# Convert to list
data2 = data.DESCRIPCION_DETALLADA.values.tolist()
type(data2)

print(data2[:1])

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
data2 = list(sent_to_words(data2))

print(data2[:1])

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data2, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data2], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data2[1]]])


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Remove Stop Words
data_words_nostops = remove_stopwords(data2)

data_words_nostops[1]

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

data_words_bigrams[2405]

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
## Desde consola de comandos python -m spacy download es_core_news_sm

nlp = es_core_news_sm.load()

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])


data_lemmatized[:1]

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# View
print(corpus[:1])
id2word[21]

#Human readable format of corpus (term-frequency)
[[(id2word[id], freq)for id, freq in cp]for cp in corpus[:1]]

















