import re
import os
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

data = pd.read_csv(r"C:\Users\LENOVO\Desktop\QUEJAS_MAYO_25.csv", sep = "~", engine = "python")
data.columns

#List unique values in the df['name'] column
#df.name.unique()


# NLTK Stop words

stop_words = stopwords.words('spanish')
#stop_words1 = stopwords.words('spanish')
stop_words.extend(['abril','adjunto','adjuntos','apellido','apellidos','apto','argumento','asesor','atlantico','ba','barrio','bello','bogota','bolivar','bucaramanga','caldas','cartagena','cesar','ciudad','contacto','contactar','contrato','celular','correo','cedula','comentarios','documento','direccion','datos','departamento','descripción','detallada','eletronico','electronico','envigado','email','envio','etp','factura','febrero','fecha','fisica','fisico','gmail','hotmail','identificador','japosada','julio','junio','linea','llamada','marzo','mayo','manizales','mes','monteria','motivo','mjgonzal','municipio','nombre','nit','numero','objeto','open','observaciones','oficina','paquete','petición','queja','reclamo','recurso','servicio','sgto','sms','sisback','solicitud','suscriptor','telefono','tigo','titular','tv','uen','usuario','valledupar','ver','virtual'])


# Convert to list
data2 = data.DESCRIPCION_DETALLADA.values.tolist()
type(data2)

print(data2[:2312]) # contacto, nombre, solicitud, correo, queja, telefono 
muestra1 = data2[:2312]

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
data2 = list(sent_to_words(data2))

print(data2[:2312])

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data2, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data2], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data2[2312]]])


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
data_words_nostops[2312]

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
id2word[6]

#Human readable format of corpus (term-frequency)
[[(id2word[id], freq)for id, freq in cp]for cp in corpus[:1]]

# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=10, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

# Print the top ten Keywords in the topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

# Visualize the topics
pyLDAvis.enable_notebook() 
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
vis
pyLDAvis.save_html(vis,'lda_Quejas_2_v2.html')

# Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
os.environ['MALLET_HOME'] = 'C:\\users\\LENOVO\\Downloads\\mallet-2.0.8'
mallet_path = 'C:/Users/LENOVO/Downloads/mallet-2.0.8/bin/mallet'
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=20, id2word=id2word)

# Show Topics
pprint(ldamallet.show_topics(formatted=False))

# Compute Coherence Score
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score: ', coherence_ldamallet)
