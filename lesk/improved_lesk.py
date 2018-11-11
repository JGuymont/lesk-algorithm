
import numpy as np

import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from lesk.stopwords import stopwords
from lesk.utils import *

#STOPWORDS = set(stopwords.words('english'))
STOPWORDS = set(stopwords)
PUNCTUATION = [',', ';', ':', '!', '^', '&', '?', '"', '|', '<', '>', ')', '(', '[', ']', '{', '}']
TOKENIZER = RegexpTokenizer(r'[\w_-]+')
LEMMATIZER = WordNetLemmatizer()
STEMMER = PorterStemmer()
VECTORIZER = CountVectorizer()

def _preprocess(sentence, remove_stowords=True, lemmatize=False, stemming=True):
    sentence = [w.lower() for w in sentence]
    if remove_stowords:
        sentence = [w for w in sentence if w not in STOPWORDS]
    for punc in PUNCTUATION:
        sentence = [w.replace(punc, '') for w in sentence] 
    if stemming:
        sentence = [STEMMER.stem(w) for w in sentence]
    if lemmatize:
        sentence = [LEMMATIZER.lemmatize(w) for w in sentence]
    if remove_stowords:
        sentence = [w for w in sentence if w not in STOPWORDS]
    return sentence

def _inverse_frequency(definitions):
    vocabulary = [w for definition in definitions for w in _preprocess(definition.split())]
    df = {w: 0 for w in vocabulary}
    N = len(definitions)
    for word in vocabulary:
        for definition in definitions:
            if word in _preprocess(definition.split()):
                df[word] += 1
    idf = {}
    for word, freq in df.items():
        idf[word] = np.log(N / freq)
    return idf

def _get_overlap(context, definition, idf):
    overlap = 0
    definition = _preprocess(definition.split())
    context = _preprocess(context)
    for _ in context:
        w = context.pop()
        try: 
            w_idx = definition.index(w)
            definition.pop(w_idx)
            overlap += idf[w]
        except ValueError:
            pass
    return overlap

def _lesk(lemma, context):
    
    synsets = get_synsets(lemma)
    syn_definitions = get_synset_definition(lemma)
    
    idf = _inverse_frequency(syn_definitions)
    best_overlap = 0
    best_synset_idx = 0
    for syn_idx, definition in enumerate(syn_definitions):
        cur_overlap = _get_overlap(context, definition, idf)
        if cur_overlap > best_overlap:
            best_synset_idx = syn_idx
    return get_sense_key(synsets[best_synset_idx])
    
    

def run_experiment(dev_instances, dev_key):
    predictions = [] 
    targets = []

    for id, wsd in dev_instances.items():
        lemma = wsd.lemma.decode("utf-8")
        context = [el.decode("utf-8") for el in wsd.context]
        pred = _lesk(lemma, context)
        predictions.append(pred)
        targets.append(dev_key[id])
    
    accuracy = evaluate_accuracy(predictions, targets)
    
    print(accuracy)

    return