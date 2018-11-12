
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

STOPWORDS = set(stopwords)
PUNCTUATION = [
    '.',',', ';', ':', '!', '^', '&', '?', '"', '|', '<', '>', ')', '(', '[', ']', '{', '}', 
    '_', ' ', '--', '-', '``'
]

VERB_TAG = ['VB', 'VBD', 'VBJ', 'VBN', 'VBP', 'VBZ']
TAG_TO_LEMMATIZE = VERB_TAG
LEMMATIZER = WordNetLemmatizer()

STEMMER = PorterStemmer()
VECTORIZER = CountVectorizer()

def _string_to_number(string):
    try:
        int(string)
    except ValueError:
        pass
    else:
        return '@card@'
    return string

def _lemmatize(word):
    if nltk.pos_tag(word) in TAG_TO_LEMMATIZE:
        return LEMMATIZER.lemmatize(word)
    return word

def _stem(word):
    if nltk.pos_tag(word) in TAG_TO_LEMMATIZE:
        return STEMMER.stem(word)
    return word

def _preprocess(sentence, remove_stowords=True, lemmatize=True, stemming=False):
    for punc in PUNCTUATION:
        sentence = sentence.replace(punc, ' ')

    sentence = sentence.split()
    
    sentence = [w.lower() for w in sentence] 
    sentence = [_string_to_number(w) for w in sentence]

    if remove_stowords:
        sentence = [w for w in sentence if w not in STOPWORDS]
    if stemming:
        sentence = [_stem(w) for w in sentence]
    if lemmatize:
        sentence = [_lemmatize(w) for w in sentence]
    if remove_stowords:
        sentence = [w for w in sentence if w not in STOPWORDS]

    return sentence

def _inverse_frequency(definitions):
    vocabulary = [w for definition in definitions for w in definition]
    df = {w: 0 for w in vocabulary}
    N = len(definitions)
    for word in vocabulary:
        for definition in definitions:
            if word in definition:
                df[word] += 1
    idf = {}
    for word, freq in df.items():
        idf[word] = np.log(N / freq)
    return idf

def _get_overlap(context, definition, idf):
    overlap = 0
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
    
    syn_definitions = [_preprocess(definition) for definition in syn_definitions]
    context = _preprocess(context)

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
        context = ' '.join([el.decode("utf-8") for el in wsd.context])
        pred = _lesk(lemma, context)
        predictions.append(pred)
        targets.append(dev_key[id])
    
    accuracy = evaluate_accuracy(predictions, targets)
    
    print(accuracy)

    return