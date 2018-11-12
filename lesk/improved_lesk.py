
import numpy as np
import string

import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords

from lesk.utils import get_sense_key, get_synsets, get_synset_definition, evaluate_accuracy
from lesk.preprocessing import string_to_number, lemmatize, stem


STOPWORDS = set(stopwords.words('english'))
PUNCTUATION = string.punctuation # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

def _preprocess(sentence, remove_stowords=True, lemmatization=False, stemming=False):
    for punc in PUNCTUATION:
        sentence = sentence.replace(punc, ' ')
    sentence = sentence.lower()
    sentence = sentence.split() 
    sentence = [string_to_number(w) for w in sentence]

    if remove_stowords:
        sentence = [w for w in sentence if w not in STOPWORDS]
    if stemming:
        sentence = [stem(w) for w in sentence]
    if lemmatization:
        sentence = [lemmatize(w) for w in sentence]
    return sentence

def _inverse_frequency(definitions):
    vocabulary = [w for definition in definitions for w in definition]
    df = {w:0. for w in vocabulary}
    N = float(len(definitions))
    for word in vocabulary:
        for definition in definitions:
            if word in definition:
                df[word] += 1.
    idf = {}
    for word, freq in df.items():
        idf[word] = np.log(0.5 + 0.5*N / freq)
    return idf, df

def _get_overlap(context, definition, idf):
    overlap = 0.
    cur_context = context.copy()
    for _ in cur_context:
        w = cur_context.pop()
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

    idf, _ = _inverse_frequency(syn_definitions)

    best_overlap = 0
    best_synset_idx = 0

    for syn_idx, definition in enumerate(syn_definitions):
        if syn_idx >= 3:
            pass
        else:
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

        '''
        if not pred in dev_key[id]:
            print('lemma:', lemma)
            print('context:', _preprocess(context), '|', pred)
            print('target:', dev_key[id])
            synstets = get_synsets(lemma)
            for syn in synstets:
                print('def:', _preprocess(syn.definition()), get_sense_key(syn))
            print('---------------------')  
        '''
    
    accuracy = evaluate_accuracy(predictions, targets)
    
    
    print(accuracy)

    return