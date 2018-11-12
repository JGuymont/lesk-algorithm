#!/usr/bin/env python3
"""
Created on Nov 11, 2018

@author: jguymont

Implementation of the most frequent sense baseline: this is the 
sense indicated as #1 in the synset according to WordNet
"""
from nltk.corpus import wordnet

from lesk.utils import *

MOST_LIKELY_SYNSET_IDX = 0 # first index correspond synset to most frequent sense

def baseline(lemma):
    """returns most common sense for input wsd.lemma

    Arg
        lemma: (string) the lemma of a word

    return: (list) list of possible sense for this lemma
    """
    synset = get_synsets(lemma)
    if len(synset) > 0:
        return get_sense_key(synset[MOST_LIKELY_SYNSET_IDX])
    else:
        print('synset empty for {}'.format(lemma))
        return None

def run_baseline_experiment(dev_instances, dev_key):
    
    predictions = [] 
    targets = []
    
    for id, wsd in dev_instances.items():
        lemma = wsd.lemma.decode("utf-8")
        pred = baseline(lemma)
        predictions.append(pred)
        targets.append(dev_key[id])
    
    accuracy = evaluate_accuracy(predictions, targets)
    
    print(accuracy)

    return 

