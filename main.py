
from lesk.utils import *
from nltk.corpus import wordnet

from lesk.loader import load_instances, load_key
from lesk.baseline import run_baseline_experiment
from lesk.lesk import run_lesk_bultin_experiment
import lesk.improved_lesk as lesk


DATA_FILE = './data/multilingual-all-words.en.xml'
KEY_FILE = './data/wordnet.en.key'

def load_data(data_file, key_file):
    """

    return: 
        dev_instances: (dict) {}
        dev_key: (dict) 

    example: dev_instances

        {d001.s001.t002: WSDInstance, ...}
            
            - value.key: e.g. d001.s001.t002
            - value.lemma: e.g. group
            - value.context: is the list of words in the sentence
                corresponding to the first part of value.key e.g. d001.s001

    example: dev_key {id: sense_key, ...}

        {d001.s001.t002: ['group%1:03:00::'], ...}
            
            
    """
    dev_instances, test_instances = load_instances(data_file)
    dev_key, test_key = load_key(key_file)

    # IMPORTANT: keys contain fewer entries than the instances; need to remove them
    dev_instances = {k:v for (k,v) in dev_instances.items() if k in dev_key}
    test_instances = {k:v for (k,v) in test_instances.items() if k in test_key}
    return dev_instances, test_instances, dev_key, test_key

if __name__ == '__main__':

    dev_instances, test_instances, dev_key, test_key = load_data(DATA_FILE, KEY_FILE)

    #run_baseline_experiment(dev_instances, dev_key)
    #run_lesk_bultin_experiment(dev_instances, dev_key)
    lesk.run_experiment(dev_instances, dev_key)
    