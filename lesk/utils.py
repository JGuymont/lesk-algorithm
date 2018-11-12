
from nltk.corpus import wordnet


# sometimes there is more then one correct sense. 
# They are all as good so we use the key SENSE_KEY = 0  
SENSE_KEY = 0

def get_sense_key(synset):
    """returns the sense key as a string for the given synset"""
    sense_keys = [sense.key() for sense in synset.lemmas()]
    return sense_keys[SENSE_KEY]

def get_synsets(lemma):
    """
    return the list of synsets corresponding to a lemma
    """
    return wordnet.synsets(lemma)

def get_synset_definition(lemma):
    synsets = wordnet.synsets(lemma)
    return [synset.definition() for synset in synsets]

def check_wordnet_version(wordnet):
    if not '3.0' == wordnet.get_version():
        raise ValueError("Wordnet version is {}. Must be 3.0".format(wordnet.get_version()))
    pass

def evaluate_accuracy(predictions, targets):
    """Evaluate accuracy
    
    Args
        predictions: (list of string)
        targets: (list of list os string) 
    """
    correct = 0
    total = len(targets)

    for prediction, target in zip(predictions, targets):
        if prediction in target:
            correct += 1
    accuracy = round((correct / total) * 100, 4)
    return accuracy