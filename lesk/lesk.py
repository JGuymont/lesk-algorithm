from nltk.corpus import wordnet

from nltk.wsd import lesk


from lesk.utils import get_sense_key, evaluate_accuracy 

def _lesk(context, lemma):
    """returns word sense for synset found using lesk's algorithm"""
    synset = lesk(context, lemma)
    if synset is not None:
        return get_sense_key(synset)
    else:
        print('synset empty for {}'.format(lemma))
        return None

def run_lesk_bultin_experiment(dev_instances, dev_key):

    predictions = [] 
    targets = []

    for id, wsd in dev_instances.items():
        
        lemma = wsd.lemma.decode("utf-8")
        context = [el.decode("utf-8") for el in wsd.context]

        #print('** processing [{}:{}:{}:{}]'.format(id, wsd.index, wsd.lemma, ' '.join(context)))

        pred = _lesk(context, lemma)
        
        predictions.append(pred)
        targets.append(dev_key[id])
    
    accuracy = evaluate_accuracy(predictions, targets)

    print(accuracy)

    return 