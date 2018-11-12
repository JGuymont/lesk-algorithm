import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)

import string

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.svm import SVC

from lesk.utils import get_sense_key, get_synsets, get_synset_definition, evaluate_accuracy
from lesk.preprocessing import string_to_number, lemmatize, stem


STOPWORDS = set(stopwords.words('english'))
PUNCTUATION = string.punctuation # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
VECTORIZER = CountVectorizer()
LE = preprocessing.LabelEncoder()

class Data:
    
    def __init__(self, wsd_instances, wsd_keys):
        self.instances = wsd_instances
        self.keys = wsd_keys
        
        self.lemmas = self._get_lemmas()
        self.definitions = self._get_definitions()
        
        self.inputs = self._get_inputs()
        self.targets = self._get_targets()
    
    def _preprocess(self, sentence):
        for punc in PUNCTUATION:
            sentence = sentence.replace(punc, '')
        sentence = sentence.lower()
        sentence = sentence.split() 
        sentence = [string_to_number(w) for w in sentence]
        sentence = [w for w in sentence if w not in STOPWORDS]
        sentence = [stem(w) for w in sentence]
        sentence = [lemmatize(w) for w in sentence]
        sentence = [w for w in sentence if w not in STOPWORDS]
        return sentence

    def _get_lemmas(self):
        """Return the set of all lemma"""
        lemmas = []
        for _, wsd in self.instances.items():
            lemma = wsd.lemma.decode("utf-8")
            lemmas.append(lemma)
        return set(lemmas)

    def _get_inputs(self):
        """return inputs in the format (lemma, context)"""
        inputs = {}
        for _, wsd in self.instances.items():
            
            lemma = wsd.lemma.decode("utf-8")
            if lemma not in inputs.keys():
                context = ' '.join([el.decode("utf-8") for el in wsd.context])
                context = ' '.join(self._preprocess(context))
                inputs[lemma] = [context]
                inputs[lemma] += [definition[0] for definition in self.definitions[lemma]]

        return inputs

    def _get_targets(self):
        targets = {}
        for id, wsd in self.instances.items():
            lemma = wsd.lemma.decode("utf-8")
            if lemma not in targets.keys():
                key = tuple(k for k in self.keys[id])
                targets[lemma] = [key[0]]
                targets[lemma] += [definition[1] for definition in self.definitions[lemma]]
        return targets

    def _get_definitions(self):
        definitions = {}
        for _, wsd in self.instances.items():
            lemma = wsd.lemma.decode("utf-8")
            lemma_definitions = get_synset_definition(lemma)
            lemma_synsets = get_synsets(lemma)
            lemma_sense_key = [get_sense_key(synset) for synset in lemma_synsets] 
            definitions[lemma] = [
                (' '.join(self._preprocess(definition)), sense) for definition, sense in zip(lemma_definitions, lemma_sense_key)
            ]
        return definitions
    
    def vocabulary(self):
        return [definition[0] for lemma in self.lemmas for definition in self.definitions[lemma]]

    def get_all_synset(self):
        return list(set(self.targets))
    


class SynsetClassifiers:

    VECTORIZER = CountVectorizer()
    LE = preprocessing.LabelEncoder

    def __init__(self, data, vectorizer=VECTORIZER, label_encoder=LE):
        self.data = data
        self.lemmas = data.lemmas
        self.inputs = data.inputs
        self.targets = data.targets

        self.vectorizer = vectorizer
        self.vectorizer.fit(data.vocabulary())

        self.label_encoder = label_encoder

        self.classifiers = {}

    def train(self):
        for lemma in self.lemmas:

            le = preprocessing.LabelEncoder()

            lemma_inputs = self.inputs[lemma] 
            lemma_targets = self.targets[lemma]

            label_encoder = le.fit(lemma_targets)
            
            X = self.vectorizer.transform(lemma_inputs)
            y = label_encoder.transform(lemma_targets)
            
            self.classifiers[lemma] = {}

            if len(set(lemma_targets)) <= 1:
                self.classifiers[lemma]['model'] = lemma_targets[0]
                self.classifiers[lemma]['le'] = None
            else:
                self.classifiers[lemma]['model'] = SVC(C=0.1, kernel='rbf')
                self.classifiers[lemma]['model'].fit(X, y)
                self.classifiers[lemma]['le'] = label_encoder

    def baseline(self, lemma):
        """returns most common sense for input wsd.lemma

        Arg
            lemma: (string) the lemma of a word

        return: (list) list of possible sense for this lemma
        """
        synset = get_synsets(lemma)
        return get_sense_key(synset[0])

    def predict(self, wsd):
        lemma = wsd.lemma.decode("utf-8")
        context = ' '.join([el.decode("utf-8") for el in wsd.context])
        context = ' '.join(self.data._preprocess(context))
        context = self.vectorizer.transform([context])

        if lemma not in self.lemmas:
            return self.baseline(lemma)

        if type(self.classifiers[lemma]['model']) is not str:
            pred = self.classifiers[lemma]['model'].predict(context)
            pred = self.classifiers[lemma]['le'].inverse_transform(pred)
        else:
            pred = self.classifiers[lemma]['model']

        return pred

    def evaluate(self, instances, keys):
        correct, total = 0, 0
        for id, wsd in instances.items():
            pred = self.predict(wsd)
            if type(pred) is list:
                pred = pred[0]
            target = keys[id]
            if pred in target:
                correct += 1
            total += 1
        print(total)
        return round((correct / total) * 100, 4)

def run_experiment(dev_instances, dev_key, test_instances, test_key):
    
    data = Data(test_instances, test_key)

    clf = SynsetClassifiers(data)
    clf.train()
    
    train_acc = clf.evaluate(dev_instances, dev_key)
    test_acc = clf.evaluate(test_instances, test_key)

    
    print(test_acc)
    print(train_acc)