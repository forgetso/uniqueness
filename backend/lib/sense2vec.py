from sense2vec import Sense2Vec
from backend.settings import SENSE2VEC_FOLDER


# TODO add more functionality for dealing with s2v
class s2vWrap:
    def __init__(self):
        self.s2v = self.get_sense_2vec()

    def get_sense_2vec(self):
        print('loading sense2vec')
        s2v = Sense2Vec().from_disk(SENSE2VEC_FOLDER)
        print('loaded sense2vec')
        return s2v

    # moved this away from main coding file to avoid loading s2v every time other modules import functions from coding

    def create_s2v_keys(self, word):
        keys = []
        word_types = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN',
                      'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
        named_entities = ['PERSON', 'NORP', 'FACILITY', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT' 'WORK_OF_ART',
                          'LANGUAGE']
        all_things = word_types + named_entities
        for word_type in all_things:
            keys.append(''.join([word, '|', word_type]))
        return keys

    def get_best_sense(self, word):
        return self.s2v.get_best_sense(word)
