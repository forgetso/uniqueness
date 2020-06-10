from sense2vec import Sense2Vec
from backend.settings import SENSE2VEC_FOLDER, DATA_FOLDER
from multiprocessing import Array
import resource
import os.path

BYTE_FILE = ''.join([DATA_FOLDER, '/s2v_bytestring'])
WORD_TYPES = ('ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN',
              'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X')
NUM_TYPES = ('CARDINAL', 'ORDINAL', 'DATE', 'TIME', 'QUANTITY', 'MONEY', 'PERCENT')
NAMED_ENTITITES = ('PERSON', 'NORP', 'FACILITY', 'ORG', 'GPE', 'LOC','LAW', 'PRODUCT', 'EVENT', 'WORK_OF_ART',
                   'LANGUAGE')


# TODO add more functionality for dealing with s2v
class s2vWrap:
    def __init__(self):
        if not os.path.isfile(BYTE_FILE):
            print('loading sense2vec')
            s2v = Sense2Vec().from_disk(SENSE2VEC_FOLDER)
            print('Used {} RAM'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024))
            print('converting sense2vec to bytestring')
            s2v_bytes = s2v.to_bytes()
            f = open(BYTE_FILE, "wb")
            print('Used {} RAM'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024))
            print('writing bytes file')
            f.write(s2v_bytes)
            print('Used {} RAM'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024))
            f.close()
            exit('Exiting to free up RAM')
        self.get_s2v()

    def get_shared_value(self):
        # f = open(BYTE_FILE, 'rb')
        # shared_value = Value(ctypes.c_char_p, f.read(), lock=False)
        # f.close()
        shared_value = Array(self.s2v.vectors)
        return shared_value

    def get_s2v(self):
        self.s2v = Sense2Vec().from_disk(SENSE2VEC_FOLDER)

    def slice_byte_string(self, byte_string):
        byte_string_len = len(byte_string)
        slices = byte_string_len % 1024
        for i in range(0, byte_string_len, slices):
            yield byte_string[i:i + slices]

    def create_s2v_keys(self, word):
        keys = []
        all_things = WORD_TYPES + NAMED_ENTITITES
        for word_type in all_things:
            keys.append(''.join([word, '|', word_type]))
        return keys
