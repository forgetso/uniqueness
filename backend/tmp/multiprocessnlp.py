import spacy


class MultiProcessNLP(object):
    def __init__(self):
        self.name = None
        self.dtype = None
        self.shape = None

    def load(self, path):
        if self.name is not None and self.dtype is not None and self.shape is not None:
            return spacy.load(path, shared={'name': self.name, 'dtype': self.dtype, 'shape': self.shape})
        else:
            return spacy.load(path)

