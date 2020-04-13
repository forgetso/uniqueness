import numpy as np
from backend.settings import SEMANTIC_SPACE_FOLDER

#TODO import this data to the database
semantic_space_data = np.load('{}/semantic_space.npz'.format(SEMANTIC_SPACE_FOLDER))
english1000_vocab = semantic_space_data['english1000_vocab']
english1000_embedding = semantic_space_data['english1000_embedding']
pragmatic_info = np.load('{}/pragmatic-info.npz'.format(SEMANTIC_SPACE_FOLDER))
lsemwts = pragmatic_info['lsemwts']

def getSemanticArea(word):
    result = ''
    found = False
    byte_word = bytes(word, encoding="ascii")
    if np.where(english1000_vocab == byte_word)[0].size != 0:
        w = np.vstack([english1000_embedding[:, np.where(english1000_vocab == byte_word)[0][0]]]).mean(0)
        response = np.dot(w, lsemwts)  # the predicted response of each of the 985 features to the list of words
        # The index area in which the word had the highest response
        index = np.where(response == response.max())
        # label assigned to area
        label = pragmatic_info['lnames'][index]
        # is this a significant area? True or False
        if pragmatic_info['lsignif'][index][0]:
            result = label[0].decode("utf-8")
            found = True
    return found, result
