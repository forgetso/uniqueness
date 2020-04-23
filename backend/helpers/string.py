import re
from backend.lib.spelling import Spelling


# Cleans a word by lowering, replacing spaces with underscores and removes all non alphanumeric characters

def clean_word(word):
    word = str(word)
    word = word.replace(' ', '_').replace('.', '').lower()
    word = re.sub('[^0-9a-z_]+', '', word)

    return word


# Returns
#   spelling incorrect (True/False)
#   list of corrections
#   dict of corrections keyed on word(s) from response
def check_spelling(cue, response):
    if not cue or not response:
        raise ValueError('Missing cue or response. Cue: {}, Response: {}'.format(cue, response))
    corrections_list = []
    if '_' in response:
        response_list = response.split('_')
    else:
        response_list = response.split(' ')
    corrections_dict = {}
    for response_word in response_list:
        speller = Spelling(cue, response_word)
        corrections = speller.get_corrections()
        if corrections:
            corrections_dict[response_word] = corrections
            corrections_list.extend(corrections)

    if len(corrections_list):
        return True, corrections_list, corrections_dict
    return False, corrections_list, corrections_dict


