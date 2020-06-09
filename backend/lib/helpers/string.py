import re
from backend.lib.cueresponsespelling import CueResponseSpelling
import numpy as np


# Cleans a word by lowering, replacing spaces with underscores and removes all non alphanumeric characters

def clean_word(word):
    word = str(word)
    word = word.replace(' ', '_').replace('.', '').lower()
    word = re.sub('[^0-9a-z_]+', '', word)

    return word


# def convert_numbers_to_text(word):
#     text_word = word
#     if re.match('^\d+$', word):
#         text_word = num2words.num2words(word)
#         print('converted {} to {}'.format(word, text_word))
#     return text_word


# Returns
#   spelling incorrect (True/False)
#   list of corrections
#   dict of corrections keyed on word(s) from response
def check_spelling(cue, response):
    if not response:
        raise ValueError('Missing response. Response: {}'.format(response))
    corrections_list = []
    if '_' in response:
        response_list = response.split('_')
    else:
        response_list = response.split(' ')
    corrections_dict = {}
    for response_word in response_list:
        speller = CueResponseSpelling(cue, response_word)
        corrections = speller.get_corrections()
        if corrections:
            corrections_dict[response_word] = corrections
            corrections_list.extend(corrections)

    if len(corrections_list):
        return True, corrections_list, corrections_dict
    return False, corrections_list, corrections_dict


def syllables(word):
    syllable_count = 0
    vowels = 'aeiouy'
    if word[0] in vowels:
        syllable_count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            syllable_count += 1
    if word.endswith('e'):
        syllable_count -= 1
    if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
        syllable_count += 1
    if syllable_count == 0:
        syllable_count += 1
    return syllable_count


def encode_word_as_int(word):
    # https://math.stackexchange.com/questions/1684783/how-can-i-convert-this-unique-string-of-characters-into-a-unique-number/1684810
    chars = str('abcdefghijklmnopqrstuvwxyz0123456789')
    word_int = np.int32(0)
    word_len = np.int(len(word)) - 1
    for idx, c in enumerate(word):
        # print('letter {} index {}'.format(c, chars.index(c)))
        # print('power is {}'.format(word_len - np.int(idx)))
        # print('calculating {} * {} ** {}'.format(chars.index(c), 36, (word_len - idx)))
        word_int = word_int + np.int(chars.index(c)) * (np.int(36) ** (word_len - np.int(idx)))
    # print(type(word_int))
    return word_int
