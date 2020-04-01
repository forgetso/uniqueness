from backend.lib.db import get_cue_response_pairs, get_affixes, get_common_norms, get_prevalences
from nltk.stem import PorterStemmer
import re
import jellyfish
import unidecode
from sense2vec import Sense2Vec
from backend.settings import SENSE2VEC_FOLDER
from backend.lib.wordnet_relations import WordnetRelations

AFFIXES = [x.get('affix') for x in get_affixes()]
PREVALENCES = get_prevalences()
COMMON_NORMS = get_common_norms()
ps = PorterStemmer()
TOTALS = {'phonology': 0, 'exact': 0, 'syntagmatic': 0, 'affix': 0}


def main():
    # TODO consider lists of films, books and music
    # TODO consider the categories found in the Gallant Research:
    # TODO (visual, tactile, outdoor, bodypart, number, place, violence, person, social, mental, time)

    s2v = Sense2Vec().from_disk(SENSE2VEC_FOLDER)

    cue_response_pairs = []
    distinct_cue_response_pairs = set()
    all_pairs = get_cue_response_pairs(limit=100)
    for pair in all_pairs:
        pair = dict(pair)
        cue = pair['cue'].lower()
        response = pair['response'].replace(' ', '_').replace('.', '').lower()
        response_raw = pair['r1raw'].replace(' ', '_').lower()
        pair['cue'] = cue
        pair['response'] = response
        cue_response_pairs.append(pair)
        distinct_cue_response_pairs.add((pair['cue'], pair['response']))
        pair['key'] = (cue, response)
        pair['basic'] = basic_coding(response, response_raw)

    erratic = set()
    # code distinct list of cue response pairs
    coded_pairs = {}

    for pair in distinct_cue_response_pairs:
        cue = pair[0]
        response = pair[1]
        print(cue, response)

        if len(cue) > 3:
            relations_coding, totals = relation_coding(cue, response)
            coded_pairs[(cue, response)]['relations'] = relations_coding
            # now check the other things
            # 'prevalence': None,
            if check_prevalence(response):
                totals['prevalent'] += 1
                coded_pairs[(cue, response)]['other']['prevalent'] = True
            # 'corpus_similarity': None,
            # 'dictionary_similarity': None,
            # 'concreteness': None,
            # 'age_of_acquisition': None,
            # 'emotion': None,
            # 'subtitle_frequency': None,
            # 'orthographic_neighbourhood': None,
            # 'phonograpic_neighbourhood': None,
            # 'part_of_speech': None,
            # 'morpheme': False,
            # 'cultural_reference': None,
            # 'active_brain_region': None,

    print(totals)
    print(sum(totals.values()))


def basic_coding(response, response_raw):
    basic = {'length': None,
             'response_time': None,
             'syllables': None,
             'capitalised': False,
             'punctuation': False,
             'number_of_words': None,
             'incorrect_spelling': False}
    basic['length'] = len(response)
    basic['syllables'] = syllables(response)
    basic['capitalised'] = str.isupper(response[0])
    basic['punctuation'] = re.match(r'[\.\-\']') or False
    basic['response'] = len(response.split('_'))
    basic['incorrect_spelling'] = response_raw.lower() != response
    return basic


def relation_coding(cue, response):
    # First do the relations
    totals = {}
    relation_obj = WordnetRelations(cue, response)
    relations = relation_obj.relations
    for type in relation_obj.relation_types:
        if relation_obj.relations[type]:
            totals.setdefault(type, 0)
            totals[type] += 1

    # phonology
    phonology, type = check_phonology(cue, response)
    if phonology:
        totals['phonology'] += 1
        relations['phonology'] = True

    # exact
    if cue == response:
        totals['exact'] += 1

    # syntagmatic
    syntagmatic, type = check_syntagmatic(cue, response)
    if syntagmatic:
        totals['syntagmatic'] += 1
        relations['syntagmatic'] = True

    # add s2v check for syntagmatic

    # add proverb check for syntagmatic

    # affix
    if check_affix(cue, response):
        totals['affix'] += 1
        relations['affix'] = True
    return relations, totals


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


def create_s2v_keys(word):
    keys = []
    word_types = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN',
                  'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
    for word_type in word_types:
        keys.append(''.join([word, '|', word_type]))
    return keys


def check_syntagmatic(cue, response):
    if '_' in response:
        response_split = response.split('_')
        if cue in response:
            cue_index = response.index(cue)

            if cue_index == 0 and response.index(response) == len(cue) + 1:
                return 2, 'consecutivexy'
            elif cue_index > 0 and len(response_split) == 2:
                return 2, 'consecutiveyx'
            else:
                return 2, 'othercollocational'

    return 0, None


def form_based(cue, response):
    if cue == response:
        return 3, 'exact'

    # check to see if y is x plus or minux an affix
    if check_affix(cue, response):
        return 3, 'affix'

    remainder = response.split(cue)
    if len(remainder) == 2:
        print(cue, response, 'other_form')
        return 3, 'other_form'

    return 0, None


# Soundex similarity check
def check_phonology(cue, response):
    # Try to get a cue soundex
    cue_decoded = unidecode.unidecode(cue)
    cue_soundex = jellyfish.soundex(cue_decoded)
    # cue_letters = re.sub(r'[0-9]', '', cue_soundex)
    # cue_numbers = cue_soundex.replace(cue_letters, '')

    # Try to get a response soundex
    response_decoded = unidecode.unidecode(response.replace('_', ''))
    response_soundex = jellyfish.soundex(response_decoded)
    # response_letters = re.sub(r'[0-9]', '', response_soundex)
    # response_numbers = response_soundex.replace(response_letters, '')

    if cue_soundex == response_soundex:
        return 3, 'soundex'

    if jellyfish.levenshtein_distance(cue, response) < 3:
        print(cue, response, 'levenshtein')
        return 3, 'levenshtein'

    return 0, None


def erratic_association(cue, response):
    return


def check_affix(cue, response):
    if cue in response:
        remainder = response.split(cue)
        for r in remainder:
            if r in AFFIXES:
                return True

    if ps.stem(cue) == ps.stem(response):
        return True


def check_prevalence(response):
    # TODO come up with a more in depth analysis of prevalence values.
    # 0,1,1.5,2,2.5
    if response in PREVALENCES:
        if PREVALENCES[response] > 0:
            return True
    return False


if __name__ == "__main__":
    main()
