from backend.lib.db import get_cue_response_pairs, get_affixes, get_common_norms, get_prevalences, get_concretenesses, \
    get_acquisition_ages, get_emotions, save_coded_pairs, delete_coded_pairs
from nltk.stem import PorterStemmer
import re
import jellyfish
import unidecode
from backend.lib.wordnet_relations import WordnetRelations
from collections import Counter
import pprint
from decimal import Decimal
from backend.lib.multiprocessor import dispatch_jobs, jobs_running
import time
import argparse
from backend.importers.semantic_atlas import getSemanticArea
from backend.lib.spelling import Spelling

AFFIXES = [x.get('affix') for x in get_affixes()]
PREVALENCES = get_prevalences()
CONCRETENESSES = get_concretenesses()
ACQUISITION_AGES = get_acquisition_ages()
EMOTIONS = get_emotions()
COMMON_NORMS = get_common_norms()
ps = PorterStemmer()
TOTALS = {'phonology': 0, 'exact': 0, 'syntagmatic': 0, 'affix': 0, 'prevalent': 0}


def main(responses=None):
    delete_coded_pairs()
    print(responses)
    all_pairs = [x for x in get_cue_response_pairs(responses=responses)]
    start = time.perf_counter()
    # job_number = mp.cpu_count() - 1
    job_number = 1
    jobs = dispatch_jobs(all_pairs, job_number, get_coding)
    while jobs_running(jobs):
        print('Jobs running')
        time.sleep(5)
    else:
        end = time.perf_counter()
        print('time taken {}'.format(end - start))


def get_coding(all_pairs):
    # TODO consider lists of films, books and music
    # TODO consider the categories found in the Gallant Research:
    # TODO (visual, tactile, outdoor, bodypart, number, place, violence, person, social, mental, time)
    # TODO consider when cue and response form one word, e.g. fear + less = fearless
    # s2v = Sense2Vec().from_disk(SENSE2VEC_FOLDER)

    erratic = set()
    # code distinct list of cue response pairs
    coded_pairs = {}
    cue_response_pairs, distinct_cue_response_pairs = format_cue_response_pairs(all_pairs)

    totals = Counter(TOTALS)
    fieldnames = None
    for pair in distinct_cue_response_pairs:
        cue = pair[0]
        response = pair[1]
        response_raw = pair[2]

        # print(cue, response)

        if len(cue) > 3:
            coded_pairs[(cue, response)] = basic_coding(response, response_raw)
            relations_coding, relations_totals = relation_coding(cue, response)
            totals = totals + relations_totals

            coded_pairs[(cue, response)].update(relations_coding)

            ############## Other Categories ##############

            # incorrect spelling
            incorrect_spelling = check_spelling(cue, response_raw)
            totals['incorrect_spelling'] += incorrect_spelling * 1
            coded_pairs[(cue, response)]['incorrect_spelling'] = incorrect_spelling
            if incorrect_spelling:
                print(cue, response_raw)

            # prevalence
            word_found, prevalent = check_prevalence(response)
            totals['prevalent'] += 1 * word_found
            coded_pairs[(cue, response)]['prevalent'] = prevalent

            # TODO 'corpus_similarity': None,
            # TODO 'dictionary_similarity': None,

            # concreteness
            word_found, concrete = check_concreteness(response)
            totals['concrete'] += 1 * word_found
            coded_pairs[(cue, response)]['concrete'] = concrete

            # age_of_acquisition
            word_found, acquisition_age = check_acquisition_age(response)
            totals['age'] += 1 * word_found
            coded_pairs[(cue, response)]['age'] = acquisition_age

            # word emotions (valence, arousal dominance)
            word_found, emotions = check_emotions(response)

            totals['emotions'] += 1 * word_found
            coded_pairs[(cue, response)]['valence'] = emotions['valence']
            coded_pairs[(cue, response)]['arousal'] = emotions['arousal']
            coded_pairs[(cue, response)]['dominance'] = emotions['dominance']
            # TODO 'subtitle_frequency': None,
            # take this from
            # TODO 'orthographic_neighbourhood': None,
            # TODO 'phonograpic_neighbourhood': None,
            # TODO 'part_of_speech': None,
            # take this from age of acquisition?
            # TODO 'morpheme': False,
            # TODO 'cultural_reference': [movies, music, famous people, books, poems, computer games, tv shows, brands]
            # active_brain_region
            # TODO check when semantic area differs from cue semantic area
            word_found, semantic_area = getSemanticArea(response)
            coded_pairs[(cue, response)]['semantic_area'] = semantic_area
            totals['semantic_areas'] += 1 * word_found

            # create the vector for the machine learning algorithm
            # coded_pairs[(cue, response)]['feature_vector'] = construct_feature_vector(v)
    if not fieldnames:
        first_key = next(iter(coded_pairs))
        fieldnames = ['cue', 'response'] + list(coded_pairs[first_key].keys())

    save_coded_pairs(coded_pairs, fieldnames)
    pprint.pprint(totals)
    c = totals
    l = all_pairs
    d = [(i, c[i] / len(l) * 100.0) for i, count in c.most_common()]
    pprint.pprint(d)
    return coded_pairs


def format_cue_response_pairs(all_pairs):
    cue_response_pairs = []
    distinct_cue_response_pairs = set()

    for pair in all_pairs:
        pair = dict(pair)
        cue = str(pair['cue'].lower())
        response = pair['response'].replace(' ', '_').replace('.', '').lower()
        response_raw = pair['response_raw'].replace(' ', '_').lower()
        pair['cue'] = cue
        pair['response'] = response
        pair['key'] = (cue, response)
        cue_response_pairs.append(pair)
        distinct_cue_response_pairs.add((pair['cue'], pair['response'], pair['response_raw']))
    return cue_response_pairs, distinct_cue_response_pairs


def get_age_bracket(age):
    bracket = None
    if age < 25:
        bracket = '<25'
    elif age >= 25 and age < 35:
        bracket = '25-34'
    elif age >= 35 and age < 45:
        bracket = '35-44'
    elif age >= 45 and age < 55:
        bracket = '45-54'
    elif age >= 55 and age < 65:
        bracket = '55-64'
    elif age >= 65:
        bracket = '>=65'
    return bracket


def basic_coding(response, response_raw):
    basic = {'length': None,
             # 'response_time': None,
             'syllables': None,
             'capitalised': False,
             'punctuation': False,
             'number_of_words': None}
    basic['length'] = len(response)
    basic['syllables'] = syllables(response)
    basic['capitalised'] = str.isupper(response[0])
    basic['punctuation'] = True if re.match(r'[\.\-\']', response_raw) else False
    basic['number_of_words'] = len(response.split('_'))

    return basic


def relation_coding(cue, response):
    # First do the relations
    totals = Counter({'phonology': 0, 'exact': 0, 'syntagmatic': 0, 'affix': 0})
    relation_obj = WordnetRelations(cue, response)
    relations = relation_obj.relations
    for type in relation_obj.relation_types:
        if relation_obj.relations[type]:
            totals.setdefault(type, 0)
            totals[type] += 1

    # phonology
    relations['phonology'] = False
    phonology, type = check_phonology(cue, response)
    if phonology:
        totals['phonology'] += 1
        relations['phonology'] = True

    # exact
    relations['exact'] = False
    if cue == response:
        totals['exact'] += 1

    # syntagmatic
    relations['syntagmatic'] = False
    syntagmatic, type = check_syntagmatic(cue, response)
    if syntagmatic:
        totals['syntagmatic'] += 1
        relations['syntagmatic'] = True

    # add s2v check for syntagmatic

    # add proverb check for syntagmatic

    # affix
    relations['affix'] = False
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

    if jellyfish.levenshtein_distance(cue, response) < 2:
        return 2, 'levenshtein'

    return 0, None


def check_affix(cue, response):
    if cue in response:
        remainder = response.split(cue)
        for r in remainder:
            if r in AFFIXES:
                return True

    remainder = response.split(cue)
    if len(remainder) == 2 and (len(remainder[0]) == 0 or len(remainder[1]) == 0):
        #print(cue, response, '... response attached to the start or to the end of the cue')
        return True

    if ps.stem(cue) == ps.stem(response):
        return True


def check_prevalence(response):
    # TODO come up with a more in depth analysis of prevalence values.
    # 0,1,1.5,2,2.5
    if response in PREVALENCES:
        if PREVALENCES[response] > 0:
            return True, True
        else:
            return False, False
    return False, False


def check_concreteness(response):
    # TODO maybe make use of other measures in concreteness table
    stem = ps.stem(response)
    if response in CONCRETENESSES:
        if CONCRETENESSES[response] >= 3:
            return True, True
        else:
            return True, False
    elif stem in CONCRETENESSES:
        if CONCRETENESSES[stem] >= 3:
            return True, True
        else:
            return True, False
    # else:
    #     raise KeyError('{} missing from concretenesses'.format(response))
    return False, False


def check_acquisition_age(response):
    if response in ACQUISITION_AGES:
        return True, round(ACQUISITION_AGES['age'], 1)
    return False, 0


def check_emotions(response):
    if response in EMOTIONS:
        return True, EMOTIONS[response]
    return False, {'valence': 0, 'arousal': 0, 'dominance': 0}


def check_spelling(cue, response):
    if '_' in response:
        wordlist = response.split('_')
    elif ' ' in response:
        wordlist = response.split(' ')
    else:
        wordlist = []
    for word in wordlist:
        spellcheck = Spelling(cue, word)
        corrections = spellcheck.get_corrections()
        if len(corrections):
            return True
    return False


def construct_feature_vector(encoding):
    # feature_vector = np.array()
    vector_list = []
    for encoding_type in encoding:
        for key in encoding[encoding_type]:
            if encoding[encoding_type][key] == True:
                vector_list.append(1)
            elif encoding[encoding_type][key] == False:
                vector_list.append(0)
            elif isinstance(encoding[encoding_type][key], Decimal):
                vector_list.append(encoding[encoding_type][key])
            elif isinstance(encoding[encoding_type][key], int):
                vector_list.append(encoding[encoding_type][key])
            elif isinstance(encoding[encoding_type][key], float):
                vector_list.append(encoding[encoding_type][key])
            else:
                print(encoding[encoding_type][key])
                print('error')

    return vector_list


def setup():
    parser = argparse.ArgumentParser(description='Import various types of word data')
    parser.add_argument('--responses', action='store_true', help='use responses table for coding')
    args = parser.parse_args()
    main(args.responses)


if __name__ == "__main__":
    setup()
