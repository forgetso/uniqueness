from backend.lib.db import get_cue_response_pairs, get_affixes, get_common_norms, get_prevalences, get_concretenesses, \
    get_acquisition_ages, get_emotions, save_coded_pairs, delete_coded_pairs, get_orthographic, get_phonographic, \
    get_all_words, get_part_of_speech
from backend.lib.contractions import replace_contraction
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
from backend.importers.semantic_atlas import BrainMap
from backend.lib.spelling import Spelling
from multiprocessing import shared_memory, cpu_count
# moved this away as it takes ages to load
from backend.lib.sense2vec import s2vWrap
import num2words
import logging
import numpy as np
import sys

AFFIXES = [x.get('affix') for x in get_affixes()]
PREVALENCES = get_prevalences()
CONCRETENESSES = get_concretenesses()
ACQUISITION_AGES = get_acquisition_ages()
PARTS_OF_SPEECH = get_part_of_speech()
EMOTIONS = get_emotions()
ORTHOGRAPHIC = get_orthographic()
PHONOGRAPHIC = get_phonographic()
COMMON_NORMS = get_common_norms()
ALL_WORDS = get_all_words()
ps = PorterStemmer()


TOTALS = {
    'phonology': 0,
    'exact': 0,
    'syntagmatic': 0,
    'affix': 0,
    'prevalent': 0,
    'part_of_speech': 0,
    'part_of_speech_changes': 0
}


def main():
    # TODO batch up the totals returned by the processes and print those instead of all the individual totals
    print('loading cue response pairs')
    all_pairs = [x for x in get_cue_response_pairs(limit=20000)]
    print('loaded cue response pairs')
    delete_coded_pairs()
    if not all_pairs:
        raise ValueError('No responses to code...')
    start = time.perf_counter()
    job_number = cpu_count() - 1
    # job_number = 1
    jobs = dispatch_jobs(all_pairs, job_number, get_coding)
    while jobs_running(jobs):
        print('Jobs running')
        time.sleep(5)
    else:
        end = time.perf_counter()
        print('time taken {}'.format(end - start))


def get_coding(all_pairs):
    # This algorithm is for coding DISTINCT cue response pairs once they have had their spelling corrected. It considers
    # categories such as prevalence, concreteness, age of acquisition, emotions, orthographic neighbourhoods,
    # phonographic neighbourhoods, active brain region.

    # TODO consider the categories found in the Gallant Research:
    # TODO (visual, tactile, outdoor, bodypart, number, place, violence, person, social, mental, time)
    # TODO consider when cue and response form one word, e.g. fear + less = fearless

    # initialise brain data
    brain_map = BrainMap()
    erratic = set()
    # code distinct list of cue response pairs
    coded_pairs = {}
    cue_response_pairs, distinct_cue_response_pairs = format_cue_response_pairs(all_pairs)

    totals = Counter(TOTALS)
    fieldnames = None
    s2v_wrap = s2vWrap()


    for pair in distinct_cue_response_pairs:
        cue = pair[0]
        response = pair[1]
        response_raw = pair[2]
        # TODO work out what to do with collisions between sources. shouldn't be that many reallky
        source = pair[3]
        response_decoded = unidecode.unidecode(response)
        cue_decoded = unidecode.unidecode(cue)
        response_stem = ps.stem(response_decoded)
        cue_stem = ps.stem(cue_decoded)
        # TODO use this throughout instead of splitting multiple times
        response_split = response.split('_')
        if len(cue) > 3:

            coded_pairs[(cue, response)] = basic_coding(response, response_raw)
            # coded_pairs[(cue, response)]['source'] = source
            relations_coding, relations_totals = relation_coding(cue, response)
            totals = totals + relations_totals

            coded_pairs[(cue, response)].update(relations_coding)

            ############## Other Categories ##############

            ### exact match
            exact = cue == response
            coded_pairs[(cue, response)]['exact'] = exact
            totals['exact'] += 1 * exact

            ### phonology (sound the same)
            phonology, type = check_phonology(cue_decoded, response_decoded)
            coded_pairs[(cue, response)]['phonology'] = phonology
            totals['phonology'] += 1 * phonology

            ### syntagmatic (x follows y, y follows x)
            syntagmatic, type = check_syntagmatic(cue, response)
            coded_pairs[(cue, response)]['syntagmatic'] = syntagmatic
            totals['syntagmatic'] += 1 * syntagmatic

            # TODO add s2v check for syntagmatic
            # TODO add proverb check for syntagmatic

            ### affix - the words are equal plus or minus an affix
            cue_affix = check_affix(cue, response)
            response_affix = check_affix(response, cue)
            affix = cue_affix or response_affix
            coded_pairs[(cue, response)]['affix'] = affix
            totals['affix'] += 1 * affix

            ### prevalence
            word_found, prevalent = check_prevalence(response)
            if not word_found:
                word_found, prevalent = check_prevalence(response_stem)
            totals['prevalent'] += 1 * word_found
            coded_pairs[(cue, response)]['prevalent'] = prevalent

            ### concreteness
            word_found, concrete = check_concreteness(response)
            if not word_found:
                word_found, concrete = check_concreteness(response_stem)
            totals['concrete'] += 1 * word_found
            coded_pairs[(cue, response)]['concrete'] = concrete

            ### age_of_acquisition
            word_found, acquisition_age = check_acquisition_age(response)
            if not word_found:
                word_found, acquisition_age = check_acquisition_age(response_stem)
            totals['age'] += 1 * word_found
            coded_pairs[(cue, response)]['age'] = acquisition_age

            ### word emotions (valence, arousal dominance)
            word_found, emotions = check_emotions(response)
            if not word_found:
                word_found, emotions = check_emotions(response_stem)
            totals['emotions'] += 1 * word_found
            coded_pairs[(cue, response)]['valence'] = emotions['valence']
            coded_pairs[(cue, response)]['arousal'] = emotions['arousal']
            coded_pairs[(cue, response)]['dominance'] = emotions['dominance']

            ### orthographic_neighbourhood
            word_found, orthographic_features = check_orthographic(cue, response)
            if not word_found:
                word_found, orthographic_features = check_orthographic(cue, response_stem)
            coded_pairs[(cue, response)].update(orthographic_features)
            totals['orthographic'] += 1 * word_found

            ### phonograpic_neighbourhood and subtitle frequency
            word_found, phonographic_features = check_phonographic(response)
            if not word_found:
                word_found, phonographic_features = check_phonographic(response_stem)
            coded_pairs[(cue, response)].update(phonographic_features)
            totals['phonographic'] += 1 * word_found

            ### semantic atlas from Gallant et. al
            word_found, response_semantic_area = brain_map.get_semantic_area(response_decoded)
            if not word_found:
                word_found, response_semantic_area = brain_map.get_semantic_area(response_stem)
            coded_pairs[(cue, response)]['semantic_area'] = response_semantic_area
            totals['semantic_areas'] += 1 * word_found
            word_found, cue_semantic_area = brain_map.get_semantic_area(cue_decoded)
            if not word_found:
                word_found, cue_semantic_area = brain_map.get_semantic_area(cue_stem)
            changes = response_semantic_area != cue_semantic_area
            coded_pairs[(cue, response)]['semantic_area_changes'] = changes
            totals['semantic_area_changes'] += 1 * changes

            # now get the coordinates of the cue and response in the brain model and calculate the euclidean distance
            # between them
            coded_pairs[(cue, response)]['semantic_distance'] = 0
            if changes and cue_semantic_area and response_semantic_area:
                coded_pairs[(cue, response)]['semantic_distance'] = brain_map.get_distances_between_semantic_areas(
                    cue_semantic_area, response_semantic_area).astype(float)
                totals['semantic_distance'] += 1

            ### part_of_speech
            word_found, part_of_speech_response = check_part_of_speech(s2v_wrap.s2v, response)
            if not word_found:
                word_found, part_of_speech_response = check_part_of_speech(s2v_wrap.s2v, response_stem)
            totals['part_of_speech'] += 1 * word_found
            coded_pairs[(cue, response)]['part_of_speech'] = part_of_speech_response
            word_found, part_of_speech_cue = check_part_of_speech(s2v_wrap.s2v, cue)
            if not word_found:
                word_found, part_of_speech_cue = check_part_of_speech(s2v_wrap.s2v, cue_stem)
            coded_pairs[(cue, response)]['part_of_speech_changes'] = False
            if word_found and part_of_speech_response != part_of_speech_cue:
                totals['part_of_speech_changes'] += 1
                coded_pairs[(cue, response)]['part_of_speech_changes'] = True

            ### list of all words ### this should come last
            coded_pairs[(cue, response)]['contraction'] = False
            if response in ALL_WORDS:
                # the word appears in the giant list of english words
                totals['all_words'] += 1
            elif replace_contraction(response) != response:
                # include the word but note that it happened
                coded_pairs[(cue, response)]['contraction'] = True
                totals['contractions'] += 1
            elif len(response.split('_')) == 1 and not part_of_speech_response:
                # this is basically a list of responses that need addressed by future changes
                print('Word missing from all_words = {}. Removing from coded_pairs.'.format(response))
                coded_pairs.pop((cue, response))

            # TODO  obscene words
            # TODO  sexual references
            # TODO  slang
            # TODO  part_of_speech: take this from age of acquisition?
            # TODO  morpheme
            # TODO  years (1960, 2010, etc.)
            # TODO  cultural_reference:
            #  [movies,
            #   music,
            #   famous people,
            #   books (1984, farenheit 451),
            #   poems,
            #   computer games,
            #   tv shows,
            #   brands,
            #   historical dates,
            #   characters,
            #   superheroes,
            #   religion,
            #   web sites,
            #   web status codes (404),
            #   emergency services numbers,
            #   scientific numbers (365, pi, gravity, phi, etc.)
            #
            #   ]
            #   The part of speech (Person, etc.) is a good start for deriving further information from the full cue/
            #   response pair. For example, (portal, chell). Chell is picked up as a person but is a character in the
            #   game, Portal.
            # TODO  company reference [walmart, asda, etc.]
            # TODO  geographical_reference [country, state, city, town, village, post code]
            # TODO  corpus_similarity
            # TODO  dictionary_similarity
    if not fieldnames:
        first_key = next(iter(coded_pairs))
        fieldnames = ['cue', 'response'] + list(coded_pairs[first_key].keys())

    # print(fieldnames)
    # save data to the db
    save_coded_pairs(coded_pairs, fieldnames)

    # pprint.pprint(totals)
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
        cue = str(pair['cue']).replace(' ', '_').replace('.', '').lower()
        response = pair['response'].replace(' ', '_').replace('.', '').lower()
        # this is only really relevant for the wordassoc table
        response_raw = pair['response_raw'].replace(' ', '_').lower()
        pair['cue'] = cue
        pair['response'] = response
        pair['key'] = (cue, response)
        cue_response_pairs.append(pair)
        distinct_cue_response_pairs.add((pair['cue'], pair['response'], pair['response_raw'], pair['source']))
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
    totals = Counter()
    relation_obj = WordnetRelations(cue, response)
    relations = relation_obj.relations
    for relation_type in relation_obj.relation_types:
        if relation_obj.relations[relation_type]:
            totals.setdefault(relation_type, 0)
            totals[relation_type] += 1
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



def check_syntagmatic(cue, response):
    # TODO make this work for compounds, common phrases that are not complete in the response
    # e.g. cue is "back" and response is "and forth" (use s2v)
    if '_' in response:
        response_split = response.split('_')
        if cue in response:
            if cue in response_split:
                cue_index = response_split.index(cue)

                if cue_index == 0:
                    return True, 'consecutivexy'
                elif cue_index == 1:
                    return True, 'consecutiveyx'
                else:
                    return True, 'othercollocational'
            else:
                return True, 'othercollocational'

    return False, None


# Soundex similarity check
def check_phonology(cue, response):
    # Try to get a cue soundex
    cue_soundex = jellyfish.soundex(cue)

    # Try to get a response soundex
    response_soundex = jellyfish.soundex(response)
    if cue_soundex == response_soundex:
        return True, 'soundex'

    levenshtein_distance = jellyfish.levenshtein_distance(cue, response)
    if cue_soundex[1:] == response_soundex[1:] and levenshtein_distance <= 3:
        return True, 'soundex'

    # if cue_soundex[-2:] == response_soundex[-2:] and levenshtein_distance <= 3:
    #     return True, 'soundex'

    # TODO add zero to the end of shorter  last two?
    try:
        # e.g. C630 M300
        if response_soundex[-3:].index(cue_soundex[-2:]) > -1 and levenshtein_distance <= 3:
            return True, 'soundex'
    except ValueError:
        pass

    try:
        # e.g. C630 M300
        if cue_soundex[-3:].index(response_soundex[-2:]) > -1 and levenshtein_distance <= 3:
            return True, 'soundex'
    except ValueError:
        pass

    if levenshtein_distance < 2:
        return True, 'levenshtein'

    return False, None


def check_affix(cue, response):
    # TODO look closer at categories of affix https://en.wikipedia.org/wiki/Affix
    if cue in response:
        remainder = response.split(cue)
        for r in remainder:
            if r in AFFIXES:
                return True

    remainder = response.split(cue)
    if len(remainder) == 2 and (len(remainder[0]) == 0 or len(remainder[1]) == 0):
        # print(cue, response, '... response attached to the start or to the end of the cue')
        return True

    if ps.stem(cue) == ps.stem(response):
        return True

    return False


def check_prevalence(response):
    # TODO come up with a more in depth analysis of prevalence values.
    # TODO handle multiple words for prevalences
    # 0,1,1.5,2,2.5
    present = False
    if response in PREVALENCES:
        if PREVALENCES[response] > 0:
            present = True
            return present, True
    return present, False


def check_concreteness(response):
    # TODO maybe make use of other measures in concreteness table
    if response in CONCRETENESSES:
        if CONCRETENESSES[response] >= 3:
            return True, True
        else:
            return True, False
    # else:
    #     raise KeyError('{} missing from concretenesses'.format(response))
    return False, False


def check_acquisition_age(response):
    if response in ACQUISITION_AGES:
        return True, round(ACQUISITION_AGES[response], 1)
    return False, 0


def check_part_of_speech(s2vobj, response):
    if response in PARTS_OF_SPEECH and PARTS_OF_SPEECH[response] != 0:
        return True, PARTS_OF_SPEECH[response].lower()
    else:
        best_sense = s2vobj.get_best_sense(response)
        if best_sense:
            return True, best_sense.split('|')[1].lower()

    return False, 0


def check_emotions(response):
    if response in EMOTIONS:
        return True, EMOTIONS[response]
    return False, {'valence': 0, 'arousal': 0, 'dominance': 0}


def check_spelling(cue, response):
    if not cue or not response:
        raise ValueError('Missing cue or response. Cue: {}, Resonse: {}'.format(cue, response))
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


def check_orthographic(cue, response):
    features = {'response_ortho_neighbours': None,
                'unigram_count': None,
                'bigram_count': None,
                'trigram_count': None,
                'unconstrained_unigram_count': None,
                'unconstrained_bigram_freq': None,
                'unconstrained_trigram_freq': None,
                'ortho_neighbours_diff': None
                }
    if response in ORTHOGRAPHIC:
        features['response_ortho_neighbours'] = ORTHOGRAPHIC[response]['ortho_neighbours']
        features['unigram_count'] = ORTHOGRAPHIC[response]['unigram_count']
        features['bigram_count'] = ORTHOGRAPHIC[response]['bigram_count']
        features['trigram_count'] = ORTHOGRAPHIC[response]['trigram_count']
        features['unconstrained_unigram_count'] = ORTHOGRAPHIC[response]['unconstrained_unigram_count']
        features['unconstrained_bigram_freq'] = ORTHOGRAPHIC[response]['unconstrained_bigram_freq']
        features['unconstrained_trigram_freq'] = ORTHOGRAPHIC[response]['unconstrained_trigram_freq']
        if cue in ORTHOGRAPHIC:
            cue_ortho_neighbours = ORTHOGRAPHIC[cue]['ortho_neighbours']
            features['ortho_neighbours_diff'] = features['response_ortho_neighbours'] - cue_ortho_neighbours
        return True, features
    return False, features


# TODO phoneme count difference between cue and response
def check_phonographic(response):
    features = {'nphon': None,
                'unsdens': None,
                'strdens': None,
                'unsbpav': None,
                'strbpav': None,
                'unstpav': None,
                'strtpav': None,
                'unspospav': None,
                'strpospav': None,
                'unslcpospav': None,
                'strlcpospav': None,
                'sfreq': None,
                'scdcnt': None
                }
    if response in PHONOGRAPHIC:
        features = {k: v for k, v in PHONOGRAPHIC[response].items() if k in features.keys()}
        return True, features
    return False, features


def convert_numbers_to_text(word):
    text_word = word
    if re.match('^\d+$', word):
        text_word = num2words.num2words(word)
        print('converted {} to {}'.format(word, text_word))
    return text_word


def setup():
    parser = argparse.ArgumentParser(description='Import various types of word data')
    args = parser.parse_args()
    main()


if __name__ == "__main__":
    setup()
