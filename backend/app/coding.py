from backend.helpers.string import syllables
from backend.lib.db import get_cue_response_pairs, get_affixes, save_coded_pairs, delete_coded_pairs, get_all_words, \
    get_all_measures
from backend.lib.setup import db_setup
from backend.lib.contractions import replace_contraction
from nltk.stem import PorterStemmer
import re
import jellyfish
import unidecode
from backend.lib.wordnet_relations import WordnetRelations
from collections import Counter
from backend.lib.multiprocessor import job_pool
import time
import argparse
from backend.importers.semantic_atlas import BrainMap
from multiprocessing import cpu_count
from sense2vec import Sense2Vec
from backend.settings import SENSE2VEC_FOLDER, SEMANTIC_SPACE_FOLDER
from backend.helpers.pairs import format_cue_response_pairs
import os
import pprint
import numpy as np

# set of affixes
AFFIXES = {x.get('affix') for x in get_affixes()}
ALL_MEASURES = get_all_measures()
print('There are {} words in ALL MEASURES'.format(len(ALL_MEASURES)))
ALL_WORDS = get_all_words()
ps = PorterStemmer()
s2v = Sense2Vec().from_disk(SENSE2VEC_FOLDER)
semantic_space_data = np.load('{}/semantic_space.npz'.format(SEMANTIC_SPACE_FOLDER))
ENGLISH1000_VOCAB = {x.decode('utf8') for x in semantic_space_data['english1000_vocab']}
TOTALS = {
    'phonology': 0,
    'exact': 0,
    'syntagmatic': 0,
    'affix': 0,
    'prevalent': 0,
    'part_of_speech': 0,
    'part_of_speech_changes': 0
}

ortho_measures = ['response_ortho_neighbours',
                  'unigram_count',
                  'bigram_count',
                  'trigram_count',
                  'unconstrained_unigram_count',
                  'unconstrained_bigram_freq',
                  'unconstrained_trigram_freq',
                  'ortho_neighbours_diff']

phonographic_measures = ['nphon',
                         'unsdens',
                         'strdens',
                         'unsbpav',
                         'strbpav',
                         'unstpav',
                         'strtpav',
                         'unspospav',
                         'strpospav',
                         'unslcpospav',
                         'strlcpospav',
                         'sfreq',
                         'scdcnt']


def main():
    # TODO batch up the totals returned by the processes and print those instead of all the individual totals
    print('loading cue response pairs')
    all_pairs = [x for x in get_cue_response_pairs()]
    print('loaded {} cue response pairs'.format(len(all_pairs)))
    delete_coded_pairs()
    # make sure all of the required columns are in the coding table
    db_setup()

    # TODO make this DISTINCT in the query? Or make a new function for that?
    cue_response_pairs, distinct_cue_response_pairs = format_cue_response_pairs(all_pairs)
    print('there are {} distinct cue response pairs'.format(len(distinct_cue_response_pairs)))
    if not all_pairs:
        raise ValueError('No responses to code...')
    start = time.perf_counter()
    job_number = cpu_count() / 2
    job_number = 2
    # job_number = 1
    # jobs = dispatch_jobs(list(distinct_cue_response_pairs), job_number, get_coding)
    jobs = job_pool(list(distinct_cue_response_pairs), job_number, get_coding)
    end = time.perf_counter()
    print('time taken {}'.format(end - start))

    # take the individual results and sum them
    totals = Counter()
    for result in jobs:
        totals = totals + result
    pprint.pprint(totals)

    # check the percentages of each categorisation
    c = totals
    l = all_pairs
    d = [(i, c[i] / len(l) * 100.0) for i, count in c.most_common()]
    pprint.pprint(d)
    exit()


def get_coding(distinct_cue_response_pairs, **kwargs):
    # This algorithm is for coding DISTINCT cue response pairs once they have had their spelling corrected. It considers
    # categories such as prevalence, concreteness, age of acquisition, emotions, orthographic neighbourhoods,
    # phonographic neighbourhoods, active brain region.

    # TODO consider the categories found in the Gallant Research:
    # TODO (visual, tactile, outdoor, bodypart, number, place, violence, person, social, mental, time)
    # TODO consider when cue and response form one word, e.g. fear + less = fearless

    # initialise brain data
    brain_map = BrainMap()
    # code distinct list of cue response pairs
    coded_pairs = {}

    totals = Counter(TOTALS)
    fieldnames = None

    for pair in distinct_cue_response_pairs:
        key = pair[0]
        cue = pair[1]
        response = pair[2]
        response_raw = pair[3]
        # TODO work out what to do with collisions between sources. shouldn't be that many reallky
        # TODO what to do about words joined with hyphens?
        response_decoded = unidecode.unidecode(response)
        cue_decoded = unidecode.unidecode(cue)
        response_stem = ps.stem(response_decoded)
        cue_stem = ps.stem(cue_decoded)
        # TODO use this throughout instead of splitting multiple times
        response_split = response.split('_')
        if len(cue) > 3 and word_filter(key[0]) and word_filter(key[1]):

            coded_pairs[key] = basic_coding(cue, response, response_raw, response_split)
            # coded_pairs[key]['source'] = source
            relations_coding, relations_totals = relation_coding(cue, response)
            totals = totals + relations_totals

            coded_pairs[key].update(relations_coding)

            ############## Other Categories ##############

            ### exact match
            exact = cue == response
            coded_pairs[key]['exact'] = exact
            totals['exact'] += 1 * exact

            ### phonology (sound the same)
            phonology, type = check_phonology(cue_decoded, response_decoded)
            coded_pairs[key]['phonology'] = phonology
            totals['phonology'] += 1 * phonology

            ### syntagmatic (x follows y, y follows x)
            syntagmatic, type = check_syntagmatic(cue, response, response_split)
            coded_pairs[key]['syntagmatic'] = type
            totals['syntagmatic'] += 1 * syntagmatic

            # TODO add s2v check for syntagmatic
            # TODO add proverb check for syntagmatic

            ### affix - the words are equal plus or minus an affix
            cue_affix = check_affix(cue, response)
            response_affix = check_affix(response, cue)
            affix = cue_affix or response_affix
            coded_pairs[key]['affix'] = affix
            totals['affix'] += 1 * affix

            ### lots of measures
            # prevalence
            # concreteness
            # age of acquisition
            # emotions
            # orthographic
            # phonographic
            measures, _totals = check_measures(response)
            if measures:
                totals = totals + _totals
                coded_pairs[key].update(measures)

            ### semantic atlas from Gallant et. al
            word_found, response_semantic_area = brain_map.get_semantic_area(response_decoded)
            if not word_found:
                word_found, response_semantic_area = brain_map.get_semantic_area(response_stem)
            coded_pairs[key]['semantic_area'] = response_semantic_area
            totals['semantic_areas'] += 1 * word_found
            word_found, cue_semantic_area = brain_map.get_semantic_area(cue_decoded)
            if not word_found:
                word_found, cue_semantic_area = brain_map.get_semantic_area(cue_stem)
            changes = response_semantic_area != cue_semantic_area
            coded_pairs[key]['semantic_area_changes'] = changes
            totals['semantic_area_changes'] += 1 * changes

            # now get the coordinates of the cue and response in the brain model and calculate the euclidean distance
            # between them
            coded_pairs[key]['semantic_distance'] = 0
            if changes and cue_semantic_area and response_semantic_area:
                coded_pairs[key]['semantic_distance'] = brain_map.get_distances_between_semantic_areas(
                    cue_semantic_area, response_semantic_area).astype(float)
                totals['semantic_distance'] += 1

            ### part_of_speech
            word_found, part_of_speech_response = check_part_of_speech(response)
            if not word_found:
                word_found, part_of_speech_response = check_part_of_speech(response_stem)
            totals['part_of_speech'] += 1 * word_found
            coded_pairs[key]['part_of_speech'] = part_of_speech_response
            word_found, part_of_speech_cue = check_part_of_speech(cue)
            if not word_found:
                word_found, part_of_speech_cue = check_part_of_speech(cue_stem)
            coded_pairs[key]['part_of_speech_changes'] = False
            if word_found and part_of_speech_response != part_of_speech_cue:
                totals['part_of_speech_changes'] += 1
                coded_pairs[key]['part_of_speech_changes'] = True

            ### list of all words ### this should come last
            coded_pairs[key]['contraction'] = False
            if response in ALL_WORDS:
                # the word(s) appear(s) in the giant list of english words
                totals['all_words'] += 1
            elif replace_contraction(response) != response:
                # include the word but note that it happened
                coded_pairs[key]['contraction'] = True
                totals['contractions'] += 1
            # elif len(response_split) == 1 and not part_of_speech_response:
            #     # this is basically a list of responses that need addressed by future changes
            #     print('Word missing from all_words = {}: {}. Removing from coded_pairs.'.format(cue, response))
            #     coded_pairs.pop(key)

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
        #       ]
        # TODO food references
        #  [cocktails,
        #   recipes,
        #   etc.
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
    print('Job with PID {} reached end'.format(os.getpid()))

    return totals


def word_filter(word):
    if word in ENGLISH1000_VOCAB:
        return True
    return False


def basic_coding(cue, response, response_raw, response_split):
    basic = {'length': None,
             # 'response_time': None,
             'syllables': None,
             'capitalised': False,
             'punctuation': False,
             'number_of_words': None}
    basic['length'] = len(response)
    basic['syllables'] = syllables(response)
    basic['syllables_diff'] = basic['syllables'] - syllables(cue)
    basic['capitalised'] = str.isupper(response[0])
    basic['punctuation'] = True if re.search(r'[\.\-\',]', response_raw) else False
    basic['number_of_words'] = len(response_split)

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


def check_syntagmatic(cue, response, response_split):
    # TODO make this work for compounds, common phrases that are not complete in the response

    if '_' in response:
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
        # if the cue is not in the response
        if len(response_split) == 2:
            # first word of response is a conjunction
            # e.g. cue is "back" and response is "and forth"
            if response_split[0] in ('for', 'and', 'nor', 'but', 'or', 'yet', 'so'):
                return True, 'consecutivexy-cue_not_entered'

    joined_word_x_y = ''.join([cue, response])
    if joined_word_x_y in ALL_WORDS:
        return True, 'consecutivexy'

    joined_word_y_x = ''.join([response, cue])
    if joined_word_y_x in ALL_WORDS:
        return True, 'consecutiveyx'

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
    # FIXME This is too loose a definition
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


def check_measures(response):
    stem = ps.stem(response)
    stem_present = stem in ALL_MEASURES
    measures = None
    totals = Counter()
    if response in ALL_MEASURES:
        measures = ALL_MEASURES[response]
        prevalence = ALL_MEASURES[response]['prevalence'] is not None
        concreteness = ALL_MEASURES[response]['concreteness_measure'] is not None
        age = ALL_MEASURES[response]['age'] is not None
        emotions = ALL_MEASURES[response]['arousal'] is not None
        orthographic = ALL_MEASURES[response]['ortho_neighbours'] is not None
        phonographic = ALL_MEASURES[response]['nphon'] is not None
        if stem_present:
            # prevalence
            if not prevalence:
                measures['pknown'] = ALL_MEASURES[stem]['pknown']
                measures['prevalence'] = ALL_MEASURES[stem]['prevalence']
                prevalence = ALL_MEASURES[response]['prevalence'] is not None
            # concreteness
            if not concreteness:
                measures['concreteness_measure'] = ALL_MEASURES[stem]['concreteness_measure']
                concreteness = ALL_MEASURES[response]['concreteness_measure'] is not None
            # age
            if not age:
                measures['age'] = ALL_MEASURES[stem]['age']
                age = ALL_MEASURES[response]['age'] is not None
            # emotions
            if not emotions:
                measures['arousal'] = ALL_MEASURES[stem]['arousal']
                measures['valence'] = ALL_MEASURES[stem]['valence']
                measures['dominance'] = ALL_MEASURES[stem]['dominance']
                emotions = ALL_MEASURES[response]['arousal'] is not None
            # orthographic
            if not orthographic:
                for o_measure in ortho_measures:
                    measures[o_measure] = ALL_MEASURES[stem][o_measure]
                orthographic = ALL_MEASURES[response]['ortho_neighbours'] is not None
            # phonographic
            if not phonographic:
                for p_measure in phonographic_measures:
                    measures[p_measure] = ALL_MEASURES[stem][p_measure]
                phonographic = ALL_MEASURES[response]['nphon'] is not None

        totals['prevalence'] += 1 * prevalence
        totals['concreteness'] += 1 * concreteness
        totals['age'] += 1 * age
        totals['emotions'] += 1 * emotions
        totals['orthographic'] += 1 * orthographic
        totals['phonographic'] += 1 * phonographic

    return measures, totals


def check_part_of_speech(response):
    if response in ALL_MEASURES and ALL_MEASURES[response]['part_of_speech'] is not None:
        return True, ALL_MEASURES[response]['part_of_speech'].lower()
    else:
        best_sense = s2v.get_best_sense(response)
        if best_sense:
            return True, best_sense.split('|')[1].lower()

    return False, 0


def setup():
    parser = argparse.ArgumentParser(description='Import various types of word data')
    args = parser.parse_args()
    main()


if __name__ == "__main__":
    setup()
