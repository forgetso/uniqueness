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
from sense2vec import Sense2Vec
from backend.settings import SENSE2VEC_FOLDER, SEMANTIC_SPACE_FOLDER, JOB_NUMBER
from backend.helpers.pairs import format_cue_response_pairs
import os
import pprint
import numpy as np
import pandas as pd
import logging

# logging.basicConfig(level=LOG_LEVEL)
MIN_CUE_LENGTH = 3
# set of affixes
AFFIXES = {x.get('affix') for x in get_affixes()}
all_measures = get_all_measures()
logging.info('There are {} words in ALL MEASURES'.format(len(all_measures)))
ALL_WORDS = get_all_words()
ps = PorterStemmer()
# s2v = Sense2Vec().from_disk(SENSE2VEC_FOLDER)
semantic_space_data = np.load('{}/semantic_space.npz'.format(SEMANTIC_SPACE_FOLDER))
ENGLISH1000_VOCAB = semantic_space_data['english1000_vocab']
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
    logging.debug('loading cue response pairs')
    cur = get_cue_response_pairs()
    all_pairs = pd.DataFrame([x for x in cur], columns=[desc[0] for desc in cur.description])
    all_pairs['cue_clean'] = all_pairs['cue_clean'].str.encode('utf8')
    all_pairs['response_clean'] = all_pairs['response_clean'].str.encode('utf8')
    # TODO create multi Index based on cue_clean and response_clean ???

    logging.debug('loaded {} cue response pairs'.format(len(all_pairs)))
    delete_coded_pairs()
    # make sure all of the required columns are in the coding table
    db_setup()

    # TODO make this DISTINCT in the query? Or make a new function for that?
    cue_response_pairs, distinct_cue_response_pairs = format_cue_response_pairs(all_pairs,
                                                                                word_filter=ENGLISH1000_VOCAB)
    logging.debug('there are {} distinct cue response pairs'.format(len(distinct_cue_response_pairs)))
    if not len(all_pairs):
        raise ValueError('No responses to code...')
    start = time.perf_counter()

    # jobs = dispatch_jobs(list(distinct_cue_response_pairs), job_number, get_coding)
    jobs = job_pool(distinct_cue_response_pairs, 1, get_coding)
    end = time.perf_counter()
    logging.debug('time taken {}'.format(end - start))

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
    #   (visual, tactile, outdoor, bodypart, number, place, violence, person, social, mental, time)
    # TODO consider when cue and response form one word, e.g. fear + less = fearless

    # FIXME only check for each response once instead of checking for each cue response PAIR!!!!
    # initialise brain data
    brain_map = BrainMap()
    # code distinct list of cue response pairs
    coded_pairs = pd.DataFrame()

    totals = Counter(TOTALS)
    fieldnames = None
    coded_responses = pd.DataFrame([])

    distinct_responses = distinct_cue_response_pairs.index.get_level_values('response_clean').drop_duplicates()
    response_measures = check_measures(all_measures, distinct_responses)

    print(response_measures)
    # TODO turn all of these loops into vector multiplications
    for pair in distinct_cue_response_pairs:
        print(pair)
        exit(0)
        key = pair[0]
        cue = pair[1]
        response = pair[2]
        response_raw = pair[3]
        # TODO what to do about words joined with hyphens?
        response_decoded = unidecode.unidecode(response)
        cue_decoded = unidecode.unidecode(cue)
        response_stem = ps.stem(response_decoded)
        cue_stem = ps.stem(cue_decoded)
        # TODO use this throughout instead of splitting multiple times
        response_split = response.split('_')
        if len(cue) > MIN_CUE_LENGTH:

            coded_pairs[key] = basic_coding(response, response_raw, response_split)

            ### exact match
            exact = cue == response
            coded_pairs[key]['exact'] = exact

            ### response only measures
            coded_pairs[key].update(coded_responses[response])

            ### wordnet relations
            relations_coding, relations_totals = relation_coding(cue, response)
            coded_pairs[key].update(relations_coding)

            ############## Other Categories ##############

            ### phonology (sound the same)
            phonology, type = check_phonology(cue_decoded, response_decoded)
            coded_pairs[key]['phonology'] = phonology

            ### syntagmatic (x follows y, y follows x)
            syntagmatic, type = check_syntagmatic(cue, response, response_split)
            coded_pairs[key]['syntagmatic'] = type

            # TODO add s2v check for syntagmatic
            # TODO add proverb check for syntagmatic

            ### affix - the words are equal plus or minus an affix
            cue_affix = check_affix(cue, response)
            response_affix = check_affix(response, cue)
            affix = cue_affix or response_affix
            coded_pairs[key]['affix'] = affix

            ### semantic atlas from Gallant et. al
            word_found, response_semantic_area = brain_map.get_semantic_area(response_decoded)
            if not word_found:
                word_found, response_semantic_area = brain_map.get_semantic_area(response_stem)
            coded_pairs[key]['semantic_area'] = response_semantic_area
            word_found, cue_semantic_area = brain_map.get_semantic_area(cue_decoded)
            if not word_found:
                word_found, cue_semantic_area = brain_map.get_semantic_area(cue_stem)
            changes = response_semantic_area != cue_semantic_area
            coded_pairs[key]['semantic_area_changes'] = changes

            # now get the coordinates of the cue and response in the brain model and calculate the euclidean distance
            # between them
            coded_pairs[key]['semantic_distance'] = 0
            if changes and cue_semantic_area and response_semantic_area:
                coded_pairs[key]['semantic_distance'] = brain_map.get_distances_between_semantic_areas(
                    cue_semantic_area, response_semantic_area).astype(float)

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
    logging.debug('Job with PID {} reached end'.format(os.getpid()))

    return totals


def basic_coding(response, response_raw, response_split):
    basic = pd.DataFrame({'length': None,
                          # 'response_time': None,
                          'syllables': None,
                          'capitalised': False,
                          'punctuation': False,
                          'number_of_words': None})
    basic['length'] = len(response)
    basic['syllables'] = syllables(response)
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


def check_measures(all_measures, distinct_responses):
    # create a column of stems based on the responses (i.e. schooling becomes school)
    stems = pd.DataFrame(
        [[response, ps.stem(response.decode('utf8')).encode('utf8')] for response in distinct_responses.values],
        columns=['response_clean', 'stem'])
    logging.debug('Stems length is {}'.format(len(stems)))
    all_measures_words = all_measures.index.get_level_values(level='word')
    response_intersect = np.intersect1d(distinct_responses, all_measures_words)
    response_measures = all_measures[all_measures.index.isin(response_intersect)]
    response_measures_full = pd.merge(pd.DataFrame(distinct_responses), pd.DataFrame(response_measures),
                                      left_on='response_clean', right_on='word', how='left')
    logging.debug('response measures shape {}'.format(response_measures_full.shape))
    stem_intersect = np.intersect1d(stems['stem'], all_measures_words)
    stem_measures = all_measures[all_measures.index.isin(stem_intersect)]
    # do i want a join instead of intersect? probably wrongly, wrongs both stem to wrong so stems intersect is
    # wrong | measures for 2 rows in the original stems dataframe
    # but that's ok beacause the left join would join wrong twice (this is what was happening!)
    # so left join then drop the stems and get the data into response_measures_full
    stem_measures_full = pd.merge(stems, pd.DataFrame(stem_measures), left_on='stem', right_on='word', how='left')
    print(stem_measures_full)
    stem_measures_full = stem_measures_full.drop('stem', axis=1)
    logging.debug('stem measures shape {}'.format(stem_measures_full.shape))
    # replace the nan values in measures with the stem measures
    response_measures_full.update(stem_measures_full)
    #FIXME stem being used as response_clean instead of the full word...
    print(response_measures_full.loc[response_measures_full['response_clean'] == b'wrongs'])
    print(response_measures_full.loc[response_measures_full['response_clean'] == b'wrong'])
    return response_measures_full


def check_part_of_speech(response):
    if response in all_measures and all_measures[response]['part_of_speech'] is not None:
        return True, all_measures[response]['part_of_speech'].lower()
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
