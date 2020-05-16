from backend.helpers.string import syllables
from backend.lib.db import get_cue_response_pairs, get_affixes, save_coded_pairs, delete_coded_pairs, get_all_words, \
    get_all_measures
from backend.lib.setup import db_setup
from nltk.stem import PorterStemmer
import re
import jellyfish
from backend.lib.wordnet_relations import WordnetRelations
from collections import Counter
from backend.lib.multiprocessor import job_pool
import time
import argparse
from backend.lib.semantic_atlas import BrainMap
from backend.settings import SEMANTIC_SPACE_FOLDER
from backend.helpers.pairs import format_cue_response_pairs
import os
import pprint
import numpy as np
import pandas as pd
import logging

UNKNOWN_AREA_LABEL = b'U'
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
punctuation_regex = re.compile(r'[\.\-\',]')

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


def get_coding(df, **kwargs):
    # var df: a dataframe of distinct cue - response pairs
    #
    # This algorithm is for coding DISTINCT cue response pairs once they have had their spelling corrected. It considers
    # categories such as prevalence, concreteness, age of acquisition, emotions, orthographic neighbourhoods,
    # phonographic neighbourhoods, active brain region.

    # TODO consider the categories found in the Gallant Research:
    #   (visual, tactile, outdoor, bodypart, number, place, violence, person, social, mental, time)
    # TODO consider when cue and response form one word, e.g. fear + less = fearless

    # initialise brain data
    brain_map = BrainMap()
    # code distinct list of cue response pairs
    coded_pairs = pd.DataFrame()

    totals = Counter(TOTALS)
    fieldnames = None
    # trim down the cues and responses so we're only running functions once per word
    distinct_responses = pd.DataFrame(df.index.get_level_values('response_clean').drop_duplicates())
    distinct_cues = pd.DataFrame(df.index.get_level_values('cue_clean').drop_duplicates())
    response_measures = check_measures(all_measures, distinct_responses)
    cue_measures = check_measures(all_measures, distinct_cues)
    # attach the various measures back to the full lists of cues and responses
    response_measures_full = pd.DataFrame(df.index.get_level_values('response_clean')).merge(
        response_measures,
        left_on='response_clean',
        right_on='response_clean',
        how='left')
    cue_measures_full = pd.DataFrame(df.index.get_level_values('cue_clean')).merge(
        cue_measures,
        left_on='cue_clean',
        right_on='cue_clean',
        how='left')

    print(cue_measures_full)
    print(response_measures_full)
    measures_diff = cue_measures_full.drop(['cue_clean', 'part_of_speech'], axis=1).subtract(
        response_measures_full.drop(
            ['response_clean', 'part_of_speech'], axis=1))
    # TODO get the parts of speech from s2v if they are missing from all_measures
    # pos_changes = pd.DataFrame([[nanindex(x, cue_measures_full['part_of_speech'][id]) == np.nan] for id, x in
    #                           enumerate(response_measures_full['part_of_speech'])])
    # print(pos_changes)
    # print(measures_diff)

    logging.debug(response_measures)
    df['exact_match'] = df.index.get_level_values(
        'response_clean') == df.index.get_level_values('cue_clean')
    df['length'] = df['response_decoded'].str.len()
    df['syllables'] = np.vectorize(syllables)(df['response_raw'])
    df['capitalised'] = df['response_raw'].str.isupper()
    df['punctuation'] = df['response_raw'].str.contains(
        punctuation_regex)
    df['number_of_words'] = df['response_decoded'].str.split(
        '_').str.len()
    # TODO do this with pandas? probably means rewriting or wrapping relation_coding to accept dataframes
    relations_array = np.vectorize(relation_coding)(df.index.get_level_values('cue_clean'),
                                                    df.index.get_level_values('response_clean'))
    other_categories = pd.json_normalize(relations_array)
    other_categories.index = df.index
    # the numpy vectorize function has no index as only the cue and response columns are passed in
    # so add the index back on to allow the dataframes to be concatenated

    assert (len(other_categories) == len(df))

    other_categories['phonology'] = np.vectorize(check_phonology)(df['cue_decoded'], df['response_decoded'])

    assert (len(other_categories) == len(df))
    df = pd.concat([df, other_categories],axis=1,sort=False)
    # check for syntagmatic and add to dataframe
    df = check_syntagmatic_pd(df)

    # check for affix and add to dataframe
    df = check_affix_pd(df)

    #     # TODO what to do about words joined with hyphens?
    #     # TODO use this throughout instead of splitting multiple times
    #     ############## Other Categories ##############
    #
    #     # TODO add s2v check for syntagmatic
    #     # TODO add proverb check for syntagmatic

    # ### semantic atlas from Gallant et. al
    # cue semantic areas are only used to determine if the area changes between cue and response
    cue_semantic_areas = brain_map.get_semantic_area(df.index.get_level_values(
        'cue_clean').unique())
    # response semantic areas
    response_semantic_areas = brain_map.get_semantic_area(df.index.get_level_values(
        'response_clean').unique())
    both_semantic_areas = pd.concat([cue_semantic_areas, response_semantic_areas]).drop_duplicates().set_index('word')
    df = df.join(both_semantic_areas, on='response_clean')
    df.rename(columns={'left_hemi_label': 'response_left_hemi_label',
                       'right_hemi_label': 'response_right_hemi_label'}, inplace=True)
    df = df.join(both_semantic_areas, on='cue_clean')
    df.rename(columns={'left_hemi_label': 'cue_left_hemi_label',
                       'right_hemi_label': 'cue_right_hemi_label'}, inplace=True)
    # TODO consider what to do when the labels are nans and get the logic to output True/False/or similar.
    #   setting them to b'' means we lose the data of which words were not found and cannot do the lookup on stem
    #   effciently. This is because words that are found that are not significant in the brain model are represented by
    #   b''
    # TODO search for stems if word is not found
    # Fill with U (Unknown) to avoid intersecting with labels that are not statistically significant
    df['response_left_hemi_label'] = df['response_left_hemi_label'].fillna(b'').replace(to_replace=b'',
                                                                                        value=UNKNOWN_AREA_LABEL)
    df['response_right_hemi_label'] = df['response_right_hemi_label'].fillna(b'').replace(to_replace=b'',
                                                                                          value=UNKNOWN_AREA_LABEL)
    df['cue_left_hemi_label'] = df['cue_left_hemi_label'].fillna(b'').replace(to_replace=b'', value=UNKNOWN_AREA_LABEL)
    df['cue_right_hemi_label'] = df['cue_right_hemi_label'].fillna(b'').replace(to_replace=b'',
                                                                                value=UNKNOWN_AREA_LABEL)
    df['r_hemi_area_differs'] = np.logical_and.reduce((df['response_right_hemi_label'] !=
                                                       df['cue_right_hemi_label'],
                                                       df['response_right_hemi_label'] != UNKNOWN_AREA_LABEL,
                                                       df['cue_right_hemi_label'] != UNKNOWN_AREA_LABEL))
    df['l_hemi_area_differs'] = np.logical_and.reduce((df['response_left_hemi_label'] !=
                                                       df['cue_left_hemi_label'],
                                                       df['response_left_hemi_label'] != UNKNOWN_AREA_LABEL,
                                                       df['cue_left_hemi_label'] != UNKNOWN_AREA_LABEL))
    right_unknown_areas = np.logical_or(df['response_right_hemi_label'] == UNKNOWN_AREA_LABEL,
                                        df['cue_right_hemi_label'] == UNKNOWN_AREA_LABEL)
    right_hemi_differs = df['r_hemi_area_differs'].copy(deep=True)
    right_hemi_differs[right_unknown_areas] = np.nan
    df['r_hemi_area_differs'] = right_hemi_differs
    left_unknown_areas = np.logical_or(df['response_left_hemi_label'] == UNKNOWN_AREA_LABEL,
                                       df['cue_left_hemi_label'] == UNKNOWN_AREA_LABEL)
    left_hemi_differs = df['l_hemi_area_differs'].copy(deep=True)
    left_hemi_differs[left_unknown_areas] = np.nan
    df['l_hemi_area_differs'] = left_hemi_differs

    left_hemi_distance = brain_map.get_distances_between_semantic_areas('left', df['cue_left_hemi_label'],
                                                                        df['response_left_hemi_label'])

    right_hemi_distance = brain_map.get_distances_between_semantic_areas('right', df['cue_right_hemi_label'],
                                                                         df['response_right_hemi_label'])
    df = df.merge(left_hemi_distance['left_hemi_distance'], how='left', left_index=True, right_index=True)
    df = df.merge(right_hemi_distance['right_hemi_distance'], how='left', left_index=True, right_index=True)
    # these are indicative of the actual words, which we do not want to model. We want to model attributes of the words.
    df = df.drop(
        ['response_left_hemi_label', 'response_right_hemi_label', 'cue_left_hemi_label', 'cue_right_hemi_label'],
        axis=1)

    ### various measures related to the response
    #df = df.merge(response_measures, left_on='response_clean', right_on='response_clean', how='left')
    print(measures_diff)
    print(df)
    df = df.reset_index().join(measures_diff)
    print(df[df.columns[-6:]])
    # TODO check if the part of speech changes from cue to response
    # TODO add comparisons between the other metrics instead of the metrics themselves (i.e. concreteness difference)
    #   this is probably more valid than modelling on the response metrics as they will be indicative of the words

    #     ### list of all words ### this should come last
    #     coded_pairs[key]['contraction'] = False
    #     if response in ALL_WORDS:
    #         # the word(s) appear(s) in the giant list of english words
    #         totals['all_words'] += 1
    #     elif replace_contraction(response) != response:
    #         # include the word but note that it happened
    #         coded_pairs[key]['contraction'] = True
    #         totals['contractions'] += 1
    #     # elif len(response_split) == 1 and not part_of_speech_response:
    #     #     # this is basically a list of responses that need addressed by future changes
    #     #     print('Word missing from all_words = {}: {}. Removing from coded_pairs.'.format(cue, response))
    #     #     coded_pairs.pop(key)
    #
    #     # TODO  obscene words
    #     # TODO  sexual references
    #     # TODO  slang
    #     # TODO  part_of_speech: take this from age of acquisition?
    #     # TODO  morpheme
    #     # TODO  years (1960, 2010, etc.)
    #     # TODO  cultural_reference:
    #     #  [movies,
    #     #   music,
    #     #   famous people,
    #     #   books (1984, farenheit 451),
    #     #   poems,
    #     #   computer games,
    #     #   tv shows,
    #     #   brands,
    #     #   historical dates,
    #     #   characters,
    #     #   superheroes,
    #     #   religion,
    #     #   web sites,
    #     #   web status codes (404),
    #     #   emergency services numbers,
    #     #   scientific numbers (365, pi, gravity, phi, etc.)
    # #       ]
    # # TODO food references
    # #  [cocktails,
    # #   recipes,
    # #   etc.
    # #   ]
    # #   The part of speech (Person, etc.) is a good start for deriving further information from the full cue/
    # #   response pair. For example, (portal, chell). Chell is picked up as a person but is a character in the
    # #   game, Portal.
    # # TODO  company reference [walmart, asda, etc.]
    # # TODO  geographical_reference [country, state, city, town, village, post code]
    # # TODO  corpus_similarity
    # # TODO  dictionary_similarity

    # save data to the db
    # save_coded_pairs(df, df.columns)
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
    relation_obj = WordnetRelations(cue, response)
    relations = relation_obj.relations
    return relations


def nanindex(list_or_string, word):
    try:
        return list_or_string.index(word)
    except ValueError:
        return np.nan


def check_syntagmatic_pd(df):
    logging.debug('Dataframe length {}'.format(len(df)))
    response_clean = df.index.get_level_values('response_clean').str.decode('utf8')
    cue_clean = df.index.get_level_values('cue_clean').str.decode('utf8')
    # having a nightmare with words as indexes so just dropping
    response_split = pd.DataFrame(response_clean.str.split('_')).reset_index()
    response_split = response_split.drop('index', axis=1)
    # find the index of the cue in the response (not present returns nan)
    cue_index = pd.DataFrame([[nanindex(x, cue_clean[id])] for id, x in enumerate(response_split['response_clean'])])
    # get the column headers onto this data
    syntagmatic = pd.DataFrame(cue_index, columns=['consecutivexy', 'consecutiveyx', 'othercollocational'])
    # if the cue is at the start then the category is x_y
    syntagmatic['consecutivexy'] = np.equal(syntagmatic['consecutivexy'], 0)
    # if the cue is second then the category is y_x
    syntagmatic['consecutiveyx'] = np.equal(syntagmatic['consecutiveyx'], 1)
    # otherwise the cue is contained somewhere else in the phrase
    syntagmatic['othercollocational'] = np.greater(syntagmatic['othercollocational'], 1)
    # check for instances of the cue NOT being present in the response, the response being 2 words, and the response
    # beginning with for|and|nor|but|or|yet|so_ (e.g. cue = back, response = and forth)
    cue_not_present = cue_index.isnull().unstack().reset_index()
    cue_not_present = cue_not_present.drop(['level_0', 'level_1'], axis=1)
    response_two_words = response_split[response_split['response_clean'].str.len() > 2]
    # intersection of a multi index with an integer index doesn't work
    more_than_two_words_and_cue_not_present_indices = cue_not_present.index[response_two_words.index]
    # TODO this could be too broad and picking up random things
    consecutivexy_cue_not_entered_matches = response_clean[more_than_two_words_and_cue_not_present_indices].str.match(
        '^(for|and|nor|but|or|yet|so)')
    consecutivexy_cue_not_entered_indices = more_than_two_words_and_cue_not_present_indices[
        consecutivexy_cue_not_entered_matches == True]
    cue_response_merged = cue_clean + response_clean
    response_cue_merged = response_clean + cue_clean
    consecutivexy = cue_response_merged.isin(ALL_WORDS)
    consecutiveyx = response_cue_merged.isin(ALL_WORDS)
    # join cue and response and check if the merge is present in ALL WORDS
    syntagmatic['consecutivexy'] = np.logical_or(syntagmatic['consecutivexy'], consecutivexy)
    # join response and cue and check if the merge is present in ALL WORDS
    syntagmatic['consecutiveyx'] = np.logical_or(syntagmatic['consecutiveyx'], consecutiveyx)
    # TODO categorise consecutivexy and consecutiveyx as 0 and 1
    syntagmatic['consecutivexy_cue_not_entered'] = syntagmatic.index.isin(consecutivexy_cue_not_entered_indices)

    syntagmatic.index = df.index
    df = pd.concat([df, syntagmatic], axis=1, sort=False)
    return df


# Soundex similarity check
def check_phonology(cue, response):
    # Try to get a cue soundex
    cue_soundex = jellyfish.soundex(cue)

    # Try to get a response soundex
    response_soundex = jellyfish.soundex(response)
    if cue_soundex == response_soundex:
        return 'soundex'

    levenshtein_distance = jellyfish.levenshtein_distance(cue, response)
    if cue_soundex[1:] == response_soundex[1:] and levenshtein_distance <= 3:
        return 'soundex'

    # if cue_soundex[-2:] == response_soundex[-2:] and levenshtein_distance <= 3:
    #     return True, 'soundex'

    # TODO add zero to the end of shorter  last two?
    # FIXME This is too loose a definition
    try:
        # e.g. C630 M300
        if response_soundex[-3:].index(cue_soundex[-2:]) > -1 and levenshtein_distance <= 3:
            return 'soundex'
    except ValueError:
        pass

    try:
        # e.g. C630 M300
        if cue_soundex[-3:].index(response_soundex[-2:]) > -1 and levenshtein_distance <= 3:
            return 'soundex'
    except ValueError:
        pass

    if levenshtein_distance < 2:
        return 'levenshtein'

    return None


def check_affix_pd(df):
    # TODO look closer at categories of affix https://en.wikipedia.org/wiki/Affix
    response_clean = df.index.get_level_values('response_clean').str.decode('utf8')
    cue_clean = df.index.get_level_values('cue_clean').str.decode('utf8')
    # TODO reduce to cue!=response initially
    cue_index = pd.DataFrame([[nanindex(x, cue_clean[int(idx)])] for idx, x in enumerate(response_clean)])
    # get rid of any cue-response pairs where the cue is not in the response
    cue_present_index = [cue_index[0] >= 0]
    response_subset = response_clean[cue_present_index[0]]
    cue_subset = cue_clean[cue_present_index[0]]
    # shouldn't need to strip x here as spaces should already be replaced by underscores
    remainder = pd.DataFrame([[x.split(cue_subset[int(idx)])] for idx, x in enumerate(response_subset)])
    # assign the resulting list to two separate columns (s1, s2)
    remainder[['s1', 's2']] = pd.DataFrame(remainder[0].tolist(), index=remainder.index)
    # we no longer need the column with the list
    remainder = remainder.drop(0, axis=1)
    # check if the remainder is in the affixes for either split1 or split2
    affix1 = np.logical_or(remainder['s1'].isin(AFFIXES), remainder['s2'].isin(AFFIXES))
    # check if the resultant word is just cueresponse or responsecue
    # TODO isn't this redundant because of syntagmatic checks?
    #   Also this overrides the above by classifying anything that starts/ends the same and ends/stars
    #   differently as being affixed.... This is not the case with anything, anyone, anywhere, whereabouts,
    #   thereabouts. Should these be classified differently or dropped?
    affix2 = np.logical_and(np.logical_and(remainder['s1'].str.len() == 0, remainder['s2'].str.len() > 0),
                            np.logical_not(remainder['s2'].str.contains('_')))
    affix3 = np.logical_and(np.logical_and(remainder['s2'].str.len() == 0, remainder['s1'].str.len() > 0),
                            np.logical_not(remainder['s1'].str.contains('_')))

    affix_indexes = np.logical_or(np.logical_or(affix1, affix2), affix3)
    cue_indexes = cue_subset[affix_indexes]
    response_indexes = response_subset[affix_indexes]
    # the resulting data is only a subset of the dataframe so we need to apply it overall
    affix_col = pd.DataFrame(np.logical_and(cue_clean.isin(cue_indexes), response_clean.isin(response_indexes)),
                             columns=['affix'])
    affix_col.index = df.index
    df = pd.concat([df, affix_col], axis=1, sort=False)
    return df


def check_measures(all_measures, distinct_words):
    # create a column of stems based on the responses (i.e. schooling becomes school)
    col_name = distinct_words.columns[0]
    stems = pd.DataFrame(
        [[response, ps.stem(response.decode('utf8')).encode('utf8')] for response in
         distinct_words[col_name]],
        columns=[col_name, 'stem'])

    logging.debug('Stems length is {}'.format(len(stems)))
    all_measures_words = all_measures.index.get_level_values(level='word')
    words_intersect = np.intersect1d(distinct_words, all_measures_words)
    words_measures = all_measures[all_measures.index.isin(words_intersect)]
    words_measures_full = pd.merge(pd.DataFrame(distinct_words), pd.DataFrame(words_measures),
                                   left_on=col_name, right_on='word', how='left')
    logging.debug('{} measures shape {}'.format(col_name, words_measures_full.shape))
    stem_measures = all_measures[all_measures.index.isin(stems['stem'])]
    stem_measures_full = pd.merge(stems, pd.DataFrame(stem_measures), left_on='stem', right_on='word', how='left')
    logging.debug('stem_measures_full')
    stem_measures_full.drop('stem', axis=1, inplace=True)
    logging.debug('stem_measures_full')
    logging.debug('stem measures shape {}'.format(stem_measures_full.shape))
    # replace the nan values in measures with the stem measures
    words_measures_full.update(stem_measures_full)
    return words_measures_full


def setup():
    parser = argparse.ArgumentParser(description='Import various types of word data')
    args = parser.parse_args()
    main()


if __name__ == "__main__":
    setup()
