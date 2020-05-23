from backend.helpers.string import syllables
from backend.db.db import get_cue_response_pairs, get_affixes, save_coded_pairs, delete_coded_pairs, \
    get_all_words, get_all_measures
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
from backend.settings import SEMANTIC_SPACE_FOLDER, JOB_NUMBER
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
ALL_MEASURES = get_all_measures()
logging.info('There are {} words in ALL MEASURES'.format(len(ALL_MEASURES)))
ALL_WORDS = get_all_words()
ps = PorterStemmer()
# s2v = Sense2Vec().from_disk(SENSE2VEC_FOLDER)
SEMANTIC_SPACE_DATA = np.load('{}/semantic_space.npz'.format(SEMANTIC_SPACE_FOLDER))
ENGLISH1000_VOCAB = SEMANTIC_SPACE_DATA['english1000_vocab']
PUNCTUATION_REGEX = re.compile(r'[\.\-\',]')


def main(save=None, source=None, filter=None):
    word_filter = None
    if filter:
        word_filter = ENGLISH1000_VOCAB
    logging.debug('loading cue response pairs')
    cur = get_cue_response_pairs(source=source)
    all_pairs = pd.DataFrame([x for x in cur], columns=[desc[0] for desc in cur.description])
    all_pairs['cue_clean'] = all_pairs['cue_clean'].str.encode('utf8')
    all_pairs['response_clean'] = all_pairs['response_clean'].str.encode('utf8')
    logging.debug('loaded {} cue response pairs'.format(len(all_pairs)))
    if save:
        delete_coded_pairs()
        # make sure all of the required columns are in the coding table
        db_setup()
    # TODO add a preprocessing spell check job to insert the corrections for all responses at once and save to the
    #   corrections table
    cue_response_pairs, distinct_cue_response_pairs = format_cue_response_pairs(all_pairs, word_filter=word_filter)
    logging.debug('there are {} distinct cue response pairs'.format(len(distinct_cue_response_pairs)))
    if not len(all_pairs):
        raise ValueError('No responses to code...')
    start = time.perf_counter()

    # jobs = dispatch_jobs(list(distinct_cue_response_pairs), job_number, get_coding)
    jobs = job_pool(distinct_cue_response_pairs, JOB_NUMBER, get_coding, save=save)
    end = time.perf_counter()
    logging.debug('time taken {}'.format(end - start))

    # take the individual results and sum them
    totals = sum(jobs)

    # check the percentages of each categorisation
    # TODO why are some of these over 100%? float errors?
    percent_totals = totals.divide(len(distinct_cue_response_pairs)) * float(100)
    print(percent_totals.to_string())
    return percent_totals


def get_coding(df, save):
    # var df: a dataframe of distinct cue - response pairs
    #
    # This algorithm is for coding DISTINCT cue response pairs once they have had their spelling corrected. It considers
    # categories such as prevalence, concreteness, age of acquisition, emotions, orthographic neighbourhoods,
    # phonographic neighbourhoods, active brain region.

    # TODO consider the categories found in the Gallant Research:
    #   (visual, tactile, outdoor, bodypart, number, place, violence, person, social, mental, time)
    # TODO consider when cue and response form one word, e.g. fear + less = fearless

    # trim down the cues and responses so we're only running functions once per word
    distinct_responses = pd.DataFrame(df.index.get_level_values('response_clean').str.decode('utf8').drop_duplicates())
    distinct_cues = pd.DataFrame(df.index.get_level_values('cue_clean').str.decode('utf8').drop_duplicates())

    response_measures = check_measures(ALL_MEASURES, distinct_responses)
    cue_measures = check_measures(ALL_MEASURES, distinct_cues)
    # attach the various measures back to the full lists of cues and responses
    response_measures_full = pd.DataFrame(df.index.get_level_values('response_clean').str.decode('utf8')).merge(
        response_measures,
        left_on='response_clean',
        right_on='response_clean',
        how='left')
    cue_measures_full = pd.DataFrame(df.index.get_level_values('cue_clean').str.decode('utf8')).merge(
        cue_measures,
        left_on='cue_clean',
        right_on='cue_clean',
        how='left')

    measures_diff = cue_measures_full.drop(['cue_clean', 'part_of_speech'], axis=1).subtract(
        response_measures_full.drop(
            ['response_clean', 'part_of_speech'], axis=1))
    # TODO get the parts of speech from s2v if they are missing from all_measures
    # pos_changes = pd.DataFrame([[nanindex(x, cue_measures_full['part_of_speech'][id]) == np.nan] for id, x in
    #                           enumerate(response_measures_full['part_of_speech'])])
    # print(pos_changes)
    # print(measures_diff)

    # df['exact_match'] = df.index.get_level_values(
    #     'response_clean') == df.index.get_level_values('cue_clean')
    df['length'] = df['response_decoded'].str.len()
    df['syllables'] = np.vectorize(syllables)(df['response_raw'])
    df['capitalised'] = df['response_raw'].str.isupper()
    df['punctuation'] = df['response_raw'].str.contains(PUNCTUATION_REGEX)
    df['number_of_words'] = df['response_decoded'].str.split('_').str.len()
    # TODO do this with pandas? probably means rewriting or wrapping relation_coding to accept dataframes
    relations_array = np.vectorize(relation_coding)(df.index.get_level_values('cue_clean'),
                                                    df.index.get_level_values('response_clean'))
    other_categories = pd.json_normalize(relations_array)
    other_categories.index = df.index
    # the numpy vectorize function has no index as only the cue and response columns are passed in
    # so add the index back on to allow the dataframes to be concatenated

    assert (len(other_categories) == len(df))

    other_categories['phonology'] = np.vectorize(check_phonology)(df['cue_decoded'], df['response_decoded'])
    other_categories['phonology'] = other_categories['phonology'].astype('category')
    assert (len(other_categories) == len(df))
    df = pd.concat([df, other_categories], axis=1, sort=False)

    # check for syntagmatic and add to dataframe
    df = check_syntagmatic(df)

    # check for affix and add to dataframe
    df = check_affix(df)

    # add the semantic brain area data
    df = check_semantic_atlas(df)

    # TODO what to do about words joined with hyphens?
    # TODO use this throughout instead of splitting multiple times
    # TODO add s2v check for syntagmatic
    # TODO add proverb check for syntagmatic

    ### various word measures related to the response and cue that have been subtracted from each other to give distances
    df = df.reset_index().join(measures_diff)
    df = df.drop(['cue', 'response', 'response_raw', 'cue_decoded', 'response_decoded'], axis=1)
    # print(df[df.columns[-6:]])

    # TODO check if the part of speech changes from cue to response
    # TODO add comparisons between the other metrics instead of the metrics themselves (i.e. concreteness difference)
    #   this is probably more valid than modelling on the response metrics as they will be indicative of the words
    # TODO  contractions (don't)
    # TODO  word is in ALL_WORDS
    # TODO  encyclopaedic associations - e.g. if the cue is war and someone responds with the name of a particular war
    # TODO  obscene words
    # TODO  sexual references
    # TODO  slang
    # TODO  part_of_speech: take this from age of acquisition?
    # TODO  morpheme
    # TODO  years (1960, 2010, etc.)
    # TODO  cultural_reference (encyclopaedic):
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

    # save data to the db
    df = df.rename({'cue_clean': 'cue', 'response_clean': 'response'}, axis=1)
    df['cue'] = df['cue'].str.decode('utf8')
    df['response'] = df['response'].str.decode('utf8')
    df = df.replace({np.nan: None})
    if save:
        save_coded_pairs(df, df.columns)
    logging.debug('Job with PID {} reached end'.format(os.getpid()))
    totals = df.count()
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


def check_syntagmatic(df):
    logging.debug('Dataframe length {}'.format(len(df)))
    response_clean = df.index.get_level_values('response_clean').str.decode('utf8')
    cue_clean = df.index.get_level_values('cue_clean').str.decode('utf8')
    # having a nightmare with words as indexes so just dropping
    response_split = pd.DataFrame(response_clean.str.split('_')).reset_index()
    response_split = response_split.drop('index', axis=1)
    # find the index of the cue in the response (not present returns nan)
    cue_index = pd.DataFrame([[nanindex(x, cue_clean[id])] for id, x in enumerate(response_split['response_clean'])])
    # get the column headers onto this data
    # if the cue is at the start then the category is x_y
    syntagmatic = pd.DataFrame([])
    syntagmatic['consecutivexy'] = np.equal(cue_index.loc[:, 0], 0)
    # if the cue is second then the category is y_x
    syntagmatic['consecutiveyx'] = np.equal(cue_index.loc[:, 0], 1)
    # otherwise the cue is contained somewhere else in the phrase
    syntagmatic['othercollocational'] = np.greater(cue_index.loc[:, 0], 1)
    # check for instances of the cue NOT being present in the response, the response being 2 words, and the response
    # beginning with for|and|nor|but|or|yet|so_ (e.g. cue = back, response = and forth)
    cue_not_present = cue_index.isnull().unstack().reset_index()
    cue_not_present = cue_not_present.drop(['level_0', 'level_1'], axis=1)
    response_two_words = response_split[response_split['response_clean'].str.len() >= 2]
    # intersection of a multi index with an integer index doesn't work
    more_than_two_words_and_cue_not_present_indices = cue_not_present.index[response_two_words.index]
    # TODO taking for out of this regex as it seems to return non-syntagmatic responses or proverbs
    consecutivexy_cue_not_entered_matches = response_clean[more_than_two_words_and_cue_not_present_indices].str.match(
        '^(and|but|or)_')
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
    # roll up the various categories into one column and give different values for each
    syntagmatic = pd.DataFrame(np.where(syntagmatic == 1, syntagmatic.columns, np.nan))
    syntagmatic = syntagmatic.fillna('')
    syntagmatic = pd.DataFrame(syntagmatic.apply(lambda x: ''.join(x), axis=1), columns=['syntagmatic'])
    syntagmatic.index = df.index
    syntagmatic['syntagmatic'] = syntagmatic['syntagmatic'].replace({'': None})
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
    # try:
    #     # e.g. C630 M300
    #     if response_soundex[-3:].index(cue_soundex[-2:]) > -1 and levenshtein_distance <= 3:
    #         return 'soundex'
    # except ValueError:
    #     pass
    #
    # try:
    #     # e.g. C630 M300
    #     if cue_soundex[-3:].index(response_soundex[-2:]) > -1 and levenshtein_distance <= 3:
    #         return 'soundex'
    # except ValueError:
    #     pass

    if levenshtein_distance < 2:
        return 'levenshtein'

    return None


def check_affix(df):
    # TODO look closer at categories of affix https://en.wikipedia.org/wiki/Affix
    response_clean = df.index.get_level_values('response_clean').str.decode('utf8')
    cue_clean = df.index.get_level_values('cue_clean').str.decode('utf8')
    # stem check first and then perform the remainder on a subset
    stems_equal = pd.DataFrame(
        np.logical_and(cue_clean != response_clean, np.equal(df['cue_stem'], df['response_stem'])),
        columns=['affix'])
    # make sure they are not identical and not already classed as affixes
    not_identical_not_affixed = np.logical_and(response_clean != cue_clean, stems_equal['affix'] == False)
    response_subset = response_clean[not_identical_not_affixed]
    cue_subset = cue_clean[not_identical_not_affixed]
    cue_index = pd.DataFrame([[nanindex(x, cue_subset[int(idx)])] for idx, x in enumerate(response_subset)])
    # get rid of any cue-response pairs where the cue is not in the response
    cue_present_index = [cue_index[0] >= 0]
    response_subset = response_subset[cue_present_index[0]]
    cue_subset = cue_subset[cue_present_index[0]]
    if len(response_subset) and len(cue_subset):
        # the cue is present in some of the responses
        # shouldn't need to strip x here as spaces should already be replaced by underscores
        # split on first occurrence of cue only
        remainder = pd.DataFrame([[x.split(cue_subset[int(idx)], 1)] for idx, x in enumerate(response_subset)])
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
        affix_col = np.logical_or(affix_col, stems_equal)
    else:
        affix_col = stems_equal
    affix_col.index = df.index
    df = pd.concat([df, affix_col], axis=1, sort=False)
    return df


def check_measures(all_measures, distinct_words):
    # TODO if multi word responses, get the measures for each word and then take the mean
    col_name = distinct_words.columns[0]

    # TODO use the stem column that is already present in the overall dataframe
    stems = pd.DataFrame(
        [[response, ps.stem(response)] for response in distinct_words[col_name]], columns=[col_name, 'stem'])

    logging.debug('Stems length is {}'.format(len(stems)))
    all_measures_words = all_measures.index.get_level_values(level='word')
    words_intersect = np.intersect1d(distinct_words, all_measures_words)
    words_measures = all_measures[all_measures.index.isin(words_intersect)]
    words_measures_full = pd.merge(pd.DataFrame(distinct_words), pd.DataFrame(words_measures),
                                   left_on=col_name, right_on='word', how='left')
    stem_measures = all_measures[all_measures.index.isin(stems['stem'])]
    stem_measures_full = pd.merge(stems, pd.DataFrame(stem_measures), left_on='stem', right_on='word', how='left')
    stem_measures_full.drop('stem', axis=1, inplace=True)
    # replace the nan values in measures with the stem measures
    words_measures_full.update(stem_measures_full, overwrite=False)
    multi_words = distinct_words[distinct_words[col_name].str.contains('_')]
    split_words = multi_words[col_name].str.split('_')
    # split is  a list of words [first, second, third] which we want to return [True, True, True] as being in all_measures
    # then we want to get all these measures and take the mean
    # this is done for each split list
    split_words_measures = pd.DataFrame(
        [all_measures[all_measures.index.isin(split)].mean().T for split in split_words.values],
        columns=all_measures.columns)
    split_words_measures.index = pd.Index(multi_words[col_name])
    words_measures_full.set_index(col_name, inplace=True)
    words_measures_full.update(split_words_measures, overwrite=False)
    words_measures_full.reset_index(inplace=True)
    return words_measures_full


def check_semantic_atlas(df):
    # ### semantic atlas from Gallant et. al
    # initialise brain data
    brain_map = BrainMap()
    # cue semantic areas are only used to determine if the area changes between cue and response
    cue_semantic_areas = brain_map.get_semantic_area(df.index.get_level_values('cue_clean').unique())
    # response semantic areas
    response_semantic_areas = brain_map.get_semantic_area(df.index.get_level_values('response_clean').unique())
    both_semantic_areas = pd.concat([cue_semantic_areas, response_semantic_areas]).drop_duplicates().set_index('word')

    df = df.join(both_semantic_areas, on='response_clean')
    df.rename(columns={'left_hemi_label': 'response_l_hemi_label',
                       'right_hemi_label': 'response_r_hemi_label'}, inplace=True)

    df = df.join(both_semantic_areas, on='cue_clean')
    df.rename(columns={'left_hemi_label': 'cue_l_hemi_label',
                       'right_hemi_label': 'cue_r_hemi_label'}, inplace=True)

    # filter the nans in cue and response data and then look for the stem instead
    cue_filter = np.logical_or(df['cue_l_hemi_label'].isna(), df['cue_r_hemi_label'].isna())
    cues_missing_hemi_label = df['cue_stem'][cue_filter]
    cue_stems = df['cue_stem'][df['cue_stem'].isin(cues_missing_hemi_label)].str.encode('utf8')
    cue_stem_semantic_areas = brain_map.get_semantic_area(cue_stems.unique())
    response_filter = np.logical_or(df['response_l_hemi_label'].isna(), df['response_r_hemi_label'].isna())
    responses_missing_hemi_label = df['response_stem'][df['response_stem'].isin(response_filter)]
    response_stems = df['response_stem'][df['response_stem'].isin(responses_missing_hemi_label)].str.encode('utf8')
    response_stem_semantic_areas = brain_map.get_semantic_area(response_stems.unique())
    both_semantic_areas_stems = pd.concat(
        [cue_stem_semantic_areas, response_stem_semantic_areas]).drop_duplicates().set_index('word')

    # TODO response_stem and cue_stem are not bytes but word is bytes
    # TODO decode the 'word' column before setting the index (above)
    df = df.join(both_semantic_areas_stems, on='response_stem')
    df.rename(columns={'left_hemi_label': 'response_stem_l_hemi_label',
                       'right_hemi_label': 'response_stem_r_hemi_label'}, inplace=True)
    df = df.join(both_semantic_areas_stems, on='cue_stem')
    df.rename(columns={'left_hemi_label': 'cue_stem_l_hemi_label',
                       'right_hemi_label': 'cue_stem_r_hemi_label'}, inplace=True)

    print(both_semantic_areas_stems)
    print(df.loc[df['response_stem_l_hemi_label'].notna()])

    # take the stem label if the original hemi label is NaN
    df['response_l_hemi_label'].fillna(df['response_stem_l_hemi_label'], inplace=True)
    df['response_r_hemi_label'].fillna(df['response_stem_r_hemi_label'], inplace=True)
    df['cue_l_hemi_label'].fillna(df['cue_stem_l_hemi_label'], inplace=True)
    df['cue_r_hemi_label'].fillna(df['cue_stem_r_hemi_label'], inplace=True)

    # then drop the stem labels
    df.drop(
        ['cue_stem_l_hemi_label', 'cue_stem_r_hemi_label', 'response_stem_l_hemi_label', 'response_stem_r_hemi_label'],
        axis=1, inplace=True)

    # Fill with U (Unknown) to avoid intersecting with labels that are not statistically significant
    df['response_l_hemi_label'] = df['response_l_hemi_label'].fillna(b'').replace(to_replace=b'',
                                                                                  value=UNKNOWN_AREA_LABEL)
    df['response_r_hemi_label'] = df['response_r_hemi_label'].fillna(b'').replace(to_replace=b'',
                                                                                  value=UNKNOWN_AREA_LABEL)
    df['cue_l_hemi_label'] = df['cue_l_hemi_label'].fillna(b'').replace(to_replace=b'', value=UNKNOWN_AREA_LABEL)
    df['cue_r_hemi_label'] = df['cue_r_hemi_label'].fillna(b'').replace(to_replace=b'',
                                                                        value=UNKNOWN_AREA_LABEL)
    df['r_hemi_area_differs'] = np.logical_and.reduce((df['response_r_hemi_label'] !=
                                                       df['cue_r_hemi_label'],
                                                       df['response_r_hemi_label'] != UNKNOWN_AREA_LABEL,
                                                       df['cue_r_hemi_label'] != UNKNOWN_AREA_LABEL))
    df['l_hemi_area_differs'] = np.logical_and.reduce((df['response_l_hemi_label'] !=
                                                       df['cue_l_hemi_label'],
                                                       df['response_l_hemi_label'] != UNKNOWN_AREA_LABEL,
                                                       df['cue_l_hemi_label'] != UNKNOWN_AREA_LABEL))
    right_unknown_areas = np.logical_or(df['response_r_hemi_label'] == UNKNOWN_AREA_LABEL,
                                        df['cue_r_hemi_label'] == UNKNOWN_AREA_LABEL)
    right_hemi_differs = df['r_hemi_area_differs'].copy(deep=True)
    right_hemi_differs[right_unknown_areas] = np.nan
    df['r_hemi_area_differs'] = right_hemi_differs
    left_unknown_areas = np.logical_or(df['response_l_hemi_label'] == UNKNOWN_AREA_LABEL,
                                       df['cue_l_hemi_label'] == UNKNOWN_AREA_LABEL)
    left_hemi_differs = df['l_hemi_area_differs'].copy(deep=True)
    left_hemi_differs[left_unknown_areas] = np.nan
    df['l_hemi_area_differs'] = left_hemi_differs

    left_hemi_distance = brain_map.get_distances_between_semantic_areas('left', df['cue_l_hemi_label'],
                                                                        df['response_l_hemi_label'])

    right_hemi_distance = brain_map.get_distances_between_semantic_areas('right', df['cue_r_hemi_label'],
                                                                         df['response_r_hemi_label'])
    df = df.merge(left_hemi_distance['left_hemi_distance'], how='left', left_index=True, right_index=True)
    df = df.merge(right_hemi_distance['right_hemi_distance'], how='left', left_index=True, right_index=True)

    # these are indicative of the actual words, which we do not want to model. We want to model attributes of the words.
    df = df.drop(
        ['response_l_hemi_label', 'response_r_hemi_label', 'cue_l_hemi_label', 'cue_r_hemi_label',
         'cue_stem', 'response_stem'],
        axis=1)
    return df


def setup():
    parser = argparse.ArgumentParser(description='Import various types of word data')
    parser.add_argument('--source', default=None, help="code responses from a particular source only")
    parser.add_argument('--filter', default=None, help="restrict responses to those that appear in a word filter list")
    parser.add_argument('--save', action="store_true", default=False,
                        help="whether to save the coding, default is False")
    args = parser.parse_args()
    main(save=args.save, source=args.source, filter=args.filter)


if __name__ == "__main__":
    setup()
