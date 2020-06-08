from backend.helpers.string import syllables
from backend.db.db import get_cue_response_pairs, get_affixes, save_coded_pairs, delete_coded_pairs, \
    get_all_words, get_all_measures
from backend.lib.setup import db_setup
from nltk.stem import PorterStemmer
import re
import jellyfish
from backend.lib.wordnet_relations import WordnetRelations
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

    # Splits are needed for many of the functions below
    df['response_split'] = df.index.get_level_values('response_clean').str.decode('utf8').str.split('_')
    df['cue_split'] = df.index.get_level_values('cue_clean').str.decode('utf8').str.split('_')

    # TODO consider the categories found in the Gallant Research:
    #   (visual, tactile, outdoor, bodypart, number, place, violence, person, social, mental, time)

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

    assert (len(cue_measures_full) == len(response_measures_full))

    measures_diff = cue_measures_full.drop(['cue_clean', 'part_of_speech'], axis=1).subtract(
        response_measures_full.drop(
            ['response_clean', 'part_of_speech'], axis=1))

    measures_diff['cue_clean'] = cue_measures_full['cue_clean'].str.encode('utf8')
    measures_diff['response_clean'] = response_measures_full['response_clean'].str.encode('utf8')
    measures_diff.set_index(['cue_clean', 'response_clean'], inplace=True)

    # TODO get the parts of speech from s2v if they are missing from all_measures
    # TODO parts of speech changes
    # pos_changes = pd.DataFrame([[nanindex(x, cue_measures_full['part_of_speech'][id]) == np.nan] for id, x in
    #                           enumerate(response_measures_full['part_of_speech'])])
    # print(pos_changes)
    # print(measures_diff)

    # df['exact_match'] = df.index.get_level_values(
    #     'response_clean') == df.index.get_level_values('cue_clean')
    df['length'] = df['cue_decoded'].str.len() - df['response_decoded'].str.len()
    df['syllables'] = np.vectorize(syllables)(df['cue']) - np.vectorize(syllables)(df['response_raw'])
    df['capitalised'] = np.equal(df['cue'].str.isupper(), df['response_raw'].str.isupper())
    df['punctuation'] = np.equal(df['cue'].str.contains(PUNCTUATION_REGEX), df['response_raw'].str.contains(
        PUNCTUATION_REGEX))
    df['number_of_words'] = df['cue_decoded'].str.split('_').str.len() - df['response_decoded'].str.split('_').str.len()
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
    df = df.join(measures_diff)
    df.reset_index(inplace=True)
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
    # df.drop(['response_split', 'cue_split'], inplace=True)
    df = df.rename({'cue_clean': 'cue', 'response_clean': 'response'}, axis=1)
    df.drop(['response_split', 'cue_split'], inplace=True, axis=1)
    df['cue'] = df['cue'].str.decode('utf8')
    df['response'] = df['response'].str.decode('utf8')
    df = df.replace({np.nan: None})
    if save:
        save_coded_pairs(df, df.columns)
    logging.debug('Job with PID {} reached end'.format(os.getpid()))
    totals = df.count()
    return totals


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
    # for was removed from this regex as it seems to return non-syntagmatic responses or proverbs
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
    # ### semantic atlas from [Huth et al., 2016](https://www.nature.com/articles/nature17637)
    # initialise brain data
    brain_map = BrainMap()
    col1 = 'cue_clean'
    col2 = 'response_clean'
    stem1 = 'cue_stem'
    stem2 = 'response_stem'
    col1_l, col1_r = create_hemi_labels(col1)
    col2_l, col2_r = create_hemi_labels(col2)
    stem1_l, stem1_r = create_hemi_labels(stem1)
    stem2_l, stem2_r = create_hemi_labels(stem2)

    # cue semantic areas are only used to determine if the area changes between cue and response
    cues = df.index.get_level_values(col1).unique()
    responses = df.index.get_level_values(col2).unique()
    df = merge_semantic_columns(brain_map, {col1: cues, col2: responses}, df)

    # gets 2% more of the responses coded with hemi labels
    # filter the nans in cue and response data and then look for the stem instead
    cue_stems = get_unlabelled(df, [col1_l, col1_r], stem1, encode=True)
    response_stems = get_unlabelled(df, [col2_l, col2_r], stem2, encode=True)
    df = merge_semantic_columns(brain_map, {stem1: cue_stems, stem2: response_stems}, df, decode=True)

    # take the stem label if the original hemi label is NaN
    df[col2_l].fillna(df[stem2_l], inplace=True)
    df[col2_r].fillna(df[stem2_r], inplace=True)
    df[col1_l].fillna(df[stem1_l], inplace=True)
    df[col1_r].fillna(df[stem1_r], inplace=True)

    # then drop the stem labels
    df.drop([stem1_l, stem1_r, stem2_l, stem2_r], axis=1, inplace=True)

    # Fill with U (Unknown) to avoid intersecting with labels that are not statistically significant
    df[col2_l] = df[col2_l].fillna(b'').replace(to_replace=b'', value=UNKNOWN_AREA_LABEL)
    df[col2_r] = df[col2_r].fillna(b'').replace(to_replace=b'', value=UNKNOWN_AREA_LABEL)
    df[col1_l] = df[col1_l].fillna(b'').replace(to_replace=b'', value=UNKNOWN_AREA_LABEL)
    df[col1_r] = df[col1_r].fillna(b'').replace(to_replace=b'', value=UNKNOWN_AREA_LABEL)
    df['r_hemi_area_differs'] = np.logical_and.reduce((df[col2_r] != df[col1_r],
                                                       df[col2_r] != UNKNOWN_AREA_LABEL,
                                                       df[col1_r] != UNKNOWN_AREA_LABEL))
    df['l_hemi_area_differs'] = np.logical_and.reduce((df[col2_l] !=
                                                       df[col1_l],
                                                       df[col2_l] != UNKNOWN_AREA_LABEL,
                                                       df[col1_l] != UNKNOWN_AREA_LABEL))
    right_unknown_areas = np.logical_or(df[col2_r] == UNKNOWN_AREA_LABEL, df[col1_r] == UNKNOWN_AREA_LABEL)
    right_hemi_differs = df['r_hemi_area_differs'].copy(deep=True)
    right_hemi_differs[right_unknown_areas] = np.nan
    df['r_hemi_area_differs'] = right_hemi_differs
    left_unknown_areas = np.logical_or(df[col2_l] == UNKNOWN_AREA_LABEL, df[col1_l] == UNKNOWN_AREA_LABEL)
    left_hemi_differs = df['l_hemi_area_differs'].copy(deep=True)
    left_hemi_differs[left_unknown_areas] = np.nan
    df['l_hemi_area_differs'] = left_hemi_differs

    # distances between hemi labels
    left_hemi_distance = brain_map.get_distances_between_semantic_areas('left', df[col1_l], df[col2_l])
    right_hemi_distance = brain_map.get_distances_between_semantic_areas('right', df[col1_r], df[col2_r])
    df = df.merge(left_hemi_distance['left_hemi_distance'], how='left', left_index=True, right_index=True)
    df = df.merge(right_hemi_distance['right_hemi_distance'], how='left', left_index=True, right_index=True)

    # get the averaged area for multi word cues and responses
    df = semantic_atlas_multi_word(df, brain_map, col1_l, col1_r, col2_l, col2_r)

    # these are indicative of the actual words, which we do not want to model. We want to model attributes of the words.
    df = df.drop([col2_l, col2_r, col1_l, col1_l, col1_r, stem1, stem2], axis=1)
    return df


def create_hemi_labels(col):
    col_l = '{}_l_hemi_label'.format(col)
    col_r = '{}_r_hemi_label'.format(col)
    return col_l, col_r


def merge_semantic_columns(brain_map, cols, df, decode=None):
    areas = []
    for col_prefix, col in cols.items():
        semantic_areas = brain_map.get_semantic_area(col)
        areas.append(semantic_areas)
        # if col_prefix == 'word':
        #     print('col_prefix is {}'.format(col_prefix))
        #     logging.debug('column looks like:')
        #     print(col)
        #     logging.debug('semantic areas that have been labelled')
        #     print(semantic_areas[semantic_areas.notnull()])

    # semantic areas
    both_semantic_areas = pd.concat(areas).drop_duplicates()

    if decode:
        col_name = both_semantic_areas.columns[0]
        both_semantic_areas[col_name] = both_semantic_areas[col_name].str.decode('utf8')

    both_semantic_areas.set_index('word', inplace=True)

    for col_prefix in cols:
        df = df.join(both_semantic_areas, on=col_prefix)
        df.rename(columns={'left_hemi_label': '{}_l_hemi_label'.format(col_prefix),
                           'right_hemi_label': '{}_r_hemi_label'.format(col_prefix)}, inplace=True)

    return df


def semantic_atlas_multi_word(df, brain_map, cue_hemi_l, cue_hemi_r, res_hemi_l, res_hemi_r):
    # TODO huge big mess, refactor to unpack all words by default and then apply semantic data back using row indexes?
    # get the individual words from multi word (len > 2) cue/responses
    cue_splits = unpack_splits(df['cue_split'])
    response_splits = unpack_splits(df['response_split'])
    # unpack the splits into one or two big columns?
    if cue_splits is not None and response_splits is None:
        unpacked_words = pd.DataFrame(np.unique(np.sort(cue_splits[:, 0], columns=['word'])))
    elif cue_splits is None and response_splits is not None:
        unpacked_words = pd.DataFrame(np.unique(np.sort(response_splits[:, 0], columns=['word'])))
    elif cue_splits is not None and response_splits is not None:
        unpacked_words = pd.DataFrame(np.unique(np.sort(np.append(cue_splits[:, 0], response_splits[:, 0].T, axis=0))),
                                      columns=['word'])
    else:
        # what are we even doing here
        return df
    unpacked_words_encode = pd.DataFrame(unpacked_words['word'].str.encode('utf8'), columns=['word'])

    # words from split columns are unencoded so we need to decode the labels returned from semantic atlas
    cols = {'word': unpacked_words_encode}
    unpacked_labelled = merge_semantic_columns(brain_map, cols=cols, df=unpacked_words, decode=True)
    # we now have a df of words against their original indices in the overall cue/response df, plus hemi labels
    # we need to attach coordinates to each of the labels
    unpacked_labelled['word_l_hemi_label'] = unpacked_labelled['word_l_hemi_label'].fillna(b'').replace(to_replace=b'',
                                                                                                        value=UNKNOWN_AREA_LABEL)
    unpacked_labelled['word_r_hemi_label'] = unpacked_labelled['word_r_hemi_label'].fillna(b'').replace(to_replace=b'',
                                                                                                        value=UNKNOWN_AREA_LABEL)
    # get the coordinates of each word in the multi word cue/response
    left = brain_map.get_semantic_coordinates(hemi='left', labels=unpacked_labelled['word_l_hemi_label'])
    right = brain_map.get_semantic_coordinates(hemi='right', labels=unpacked_labelled['word_r_hemi_label'])
    unpacked_labelled = unpacked_labelled.merge(left, left_on='word_l_hemi_label', right_index=True)
    unpacked_labelled.rename({'coord_x': 'l_coord_x', 'coord_y': 'l_coord_y', 'coord_z': 'l_coord_z'}, inplace=True,
                             axis=1)
    unpacked_labelled = unpacked_labelled.merge(right, left_on='word_r_hemi_label', right_index=True)
    unpacked_labelled.rename({'coord_x': 'r_coord_x', 'coord_y': 'r_coord_y', 'coord_z': 'r_coord_z'}, inplace=True,
                             axis=1)
    unpacked_words = unpacked_words.merge(unpacked_labelled, left_on='word', right_on='word')
    # take the mean of the multi word coordinates and attach them back to the original cue/responses
    cue_splits = merge_coordinates_with_splits(unpacked_words, cue_splits)
    response_splits = merge_coordinates_with_splits(unpacked_words, response_splits)
    df.reset_index(inplace=True)
    cue_splits.index = cue_splits.index.astype(int)
    response_splits.index = response_splits.index.astype(int)
    # put the averaged coordinates in the main df
    df = df.merge(cue_splits, left_index=True, right_index=True, how='left')
    df = df.rename({'l_coord_x': 'cue_l_coord_x', 'l_coord_y': 'cue_l_coord_y', 'l_coord_z': 'cue_l_coord_z'}, axis=1)
    df = df.rename({'r_coord_x': 'cue_r_coord_x', 'r_coord_y': 'cue_r_coord_y', 'r_coord_z': 'cue_r_coord_z'}, axis=1)
    df = df.merge(response_splits, left_index=True, right_index=True, how='left')
    df = df.rename(
        {'l_coord_x': 'response_l_coord_x', 'l_coord_y': 'response_l_coord_y', 'l_coord_z': 'response_l_coord_z'},
        axis=1)
    df = df.rename(
        {'r_coord_x': 'response_r_coord_x', 'r_coord_y': 'response_r_coord_y', 'r_coord_z': 'response_r_coord_z'},
        axis=1)
    # cue may be single word and response may be multi word and vice versa so we need to identify rows where
    # cue is single word and response is multi word or vice versa
    # cue is multi word and response is multi word
    # and coordinates exist for l or r hemi
    # Create filters to find rows fitting the above criteria
    # vars are cue_l, cue_r, response_l, response_r and their coordinates
    cue_l_multi = np.logical_and(df[res_hemi_l] != UNKNOWN_AREA_LABEL, ~np.isnan(df['cue_l_coord_x']))
    cue_r_multi = np.logical_and(df[res_hemi_r] != UNKNOWN_AREA_LABEL, ~np.isnan(df['cue_r_coord_x']))
    res_l_multi = np.logical_and(df[cue_hemi_l] != UNKNOWN_AREA_LABEL, ~np.isnan(df['response_l_coord_x']))
    res_r_multi = np.logical_and(df[cue_hemi_r] != UNKNOWN_AREA_LABEL, ~np.isnan(df['response_r_coord_x']))
    cueres_l_multi = np.logical_and(
        np.logical_not(df['cue_l_coord_x'].isna()), np.logical_not(df['response_l_coord_x'].isna()))
    cueres_r_multi = np.logical_and(
        np.logical_not(df['cue_r_coord_x'].isna()), np.logical_not(df['response_r_coord_x'].isna()))
    distances = []
    distances.append(get_distance_between_coords(brain_map, df, res_hemi_l, cue_hemi_l, row_filter=cue_l_multi,
                                                 hemi='left'))
    distances.append(get_distance_between_coords(brain_map, df, res_hemi_r, cue_hemi_r, row_filter=cue_r_multi,
                                                 hemi='right'))
    distances.append(get_distance_between_coords(brain_map, df, cue_hemi_l, res_hemi_l, row_filter=res_l_multi,
                                                 hemi='left'))
    distances.append(get_distance_between_coords(brain_map, df, cue_hemi_r, res_hemi_r, row_filter=res_r_multi,
                                                 hemi='right'))
    distances.append(get_distance_between_coords(brain_map, df, cue_hemi_l, res_hemi_l, row_filter=cueres_l_multi,
                                                 hemi='left'))
    distances.append(get_distance_between_coords(brain_map, df, cue_hemi_r, res_hemi_r, row_filter=cueres_r_multi,
                                                 hemi='right'))

    for distance_col in distances:
        df.update(distance_col, overwrite=False)

    for side in ['l', 'r']:
        df.drop(['cue_{}_coord_x'.format(side),
                 'cue_{}_coord_y'.format(side),
                 'cue_{}_coord_z'.format(side),
                 'response_{}_coord_x'.format(side),
                 'response_{}_coord_y'.format(side),
                 'response_{}_coord_z'.format(side)], inplace=True, axis=1)
    # set the index back to what it was
    df.set_index(['cue_clean', 'response_clean'], inplace=True)
    # print(df[df.columns[-6:]])
    return df


def merge_coordinates_with_splits(unpacked_words, split_col):
    split_col = pd.DataFrame(split_col, columns=['word', 'rowid']).merge(unpacked_words, left_on='word',
                                                                         right_on='word').sort_values('rowid')
    split_col.drop(['word', 'word_l_hemi_label', 'word_r_hemi_label'], inplace=True, axis=1)
    split_col = split_col.astype('float')
    split_col_centre = split_col.groupby('rowid').mean()

    return split_col_centre


def get_unlabelled(df, should_be_labelled, other_col, encode=None):
    # takes a list of columns and checks if there are any nans present
    # returns values from other_col where nans are present in should_be_labelled
    nancheck = []
    for col in should_be_labelled:
        nancheck.append(df[col].isna())
    if len(nancheck) > 2:
        # These would need to be unpacked into wrapping logical_ors or(a, or(b, or...)
        raise NotImplemented('Checks for more than 2 columns have not yet been implmented')
    filter = np.logical_or(nancheck[0], nancheck[1])
    missing_hemi_label = df[other_col][filter]
    if encode:
        other_col = df[other_col][df[other_col].isin(missing_hemi_label)].str.encode('utf8').unique()
    else:
        other_col = df[other_col][df[other_col].isin(missing_hemi_label)].str.unique()
    return other_col


def unpack_splits(splits):
    # return a list of individual words from columns of word lists and include the original column index of the list
    unpacked = None
    try:
        unpacked = np.vstack([np.c_[x, [idx] * len(x)] for idx, x in enumerate(splits) if len(x) > 2])
    except ValueError:
        # the array may be empty if there are no multi word strings, causing concatenate to fail with
        # "ValueError: need at least one array to concatenate"
        pass
    return unpacked


def get_distance_between_coords(brain_map, df, label_col_left, label_col_right, row_filter, hemi, both_multi=None):
    # the left column is the one with the individual words
    # the right column contains multi word cues/responses
    # UNLESS both_multi is set, which implies both cue and response are multi word
    distances = np.empty(0)
    prefix_left = label_col_left.split('_')[0]
    prefix_right = label_col_right.split('_')[0]
    distance_label = '{}_hemi_distance'.format(hemi)
    if not df[row_filter][label_col_left].empty:
        # if one of the columns contains single word responses
        if both_multi is None:
            # label is set as index when returned from get_semantic_coordinates so have we reset it
            left_labels = brain_map.get_semantic_coordinates('left', df[row_filter][label_col_left])
            if not left_labels.empty:
                left_coords = pd.merge(df, left_labels, left_on=label_col_left, right_index=True, how='left')
                distances = pd.DataFrame(brain_map.get_distance(
                    (left_coords[row_filter]['coord_x'.format(hemi)],
                     left_coords[row_filter]['coord_y'.format(hemi)],
                     left_coords[row_filter]['coord_z'.format(hemi)]),
                    # format for
                    (df[row_filter]['{}_{}_coord_x'.format(prefix_right, hemi[0])],
                     df[row_filter]['{}_{}_coord_y'.format(prefix_right, hemi[0])],
                     df[row_filter]['{}_{}_coord_z'.format(prefix_right, hemi[0])])), columns=[distance_label])

                # need the indexes of the original filter to attach back to the df
                distances.index = row_filter.index[row_filter == True]
        # both columns contain hemiword responses
        # we're doing distance between hemi word cue coords (l|r) and response coords (l|r)
        else:
            distances = pd.DataFrame(brain_map.get_distance(
                (df[row_filter]['{}_{}_coord_x'.format(prefix_left, hemi[0])],
                 df[row_filter]['{}_{}_coord_y'.format(prefix_left, hemi[0])],
                 df[row_filter]['{}_{}_coord_z'.format(prefix_left, hemi[0])]),
                (df[row_filter]['{}_{}_coord_x'.format(prefix_right, hemi[0])],
                 df[row_filter]['{}_{}_coord_y'.format(prefix_right, hemi[0])],
                 df[row_filter]['{}_{}_coord_z'.format(prefix_right, hemi[0])])), columns=[distance_label])

            # need the indexes of the original filter to attach back to the df
            distances.index = row_filter.index[row_filter == True]
    return distances


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
