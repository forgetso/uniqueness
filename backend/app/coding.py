from backend.lib.helpers.string import syllables
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
from backend.settings import SEMANTIC_SPACE_FOLDER, JOB_NUMBER, LOG_LEVEL
from backend.lib.helpers.pairs import format_cue_response_pairs
import os
import numpy as np
import pandas as pd
import logging
from backend.lib.sense2vec import NUM_TYPES, NAMED_ENTITITES
from backend.app.semantic_atlas import check_semantic_atlas

logging.basicConfig(level=LOG_LEVEL)
MIN_CUE_LENGTH = 3
# set of affixes
AFFIXES = {x.get('affix') for x in get_affixes()}
ALL_MEASURES = get_all_measures()
logging.info('There are {} words in ALL MEASURES'.format(len(ALL_MEASURES)))
ALL_WORDS = get_all_words()
ps = PorterStemmer()
SEMANTIC_SPACE_DATA = np.load('{}/semantic_space.npz'.format(SEMANTIC_SPACE_FOLDER))
ENGLISH1000_VOCAB = SEMANTIC_SPACE_DATA['english1000_vocab']
PUNCTUATION_REGEX = re.compile(r'[\.\-\',]')
DROP_BEFORE_SAVING = ['{}_clean', '{}_decoded', '{}_split', '{}_correction', '{}_pos', '{}_sense', '{}_bad_spell']


# TODO get rid of the slice that's causing this error
# pd.options.mode.chained_assignment = 'raise'


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
        logging.debug('deleting existing coding')
        delete_coded_pairs()
        # make sure all of the required columns are in the coding table
        logging.debug('setting up db')
        try:
            db_setup()
        except Exception as e:
            logging.error('Database setup error {}'.format(e))
            exit(0)
    distinct_cue_response_pairs = format_cue_response_pairs(all_pairs, word_filter=word_filter)
    logging.debug('there are {} distinct cue response pairs'.format(len(distinct_cue_response_pairs)))
    if not len(all_pairs):
        raise ValueError('No responses to code...')
    start = time.perf_counter()
    logging.debug('dispatching jobs...')
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
    logging.debug('in job...')
    # var df: a dataframe of distinct cue - response pairs
    #
    # This algorithm is for coding DISTINCT cue response pairs once they have had their spelling corrected. It considers
    # categories such as prevalence, concreteness, age of acquisition, emotions, orthographic neighbourhoods,
    # phonographic neighbourhoods, active brain region.

    # store the len for checks
    df_len = len(df)

    # First, work out which spelling corrections should be used
    df = check_corrections(df)

    # Splits are needed for many of the functions below
    df['response_split'] = df.index.get_level_values('response_clean').str.decode('utf8').str.split('_')
    df['cue_split'] = df.index.get_level_values('cue_clean').str.decode('utf8').str.split('_')

    # attach the measures from all measures to the cues/responses and then subtract one from the other to get the diff
    measures_diff = get_measures_diff(df)

    # should exact match be considered?
    df = basic_coding(df)

    relations_array = np.vectorize(relation_coding)(df.index.get_level_values('cue_clean'),
                                                    df.index.get_level_values('response_clean'))

    wordnet_categories = pd.json_normalize(relations_array)
    wordnet_categories.index = df.index

    assert (len(wordnet_categories) == df_len)

    # why is phonology lumped in here?
    wordnet_categories['phonology'] = np.vectorize(check_phonology)(df['cue_decoded'], df['response_decoded'])
    wordnet_categories['phonology'] = wordnet_categories['phonology'].astype('category')

    assert (len(wordnet_categories) == df_len)
    df = pd.concat([df, wordnet_categories], axis=1, sort=False)

    # check for syntagmatic and add to dataframe
    df = check_syntagmatic(df)
    assert (len(df) == df_len)

    # check for affix and add to dataframe
    df = check_affix(df)
    assert (len(df) == df_len)

    # add the semantic brain area data
    df = check_semantic_atlas(df)
    assert (len(df) == df_len)

    # various word measures related to the response and cue that have been subtracted from each other to give distances
    df = pd.concat([df.reset_index(), measures_diff], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    assert (len(df) == df_len)

    # check if the part of speech changes from cue to response, based on pos column and s2vsense
    df = check_senses_and_pos(df)
    assert (len(df) == df_len)

    # get rid of unnecessary columns
    drop_before_saving = []
    for x in ['cue', 'response']:
        for col in DROP_BEFORE_SAVING:
            drop_before_saving.append(col.format(x))
    df = df.drop(drop_before_saving, axis=1)

    # look at the final 6 columns
    # print(df[df.columns[-6:]])

    # save data to the db
    df = df.rename({'cue_raw': 'cue', 'response_raw': 'response'}, axis=1)

    df = df.replace({np.nan: None})
    # df = df.drop_duplicates()
    if save:
        save_coded_pairs(df, df.columns)

    logging.debug('Job with PID {} reached end'.format(os.getpid()))
    totals = df.count()
    return totals


def basic_coding(df):
    print(df)
    df['length'] = df['cue_decoded'].str.len() - df['response_decoded'].str.len()
    df['syllables'] = np.vectorize(syllables)(df['cue_raw']) - np.vectorize(syllables)(df['response_raw'])
    df['capitalised'] = np.equal(df['cue_raw'].str[0].str.isupper(), df['response_raw'].str[0].str.isupper())
    df['punctuation'] = np.equal(df['cue_raw'].str.contains(PUNCTUATION_REGEX), df['response_raw'].str.contains(
        PUNCTUATION_REGEX))
    df['number_of_words'] = df['cue_decoded'].str.split('_').str.len() - df['response_decoded'].str.split('_').str.len()
    return df


def get_measures_diff(df):
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
    # add the parts of speech back into the measures
    measures_diff['cue_pos'] = cue_measures_full['part_of_speech']
    measures_diff['response_pos'] = response_measures_full['part_of_speech']

    # measures_diff.set_index(['cue_clean', 'response_clean'], inplace=True)
    return measures_diff


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
    # make sure they are not identical and not already classed as affixes and also long enough to add an affix to
    not_identical_not_affixed = np.logical_and.reduce((response_clean != cue_clean,
                                                       stems_equal['affix'] == False,
                                                       np.greater(response_clean.str.len(), 1),
                                                       np.greater(cue_clean.str.len(), 1)))
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
    print([all_measures[all_measures.index.isin(split)] for split in split_words.values])
    # this is done for each split list
    split_words_measures = pd.DataFrame(
        [all_measures[all_measures.index.isin(split)].mean().T for split in split_words.values],
        columns=all_measures.columns)
    split_words_measures.index = pd.Index(multi_words[col_name])
    words_measures_full.set_index(col_name, inplace=True)
    words_measures_full.update(split_words_measures, overwrite=False)
    words_measures_full.reset_index(inplace=True)
    return words_measures_full


def check_corrections(df):
    # rules for handling spelling corrections
    # - check if there are spelling corrections for cue/response
    # - check if the sense, as defined by sense2vec, is numeric or a named_entity
    # - only use the spelling correction if the sense is not a numeric or named_entity
    # - if both cue and response are incorrect/correct, return False
    # - if one of cue/response is incorrect and one is correct, return True
    unfiltered_senses = np.array([x.lower() for x in NUM_TYPES + NAMED_ENTITITES])
    for x in ['cue', 'response']:
        sensecol = '{}_sense'.format(x)
        decodedcol = '{}_decoded'.format(x)
        corr_col = '{}_correction'.format(x)
        spellcol = '{}_bad_spell'.format(x)
        indexcol = '{}_clean'.format(x)
        df[spellcol] = np.logical_and.reduce((~df[sensecol].isin(unfiltered_senses),
                                              df[corr_col].notnull(),
                                              df[corr_col] != df[decodedcol]))

        rename_index = np.where(df[spellcol] == True)
        rename_dict = dict(zip(df.index.get_level_values(indexcol)[rename_index].values,
                               df[corr_col].str.encode('utf8').values[rename_index]))
        if rename_dict:
            df.rename(rename_dict, inplace=True, axis=0)
    df['bad_spell'] = ~np.logical_or(
        np.logical_and(df['cue_bad_spell'], df['response_bad_spell']),
        np.logical_and(~df['cue_bad_spell'], ~df['response_bad_spell'])
    )
    return df


def check_senses_and_pos(df):
    cue_pos = df['cue_pos'].copy()
    cue_sense = df['cue_sense'].copy()
    response_pos = df['response_pos'].copy()
    response_sense = df['response_sense'].copy()
    # part of speech is the same for cue and response if... (NaNs are not equal so don't need to worry about these)
    pos_equal = np.logical_or.reduce((
        np.equal(cue_pos, response_pos),
        np.equal(cue_pos, response_sense),
        np.equal(cue_sense, response_pos),
        np.equal(cue_sense, response_sense)
    ))
    # the remaining records to be checked
    remaining = df[~pos_equal].copy(deep=True)
    # not nans
    cue_pos_not_null = ~remaining['cue_pos'].copy().isnull()
    cue_sense_not_null = ~remaining['cue_sense'].copy().isnull()
    response_pos_not_null = ~remaining['response_pos'].copy().isnull()
    response_sense_not_null = ~remaining['response_sense'].copy().isnull()
    # we need at least one of the cue pos/sense and one of the res pos/sense to not be NaN and also not equal
    pos_diff = np.logical_or.reduce((
        np.logical_and(cue_pos_not_null, cue_sense_not_null),
        np.logical_and(cue_pos_not_null, response_sense_not_null),
        np.logical_and(response_pos_not_null, cue_sense_not_null),
        np.logical_and(response_pos_not_null, response_sense_not_null)
    ))
    # everything else we cannot tell as the pos/senses are NaN for both cue and/or response cols

    pos_diff_indexes = remaining.index[pos_diff].copy()
    pos_equal_indexes = df.index[pos_equal].copy()
    # pos_unknown_indexes = remaining[~pos_diff]
    df['pos_changes'] = np.nan
    df['pos_changes'].loc[pos_diff_indexes] = True
    df['pos_changes'].loc[pos_equal_indexes] = False

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
    # TODO wordnet stuff with pandas? probably means rewriting or wrapping relation_coding to accept dataframes
    # TODO consider the categories found in the Gallant Research:
    #   (visual, tactile, outdoor, bodypart, number, place, violence, person, social, mental, time)
    # TODO what to do about words joined with hyphens?
    # TODO use this throughout instead of splitting multiple times
    # TODO add s2v check for syntagmatic
    # TODO add proverb check for syntagmatic
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
