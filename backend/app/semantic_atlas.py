from backend.lib.semantic_atlas import BrainMap
from backend.lib.measures import unpack_splits
import numpy as np
import pandas as pd
import logging

UNKNOWN_AREA_LABEL = b'U'


# this is quite a large process so its in its own file

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
        unpacked_words = pd.DataFrame(np.unique(np.sort(cue_splits[:, 0])), columns=['word'])
    elif cue_splits is None and response_splits is not None:
        unpacked_words = pd.DataFrame(np.unique(np.sort(response_splits[:, 0])), columns=['word'])
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
                # set to float because they were being interpreted as objects, not sure why
                distances = brain_map.get_distance(
                    np.array([left_coords[row_filter]['coord_x'.format(hemi)].astype(float).values,
                              left_coords[row_filter]['coord_y'.format(hemi)].astype(float).values,
                              left_coords[row_filter]['coord_z'.format(hemi)].astype(float).values]),
                    # format for
                    np.array([df[row_filter]['{}_{}_coord_x'.format(prefix_right, hemi[0])].astype(float).values,
                              df[row_filter]['{}_{}_coord_y'.format(prefix_right, hemi[0])].astype(float).values,
                              df[row_filter]['{}_{}_coord_z'.format(prefix_right, hemi[0])].astype(float).values]))

                # case when there is only 1 results
                if isinstance(distances, float):
                    logging.debug('there is only one distance {} '.format(distances))
                    distances = [distances]
                distances = pd.DataFrame(distances, columns=[distance_label])

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
