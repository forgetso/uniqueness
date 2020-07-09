from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from backend.db.db import get_cue_response_pairs_coded
import pandas as pd
import argparse
import numpy as np
import time
import logging
from backend.settings import LOG_LEVEL

SAMPLE_SIZE = 8
TO_PREDICT = 'user_id'
MIN_RESPONSES = 3 * SAMPLE_SIZE
logging.basicConfig(level=LOG_LEVEL)
USER_COLUMNS = ['nativelanguage', 'gender', 'age', 'city', 'country', 'education']


def main(sample_size=None, source=None):
    try:
        sample_size = int(sample_size)
    except TypeError:
        sample_size = SAMPLE_SIZE
    start = time.perf_counter()
    logging.debug('Getting data')

    min_responses = 50 * sample_size
    data = getData(source=source, min_responses=min_responses)

    batched_data = data
    if sample_size > 1:
        batched_data = batch_data(data, sample_size)
    logging.info('Predicting for {}'.format(TO_PREDICT))
    classes, coefs, columns = logisticRegression(batched_data)
    end = time.perf_counter()
    logging.debug('time taken {}'.format(end - start))
    return classes, coefs, columns


def logisticRegression(data):
    # define feature selection
    fs = SelectKBest(score_func=f_classif, k=40)
    # apply feature selection
    X_selected = fs.fit_transform(data.drop(TO_PREDICT, axis=1), data[TO_PREDICT])

    X_train, X_test, y_train, y_test = train_test_split(X_selected, data[TO_PREDICT],
                                                        test_size=0.30,
                                                        random_state=101)

    X__train_scaled = preprocessing.scale(X_train)
    X_test_scaled = preprocessing.scale(X_test)
    # TODO maybe fill nans with the mean?
    # create an instance and fit the model
    logmodel = LogisticRegression()
    logmodel.fit(X__train_scaled, y_train)
    # predictions
    predictions = logmodel.predict(X_test_scaled)
    report = classification_report(y_test, predictions, output_dict=True)
    print(classification_report(y_test, predictions))
    chance = 1 / len(logmodel.classes_)
    logging.info('Random chance would have been {}'.format(str(chance)))
    logging.info('This is {} times chance'.format(str(float(report['macro avg']['precision']) / chance)))
    return logmodel.classes_, logmodel.coef_, data.columns


def getData(source=None, min_responses=None):
    # TODO make this multiprocess the formatting
    min_responses = min_responses or MIN_RESPONSES

    cur = get_cue_response_pairs_coded(source=source)
    df = pd.DataFrame([x for x in cur], columns=[desc[0] for desc in cur.description])
    df.drop(columns=['rowid'], inplace=True)
    logging.info('there are {} records'.format(len(df)))
    # full wordssoc table
    predict_group = df.groupby(by=TO_PREDICT).agg(['count'])
    logging.debug('min responses set to {}'.format(min_responses))
    min_responses_indices = predict_group['response_clean'].gt(min_responses)
    groups_with_min_responses = predict_group.index[min_responses_indices['count']].values
    if not len(groups_with_min_responses):
        exit('not enough responses')
    df = df[df[TO_PREDICT].isin(groups_with_min_responses)]
    df = df.drop(
        columns=['cue_clean', 'response_clean',
                 'cue_raw', 'response_raw',
                 'cue_correction', 'response_correction',
                 'cue_sense', 'response_sense', 'source'])
    user_columns = USER_COLUMNS
    if TO_PREDICT in USER_COLUMNS:
        user_columns.remove(TO_PREDICT)
    df = df.drop(columns=user_columns)

    # converts True/False to 1/0
    df = df * 1
    # drop any columns with only zeroes
    df = df.loc[:, (df != 0).any(axis=0)]

    logging.debug('replacing nans with strings in str columns')
    df['phonology'] = df['phonology'].replace({np.nan: ''})
    df['syntagmatic'] = df['syntagmatic'].replace({np.nan: ''})
    # print(np.where(np.isnan(df)))
    logging.debug('encoding non-numeric columns')
    label_encoder = preprocessing.LabelEncoder()
    df['phonology'] = label_encoder.fit_transform(df['phonology'])
    df['syntagmatic'] = label_encoder.fit_transform(df['syntagmatic'])

    df[TO_PREDICT] = label_encoder.fit_transform(df[TO_PREDICT])
    df = pd.DataFrame(df).fillna(value=0)
    df = df.astype(float)
    df = df.sort_values(by=TO_PREDICT)
    logging.debug('columns left in dataframe {}'.format(sorted(df.columns)))
    return df


def batch_data(df, size):
    # GROUP BY variable to predict
    # GROUP BY n rows and calculate mean
    logging.info('There are {} samples to batch'.format(len(df)))
    groups = {key: df.loc[value] for key, value in df.groupby(TO_PREDICT).groups.items()}
    batched = pd.DataFrame()
    logging.debug('batching data...')
    for key, group in groups.items():
        logging.debug('batching {}'.format(key))
        group.reset_index(drop=True, inplace=True)
        group = group.groupby(group.index // size).mean()
        logging.info('There are {} samples for key {}'.format(len(group), key))
        # TODO MemoryError: Unable to allocate 90.5 GiB for an array with shape (154023, 78895) and data type int64
        #  check what size of output is coming from this function
        #  it should be smaller than what was passed in
        batched = pd.concat([batched, group])
    batched = batched.loc[:, ~df.columns.duplicated()]
    return batched


def drop_wordnet_features(df):
    df = df.drop(columns=['derivationally_related_forms', 'pertainyms',
                          'instance_hypernyms', 'hyponyms', 'instance_hyponyms',
                          'member_holonyms', 'substance_holonyms', 'part_holonyms',
                          'member_meronyms', 'substance_meronyms', 'part_meronyms',
                          'topic_domains', 'region_domains', 'usage_domains', 'attributes',
                          'entailments', 'causes', 'also_sees', 'verb_groups', 'similar_tos',
                          'hypernyms', 'root_hypernyms'])
    return df


def setup():
    parser = argparse.ArgumentParser(description='Import various types of word data')
    parser.add_argument('--source', default=None, help="code responses from a particular source only")
    parser.add_argument('--samplesize', default=None, help="choose the number of responses to batch as sample")
    args = parser.parse_args()
    main(source=args.source, sample_size=args.samplesize)


if __name__ == "__main__":
    setup()
