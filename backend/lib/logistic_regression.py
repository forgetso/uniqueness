from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import preprocessing
from backend.db.db import get_cue_response_pairs_coded
import pandas as pd
import argparse
import numpy as np
import time
import logging

SAMPLE_SIZE = 3
TO_PREDICT = 'user_id'

logging.basicConfig(level=logging.DEBUG)


def main(sample_size=None):
    start = time.perf_counter()
    logging.debug('Getting data')
    data = getData()
    sample_size = sample_size or SAMPLE_SIZE
    batched_data = batch_data(data, sample_size)
    classes, coefs, columns = logisticRegression(batched_data)
    end = time.perf_counter()
    logging.debug('time taken {}'.format(end - start))
    return classes, coefs, columns


def logisticRegression(data):
    X_train, X_test, y_train, y_test = train_test_split(data.drop(TO_PREDICT, axis=1),
                                                        data[TO_PREDICT], test_size=0.30,
                                                        random_state=101)

    X__train_scaled = preprocessing.scale(X_train)
    X_test_scaled = preprocessing.scale(X_test)
    # create an instance and fit the model
    logmodel = LogisticRegression()
    logmodel.fit(X__train_scaled, y_train)
    # predictions
    predictions = logmodel.predict(X_test_scaled)
    print(classification_report(y_test, predictions))
    return logmodel.classes_, logmodel.coef_, data.columns


def getData():
    cur = get_cue_response_pairs_coded()
    df = pd.DataFrame([x for x in cur], columns=[desc[0] for desc in cur.description])
    # TODO put the incorrect spelling check in here
    df = df.drop(columns=['cue', 'response', 'cue_clean', 'response_clean', 'response_raw'])

    # df = df.drop(columns=['derivationally_related_forms', 'pertainyms',
    #                  'instance_hypernyms', 'hyponyms', 'instance_hyponyms',
    #                  'member_holonyms', 'substance_holonyms', 'part_holonyms',
    #                  'member_meronyms', 'substance_meronyms', 'part_meronyms',
    #                  'topic_domains', 'region_domains', 'usage_domains', 'attributes',
    #                  'entailments', 'causes', 'also_sees', 'verb_groups', 'similar_tos',
    #                  'hypernyms', 'root_hypernyms', 'synonym', 'antonyms', 'phonology',
    #                  'exact', 'syntagmatic'])
    # converts True/False to 1/0
    df = df * 1
    # drop any columns with only zeroes
    df = df.loc[:, (df != 0).any(axis=0)]

    logging.debug('replacing nans with strings in str columns')
    label_encoder = preprocessing.LabelEncoder()
    df['phonology'] = df['phonology'].replace({np.nan: ''})
    df['syntagmatic'] = df['syntagmatic'].replace({np.nan: ''})

    logging.debug('encoding non-numeric columns')
    df['phonology'] = label_encoder.fit_transform(df['phonology'])
    df['syntagmatic'] = label_encoder.fit_transform(df['syntagmatic'])

    df = pd.DataFrame(df).fillna(value=0)
    df = df.sort_values(by='source')
    return df


def batch_data(df, size):
    # GROUP BY variable to predict
    # GROUP BY n rows and calculate mean
    groups = {key: df.loc[value] for key, value in df.groupby(TO_PREDICT).groups.items()}
    batched = pd.DataFrame()
    logging.debug('batching data...')
    for key, group in groups.items():
        logging.debug('batching {}'.format(key))
        group = group.reset_index(drop=True)
        group = group.groupby(group.index // size).mean()
        group[TO_PREDICT] = key
        column_order = [TO_PREDICT] + group.columns.tolist()[:-1]
        group = group[column_order]
        #TODO MemoryError: Unable to allocate 90.5 GiB for an array with shape (154023, 78895) and data type int64
        #  check what size of output is coming from this function
        #  it should be smaller than what was passed in
        batched = pd.concat([batched, group])
    return batched


def setup():
    parser = argparse.ArgumentParser(description='Import various types of word data')
    args = parser.parse_args()
    main()


if __name__ == "__main__":
    setup()
