from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import preprocessing
from backend.lib.db import get_coded_pairs, get_cue_response_pairs
from backend.app.coding import format_cue_response_pairs
import pandas as pd
import argparse
import numpy as np

SAMPLE_SIZE = 20


def main(responses=None, sample_size=None):
    data = getData(responses=responses)
    sample_size = sample_size or SAMPLE_SIZE
    batched_data = batch_data(data, sample_size)
    coefs, columns = logisticRegression(batched_data)
    return coefs, columns


def logisticRegression(data):
    X_train, X_test, y_train, y_test = train_test_split(data.drop('participantid', axis=1),
                                                        data['participantid'], test_size=0.30,
                                                        random_state=101)

    X__train_scaled = preprocessing.scale(X_train)
    X_test_scaled = preprocessing.scale(X_test)
    # create an instance and fit the model
    logmodel = LogisticRegression()
    logmodel.fit(X__train_scaled, y_train)
    # create an instance and fit the model
    logmodel = LogisticRegression()
    logmodel.fit(X__train_scaled, y_train)
    # predictions
    predictions = logmodel.predict(X_test_scaled)
    print(classification_report(y_test, predictions))
    return logmodel.coef_, data.columns


def getData(responses=None):
    dataset = []
    coded_pairs = get_coded_pairs()
    all_pairs = [x for x in get_cue_response_pairs(responses=responses)]
    # TODO put the incorrect spelling check in here
    cue_response_pairs, distinct_cue_response_pairs = format_cue_response_pairs(all_pairs)
    missing_pair = 0
    for pair in cue_response_pairs:
        coded_pair = coded_pairs.get(pair['key'])
        if coded_pair:
            data_dict = dict(coded_pair)
            data_dict.update({'participantid': pair['participantid']})
            dataset.append(data_dict)
        else:
            print('Key pair {} not present in coding table'.format(pair['key']))
            missing_pair += 1
    if missing_pair > 100:
        print('Rerun the coding algorithm')
        exit()
    df = pd.DataFrame(dataset)

    df = df.drop(columns=['cue', 'response', 'id', 'semantic_area'])
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
    label_encoder = preprocessing.LabelEncoder()
    # print(df['semantic_area'])
    # df['semantic_area'] = label_encoder.fit_transform(df['semantic_area'])
    df = pd.DataFrame(df).fillna(value=0)
    df = df.sort_values(by='participantid')
    return df


def batch_data(df, size):
    # GROUP BY participantid
    # GROUP BY n rows and calculate mean
    groups = {key: df.loc[value] for key, value in df.groupby("participantid").groups.items()}
    batched = pd.DataFrame()
    for key, group in groups.items():
        group = group.reset_index(drop=True)
        group = group.groupby(group.index // size).mean()
        group['participantid'] = key
        column_order = ['participantid'] + group.columns.tolist()[:-1]
        group = group[column_order]
        batched = pd.concat([batched, group])
    return batched


def setup():
    parser = argparse.ArgumentParser(description='Import various types of word data')
    parser.add_argument('--responses', action='store_true', help='use responses table for coding')
    args = parser.parse_args()
    main(responses=args.responses)


if __name__ == "__main__":
    setup()
