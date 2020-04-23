from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import preprocessing
from backend.lib.db import get_coded_pairs, get_cue_response_pairs
import pandas as pd
import argparse
from backend.helpers.pairs import format_cue_response_pairs

SAMPLE_SIZE = 10


def main(sample_size=None):
    data = getData()
    sample_size = sample_size or SAMPLE_SIZE
    batched_data = batch_data(data, sample_size)
    coefs, columns = logisticRegression(batched_data)
    return coefs, columns


def logisticRegression(data):
    X_train, X_test, y_train, y_test = train_test_split(data.drop('source', axis=1),
                                                        data['source'], test_size=0.30,
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


def getData():
    dataset = []
    coded_pairs = get_coded_pairs()
    all_pairs = [x for x in get_cue_response_pairs(coded=True)]
    # TODO put the incorrect spelling check in here
    cue_response_pairs, distinct_cue_response_pairs = format_cue_response_pairs(all_pairs)
    missing_pair = 0
    for pair in cue_response_pairs:
        coded_pair = coded_pairs.get(pair['key'])
        if coded_pair:
            data_dict = dict(coded_pair)
            data_dict.update({'user_id': pair['user_id']})
            data_dict.update({'source': pair['source']})
            dataset.append(data_dict)
        else:
            print('Key pair {} not present in coding table. Source {}'.format(pair['key'], pair['source']))
            missing_pair += 1
    print('{} missing pairs'.format(missing_pair))
    print('{} coded pairs'.format(len(dataset)))
    df = pd.DataFrame(dataset)
    df = df.drop(columns=['cue', 'response', 'id', 'semantic_area', 'user_id'])
    df = df.drop(columns=['length', 'syllables', 'number_of_words', 'nphon'])

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
    label_encoder = preprocessing.LabelEncoder()
    # print(df['semantic_area'])
    # df['semantic_area'] = label_encoder.fit_transform(df['semantic_area'])
    df = pd.DataFrame(df).fillna(value=0)
    df = df.sort_values(by='source')
    return df


def batch_data(df, size):
    # GROUP BY participantid
    # GROUP BY n rows and calculate mean
    groups = {key: df.loc[value] for key, value in df.groupby("source").groups.items()}
    batched = pd.DataFrame()
    for key, group in groups.items():
        group = group.reset_index(drop=True)
        group = group.groupby(group.index // size).mean()
        group['source'] = key
        column_order = ['source'] + group.columns.tolist()[:-1]
        group = group[column_order]
        batched = pd.concat([batched, group])
    return batched


def setup():
    parser = argparse.ArgumentParser(description='Import various types of word data')
    args = parser.parse_args()
    main()


if __name__ == "__main__":
    setup()
