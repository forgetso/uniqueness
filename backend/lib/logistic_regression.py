from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import preprocessing
from backend.lib.db import get_coded_pairs, get_cue_response_pairs
from backend.app.coding import format_cue_response_pairs
import pandas as pd
import argparse


def main(responses=None):
    data = getData(responses=responses)
    logisticRegression(data)


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
    print(logmodel.coef_)
    print(data.columns)

def getData(responses=None):
    dataset = []
    coded_pairs = get_coded_pairs()
    all_pairs = [x for x in get_cue_response_pairs(responses=responses)]

    cue_response_pairs, distinct_cue_response_pairs = format_cue_response_pairs(all_pairs)
    print(cue_response_pairs[0])
    for pair in cue_response_pairs:
        data_dict = dict(coded_pairs[pair['key']])
        data_dict.update({'participantid': pair['participantid']})
        dataset.append(data_dict)
    df = pd.DataFrame(dataset)
    # converts True/False to 1/0
    df = df.drop(columns=['cue', 'response', 'id'])
    df = df * 1
    label_encoder = preprocessing.LabelEncoder()
    print(df['semantic_area'])
    df['semantic_area'] = label_encoder.fit_transform(df['semantic_area'])
    print(df)
    return df


def setup():
    parser = argparse.ArgumentParser(description='Import various types of word data')
    parser.add_argument('--responses', action='store_true', help='use responses table for coding')
    args = parser.parse_args()
    main(args.responses)


if __name__ == "__main__":
    setup()
