# random forest
from sklearn import metrics
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

columns = list(range(0, 127))
columns.extend(['mero'])
train = pd.read_csv('/home/chris/dev/uniqueness/data/LREC 2020/Experiment3/SWOW_DT_mero_experiment.csv', names=columns)


def rfcc(df, ycol):
    X = df[list(range(0, 127))]
    y = df[ycol] * 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # no scaling required
    classifier = RandomForestClassifier(n_estimators=20, random_state=0)
    classifier.fit(X_train, y_train)
    # y_pred = classifier.predict(X_test)
    # print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    # print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    # print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    # predictions = classifier.predict(X_test)
    # print(classification_report(y_test, predictions))
    return classifier
