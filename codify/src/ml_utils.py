from codify.config.strings import *
from copy import deepcopy
from gen_utils import *
from sklearn import base, cross_validation, metrics

import os


def kfold_cross_val(classifiers, x, y, train_test_indices_pair=None, \
        cv=5, do_shuffle=True):
    if type(classifiers) is not list:
        classifiers = [classifiers]
    scores = kfold_cross_val_multi_clf(classifiers, x, y, \
            train_test_indices_pair, cv=cv, do_shuffle=do_shuffle)
    if len(classifiers) == 1:
        return scores[classifiers[0]]
    return scores


def kfold_cross_val_multi_clf(classifiers, x, y, train_test_indices_pair=None,\
        cv=5, do_shuffle=True):
    default_score = {ACCURACY:[], F1_SCORE:[], PRECISION:[], RECALL:[]}
    result = dict((clf, deepcopy(default_score)) for clf in classifiers)
    if len(x) < cv:
        result = dict((k,[0 for i in range(cv)]) for k in result.keys())
        return result
    if train_test_indices_pair is None:
        kf = cross_validation.KFold(len(x), n_folds=cv, shuffle=do_shuffle)
    else:
        kf = [train_test_indices_pair]
    for train_i, test_i in kf:
        train_x, train_y, test_x, test_y = [], [], [], []
        for index in train_i:
            train_x.append(x[index])
            train_y.append(y[index])
        for index in test_i:
            test_x.append(x[index])
            test_y.append(y[index])
        for clf in classifiers:
            # Train classifier for actual scores
            temp_classifier = base.clone(clf)
            #If there is only one class in training, it cannot learn anything
            if len(unique(train_y)) == 1:
                label = train_y[0]
                test_y_pred = [label for item in test_y]
            else:
                temp_classifier.fit(train_x, train_y)
                test_y_pred = temp_classifier.predict(test_x)
            result[clf][ACCURACY].append(float(\
                    metrics.accuracy_score(test_y, test_y_pred)))
            result[clf][PRECISION].append(float(\
                    metrics.precision_score(test_y, test_y_pred, average='binary')))
            result[clf][RECALL].append(float(\
                    metrics.recall_score(test_y, test_y_pred, average='binary')))
            result[clf][F1_SCORE].append(float(\
                    metrics.f1_score(test_y, test_y_pred, average='binary')))
    return result


