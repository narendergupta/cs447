from codify.config.strings import *
from codify.config.settings import *
from collections import defaultdict
from wordvecs import WordVectors
from datamodel import DataModel
from gen_utils import *
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

import csv
import os
import logging
import numpy as np
import pickle


class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        tokens = [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
        tokens = [w for w in tokens if w not in stopwords.words('english')]
        return tokens
#endclass


class Experimenter:
    """Execute and manage experiments"""
    def __init__(self, dm, process_datamodel, serialise):
        self.logger = logging.getLogger(LOGGER)
        self.set_datamodel(dm, process_datamodel, serialise)
        #self.wv = WordVectors()
        return


    def set_datamodel(self, dm, process_datamodel, serialise):
        self.dm = dm
        if (not os.path.isfile(DATAMODEL)) or process_datamodel:
            # self.wv.set_embeddings()
            self.logger.info('processing recipes ...')
            if serialise:
                self.logger.info('serialising processed data model')
                with open(DATAMODEL, 'w') as data_model:
                    pickle.dump(self.dm.data, data_model)
                #endwith
            #endif
        else:
            with open(DATAMODEL, 'r') as data_model:
                self.logger.info('loading serialised data model')
                self.dm.data = pickle.load(data_model)
            #endwith
        return None


    def perform_multiclass_experiment(self, pred_mode=JOINT_EVERYTHING,
            use_exclusion=True,
            need_to_extract_features=False,
            prediction_file='../data/binary_predictions.csv',
            result_file='../data/binary_results.txt',
            english_only=True):
        train_data = self.dm.get_training_data(english_only=english_only)
        test_data = self.dm.get_testing_data(english_only=english_only)
        self.logger.info('Data Size = Training:Testing::%d:%d' % (len(train_data), len(test_data)))
        if need_to_extract_features is True:
            self.logger.info('Extracting Bag of Words features for multiclass classification')
            self.dm.extract_bow_features(analyzer='char', ngram_range=(3,3), max_features=2000)
            self.dm.extract_bow_features(analyzer='word', ngram_range=(1,2), max_features=2000)
        classifiers = self.multiclass_train(train_data)
        predictions = self.multiclass_predict(classifiers, test_data,
                pred_mode=pred_mode, top_k=5, use_exclusion=use_exclusion)
        self.save_predictions(predictions, output_file=prediction_file)
        self.evaluate(test_data, predictions, output_file=result_file)
        return


    def perform_binary_experiment(self, pred_mode=JOINT_EVERYTHING,
            use_exclusion=True,
            need_to_extract_features=False,
            prediction_file='../data/multiclass_predictions.csv',
            result_file='../data/multiclass_results.txt',
            english_only=True):
        train_data = self.dm.get_training_data(english_only=english_only)
        test_data = self.dm.get_testing_data(english_only=english_only)
        self.logger.info('Data Size = Training:Testing::%d:%d' % (len(train_data), len(test_data)))
        if need_to_extract_features is True:
            self.logger.info('Extracting Bag of Words features for binary classification')
            self.dm.extract_bow_features(analyzer='char', ngram_range=(3,3), max_features=2000)
            self.dm.extract_bow_features(analyzer='word', ngram_range=(1,2), max_features=2000)
        classifiers = self.binary_train(train_data)
        predictions = self.binary_predict(classifiers, test_data,
                pred_mode=JOINT_EVERYTHING, top_k=5, use_exclusion=use_exclusion)
        self.save_predictions(predictions, output_file=prediction_file)
        self.evaluate(test_data, predictions, output_file=result_file)
        return


    def perform_hierarchical_multiclass_experiment(self, use_exclusion=True,
            need_to_extract_features=False,
            prediction_file='../data/multiclass_predictions.csv',
            result_file='../data/multiclass_results.txt',
            english_only=True):
        train_data = self.dm.get_training_data(english_only=english_only)
        test_data = self.dm.get_testing_data(english_only=english_only)
        self.logger.info('Data Size = Training:Testing::%d:%d' % (len(train_data), len(test_data)))
        if need_to_extract_features is True:
            self.logger.info('Extracting Bag of Words features for hierarchical multiclass classification')
            #self.dm.extract_bow_features(analyzer='char', ngram_range=(3,3), max_features=2000)
            self.dm.extract_bow_features(analyzer='word', ngram_range=(1,2), max_features=2000)
        (channel_classifiers, func_classifiers) = self.hierarchical_multiclass_train(train_data)
        predictions = self.hierarchical_multiclass_predict(
                channel_classifiers, func_classifiers, test_data, use_exclusion=use_exclusion)
        self.save_predictions(predictions, output_file=prediction_file)
        self.evaluate(test_data, predictions, output_file=result_file)
        return


    def multiclass_train(self, train_data):
        self.logger.info('Training using multiclass classifiers')
        classifiers = defaultdict(lambda : defaultdict(None))
        label_types = [TRIGGER_CHANNEL, TRIGGER_FUNC, ACTION_CHANNEL, ACTION_FUNC]
        for label_type in label_types:
            classifiers[label_type] = \
                    self.__multiclass_get_classifier(train_data, label_type)
        #end for
        return classifiers


    def multiclass_predict(self, classifiers, test_data, \
            pred_mode=JOINT_EVERYTHING, top_k=5, use_exclusion=False):
        predictions = [defaultdict(str) for i in range(len(test_data))]
        self.logger.info('Predicting using multiclass classifiers')
        test_X = []
        all_pred_probas = {}
        for recipe in test_data:
            test_X.append(recipe.feats)
        for label_type in classifiers:
            test_Y_proba = classifiers[label_type][0].predict_proba(test_X)
            all_pred_probas[label_type] = test_Y_proba
        if pred_mode == INDEPENDENT:
            top_k = 1
        for i in range(len(test_data)):
            max_pred_proba = 0.0
            top_k_labels = {}
            all_labels = {}
            for label_type in classifiers.keys():
                top_k_labels[label_type] = self.__get_top_k_labels(\
                        all_pred_probas[label_type][i].tolist(), \
                        classifiers[label_type][1].classes_,
                        k=top_k)
                all_labels[label_type] = self.__get_top_k_labels(\
                        all_pred_probas[label_type][i].tolist(), \
                        classifiers[label_type][1].classes_,
                        k=len(all_pred_probas[label_type][i].tolist()))
            #endfor
            if pred_mode == JOINT_EVERYTHING:
                predictions[i] = self.__predict_joint_everything(\
                        all_labels, top_k_labels, use_exclusion=use_exclusion)
            elif pred_mode == JOINT_CHANNEL_FUNCS:
                predictions[i] = self.__predict_joint_channel_funcs(\
                        all_labels, top_k_labels, use_exclusion=use_exclusion)
            elif pred_mode == INDEPENDENT:
                predictions[i] = self.__predict_independent(\
                        all_labels, top_k_labels, use_exclusion=use_exclusion)
        #endfor i
        return predictions


    def binary_train(self, train_data):
        self.logger.info('Training using binary classifiers')
        classifiers = defaultdict(lambda : defaultdict(None))
        label_types = [TRIGGER_CHANNEL, TRIGGER_FUNC, ACTION_CHANNEL, ACTION_FUNC]
        for label_type in label_types:
            classifiers[label_type] = \
                    self.__binary_get_classifiers(train_data, label_type)
        #end for
        return classifiers


    def binary_predict(self, classifiers, test_data, \
            pred_mode=JOINT_EVERYTHING, top_k=5, use_exclusion=False):
        predictions = [defaultdict(str) for i in range(len(test_data))]
        self.logger.info('Predicting using binary classifiers')
        test_X = []
        for recipe in test_data:
            test_X.append(recipe.feats)
        all_pred_probas = {}
        label_type_tokens = {}
        label_types = [TRIGGER_CHANNEL, TRIGGER_FUNC, ACTION_CHANNEL, ACTION_FUNC]
        for label_type in label_types:
            all_pred_probas[label_type] = []
            label_type_tokens[label_type] = list(classifiers[label_type].keys())
            label_type_preds = np.zeros(\
                    shape=(len(test_data),len(label_type_tokens[label_type])))
            i = 0
            for token in label_type_tokens[label_type]:
                clf = classifiers[label_type][token]
                test_Y_proba = clf.predict_proba(test_X)
                test_pos_proba = test_Y_proba[:,1]
                label_type_preds[:,i] = test_pos_proba
                i += 1
            all_pred_probas[label_type] = label_type_preds
        if pred_mode == INDEPENDENT:
            top_k = 1
        for i in range(len(test_data)):
            max_pred_proba = 0.0
            top_k_labels = {}
            all_labels = {}
            for label_type in label_types:
                top_k_labels[label_type] = self.__get_top_k_labels(\
                        all_pred_probas[label_type][i].tolist(), \
                        label_type_tokens[label_type],
                        k=top_k)
                all_labels[label_type] = self.__get_top_k_labels(\
                        all_pred_probas[label_type][i].tolist(), \
                        label_type_tokens[label_type],
                        k=len(all_pred_probas[label_type][i].tolist()))
            #endfor
            if pred_mode == JOINT_EVERYTHING:
                predictions[i] = self.__predict_joint_everything(\
                        all_labels, top_k_labels, use_exclusion=use_exclusion)
            elif pred_mode == JOINT_CHANNEL_FUNCS:
                predictions[i] = self.__predict_joint_channel_funcs(\
                        all_labels, top_k_labels, use_exclusion=use_exclusion)
            elif pred_mode == INDEPENDENT:
                predictions[i] = self.__predict_independent(\
                        all_labels, top_k_labels, use_exclusion=use_exclusion)
        #endfor i
        return predictions


    def hierarchical_multiclass_train(self, train_data):
        self.logger.info('Training using hierarchical multiclass classifiers')
        channel_label_types = {TRIGGER:TRIGGER_CHANNEL, ACTION:ACTION_CHANNEL}
        func_label_types = {TRIGGER:TRIGGER_FUNC, ACTION:ACTION_FUNC}
        channel_classifiers = defaultdict(None)
        func_classifiers = defaultdict(lambda : defaultdict(None))
        # Get Channel Classifiers
        for label_type in channel_label_types:
            channel_classifiers[label_type] = \
                    self.__multiclass_get_classifier(
                            train_data,
                            channel_label_types[label_type])
        # Get Func  Classifiers
        channel_wise_train_data = {TRIGGER:defaultdict(list), ACTION:defaultdict(list)}
        for recipe in train_data:
            channel_wise_train_data[TRIGGER][recipe.trigger_channel].append(recipe)
            channel_wise_train_data[ACTION][recipe.action_channel].append(recipe)
        for func_type in func_label_types:
            for channel in channel_wise_train_data[func_type]:
                funcs = [recipe[func_label_types[func_type]] for recipe in channel_wise_train_data[func_type][channel]]
                funcs = unique(funcs)
                if len(funcs) <= 1:
                    continue
                func_classifiers[func_type][channel] = \
                        self.__multiclass_get_classifier(
                                channel_wise_train_data[func_type][channel],
                                func_label_types[func_type])
            #endfor channel
        #endfor func_type
        return (channel_classifiers, func_classifiers)


    def hierarchical_multiclass_predict(self, channel_classifiers, func_classifiers, 
            test_data, top_k=5, use_exclusion=False):
        predictions = [defaultdict(str) for i in range(len(test_data))]
        self.logger.info('Predicting using hierarchical multiclass classifiers')
        test_X = []
        for recipe in test_data:
            test_X.append(recipe.feats)
        channel_pred_probas = {}
        func_pred_probas = defaultdict(dict)
        channel_label_types = {TRIGGER:TRIGGER_CHANNEL, ACTION:ACTION_CHANNEL}
        func_label_types = {TRIGGER:TRIGGER_FUNC, ACTION:ACTION_FUNC}
        for channel_type in channel_classifiers:
            (clf, le) = channel_classifiers[channel_type]
            test_Y_proba = clf.predict_proba(test_X)
            channel_pred_probas[channel_type] = test_Y_proba
        for func_type in func_classifiers:
            clf_map = func_classifiers[func_type]
            for channel in clf_map:
                (clf, le) = clf_map[channel]
                func_pred_probas[func_type][channel] = clf.predict_proba(test_X)
            #endfor channel
        for i in range(len(test_data)):
            for infer_type in channel_label_types:
                max_pred_proba = 0.0
                top_k_channels = self.__get_top_k_labels(
                        channel_pred_probas[infer_type][i].tolist(),
                        channel_classifiers[infer_type][1].classes_,
                        k=top_k)
                all_channels = self.__get_top_k_labels(
                        channel_pred_probas[infer_type][i].tolist(),
                        channel_classifiers[infer_type][1].classes_,
                        k=len(channel_classifiers[infer_type][1].classes_))
                for channel in top_k_channels:
                    if channel not in func_pred_probas[infer_type]:
                        continue
                    top_k_funcs = self.__get_top_k_labels(
                            func_pred_probas[infer_type][channel][i].tolist(),
                            func_classifiers[infer_type][channel][1].classes_,
                            k=top_k)
                    all_funcs = self.__get_top_k_labels(
                            func_pred_probas[infer_type][channel][i].tolist(),
                            func_classifiers[infer_type][channel][1].classes_,
                            k=len(func_classifiers[infer_type][channel][1].classes_))
                    for func in top_k_funcs:
                        pred_proba = top_k_channels[channel]
                        if use_exclusion is True:
                            pred_proba *= self.__get_exclusion_proba(all_channels, channel)
                            pred_proba *= self.__get_exclusion_proba(all_funcs, func)
                        pred_proba *= top_k_funcs[func]
                        if pred_proba > max_pred_proba:
                            max_pred_proba = pred_proba
                            channel_val = channel_label_types[infer_type]
                            func_val = func_label_types[infer_type]
                            predictions[i][channel_val] = channel
                            predictions[i][func_val] = func
                        #endif
                    #endfor func
                #endfor channel
            #end infer_type
        #endfor i
        return predictions


    def __binary_get_classifiers(self, recipes, recipe_label_type):
        classifiers = defaultdict(None)
        unique_labels = []
        for recipe in recipes:
            unique_labels.append(recipe[recipe_label_type])
        unique_labels = unique(unique_labels)
        for label in unique_labels:
            X, Y = [], []
            for recipe in recipes:
                X.append(recipe.feats)
                Y.append(1 if recipe[recipe_label_type]==label else 0)
            clf = linear_model.LogisticRegression(class_weight='balanced')
            clf.fit(X, Y)
            classifiers[label] = clf
        return classifiers


    def __multiclass_get_classifier(self, recipes, recipe_label_type):
        X = []
        Y = []
        for recipe in recipes:
            X.append(recipe.feats)
            Y.append(recipe[recipe_label_type])
        le = LabelEncoder()
        Y = le.fit_transform(Y)
        clf = linear_model.LogisticRegression(\
                multi_class='multinomial', \
                class_weight='balanced', \
                solver='lbfgs')
        clf.fit(X, Y)
        return (clf, le)


    def __get_top_k_labels(self, pred_probas, labels, k):
        max_k_inds = index_max_k(pred_probas,k)
        max_labels = {}
        for i in max_k_inds:
            max_labels[labels[i]] = pred_probas[i]
        return max_labels


    def __get_exclusion_proba(self, all_labels, inclusive_label, label_type=None):
        proba = 1.0
        if label_type is not None:
            for label in all_labels[label_type]:
                if label == inclusive_label:
                    continue
                proba *= (1 - all_labels[label_type][label])
        else:
            for label in all_labels:
                if label == inclusive_label:
                    continue
                proba *= (1 - all_labels[label])
        return proba


    def __predict_joint_everything(self, all_labels, top_k_labels, use_exclusion=False):
        prediction = defaultdict(str)
        channel_func_priors = self.dm.get_channel_func_priors()
        trigger_action_priors = self.dm.get_trigger_action_priors()
        max_pred_proba = 0.0
        for t_channel in top_k_labels[TRIGGER_CHANNEL].keys():
            for t_func in top_k_labels[TRIGGER_FUNC].keys():
                for a_channel in top_k_labels[ACTION_CHANNEL].keys():
                    for a_func in top_k_labels[ACTION_FUNC].keys():
                        pred_proba = 1.0
                        pred_proba *= top_k_labels[TRIGGER_CHANNEL][t_channel]
                        pred_proba *= top_k_labels[TRIGGER_FUNC][t_func]
                        pred_proba *= top_k_labels[ACTION_CHANNEL][a_channel]
                        pred_proba *= top_k_labels[ACTION_FUNC][a_func]
                        pred_proba *= channel_func_priors[t_channel][t_func]
                        pred_proba *= channel_func_priors[a_channel][a_func]
                        pred_proba *= trigger_action_priors[t_channel][a_channel]
                        if use_exclusion is True:
                            pred_proba *= self.__get_exclusion_proba(all_labels, t_channel, TRIGGER_CHANNEL)
                            pred_proba *= self.__get_exclusion_proba(all_labels, t_func, TRIGGER_FUNC)
                            pred_proba *= self.__get_exclusion_proba(all_labels, a_channel, ACTION_CHANNEL)
                            pred_proba *= self.__get_exclusion_proba(all_labels, a_func, ACTION_FUNC)
                        if pred_proba > max_pred_proba:
                            prediction[TRIGGER_CHANNEL] = t_channel
                            prediction[TRIGGER_FUNC] = t_func
                            prediction[ACTION_CHANNEL] = a_channel
                            prediction[ACTION_FUNC] = a_func
                        #endif
                    #endfor a_func
                #endfor a_channel
            #endfor t_func
        #endfor t_channel
        return prediction


    def __predict_joint_channel_funcs(self, all_labels, top_k_labels, use_exclusion=False):
        prediction = defaultdict(str)
        channel_func_priors = self.dm.get_channel_func_priors()
        max_pred_proba = 0.0
        for t_channel in top_k_labels[TRIGGER_CHANNEL].keys():
            for t_func in top_k_labels[TRIGGER_FUNC].keys():
                pred_proba = 1.0
                pred_proba *= top_k_labels[TRIGGER_CHANNEL][t_channel]
                pred_proba *= top_k_labels[TRIGGER_FUNC][t_func]
                pred_proba *= channel_func_priors[t_channel][t_func]
                if use_exclusion is True:
                    pred_proba *= self.__get_exclusion_proba(all_labels, t_channel, TRIGGER_CHANNEL)
                    pred_proba *= self.__get_exclusion_proba(all_labels, t_func, TRIGGER_FUNC)
                if pred_proba > max_pred_proba:
                    prediction[TRIGGER_CHANNEL] = t_channel
                    prediction[TRIGGER_FUNC] = t_func
                #endif
            #endfor t_func
        #endfor t_channel
        for a_channel in top_k_labels[ACTION_CHANNEL].keys():
            for a_func in top_k_labels[ACTION_FUNC].keys():
                pred_proba = 1.0
                pred_proba *= top_k_labels[ACTION_CHANNEL][a_channel]
                pred_proba *= top_k_labels[ACTION_FUNC][a_func]
                pred_proba *= channel_func_priors[a_channel][a_func]
                if use_exclusion is True:
                    pred_proba *= self.__get_exclusion_proba(all_labels, a_channel, ACTION_CHANNEL)
                    pred_proba *= self.__get_exclusion_proba(all_labels, a_func, ACTION_FUNC)
                if pred_proba > max_pred_proba:
                    prediction[ACTION_CHANNEL] = a_channel
                    prediction[ACTION_FUNC] = a_func
                #endif
            #endfor a_func
        #endfor a_channel
        return prediction


    def __predict_independent(self, all_labels, top_k_labels, use_exclusion=False):
        prediction = defaultdict(str)
        max_pred_proba = 0.0
        for t_channel in top_k_labels[TRIGGER_CHANNEL].keys():
            pred_proba = 1.0
            pred_proba *= top_k_labels[TRIGGER_CHANNEL][t_channel]
            if use_exclusion is True:
                pred_proba *= self.__get_exclusion_proba(all_labels, t_channel, TRIGGER_CHANNEL)
            if pred_proba > max_pred_proba:
                prediction[TRIGGER_CHANNEL] = t_channel
            #endif
        #endfor t_channel
        for t_func in top_k_labels[TRIGGER_FUNC].keys():
            pred_proba = 1.0
            pred_proba *= top_k_labels[TRIGGER_FUNC][t_func]
            if use_exclusion is True:
                pred_proba *= self.__get_exclusion_proba(all_labels, t_func, TRIGGER_FUNC)
            if pred_proba > max_pred_proba:
                prediction[TRIGGER_FUNC] = t_func
            #endif
        #endfor t_func
        for a_channel in top_k_labels[ACTION_CHANNEL].keys():
            pred_proba = 1.0
            pred_proba *= top_k_labels[ACTION_CHANNEL][a_channel]
            if use_exclusion is True:
                pred_proba *= self.__get_exclusion_proba(all_labels, a_channel, ACTION_CHANNEL)
            if pred_proba > max_pred_proba:
                prediction[ACTION_CHANNEL] = a_channel
            #endif
        #endfor a_channel
        for a_func in top_k_labels[ACTION_FUNC].keys():
            pred_proba = 1.0
            pred_proba *= top_k_labels[ACTION_FUNC][a_func]
            if use_exclusion is True:
                pred_proba *= self.__get_exclusion_proba(all_labels, a_func, ACTION_FUNC)
            if pred_proba > max_pred_proba:
                prediction[ACTION_FUNC] = a_func
            #endif
        #endfor a_func
        return prediction
    

    def save_predictions(self, predictions, output_file):
        fieldnames = [URL]
        label_types = [TRIGGER_CHANNEL, TRIGGER_FUNC, ACTION_CHANNEL, ACTION_FUNC]
        test_data = self.dm.get_testing_data()
        for label_type in label_types:
            fieldnames.append(PRED_ + label_type)
            fieldnames.append(GOLD_ + label_type)
        fieldnames = sorted(fieldnames)
        with open(output_file,'w') as output_f:
            self.logger.info('Writing prediction and gold labels to: %s ' % output_file)
            output_writer = csv.DictWriter(output_f, fieldnames=fieldnames)
            output_writer.writeheader()
            for i in range(len(predictions)):
                out_dict = {URL:test_data[i].url, \
                        PRED_TRIGGER_CHANNEL:predictions[i][TRIGGER_CHANNEL], \
                        PRED_TRIGGER_FUNC:predictions[i][TRIGGER_FUNC], \
                        PRED_ACTION_CHANNEL:predictions[i][ACTION_CHANNEL], \
                        PRED_ACTION_FUNC:predictions[i][ACTION_FUNC], \
                        GOLD_TRIGGER_CHANNEL:test_data[i][TRIGGER_CHANNEL], \
                        GOLD_TRIGGER_FUNC:test_data[i][TRIGGER_FUNC], \
                        GOLD_ACTION_CHANNEL:test_data[i][ACTION_CHANNEL], \
                        GOLD_ACTION_FUNC:test_data[i][ACTION_FUNC]
                        }
                output_writer.writerow(out_dict)
            #endfor
        #endwith
        return


    def evaluate(self, test_data, predictions, output_file):
        self.logger.info('Evaluating Prediction Scores')
        channel_labels = []
        channel_preds = []
        func_labels = []
        func_preds = []
        for i in range(len(test_data)):
            channel_labels.append(test_data[i].trigger_channel)
            channel_labels.append(test_data[i].action_channel)
            func_labels.append(test_data[i].trigger_func)
            func_labels.append(test_data[i].action_func)
            channel_preds.append(predictions[i][TRIGGER_CHANNEL])
            channel_preds.append(predictions[i][ACTION_CHANNEL])
            func_preds.append(predictions[i][TRIGGER_FUNC])
            func_preds.append(predictions[i][ACTION_FUNC])
        channel_accuracy = metrics.accuracy_score(channel_labels, channel_preds)
        func_accuracy = metrics.accuracy_score(func_labels, func_preds)
        channel_f1 = metrics.f1_score(channel_labels, channel_preds, average='weighted')
        func_f1 = metrics.f1_score(func_labels, func_preds, average='weighted')
        self.logger.info('Accuracy Scores (Channel, Func) : (%f, %f)' % \
                (channel_accuracy, func_accuracy))
        self.logger.info('F1 Scores (Channel, Func) : (%f, %f)' % \
                (channel_f1, func_f1))
        with open(output_file,'w') as output_f:
            self.logger.info('Writing scores to: %s ' % output_file)
            output_f.write('Accuracy Scores (Channel, Func) : (%f, %f)' % \
                    (channel_accuracy, func_accuracy))
            output_f.write('F1 Scores (Channel, Func) : (%f, %f)' % \
                    (channel_f1, func_f1))
        return


