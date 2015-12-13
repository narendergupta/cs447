import csv
import os
import logging
import numpy as np
import pickle
from collections import defaultdict
from codify.config.strings import *
from codify.config.settings import *
from wordvecs import WordVectors
from datamodel import DataModel
from gen_utils import *
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn import linear_model, svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics


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


    def perform_multiclass_experiment(self):
        self.multiclass = True
        test_data = self.dm.get_testing_data()
        self.dm.extract_bow_features(analyzer='char', ngram_range=(3,3), max_features=2000)
        self.dm.extract_bow_features(analyzer='word', ngram_range=(1,2), max_features=2000)
        train_data = self.dm.get_training_data()
        test_data = self.dm.get_testing_data()
        classifiers = self.train(train_data)
        predictions = self.predict_joint_everything(classifiers, test_data)
        self.save_predictions(predictions, output_file='../data/predictions_multiclass.csv')
        self.evaluate(test_data, predictions, output_file='../data/results_multiclass.txt')
        return


    def __get_multiclass_classifier(self, recipes, recipe_label_type):
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


    def train(self, train_data):
        self.logger.info('Training')
        classifiers = defaultdict(lambda : defaultdict(None))
        label_types = [TRIGGER_CHANNEL, TRIGGER_FUNC, ACTION_CHANNEL, ACTION_FUNC]
        if self.multiclass is True:
            for label_type in label_types:
                classifiers[label_type] = \
                        self.__get_multiclass_classifier(train_data, label_type)
            #end for
            return classifiers
        else:
            #TODO: Implementing multiclass classifier implementation as of now.
            # Need to think more if binary classifier implementation can or should be here.
            pass
        return None


    def predict_joint_everything(self, classifiers, test_data):
        predictions = [defaultdict(str) for i in range(len(test_data))]
        self.logger.info('Predicting for Test')
        test_X = []
        all_pred_probas = {}
        if self.multiclass is True:
            for recipe in test_data:
                test_X.append(recipe.feats)
            for label_type in classifiers:
                test_Y_proba = classifiers[label_type][0].predict_proba(test_X)
                all_pred_probas[label_type] = test_Y_proba
            trigger_channel_labels = classifiers[TRIGGER_CHANNEL][1].classes_
            trigger_func_labels = classifiers[TRIGGER_FUNC][1].classes_
            action_channel_labels = classifiers[ACTION_CHANNEL][1].classes_
            action_func_labels = classifiers[ACTION_FUNC][1].classes_
            for i in range(len(test_data)):
                max_pred_proba = 0.0
                max_k_labels = {}
                for label_type in classifiers.keys():
                    max_k_labels[label_type] = self.get_max_k_labels(\
                            all_pred_probas[label_type][i].tolist(), \
                            classifiers[label_type][1].classes_,
                            k=5)
                #endfor
                channel_func_priors = self.dm.get_channel_func_priors()
                trigger_action_priors = self.dm.get_trigger_action_priors()
                for t_channel in max_k_labels[TRIGGER_CHANNEL].keys():
                    for t_func in max_k_labels[TRIGGER_FUNC].keys():
                        for a_channel in max_k_labels[ACTION_CHANNEL].keys():
                            for a_func in max_k_labels[ACTION_FUNC].keys():
                                pred_proba = 1.0
                                pred_proba *= max_k_labels[TRIGGER_CHANNEL][t_channel]
                                pred_proba *= max_k_labels[TRIGGER_FUNC][t_func]
                                pred_proba *= max_k_labels[ACTION_CHANNEL][a_channel]
                                pred_proba *= max_k_labels[ACTION_FUNC][a_func]
                                pred_proba *= channel_func_priors[t_channel][t_func]
                                pred_proba *= channel_func_priors[a_channel][a_func]
                                pred_proba *= trigger_action_priors[t_channel][a_channel]
                                if pred_proba > max_pred_proba:
                                    predictions[i][TRIGGER_CHANNEL] = t_channel
                                    predictions[i][TRIGGER_FUNC] = t_func
                                    predictions[i][ACTION_CHANNEL] = a_channel
                                    predictions[i][ACTION_FUNC] = a_func
                                #endif
                            #endfor a_func
                        #endfor a_channel
                    #endfor t_func
                #endfor t_channel
            #endfor i
            return predictions
        else:
            #TODO: Implementing multiclass classifier implementation as of now.
            # Need to think more if binary classifier implementation can or should be here.
            pass
        return


    def predict_joint_channel_funcs(self, classifiers, test_data):
        predictions = [defaultdict(str) for i in range(len(test_data))]
        self.logger.info('Predicting for Test')
        test_X = []
        all_pred_probas = {}
        if self.multiclass is True:
            for recipe in test_data:
                test_X.append(recipe.feats)
            for label_type in classifiers:
                test_Y_proba = classifiers[label_type][0].predict_proba(test_X)
                all_pred_probas[label_type] = test_Y_proba
            trigger_channel_labels = classifiers[TRIGGER_CHANNEL][1].classes_
            trigger_func_labels = classifiers[TRIGGER_FUNC][1].classes_
            action_channel_labels = classifiers[ACTION_CHANNEL][1].classes_
            action_func_labels = classifiers[ACTION_FUNC][1].classes_
            for i in range(len(test_data)):
                max_pred_proba = 0.0
                max_k_labels = {}
                for label_type in classifiers.keys():
                    max_k_labels[label_type] = self.get_max_k_labels(\
                            all_pred_probas[label_type][i].tolist(), \
                            classifiers[label_type][1].classes_,
                            k=5)
                #endfor
                channel_func_priors = self.dm.get_channel_func_priors()
                for t_channel in max_k_labels[TRIGGER_CHANNEL].keys():
                    for t_func in max_k_labels[TRIGGER_FUNC].keys():
                        pred_proba = 1.0
                        pred_proba *= max_k_labels[TRIGGER_CHANNEL][t_channel]
                        pred_proba *= max_k_labels[TRIGGER_FUNC][t_func]
                        pred_proba *= channel_func_priors[t_channel][t_func]
                        if pred_proba > max_pred_proba:
                            predictions[i][TRIGGER_CHANNEL] = t_channel
                            predictions[i][TRIGGER_FUNC] = t_func
                        #endif
                    #endfor t_func
                #endfor t_channel
                for a_channel in max_k_labels[ACTION_CHANNEL].keys():
                    for a_func in max_k_labels[ACTION_FUNC].keys():
                        pred_proba = 1.0
                        pred_proba *= max_k_labels[ACTION_CHANNEL][a_channel]
                        pred_proba *= max_k_labels[ACTION_FUNC][a_func]
                        pred_proba *= channel_func_priors[a_channel][a_func]
                        if pred_proba > max_pred_proba:
                            predictions[i][ACTION_CHANNEL] = a_channel
                            predictions[i][ACTION_FUNC] = a_func
                        #endif
                    #endfor a_func
                #endfor a_channel
            #endfor i
            return predictions
        else:
            #TODO: Implementing multiclass classifier implementation as of now.
            # Need to think more if binary classifier implementation can or should be here.
            pass
        return



    def predict_independent(self, classifiers, test_data):
        predictions = [defaultdict(str) for i in range(len(test_data))]
        self.logger.info('Predicting for Test')
        test_X = []
        all_pred_probas = {}
        if self.multiclass is True:
            for recipe in test_data:
                test_X.append(recipe.feats)
            for label_type in classifiers:
                test_Y_proba = classifiers[label_type][0].predict_proba(test_X)
                all_pred_probas[label_type] = test_Y_proba
            trigger_channel_labels = classifiers[TRIGGER_CHANNEL][1].classes_
            trigger_func_labels = classifiers[TRIGGER_FUNC][1].classes_
            action_channel_labels = classifiers[ACTION_CHANNEL][1].classes_
            action_func_labels = classifiers[ACTION_FUNC][1].classes_
            for i in range(len(test_data)):
                max_pred_proba = 0.0
                max_k_labels = {}
                for label_type in classifiers.keys():
                    max_k_labels[label_type] = self.get_max_k_labels(\
                            all_pred_probas[label_type][i].tolist(), \
                            classifiers[label_type][1].classes_,
                            k=1)
                    predictions[i][label_type] = list(max_k_labels.keys())[0]
                #endfor
            #endfor i
            return predictions
        else:
            #TODO: Implementing multiclass classifier implementation as of now.
            # Need to think more if binary classifier implementation can or should be here.
            pass
        return


    def get_max_k_labels(self, pred_probas, labels, k):
        max_k_inds = index_max_k(pred_probas,k)
        max_labels = {}
        for i in max_k_inds:
            max_labels[labels[i]] = pred_probas[i]
        return max_labels


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


