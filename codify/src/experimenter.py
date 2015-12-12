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
            self.process_recipes()
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
        self.classifiers = defaultdict(lambda : defaultdict(None))
        test_data = self.dm.get_testing_data()
        self.predictions = [defaultdict(str) for i in range(len(test_data))]
        self.train()
        self.predict()  # Set predictions for test_data to self.predictions
        self.save_predictions(output_file='../data/predictions_multiclass.csv')
        self.evaluate(output_file='../data/results_multiclass.txt')
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


    def train(self):
        self.logger.info('Training')
        label_types = [TRIGGER_CHANNEL, TRIGGER_FUNC, ACTION_CHANNEL, ACTION_FUNC]
        train_data = self.dm.get_training_data()
        if self.multiclass is True:
            for label_type in label_types:
                self.classifiers[label_type] = \
                        self.__get_multiclass_classifier(train_data, label_type)
        else:
            #TODO: Implementing multiclass classifier implementation as of now.
            # Need to think more if binary classifier implementation can or should be here.
            pass
        return


    def predict(self):
        self.logger.info('Predicting for Test')
        test_X = []
        all_pred_probas = {}
        test_data = self.dm.get_testing_data()
        if self.multiclass is True:
            for recipe in test_data:
                test_X.append(recipe.feats)
            for label_type in self.classifiers:
                test_Y_proba = self.classifiers[label_type][0].predict_proba(test_X)
                all_pred_probas[label_type] = test_Y_proba
            trigger_channel_labels = self.classifiers[TRIGGER_CHANNEL][1].classes_
            trigger_func_labels = self.classifiers[TRIGGER_FUNC][1].classes_
            action_channel_labels = self.classifiers[ACTION_CHANNEL][1].classes_
            action_func_labels = self.classifiers[ACTION_FUNC][1].classes_
            for i in range(len(test_data)):
                max_pred_proba = 0.0
                max_k_labels = {}
                for label_type in self.classifiers.keys():
                    max_k_labels[label_type] = self.get_max_k_labels(\
                            all_pred_probas[label_type][i].tolist(), \
                            self.classifiers[label_type][1].classes_,
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
                                    self.predictions[i][TRIGGER_CHANNEL] = t_channel
                                    self.predictions[i][TRIGGER_FUNC] = t_func
                                    self.predictions[i][ACTION_CHANNEL] = a_channel
                                    self.predictions[i][ACTION_FUNC] = a_func
                                #endif
                            #endfor a_func
                        #endfor a_channel
                    #endfor t_func
                #endfor t_channel
            #endfor i
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


    def save_predictions(self, output_file):
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
            for i in range(len(self.predictions)):
                out_dict = {URL:test_data[i].url, \
                        PRED_TRIGGER_CHANNEL:self.predictions[i][TRIGGER_CHANNEL], \
                        PRED_TRIGGER_FUNC:self.predictions[i][TRIGGER_FUNC], \
                        PRED_ACTION_CHANNEL:self.predictions[i][ACTION_CHANNEL], \
                        PRED_ACTION_FUNC:self.predictions[i][ACTION_FUNC], \
                        GOLD_TRIGGER_CHANNEL:test_data[i][TRIGGER_CHANNEL], \
                        GOLD_TRIGGER_FUNC:test_data[i][TRIGGER_FUNC], \
                        GOLD_ACTION_CHANNEL:test_data[i][ACTION_CHANNEL], \
                        GOLD_ACTION_FUNC:test_data[i][ACTION_FUNC]
                        }
                output_writer.writerow(out_dict)
            #endfor
        #endwith
        return


    def evaluate(self, output_file):
        self.logger.info('Evaluating Prediction Scores')
        channel_labels = []
        channel_preds = []
        func_labels = []
        func_preds = []
        test_data = self.dm.get_testing_data()
        for i in range(len(test_data)):
            channel_labels.append(test_data[i].trigger_channel)
            channel_labels.append(test_data[i].action_channel)
            func_labels.append(test_data[i].trigger_func)
            func_labels.append(test_data[i].action_func)
            channel_preds.append(self.predictions[i][TRIGGER_CHANNEL])
            channel_preds.append(self.predictions[i][ACTION_CHANNEL])
            func_preds.append(self.predictions[i][TRIGGER_FUNC])
            func_preds.append(self.predictions[i][ACTION_FUNC])
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


    def process_recipes(self):
        desc_list = []
        title_list = []
        for recipe in self.dm.data:
            desc_list.append(recipe.desc)
            title_list.append(recipe.title)
        # count_vect = CountVectorizer(tokenizer=LemmaTokenizer(), max_features=10000)

        tokenizer = LemmaTokenizer()
        self.logger.info('Applying count vectorizer to the data')
        count_vect = CountVectorizer(analyzer='char_wb', ngram_range=(1,3), max_features=2000)
        #count_vect_title = CountVectorizer(tokenizer=tokenizer, max_features=1000)
        desc_term_mat = count_vect.fit_transform(desc_list)
        #title_term_mat = count_vect_title.fit_transform(title_list)
        self.logger.info('Applying TF-IDF transform')
        tfidf_transformer_desc = TfidfTransformer()
        #tfidf_transformer_title = TfidfTransformer()
        #desc_term_tfidf = tfidf_transformer_desc.fit_transform(desc_term_mat)
        # title_term_tfidf = tfidf_transformer_title.fit_transform(title_term_mat)
        # self.logger.info('TF-IDF output shape - ' + str(desc_term_tfidf.shape))
        # self.logger.info('TF-IDF output shape of title term matrix - ' + str(title_term_tfidf.shape))

        inv_vocab_desc = {v: k for k, v in count_vect.vocabulary_.items()}
        #inv_vocab_title = {v: k for k, v in count_vect_title.vocabulary_.items()}


        self.logger.info('Extracting features')
        for i in range(len(self.dm.data)):
            self.dm.data[i].feats = desc_term_mat.getrow(i).toarray().flatten()
            # self.dm.data[i].set_feats()
            if i % 1000 == 0: self.logger.info('Features extracted for %d recipes' % i)

            # cur_row_desc = desc_term_tfidf.getrow(i).toarray()
            # desc_word_indices = np.where(cur_row_desc != 0)[1]

            # cur_row_title = title_term_tfidf.getrow(i).toarray()
            # title_word_indices = np.where(cur_row_title != 0)[1]

            # if len(desc_word_indices) < DESC_LEN_THRESHOLD:
            #     continue

            # desc_words = map(lambda x: inv_vocab_title[x], desc_word_indices)

            # wordvec_feats = np.zeros(EMBEDDIGS_DIM)
            # n = 0
            # for word in words:
            #     if word in self.wv.embeddings:
            #         n += 1
            #         wordvec_feats += np.array(self.wv.embeddings[word]) * doc_term_tfidf[i, count_vect.vocabulary_[word]]
            # if n == 0:
            #     continue
            # wordvec_feats = wordvec_feats / n

            # self.dm.data[i].feats = np.concatenate((wordvec_feats, cur_row[0]))

            # self.dm.data[i].feats = cur_row_title[0]

        return None


