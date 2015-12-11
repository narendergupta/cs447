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
from ml_utils import *
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn import linear_model, svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, ClassifierMixin


class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        tokens = [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
        tokens = [w for w in tokens if w not in stopwords.words('english')]
        return tokens
#endclass

class UnaryClassifier(BaseEstimator, ClassifierMixin):
    # A simple unary classifier which return the one class that the data as the prediction
    def __init__(self, label):
        self.label = label

    def fit(self, X, y):
        assert len(np.unique(y)) == 1
        self.label = np.unique(y)[0]
        return self

    def predict(self, X):
        preds = [self.label] * len(X)
        return np.array(preds)

    def predict_proba(self, X):
        return np.ones((len(X), 1))

    def predict_log_proba(self, X):
        return np.zeros((len(X), 1))

class Experimenter:
    """Execute and manage experiments"""
    def __init__(self, dm, train_file, test_file, process_datamodel, serialise):
        self.logger = logging.getLogger(LOGGER)

        self.wv = WordVectors()

        self.__read_train_test_urls(train_file, test_file)
        self.dm = None
        self.set_datamodel(dm, process_datamodel, serialise)

        self.productions_data = defaultdict(lambda : defaultdict(list))
        # list of all possible trigger_channel, action_channel, trigger_function, action_function
        self.node_domain = {TRIGGER_CHANNEL: [], ACTION_CHANNEL: [], TRIGGER_FUNC: [], ACTION_FUNC: []}
        self.__extract_productions_data()

        self.classifiers = {}
        self.train_productions()

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
        else:
            with open(DATAMODEL, 'r') as data_model:
                self.logger.info('loading serialised data model')
                self.dm.data = pickle.load(data_model)

        # self.__enumerate_channels_funcs()
        return None


    def __read_train_test_urls(self, train_file, test_file):
        with open(train_file, 'r') as train_f:
            self.train_urls = train_f.readlines()
        with open(test_file, 'r') as test_f:
            self.test_urls = test_f.readlines()
        return None

    def __extract_productions_data(self):
        self.logger.info('Extracting productions\' data')
        for recipe in self.dm.data:
            self.productions_data[TRIGGER][recipe.trigger_channel].append(recipe)
            self.productions_data[recipe.trigger_channel][recipe.trigger_func].append(recipe)
            self.productions_data[ACTION][recipe.action_channel].append(recipe)
            self.productions_data[recipe.action_channel][recipe.action_func].append(recipe)

            if recipe.trigger_channel not in self.node_domain[TRIGGER_CHANNEL]:
                self.node_domain[TRIGGER_CHANNEL].append(recipe.trigger_channel)
            if recipe.trigger_func not in self.node_domain[TRIGGER_FUNC]:
                self.node_domain[TRIGGER_FUNC].append(recipe.trigger_func)
            if recipe.action_channel not in self.node_domain[ACTION_CHANNEL]:
                self.node_domain[ACTION_CHANNEL].append(recipe.action_channel)
            if recipe.action_func not in self.node_domain[ACTION_FUNC]:
                self.node_domain[ACTION_FUNC].append(recipe.action_func)
        return None

    def __get_classifier(self, nt, train_dict):
        # given the non-terminal and all the samples with productions having that nt, return a classifier

        label_map = {}
        i = 0
        for key in train_dict.keys():
            label_map[key] = i
            i += 1
        inv_label_map = {}
        for k in label_map.keys():
            inv_label_map[label_map[k]] = k

        recipes = []
        X = []
        Y = []
        # RHS of the production with nt as LHS
        for rhs in train_dict:
            for recipe in train_dict[rhs]:
                if recipe.feats is not None:
                    recipes.append(recipe)
                    X.append(recipe.feats)
                    Y.append(label_map[rhs])

        Y_unique = np.unique(Y)
        if len(Y_unique) == 0:
            # no training data found. Return any label as the prediction.
            label_index = inv_label_map.keys()[0]
            clf = UnaryClassifier(label_index)
            return {label_index: inv_label_map[label_index]}, clf
        if len(Y_unique) == 1:
            label_index = Y_unique[0]
            clf = UnaryClassifier(label_index)
            return {label_index: inv_label_map[label_index]}, clf

        self.logger.info('Learning classifier for %s' % nt)
        # clf = linear_model.LogisticRegression(C = 10, multi_class='multinomial', class_weight=defaultdict(lambda : 1), solver='lbfgs')
        clf = linear_model.LogisticRegression(C = 10, class_weight=defaultdict(lambda : 1))
        clf.fit(X, Y)
        return inv_label_map, clf

    # Note - there could be overlap between channel names and function names (there is just one in the data - Is_It_Christmas? which is both trigger function and trigger channel). Handle it later!
    def inference(self, X_i):
        """
        Given the probabilities of productions, it defines a bayesian network. With S at root and S->trigger, S->action as the first level.
        trigger->trigger_channel, action->action_channel as the second level. trigger_channel->trigger_function, action_channel->action_function as the third level.
        This method does inference on the tree given the transition probabilities
        """

        result = {}

        for root in [TRIGGER, ACTION]:
            label_map, clf = self.classifiers[root]
            log_proba = clf.predict_log_proba(X_i)[0]
            # labels and log_proba indices should be same. But for future extension, the following dict
            label_to_index = {clf.classes_[i] : i for i in xrange(len(clf.classes_))}

            best_channel_func = {'total_log_prob': float('-inf'), CHANNEL: None, FUNC: None}

            for channel_label in clf.classes_:
                channel = label_map[channel_label]
                label_map_channel, clf_channel = self.classifiers[channel]
                best_func = label_map_channel[clf_channel.predict(X_i)[0]]
                best_log_proba = np.max(clf_channel.predict_log_proba(X_i))

                total_log_prob = log_proba[label_to_index[channel_label]] + best_log_proba

                if total_log_prob > best_channel_func['total_log_prob']:
                    best_channel_func['total_log_prob'] = total_log_prob
                    best_channel_func[CHANNEL] = channel
                    best_channel_func[FUNC] = best_func

            result[root + '_channel'] = best_channel_func[CHANNEL]
            result[root + '_func'] = best_channel_func[FUNC]

        return result



    def train_productions(self):
        # for all non-terminals get a classifier of posterior distribution given the nt
        for nt in self.productions_data:
            # if nt != TRIGGER:
            #     self.classifiers[nt] = None
            #     continue
            self.classifiers[nt] = self.__get_classifier(nt, self.productions_data[nt])

    # unused
    def __enumerate_channels_funcs(self):
        self.trigger_channels = {}
        self.trigger_funcs = {}
        self.action_channels = {}
        self.action_funcs = {}
        self.logger.info('enumerating channels and functions')
        for recipe in self.dm.data:
            if recipe.trigger_channel not in self.trigger_channels:
                self.trigger_channels[recipe.trigger_channel] = []
            self.trigger_channels[recipe.trigger_channel].append(recipe)
            if recipe.trigger_func not in self.trigger_funcs:
                self.trigger_funcs[recipe.trigger_func] = []
            self.trigger_funcs[recipe.trigger_func].append(recipe)
            if recipe.action_channel not in self.action_channels:
                self.action_channels[recipe.action_channel] = []
            self.action_channels[recipe.action_channel].append(recipe)
            if recipe.action_func not in self.action_funcs:
                self.action_funcs[recipe.action_func] = []
            self.action_funcs[recipe.action_func].append(recipe)
        self.logger.info('done')
        return None

    # unused
    def get_classifiers(self):
        clfs = [{CLASSIFIER:svm.SVC(kernel='linear'), STRING:'SVC_LINEAR'},\
                {CLASSIFIER:svm.SVC(kernel='rbf'), STRING:'SVC_RBF'},\
                {CLASSIFIER:linear_model.LogisticRegression(), \
                STRING:'LOGISTIC_REGRESSION'},\
                ]
        return clfs

    def process_recipes(self):
        desc_list = []
        title_list = []
        for recipe in self.dm.data:
            desc_list.append(recipe.desc)
            title_list.append(recipe.title)
        # count_vect = CountVectorizer(tokenizer=LemmaTokenizer(), max_features=10000)

        tokenizer = LemmaTokenizer()
        self.logger.info('Applying count vectorizer to the data')
        count_vect = CountVectorizer(tokenizer=tokenizer)
        count_vect_title = CountVectorizer(tokenizer=tokenizer, max_features=10000)
        desc_term_mat = count_vect.fit_transform(desc_list)
        title_term_mat = count_vect_title.fit_transform(title_list)
        self.logger.info('Applying TF-IDF transform')
        tfidf_transformer_desc = TfidfTransformer()
        tfidf_transformer_title = TfidfTransformer()
        # desc_term_tfidf = tfidf_transformer_desc.fit_transform(desc_term_mat)
        title_term_tfidf = tfidf_transformer_title.fit_transform(title_term_mat)
        # self.logger.info('TF-IDF output shape - ' + str(desc_term_tfidf.shape))
        self.logger.info('TF-IDF output shape of title term matrix - ' + str(title_term_tfidf.shape))

        inv_vocab_desc = {v: k for k, v in count_vect.vocabulary_.items()}
        inv_vocab_title = {v: k for k, v in count_vect_title.vocabulary_.items()}


        self.logger.info('Extracting features')
        for i in range(len(self.dm.data)):
            # self.dm.data[i].desc_vector = doc_term_tfidf.getrow(i)
            # self.dm.data[i].set_feats()
            if i % 1000 == 0: self.logger.info('Features extracted for %d recipes' % i)

            # cur_row_desc = desc_term_tfidf.getrow(i).toarray()
            # desc_word_indices = np.where(cur_row_desc != 0)[1]

            cur_row_title = title_term_tfidf.getrow(i).toarray()
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

            self.dm.data[i].feats = cur_row_title[0]

        return None


