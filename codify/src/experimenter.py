from codify.config.strings import *
from datamodel import DataModel
from gen_utils import *
from ml_utils import *
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn import linear_model, svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


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
    def __init__(self, dm, train_file, test_file):
        self.__read_train_test_urls(train_file, test_file)
        self.set_datamodel(dm)


    def set_datamodel(self, dm):
        self.dm = dm
        self.process_recipes()
        self.__enumerate_channels_funcs()
        return None


    def __read_train_test_urls(self, train_file, test_file):
        with open(train_file, 'r') as train_f:
            self.train_urls = train_f.readlines()
        with open(test_file, 'r') as test_f:
            self.test_urls = test_f.readlines()
        return None


    def __enumerate_channels_funcs(self):
        self.trigger_channels = {}
        self.trigger_funcs = {}
        self.action_channels = {}
        self.action_funcs = {}
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
        return None


    def get_classifiers(self):
        clfs = [{CLASSIFIER:svm.SVC(kernel='linear'), STRING:'SVC_LINEAR'},\
                {CLASSIFIER:svm.SVC(kernel='rbf'), STRING:'SVC_RBF'},\
                {CLASSIFIER:linear_model.LogisticRegression(), \
                STRING:'LOGISTIC_REGRESSION'},\
                ]
        return clfs


    def process_recipes(self):
        desc_list = []
        for recipe in self.dm.data:
            desc_list.append(recipe.desc)
        count_vect = CountVectorizer(tokenizer=LemmaTokenizer(), max_features=10000)
        doc_term_mat = count_vect.fit_transform(desc_list)
        tfidf_transformer = TfidfTransformer()
        doc_term_tfidf = tfidf_transformer.fit_transform(doc_term_mat)
        print(doc_term_tfidf.shape)
        for i in range(len(self.dm.data)):
            self.dm.data[i].desc_vector = doc_term_tfidf.getrow(i)
        return None


