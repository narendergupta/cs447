import logging
from collections import defaultdict
from codify.config.strings import *
from codify.config.settings import *
from recipe import Recipe

import csv


class DataModel:
    """Class for reading and managing raw data"""
    def __init__(self, data_file='../data/recipe_parse.tsv', delimiter='\t', \
            train_urls_file = '../data/train.urls', \
            test_urls_file = '../data/test.urls'):
        self.data_file = data_file
        self.train_urls_file = train_urls_file
        self.test_urls_file = test_urls_file
        self.data_file_delimiter = delimiter
        self.logger = logging.getLogger(LOGGER)
        self.data = None


    def read_data(self, to_read_count=-1):
        self.logger.info('Reading data ...')
        self.data = []
        """Reads data file"""
        read_count = 0
        with open(self.data_file,'r') as data_f:
            reader = csv.DictReader(data_f, delimiter=self.data_file_delimiter)
            for row in reader:
                read_count += 1
                try:
                    recipe = Recipe(row[URL], row[ID], row[TITLE], row[DESC],
                            row[AUTHOR], row[FEATURED], row[USES], row[FAVS],
                            row[CODE])
                    self.data.append(recipe)
                except ValueError:
                    pass
                if to_read_count > 0 and read_count >= to_read_count:
                    break
                if read_count % 10000 == 0:
                    self.logger.info('Read ' + str(read_count) + ' recipes')
            #endfor
        #endwith
        self.logger.info('Read all recipes (' + str(read_count) + ')')
        return None


    def __read_train_test_urls(self):
        with open(self.train_urls_file, 'r') as train_f:
            self.train_urls = train_f.readlines()
        with open(self.test_urls_file, 'r') as test_f:
            self.test_urls = test_f.readlines()
        # Change lists into dict for quick access
        self.train_urls = dict((url.strip(),1) for url in self.train_urls)
        self.test_urls = dict((url.strip(),1) for url in self.test_urls)
        return None


    def get_training_data(self):
        try:
            return self.train_data
        except AttributeError:
            try:
                self.train_data = []
                for recipe in self.data:
                    if recipe.url in self.train_urls:
                        self.train_data.append(recipe)
                #endfor
                return self.train_data
            except AttributeError:
                del self.train_data
                self.__read_train_test_urls()
                return self.get_training_data()
        #end try except


    def get_testing_data(self):
        try:
            return self.test_data
        except AttributeError:
            try:
                self.test_data = []
                for recipe in self.data:
                    if recipe.url in self.test_urls:
                        self.test_data.append(recipe)
                #endfor
                return self.test_data
            except AttributeError:
                del self.test_data
                self.__read_train_test_urls()
                return self.get_testing_data()
        #end try except


    def get_channel_func_priors(self):
        try:
            return self.channel_func_priors
        except AttributeError:
            self.channel_func_priors = defaultdict(lambda : defaultdict(float))
            train_data = self.get_training_data()
            for recipe in train_data:
                self.channel_func_priors[recipe.trigger_channel][recipe.trigger_func] += 1.0
                self.channel_func_priors[recipe.action_channel][recipe.action_func] += 1.0
            #endfor data loop
            for channel in self.channel_func_priors:
                func_count = 0
                for func in self.channel_func_priors[channel]:
                    func_count += self.channel_func_priors[channel][func]
                # If any function is present, turn it into probability distribution
                if func_count == 0:
                    continue
                else:
                    for func in self.channel_func_priors[channel]:
                        self.channel_func_priors[channel][func] /= func_count
                #endif
            #endfor
            return self.channel_func_priors
        #end try except


    def get_trigger_action_priors(self):
        try:
            return self.trigger_action_priors
        except AttributeError:
            self.trigger_action_priors = defaultdict(lambda : defaultdict(float))
            train_data = self.get_training_data()
            for recipe in train_data:
                self.trigger_action_priors[recipe.trigger_channel][recipe.action_channel] += 1.0
            for trigger in self.trigger_action_priors:
                action_count = 0
                for action in self.trigger_action_priors[trigger]:
                    action_count += self.trigger_action_priors[trigger][action]
                if action_count == 0:
                    continue
                else:
                    for action in self.trigger_action_priors[trigger]:
                        self.trigger_action_priors[trigger][action] /= action_count
                #endif
            return self.trigger_action_priors
        #end try except
        

