import logging
from codify.config.strings import *
from codify.config.settings import *
from recipe import Recipe

import csv


class DataModel:
    """Class for reading and managing raw data"""
    def __init__(self, data_file='../data/recipe_parse.tsv', delimiter='\t'):
        self.data_file = data_file
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


