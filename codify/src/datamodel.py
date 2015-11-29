from codify.config.strings import *
from recipe import Recipe

import csv


class DataModel:
    """Class for reading and managing raw data"""
    def __init__(self, data_file='../data/recipe_parse.tsv', delimiter='\t'):
        self.data_file = data_file
        self.data_file_delimiter = delimiter


    def read_data(self, to_read_count=-1):
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
            #endfor
        #endwith
        return None


