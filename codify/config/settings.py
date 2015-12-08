import logging

GLOVE_DIR = '/Users/pavankumar/Developer/glove.6B'
EMBEDDIGS_FILE = 'glove.6B.100d.txt'
EMBEDDIGS_DIM = 100

LOGGER = 'codify'
LOG_LEVEL = logging.DEBUG
DEFAULT_DATA_PATH = '../data/recipe_parse.tsv'
DATAMODEL = '../model/data_model.pickle'
ROOT_DIR = '/Users/pavankumar/Onedrive/cogcomp/ifttt/public_release/cs447/'
DESC_LEN_THRESHOLD = 2  # minimum length of the description to be considered for train or test