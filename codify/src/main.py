from codify.config.strings import *
from codify.config.settings import *
from datamodel import DataModel
from experimenter import Experimenter

import argparse
import logging


def main(args):
    dm = DataModel(args.data_file)
    dm.read_data(to_read_count=100)
    exp = Experimenter(dm, \
            train_file='../data/train.urls', \
            test_file='../data/test.urls', \
            process_datamodel=True, \
            serialise=False)
    return exp


if __name__ == '__main__':
    logging.basicConfig(level=LOG_LEVEL)
    logger = logging.getLogger(LOGGER)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default=DEFAULT_DATA_PATH, required=False)
    args = parser.parse_args()
    exp = main(args)

