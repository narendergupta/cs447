from codify.config.strings import *
from codify.config.settings import *
from datamodel import DataModel
from experimenter import Experimenter

import argparse
import logging
import time


def main(args):
    dm = DataModel(args.data_file)
    dm.read_data(to_read_count=100)
    exp = Experimenter(dm, \
            train_file='../data/train.urls', \
            test_file='../data/test.urls', \
            process_datamodel=True, \
            serialise=False)
    t1 = time.time()
    exp.perform_multiclass_experiment()
    t2 = time.time()
    timeused = t2 - t1
    logging.getLogger(LOGGER).info('Time used in experiment: %f seconds' % timeused)
    return exp


if __name__ == '__main__':
    logging.basicConfig(level=LOG_LEVEL)
    logger = logging.getLogger(LOGGER)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default=DEFAULT_DATA_PATH, required=False)
    args = parser.parse_args()
    exp = main(args)

