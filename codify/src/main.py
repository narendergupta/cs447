from codify.config.strings import *
from codify.config.settings import *
from datamodel import DataModel
from experimenter import Experimenter

import argparse
import logging
import time


def main(args):
    dm = DataModel(args.data_file)
    dm.read_data(to_read_count=2000)
    exp = Experimenter(dm, \
            process_datamodel=True, \
            serialise=False)
    t1 = time.time()
    exp.perform_hierarchical_multiclass_experiment(
            need_to_extract_features=True, english_only=False)
    t2 = time.time()
    timeused = t2 - t1
    logging.getLogger(LOGGER).info('Time used in experiment (hour:min:sec): %d:%d:%d' % \
            (timeused/3600, timeused/60, timeused%60))
    return exp


if __name__ == '__main__':
    logging.basicConfig(level=LOG_LEVEL)
    logger = logging.getLogger(LOGGER)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default=DEFAULT_DATA_PATH, required=False)
    args = parser.parse_args()
    exp = main(args)

