from codify.config.strings import *
from datamodel import DataModel
from experimenter import Experimenter

import argparse


def main(args):
    dm = DataModel(args.data_file)
    dm.read_data(to_read_count=-1)
    exp = Experimenter(dm, train_file='../data/train.urls', test_file='../data/test.urls')
    return exp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", required=True)
    args = parser.parse_args()
    exp = main(args)

