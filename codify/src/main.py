from codify.config.strings import *
from datamodel import DataModel
from experimenter import Experimenter

import argparse


def main(args):
    dm = DataModel(args.data_file)
    dm.read_data(to_read_count=-1)
    exp = Experimenter(dm)
    print(len(exp.channels))
    print(len(exp.funcs))
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", required=True)
    args = parser.parse_args()
    main(args)

