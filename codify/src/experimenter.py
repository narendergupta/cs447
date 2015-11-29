from codify.config.strings import *
from datamodel import DataModel


class Experimenter:
    """Execute and manage experiments"""
    def __init__(self, dm):
        self.dm = dm


    def set_datamodel(self, dm):
        self.dm = dm
        return None


