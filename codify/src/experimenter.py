from codify.config.strings import *
from datamodel import DataModel


class Experimenter:
    """Execute and manage experiments"""
    def __init__(self, dm):
        self.set_datamodel(dm)


    def set_datamodel(self, dm):
        self.dm = dm
        self.channels = []
        self.funcs = []
        self.__enumerate_channels_funcs()
        return None


    def __enumerate_channels_funcs(self):
        for recipe in self.dm.data:
            if recipe.trigger_channel not in self.channels:
                self.channels.append(recipe.trigger_channel)
            if recipe.action_channel not in self.channels:
                self.channels.append(recipe.action_channel)
            if recipe.trigger_func not in self.funcs:
                self.funcs.append(recipe.trigger_func)
            if recipe.action_func not in self.funcs:
                self.funcs.append(recipe.action_func)
        return None


