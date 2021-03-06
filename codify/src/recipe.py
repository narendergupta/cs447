from codify.config.strings import *


class Recipe:
    """Class representing an IFTTT recipe"""
    def __init__(self, url, ID, title, desc, author, featured, uses, favs, code):
        if url == '':
            raise ValueError('URL cannot be empty')
        self.url = url
        self.ID = int(ID)
        self.title = title
        self.desc = desc
        self.desc_vector = None
        self.author = author
        self.featured = True if featured == 'TRUE' else False
        self.uses = int(uses)
        self.favs = int(favs)
        self.code = code
        self.feats = []       # Features to be extracted later
        self.trigger_channel, self.trigger_func, self.trigger_params = '', '', ''
        self.action_channel, self.action_func, self.action_params = '', '', ''
        self.is_english = None
        self.is_legible = None
        self.trigger_turk_agreements = None
        self.action_turk_agreements = None
        self.parse_code()


    def __getitem__(self, attr):
        if attr == TRIGGER_CHANNEL:
            return self.trigger_channel
        elif attr == TRIGGER_FUNC:
            return self.trigger_func
        elif attr == ACTION_CHANNEL:
            return self.action_channel
        elif attr == ACTION_FUNC:
            return self.action_func
        raise AttributeError('Unknown attribute type')


    def parse_code(self):
        code = self.code[6:-1]
        if_then = code.split(' (THEN) ')
        if_str = if_then[0][6:-1]
        then_str = if_then[1][1:-1]
        if_parts = if_str.split(' (FUNC ')
        self.trigger_channel = if_parts[0][9:-1]
        if_parts[1] = if_parts[1][:-1]
        then_parts = then_str.split(' (FUNC ')
        self.action_channel = then_parts[0][8:-1]
        then_parts[1] = then_parts[1][:-1]
        self.trigger_func, self.trigger_params = self.__get_func_params(if_parts[1])
        self.action_func, self.action_params = self.__get_func_params(then_parts[1])
        return None


    def __get_func_params(self, string):
        func, params = '', ''
        parts = string.split(' (PARAMS')
        func = parts[0][1:-1]
        params = parts[1][1:-1]
        return (func, params)



    def print_trigger_action(self):
        print(self.trigger_channel, self.trigger_func, self.trigger_params)
        print(self.action_channel, self.action_func, self.action_params)
        return None


