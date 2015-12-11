import os

def lowercase(source):
    if type(source) is str:
        return source.lower()
    elif type(source) is dict:
        return dict((k,lowercase(v)) for k,v in source.items())
    elif type(source) is list:
        return [lowercase(x) for x in source]
    else:
        return source


def unique(source):
    if type(source) is list:
        return list(set(source))
    elif type(source) is dict:
        return dict((k,unique(v)) for k,v in source.items())
    else:
        return source


def float_precise_str(source, precision=2):
    if type(source) is float:
        precision_str = '%.' + str(precision) + 'f'
        return precision_str % source
    elif type(source) is dict:
        return dict((k,float_precise_str(v,precision)) for k,v in source.items())
    elif type(source) is list:
        return [float_precise_str(x,precision) for x in source]
    else:
        return source


def skewness(items, label1, label2):
    numer = min(items.count(label1),items.count(label2))
    denom = max(items.count(label1),items.count(label2))
    skewness = float(numer)/float(denom)
    if skewness < 0.5:
        skewness = 1.0 - skewness
    return skewness


def index_min(items):
    if len(items) > 0:
        return min(range(len(items)), key=items.__getitem__)
    return None


def index_max(items):
    if len(items) > 0:
        return max(range(len(items)), key=items.__getitem__)
    return None


def index_max_k(items, k, reverse=True):
    if type(items) is not list:
        return None
    desc_sorted = [(items[i],i) for i in range(len(items))]
    desc_sorted = sorted(desc_sorted, reverse=reverse, key=lambda item : item[0])
    desc_sorted = desc_sorted[:k]
    return [desc_sorted[i][1] for i in range(len(desc_sorted))]


def index_min_k(items, k):
    return index_max_k(items, k, reverse=False)


def ensure_dir_exists(dir_path):
    if os.path.exists(dir_path) is False or\
            os.path.isdir(dir_path) is False:
                os.makedirs(dir_path)


def does_file_exist(file_path):
    return os.path.exists(file_path) and os.path.isfile(file_path)


