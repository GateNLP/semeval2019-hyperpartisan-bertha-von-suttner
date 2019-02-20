"""
Constants and functions related to the features.
"""

FEATURES = [
]


def doc2features(doc, features):
    """
    Extract the features from the document. Each feature is either just the name
    or a tuple of (name,flag,type) where flag indicates if the feature should get
    selected. This returns the features in their original representation.
    :param doc: to extract the features from
    :param features: list of feature names or 3-tuples
    :return: list of values for the selected features
    """
    ret = []
    for f in features:
        if isinstance(f, str):
            val = doc.get(f)
            ret.append(val)
        else:
            name, flag, ftype = f
            if flag:
                val = doc.get(name)
                ret.append(val)
    return ret

def features2use(features):
    """
    Return just those features which have flag true or are just the name, not the tuple
    :param features:
    :return:
    """
    ret = []
    for f in features:
        if isinstance(f, str):
            ret.append(f)
        else:
            name, flag, ftype = f
            if flag:
                ret.append(f)
    return ret
