'''
Various methods for doing nlp stuff (sentence splitting, tokenization) in a consistent way across scripts.
'''

import sys
import re

punctuation = '.,:;?!"\'\'``+={}[]()#~$--'
stopwords = None


def init_stopwords():
    print("Loading stopwords...", file=sys.stderr)
    from nltk.corpus import stopwords as NLTK_STOPWORDS
    from gensim.parsing.preprocessing import STOPWORDS as GENSIM_STOPWORDS
    global stopwords
    stopwords = set(NLTK_STOPWORDS.words('english'))
    for w in GENSIM_STOPWORDS:
        stopwords.add(w)
    stopwords.add("'s")
    stopwordlist = list(stopwords)
    for s in stopwordlist:
        stopwords.add(s.capitalize())
    print("Stopwords loaded", file=sys.stderr)


re_num_simple = re.compile('^-?[0-9.,]+([eE^][0-9]+)?(th)?$')


def filter_tokens(tokens):
    if not stopwords:
        init_stopwords()
    filtered = [token for token in tokens if token not in punctuation and "_" not in token and token not in stopwords]
    filtered = [re_num_simple.subn("<num>", token)[0] for token in filtered]
    return filtered
