import pickle
import sys
import os
import sqlite3
import xml.etree.ElementTree as ET
import time


def load_targets_file(targetsfile, cache=None):
    """
    Load targets file from file, optionally using a cache directory.
    Return a2target, a2bias, a2url mappings.
    If cache is None, do not use cache.
    """
    if cache:
        pickledtargetsfile = os.path.join(cache.dir, os.path.splitext(os.path.basename(targetsfile))[0] + ".pickle")

    # load the target info into memory. If we already created our pickled version in the cache, use
    # that instead
    print("Loading target info ... ", file=sys.stderr)
    if cache is not None and os.path.exists(pickledtargetsfile):
        print("Found cache file, loading from ", pickledtargetsfile)
        with open(pickledtargetsfile, "rb") as inputstream:
            (a2target, a2bias, a2url) = pickle.load(inputstream)
    else:
        a2target = {}
        a2bias = {}
        a2url = {}
        tree = ET.parse(targetsfile)
        root = tree.getroot()
        for child in root:
            attribs = child.attrib
            id = attribs["id"]
            a2target[id] = attribs["hyperpartisan"]
            a2bias[id] = attribs.get("bias")
            a2url[id] = attribs["url"]
        if cache is not None:
            print("Saving to cache: ", pickledtargetsfile)
            with open(pickledtargetsfile, "wb") as outstream:
                pickle.dump((a2target, a2bias, a2url), outstream)
    print("Target info loaded, number of targets: ", len(a2target), file=sys.stderr)
    return a2target, a2bias, a2url


def run_pipeline(functionlist, article, **kwargs):
    """
    Run the given pipeline on the article. Return nerror.
    :param functionlist:
    :param article:
    :param kwargs:
    :return:
    """
    nerror = 0
    if kwargs.get("debug"):
        print("DEBUG: running pipeline on ", article['id'], file=sys.stderr)
    for func in functionlist:
        ret = func(article, **kwargs)
        if ret:
            nerror += 1
    return nerror




def process_articles_xml(articlesfile, functionlist, maxn=None, **kwargs):
    """
    Process the given article XML file and run the list of closures on each article.
    This always creates an initial article representation that only contains the
    article-XML as 'xml' and the id as 'id' before invoking the first closure.
    Instead of a closure it could also be any class that implements __call__
    All cosures should accept the following arguments: the article representation,
    and arbitrary **kwargs.
    All closure should return true if the article was processed successfully or false
    if there was an error.
    :return: a tuple with the total number of articles and number of errors
    """
    if len(functionlist) == 0:
        raise Exception("Need at least one function to run in the pipeline list")
    tree = ET.iterparse(articlesfile)
    nprocessed = 0
    nerror = 0
    debug = kwargs.get("debug")
    for event, element in tree:
        if element.tag == "article":
            attrs = element.attrib
            articleid = attrs['id']
            published = attrs.get("published-at")
            if debug:
                print("DEBUG: processing article", articleid)
            # get the XML and store it in a dictionary as "xml"
            xml = ET.tostring(element, encoding="utf-8", method="xml").decode()
            article = {
                'id': articleid,
                'xml': xml,
                'published-at': published,
                'et': element
            }
            nerror += run_pipeline(functionlist, article, **kwargs)
            del article['et']
            nprocessed += 1
            if nprocessed % 1000 == 0:
                print("Processed articles:", nprocessed, " errors:", nerror)
            if maxn is not None and nprocessed >= maxn:
                print("Stopping after maximum number of articles: ", maxn, file=sys.stderr)
                break
    return nprocessed, nerror
