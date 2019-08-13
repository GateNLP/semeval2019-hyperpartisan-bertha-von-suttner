#!/usr/bin/env python
'''
Process the XML articles with a pipeline, extract a list of features and write
the features to one line per document in an output tsv file.
'''

import argparse

from . import utils
from . import processingresources
from . import features


def create_pipeline():
    """
    Create a fresh pipeline (a list of functions) that
    does the usual XML processing.
    """

    return [
        processingresources.PrAddTitle(),
        processingresources.PrAddText(),
        processingresources.PrNlpSpacy01(),
        processingresources.PrFilteredText(),
        processingresources.PrSeqSentences(),
        processingresources.PrRemovePars(),
    ]


def main():
    default_F = "FEATURES_TOKENS_ORIG"

    parser = argparse.ArgumentParser()
    parser.add_argument("-A", default=None, type=str, required=True, help="Article XML file")
    parser.add_argument("-T", default=None, type=str, help="Targets XML file, if missing, targets etc are None")
    parser.add_argument("-F", default=default_F, help="Feature list to use, or comma separated list of features ({})".format(default_F))
    parser.add_argument("outfile", type=str, help="Output (tsv) file")
    args = parser.parse_args()

    features_string = args.F

    if "," in features_string:
        tmp = features_string.split(",")
        features = [f for f in tmp if f]
    else:
        features = getattr(features, features_string)

    if args.T is None:
        a2target, a2bias, a2url = (None, None, None)
        prefix_pipeline = []
    else:
        print("Loading targets")
        a2target, a2bias, a2url = utils.load_targets_file(args.T, cache=None)
        prefix_pipeline = [processingresources.PrAddTarget(a2target, a2bias, a2url)]


    pipeline = create_pipeline()
    pipeline = prefix_pipeline + pipeline

    with open(args.outfile, "wt", encoding="utf8") as outstream:
        pipeline.append(processingresources.PrArticle2Line(outstream, features, addtargets=True))
        print("Pipeline:")
        for p in pipeline:
            print(p)
        ntotal, nerror = utils.process_articles_xml(args.A, pipeline)
        print("Total processed articles:", ntotal)
        print("Errors (could be >1 per article):", nerror)

if __name__ == "__main__":
    main()
