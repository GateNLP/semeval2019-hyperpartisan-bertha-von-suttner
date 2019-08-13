#!/usr/bin/env python3

"""
hyperpartisan text-to-process

hyperpartisan is a command line tool that evaluates an ensemble
model that is trained to estimate the "hyperpartisan" category
of a news document.
"""

import json
# https://docs.python.org/3.5/library/sys.html
import sys
# https://docs.python.org/3.5/library/xml.etree.elementtree.html
import xml.etree.ElementTree

from Preprocessing import xml2line
from Preprocessing import utils


def hyperpartisan(text):
    pipeline = xml2line.create_pipeline()

    article = {
        "id": "command line text",
        "xml": text,
        "et": xml.etree.ElementTree.fromstring(
            '<article title="command line text" />')
    }
    nerror = utils.run_pipeline(pipeline, article)
    article["nerror"] = nerror

    del article["et"]

    return article


def main(argv=None):
    if argv is None:
        argv = sys.argv

    arg = argv[1:]

    text = " ".join(arg)
    result = hyperpartisan(text)
    json.dump(result, sys.stdout, indent=2)


if __name__ == "__main__":
    sys.exit(main())
