# hyperpartisan_gate.py
# This (Python) script is intended to be launched via
# gateplugin-python https://github.com/GateNLP/gateplugin-python

import os
import sys

from gate import executable

import hyperpartisan

@executable
def hyper(document, outputAS):
    score = hyperpartisan.hyperpartisan(document.text)
    document.annotationSets[outputAS].add(
        0, len(document.text), "hyperpartisan",
        dict(hyperpartisan_probability=score)
    )


if __name__ == "__main__":
    hyper.start()
