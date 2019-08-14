#!/usr/bin/env python3

"""
hyperpartisan text-to-process

hyperpartisan is a command line tool that evaluates an ensemble
model that is trained to estimate the "hyperpartisan" category
of a news document.
"""



# https://docs.python.org/3.5/library/glob.html
import glob
# https://docs.python.org/3.5/library/json.html
import json
# https://docs.python.org/3.5/library/sys.html
import sys
# https://docs.python.org/3.5/library/xml.etree.elementtree.html
import xml.etree.ElementTree

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

# Avoid "Using TensorFlow backend" stderr stutter.
# Thanks, random GitHub person: https://github.com/keras-team/keras/issues/1406#issuecomment-466135951
import contextlib
with contextlib.redirect_stderr(open(os.devnull, "w")):
    import keras.preprocessing

import numpy
import sklearn.preprocessing

from Preprocessing import utils
from Preprocessing import xml2line
from Preprocessing import line2elmo2

import ensemble_pred


def hyperpartisan(text):
    """
    Given the text of an article, as a Python string,
    return a probability that the article is hyperpartisan.
    As estimated by the models in the directory "prediction_models".
    """

    article = {
        "id": "command line text",
        "xml": text,
    }

    add_article_sent(article)

    vectors = elmo_embedding(article)

    prediction = apply_ensemble_model(vectors)

    return prediction


def add_article_sent(article):
    """
    To the article represented by `article`, add
    a "article_sent" key which is the tokenised text
    of the article joined with spaces into a single string.
    (and perform other processing necessary to get to this point)
    """

    article["et"] = xml.etree.ElementTree.fromstring(
            '<article title="command line text" />')

    pipeline = xml2line.create_pipeline()
    nerror = utils.run_pipeline(pipeline, article)
    article["nerror"] = nerror

    del article["et"]

    return article


def elmo_embedding(article):
    """
    Convert an article (using the text in its "article_sent" key)
    to embedded ELMo vectors.

    A 2-dimensional array is returned,
    with one row per sentence.
    """

    sents = article["article_sent"].split(" <splt> ")

    elmo = line2elmo2.create_elmo("original", False)
    vectors = line2elmo2.elmo_one_article(elmo, sents, 200, 200, batchsize=50,)
    return vectors


def apply_ensemble_model(vectors):
    """
    Given a list of vectors representing an article,
    apply the ensemble model and return a score.
    """

    model = ensemble_pred.create_ensemble_from_files(
      sorted(glob.glob("prediction_models/*.hdf5")))

    padded_vectors = keras.preprocessing.sequence.pad_sequences(
      [vectors], maxlen=200, dtype='float32')[0]

    data = numpy.array([padded_vectors])

    predicts = model.predict(data)
    return predicts[0].tolist()[0]


def main(argv=None):
    if argv is None:
        argv = sys.argv

    arg = argv[1:]

    text = " ".join(arg)
    score = hyperpartisan(text)
    print(json.dumps({"hyperpartisan_probability": score}))


if __name__ == "__main__":
    sys.exit(main())
