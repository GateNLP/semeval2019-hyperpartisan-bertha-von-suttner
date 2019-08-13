#!/usr/bin/env python
'''
NOTE: This uses the alannlp PyTorch implementation of Elmo!
Process a line corpus and convert text to elmo embeddings, save as json array of sentence vectors.
This expects the sents corpus which has fields title, article, domains. and treates
title as the first sentence, then splits the article sentences. The domains are ignored
'''

import argparse
import os
import json
import math
# https://docs.python.org/3.5/library/pathlib.html
import pathlib

from allennlp.commands.elmo import ElmoEmbedder
import numpy as np

configs = {
  "small": "elmo_2x1024_128_2048cnn_1xhighway_options.json",
  "medium": "elmo_2x2048_256_2048cnn_1xhighway_options.json",
  "original5b": "elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json",
  "original": "elmo_2x4096_512_2048cnn_2xhighway_options.json"
}

weights = {
  "small": "elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5",
  "medium": "elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5",
  "original5b": "elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5",
  "original": "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

}

def main():
    default_m = "original"
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, help="Input file, should be in sent1 format")
    parser.add_argument("outfile", type=str, help="Output file, contains standard cols 0.3, plus json vectors")
    parser.add_argument("-b", type=int, default=50, help="Batchsize (50)")
    parser.add_argument("-l", type=int, default=1000, help="Log every (1000)")
    parser.add_argument("--maxtoks", type=int, default=200, help="Maximum number of tokens per sentence to use (200)")
    parser.add_argument("--maxsents", type=int, default=200, help="Maximum number of sentences per article to use (200)")
    parser.add_argument("-m", type=str, default=default_m,
                        help="Model (small, medium, original, original5b ({})".format(default_m))
    parser.add_argument("-g", action='store_true', help="Use the GPU (default: don't)")
    parser.add_argument("--concat", action='store_true', help="Concatenate representations instead of averaging")
    args = parser.parse_args()

    outfile = args.outfile
    infile = args.infile
    batchsize = args.b
    every = args.l
    concat = args.concat
    maxtoks = args.maxtoks
    maxsents = args.maxsents

    print("Loading model {}...".format(args.m))
    elmo = create_elmo(args.m, args.g)

    print("Processing lines...")
    with open(infile, "rt", encoding="utf8") as inp:
        with open(outfile, "wt", encoding="utf8") as outp:
            for nlines, line in enumerate(inp):
                fields = line.split("\t")
                title = fields[5]
                tmp = fields[4]
                tmp = tmp.split(" <splt> ")
                sents = [title]
                sents.extend(tmp)

                outs = elmo_one_article(
                    elmo, sents, maxsents,
                    maxtoks, batchsize, concat=concat)

                # print("Result lines:", len(outs))
                outs = [a.tolist() for a in outs]
                print(fields[0], fields[1], fields[2], fields[3], json.dumps(outs), sep="\t", file=outp)
                if (1+nlines) % every == 0:
                    print("Processed lines:", nlines)
        print("Total processed lines:", nlines+1)


def create_elmo(model, gpu):
    """
    Create an ELMo embedder.
    `model` is a string specifying the model
    (it's an index into the configs and weights dicts in this module);
    `gpu` is True if CUDA device 0 should be used.
    """

    if gpu:
        device = 0
    else:
        device = -1

    elmo_path = pathlib.Path("elmo")
    config_path = elmo_path / configs[model]
    weight_path = elmo_path / weights[model]
    elmo = ElmoEmbedder(
      options_file=str(config_path),
      weight_file=str(weight_path),
      cuda_device=device)
    return elmo


def elmo_one_article(elmo, sents, maxsents, maxtoks, batchsize, concat=False):
    """
    Convert a single article into ELMo embeddings
    (using the embedder `elmo`).

    `sents` is the article is modelled as a sequence (list) of sentences;
    each sentence being a single string that
    has already been tokenised with tokens separated by spaces.

    The first sentence is assumed to be a title;
    subsequent to that, only `maxsents` sentences are considered.
    Within each sentence, only maxtoks tokens are considered.

    The result is one vector per sentence;
    having combined the vectors for a word by
    either concatenation (if concat is True) or averaging (the default),
    then averaging each word vector in a sentence.
    """

    sents = sents[:1+maxsents]

    # processes the sents in batches
    outs = []
    # unlike the tensorflow version we can have dynamic batch sizes here!
    for batchnr in range(math.ceil(len(sents)/batchsize)):
        fromidx = batchnr * batchsize
        toidx = (batchnr+1) * batchsize
        actualtoidx = min(len(sents), toidx)
        # print("Batch: from=",fromidx,"toidx=",toidx,"actualtoidx=",actualtoidx)
        sentsbatch = sents[fromidx:actualtoidx]
        sentsbatch = [s.split()[:maxtoks] for s in sentsbatch]
        for s in sentsbatch:
            if len(s) == 0:
                s.append("")  # otherwise we get a shape (3,0,dims) result
        ret = list(elmo.embed_sentences(sentsbatch))
        # the ret is the original representation of three vectors per word
        # We first combine per word through concatenation or average, then average
        if concat:
            ret = [np.concatenate(x, axis=1) for x in ret]
        else:
            ret = [np.average(x, axis=1) for x in ret]
        # print("DEBUG tmpembs=", [l.shape for l in tmpembs])
        ret = [np.average(x, axis=0) for x in ret]
        # print("DEBUG finalreps=", [l.shape for l in finalreps])
        outs.extend(ret)
    return outs


if __name__ == '__main__':
    main()
