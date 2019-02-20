#!/usr/bin/env python
'''
NOTE: This uses the alannlp PyTorch implementation of Elmo!
Process a line corpus and convert text to elmo embeddings, save as json array of sentence vectors.
This expects the sents corpus which has fields title, article, domains. and treates
title as the first sentence, then splits the article sentences. The domains are ignored
'''

import argparse
from allennlp.commands.elmo import ElmoEmbedder
import os
import json
import math
import numpy as np

configs = {
  "small": "elmo_2x1024_128_2048cnn_1xhighway_options.json",
  "medium": "elmo_2x2048_256_2048cnn_1xhighway_options.json",
  "original5b": "elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json",
  "original": "elmo_2x4096_512_2048cnn_2xhighway_options.json"
}

models = {
  "small": "elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5",
  "medium": "elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5",
  "original5b": "elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5",
  "original": "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

}

if __name__ == '__main__':

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
    use_gpu = args.g
    model = os.path.join("elmo", models[args.m])
    config = os.path.join("elmo", configs[args.m])
    concat = args.concat
    maxtoks = args.maxtoks
    maxsents = args.maxsents

    print("Loading model {}...".format(args.m))
    if use_gpu:
        device = 0
    else:
        device = -1
    elmo = ElmoEmbedder(options_file=config, weight_file=model, cuda_device=device)


    print("Processing lines...")
    with open(infile, "rt", encoding="utf8") as inp:
        nlines = 0
        with open(outfile, "wt", encoding="utf8") as outp:
            for line in inp:
                fields = line.split("\t")
                title = fields[5]
                tmp = fields[4]
                tmp = tmp.split(" <splt> ")[:maxsents]
                sents = [title]
                sents.extend(tmp)
                # now processes the sents in batches
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

                # print("Result lines:", len(outs))
                outs = [a.tolist() for a in outs]
                print(fields[0], fields[1], fields[2], fields[3], json.dumps(outs), sep="\t", file=outp)
                nlines += 1
                if nlines % every == 0:
                    print("Processed lines:", nlines)
        print("Total processed lines:", nlines)
