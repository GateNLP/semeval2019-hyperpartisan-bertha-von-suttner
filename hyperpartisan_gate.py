# hyperpartisan_gate.py
# This (Python) script is intended to be launched via
# gateplugin-python https://github.com/GateNLP/gateplugin-python

import os
import sys

import hyperpartisan

noisy = False
if os.environ.get("USER") == "ac1xdrj":
    noisy = True

if noisy: print("hyper starting", file=sys.stderr, flush=True)

if noisy:
    with open(__file__) as me:
        sys.stderr.write(me.read())

from gate import executable

if noisy: print("imported gate", file=sys.stderr)

print("Python version", sys.version, file=sys.stderr, flush=True)

# This version adds a dummy annotation,
# it doesn't do any useful work yet.
# But does demonstrate the interoperation.

@executable
def hyper(document, outputAS):
    score = hyperpartisan.hyperpartisan(document.text)
    document.annotationSets[outputAS].add(
        0, len(document.text), "hyperpartisan",
        dict(hyperpartisan_probability=score)
    )


if __name__ == "__main__":
    hyper.start()
