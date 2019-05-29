#!/bin/bash

inelmo="$1"
outpreds="$2"

if [[ "x$outpreds" == "x" ]]
then
  echo Need two parameters, inpout elmo embeddings file and output predictions file
  exit 1
fi

models=`find saved_models/ -name '*.hdf5' | sort -r | head -3`
amodels=($models)
m1=${amodels[0]}
m2=${amodels[1]}
m3=${amodels[2]}

echo 'Running: ' python ensemble_pred.py --saved_model1 $m1 --saved_model2 $m2 --saved_model3 $m3 --inputTSV $inelmo --output $outpreds
KERAS_BACKEND=tensorflow python ensemble_pred.py --saved_model1 $m1 --saved_model2 $m2 --saved_model3 $m3 --inputTSV $inelmo --output $outpreds

