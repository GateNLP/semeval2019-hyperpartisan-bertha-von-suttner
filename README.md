# SemEval2019/Task4 Team Bertha-von-Suttner submission

This is the code for the [SemEval 2019 Task 4, Hyperpartisan News Detection](https://pan.webis.de/semeval19/semeval19-web/)
submitted by team `Bertha von Suttner`:
* [Ye Jiang](https://ye-jiang.github.io/)
* [Johann Petrak](http://johann-petrak.github.io) 
* [Xingyi Song](http://staffwww.dcs.shef.ac.uk/people/X.Song/)

All are members of the [GATE](https://gate.ac.uk) team of the [University of Sheffield Natural Language Processing group](https://www.sheffield.ac.uk/dcs/research/groups/nlp)

The model created with this was the winning entry, see the public leaderboard (sort by accuracy column, descending):
https://www.tira.io/task/hyperpartisan-news-detection/dataset/pan19-hyperpartisan-news-detection-by-article-test-dataset-2018-12-07/

## Preparation / Requirements

* Python 3.6 (Anaconda will work best)
* Tensorflow version 1.12.0
* Keras version 2.2.4
* PyTorch version 0.4.1
* Spacy version 2.0.16

Preparation steps:
* create a directory `elmo` and store the ELMo model files in that directory:
  * https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5
  * https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json
* create a directory `data` and save the by-article training files into it:
  * articles-training-byarticle-20181122.xml
  * ground-truth-training-byarticle-20181122.xml 


## Training

Run the following steps

* Convert the XML file into a tsv file with one article per line:
  * `python Preprocessing/xml2line.py -A data/articles-training-byarticle-20181122.xml -T data/ground-truth-training-byarticle-20181122.xml -F article_sent,title_sent work/train.text.tsv`
* Convert the tsv file containing text into a tsv file containing elmo embeddings:
  * If you have a GPU: `python Preprocessing/line2elmo2.py -g -l 100  work/train.text.tsv work/train.elmo.tsv`
  * Otherwise: `python Preprocessing/line2elmo2.py -l 100 work/train.text.tsv work/train.elmo.tsv`
  If you get problems with the GPU memory or RAM, use the -b option to reduce the batch size
* Make sure the directory `saved_models` does not contain any model files from previous runs:
  * `rm saved_models/*.hdf5`
* Train the actual model: 
  `python CNN_elmo.py work/train.elmo.tsv`
  This will create a number of model files in the `saved_models` directory. The file names contain the validation accuracy.


## Application

* Convert the XML to tsv:
  `python Preprocessing/xml2line.py -A $TESTXMLFILE -F article_sent,title_sent work/test.text.tsv`
* Convert the text to elmo embeddings:
  * If you have a GPU: `python Preprocessing/line2elmo2.py -g -l 100  work/test.text.tsv work/test.elmo.tsv`
  * Otherwise: `python Preprocessing/line2elmo2.py -l 100 work/test.text.tsv work/test.elmo.tsv`
* Run the actual application of the model ensemble
  * `./ensemble_pred.sh work/test.elmo.tsv work/test.preds.txt` 
