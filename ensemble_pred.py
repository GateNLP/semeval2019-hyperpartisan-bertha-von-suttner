import ast
import optparse

import keras.models
import numpy as np

from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import sequence
from keras.layers import Average, Input


def toEvaluationFormat(all_doc_ids, all_prediction):
    evaluationFormatList = []
    for i in range(len(all_doc_ids)):
        current_doc_id = all_doc_ids[i]
        current_prob = all_prediction[i][0]
        #current_prob = all_prediction[i]
        if current_prob > 0.5:
            current_pred = 'true'
        else:
            current_prob = 1 - current_prob
            current_pred = 'false'
        evaluationFormat = str(current_doc_id) + ' ' + str(current_pred) + ' ' + str(current_prob) + '\n'
        evaluationFormatList.append(evaluationFormat)
    return evaluationFormatList

def load_data(data_path, max_len=200):
    data = []
    l = []
    ids = []
    i = 0
    l_encoder = LabelEncoder()
    with open(data_path, 'rb') as inf:
        for line in inf:
            gzip_fields = line.decode('utf-8').split('\t')
            gzip_id = gzip_fields[0]
            gzip_label = gzip_fields[1]
            elmo_embd_str = gzip_fields[4].strip()
            elmo_embd_list = ast.literal_eval(elmo_embd_str)
            elmo_embd_array = np.array(elmo_embd_list)
            padded_seq = sequence.pad_sequences([elmo_embd_array], maxlen=max_len, dtype='float32')[0]
            data.append(padded_seq)
            l.append(gzip_label)
            ids.append(gzip_id)
            i += 1
            print(i)
    label = l_encoder.fit_transform(l)
    return np.array(data), np.array(label), np.array(ids)


def ensemble_model(models, model_input):
    outputs = [model(model_input) for model in models]
    y = Average()(outputs)

    model = keras.models.Model(model_input, y, name='ensemble')

    return model


def load_models(paths):
    """
    Load keras models from the paths, returning a list of
    models.
    """

    models = []

    for i, path in enumerate(paths, start=1):
        model = keras.models.load_model(path)
        model.name = "model{}".format(i)
        models.append(model)

    return models


def create_ensemble_from_files(paths):
    """
    Create an ensemble model from the keras models located at
    `paths`, which is a sequence of pathnames.
    """

    models = load_models(paths)
    model_input = Input(shape=models[0].input_shape[1:], dtype='float32')
    return ensemble_model(models, model_input)



def main():
    parser = optparse.OptionParser()
    parser.add_option("--inputTSV", help="load saved cache", type=str)
    parser.add_option("--output", help="load saved cache", type=str)
    parser.add_option("--saved_model1", help="load saved cache", type=str)
    parser.add_option("--saved_model2", help="load saved cache", type=str)
    parser.add_option("--saved_model3", help="load saved cache", type=str)

    options, arguments = parser.parse_args()

    max_len = 200
    embed_size = 1024
    seed = 7

    x_data, y_data, doc_id = load_data(options.inputTSV,max_len=max_len)

    ensemble = create_ensemble_from_files(
        [options.saved_model1,
        options.saved_model2,
        options.saved_model3])

    pred = ensemble.predict(x_data)

    all_pred = toEvaluationFormat(doc_id, pred)
    with open(options.output, 'w') as fo:
        for item in all_pred:
            fo.write(item)

if __name__ == "__main__":
    main()
