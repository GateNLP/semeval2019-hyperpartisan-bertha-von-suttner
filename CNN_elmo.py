import ast
import numpy as np
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from keras.layers import Input, Flatten, Dense, Activation,Average
from keras.layers import Concatenate,Dropout,Conv1D,MaxPooling1D,BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from argparse import ArgumentParser


def load_elmo(path, max_len=200):
    '''
    load ELMo embedding from tsv file.
    :param path: tsv file path.
    :param to_pickle: Convert elmo embeddings to .npy file, avoid read and pad every time.
    :return: elmo embedding and its label.
    '''
    X = []
    label = []
    ids = []
    i = 0
    l_encoder = LabelEncoder()
    with open(path, 'rb') as inf:
        for line in inf:
            gzip_fields = line.decode('utf-8').split('\t')
            gzip_id = gzip_fields[0]
            gzip_label = gzip_fields[1]
            elmo_embd_str = gzip_fields[4].strip()
            elmo_embd_list = ast.literal_eval(elmo_embd_str)
            elmo_embd_array = np.array(elmo_embd_list)
            padded_seq = sequence.pad_sequences([elmo_embd_array], maxlen=max_len, dtype='float32')[0]
            X.append(padded_seq)
            label.append(gzip_label)
            ids.append(gzip_id)
            i += 1
            print(i)
    Y = l_encoder.fit_transform(label)

    return np.array(X), np.array(Y), np.array(ids)

def conv1d(max_len, embed_size):
    '''
    CNN without Batch Normalisation.
    :param max_len: maximum sentence numbers, default=200
    :param embed_size: ELMo embeddings dimension, default=1024
    :return: CNN without BN model
    '''
    filter_sizes = [2, 3, 4, 5, 6]
    num_filters = 128
    drop = 0.5
    inputs = Input(shape=(max_len,embed_size), dtype='float32')

    conv_0 = Conv1D(num_filters, kernel_size=(filter_sizes[0]))(inputs)
    act_0 = Activation('relu')(conv_0)
    conv_1 = Conv1D(num_filters, kernel_size=(filter_sizes[1]))(inputs)
    act_1 = Activation('relu')(conv_1)
    conv_2 = Conv1D(num_filters, kernel_size=(filter_sizes[2]))(inputs)
    act_2 = Activation('relu')(conv_2)
    conv_3 = Conv1D(num_filters, kernel_size=(filter_sizes[3]))(inputs)
    act_3 = Activation('relu')(conv_3)
    conv_4 = Conv1D(num_filters, kernel_size=(filter_sizes[4]))(inputs)
    act_4 = Activation('relu')(conv_4)

    maxpool_0 = MaxPooling1D(pool_size=(max_len - filter_sizes[0]))(act_0)
    maxpool_1 = MaxPooling1D(pool_size=(max_len - filter_sizes[1]))(act_1)
    maxpool_2 = MaxPooling1D(pool_size=(max_len - filter_sizes[2]))(act_2)
    maxpool_3 = MaxPooling1D(pool_size=(max_len - filter_sizes[3]))(act_3)
    maxpool_4 = MaxPooling1D(pool_size=(max_len - filter_sizes[4]))(act_4)

    concatenated_tensor = Concatenate()([maxpool_0, maxpool_1, maxpool_2, maxpool_3, maxpool_4])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=1, activation='sigmoid')(dropout)

    model = Model(inputs=inputs, outputs=output)
    #model = multi_gpu_model(model, gpus=gpus)
    model.summary()
    model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer='adam')
    return model

def conv1d_BN(max_len, embed_size):
    '''
    CNN with Batch Normalisation.
    :param max_len: maximum sentence numbers, default=200
    :param embed_size: ELMo embeddings dimension, default=1024
    :return: CNN with BN model
    '''
    filter_sizes = [2, 3, 4, 5, 6]
    num_filters = 128
    inputs = Input(shape=(max_len,embed_size), dtype='float32')
    conv_0 = Conv1D(num_filters, kernel_size=(filter_sizes[0]))(inputs)
    act_0 = Activation('relu')(conv_0)
    bn_0 = BatchNormalization(momentum=0.7)(act_0)

    conv_1 = Conv1D(num_filters, kernel_size=(filter_sizes[1]))(inputs)
    act_1 = Activation('relu')(conv_1)
    bn_1 = BatchNormalization(momentum=0.7)(act_1)

    conv_2 = Conv1D(num_filters, kernel_size=(filter_sizes[2]))(inputs)
    act_2 = Activation('relu')(conv_2)
    bn_2 = BatchNormalization(momentum=0.7)(act_2)

    conv_3 = Conv1D(num_filters, kernel_size=(filter_sizes[3]))(inputs)
    act_3 = Activation('relu')(conv_3)
    bn_3 = BatchNormalization(momentum=0.7)(act_3)

    conv_4 = Conv1D(num_filters, kernel_size=(filter_sizes[4]))(inputs)
    act_4 = Activation('relu')(conv_4)
    bn_4 = BatchNormalization(momentum=0.7)(act_4)

    maxpool_0 = MaxPooling1D(pool_size=(max_len - filter_sizes[0]))(bn_0)
    maxpool_1 = MaxPooling1D(pool_size=(max_len - filter_sizes[1]))(bn_1)
    maxpool_2 = MaxPooling1D(pool_size=(max_len - filter_sizes[2]))(bn_2)
    maxpool_3 = MaxPooling1D(pool_size=(max_len - filter_sizes[3]))(bn_3)
    maxpool_4 = MaxPooling1D(pool_size=(max_len - filter_sizes[4]))(bn_4)

    concatenated_tensor = Concatenate()([maxpool_0, maxpool_1, maxpool_2, maxpool_3, maxpool_4])
    flatten = Flatten()(concatenated_tensor)
    output = Dense(units=1, activation='sigmoid')(flatten)

    model = Model(inputs=inputs, outputs=output)
    #model = multi_gpu_model(model, gpus=gpus)
    model.summary()
    model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer='adam')
    return model


parser = ArgumentParser()
parser.add_argument("inputTSV", help="Elmo format input file")
args = parser.parse_args()

seed = 7
max_len = 200
embed_size = 1024

x_data, y_data, ids = load_elmo(args.inputTSV, max_len=max_len)

# sk-learn provides 10-fold CV wrapper.
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# list of validation accuracy from each fold.
cvscores = []
# counter to tell which fold.
i = 0
for train, test in kfold.split(x_data, y_data):
    i += 1
    print("current fold is : %s " % i)
    model = conv1d_BN(max_len, embed_size)
    checkpoints = ModelCheckpoint(filepath='./saved_models/BNCNN_vacc{val_acc:.4f}_f%s_e{epoch:02d}.hdf5' % str(i),
                                  verbose=1,monitor='val_acc', save_best_only=True)
    history = model.fit(x_data[train],y_data[train],batch_size=32,verbose=1, epochs=30,
              validation_data=[x_data[test],y_data[test]],callbacks=[checkpoints])
    # use the last validation accuracy from the 30 epochs
    his_val = history.history['val_acc'][-1]
    cvscores.append(his_val)
    # clear memory
    K.clear_session()
print("Final score: %.4f%% (+/- %.4f%%)" % (np.mean(cvscores), np.std(cvscores)))
