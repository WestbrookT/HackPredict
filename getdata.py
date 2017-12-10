from gensim.models import KeyedVectors
from nltk import word_tokenize as tokenize
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense




def load_w2v():
    _fname = "GoogleNews-vectors-negative300.bin"
    w2vModel = KeyedVectors.load_word2vec_format(_fname, binary=True)
    return w2vModel

w2v = load_w2v()
print('loaded vectors')





import numpy as np
import os

def get_data_file_list():
    pos_paths = os.listdir('winners')
    neg_paths = os.listdir('losers')

    np.random.shuffle(neg_paths)
    neg_paths = neg_paths[:8000]

    pos_examples = []
    neg_examples = []

    for path in pos_paths:

        pos_examples.append('winners/{}'.format(path))

    for path in neg_paths:

        neg_examples.append('losers/{}'.format(path))

    return pos_examples, neg_examples

def load_file(path):

    text = ''
    with open(path, 'r') as f:
        text = f.read()



    tokens = tokenize(text)



    filteredTokens = filter(lambda x: x in w2v.vocab, tokens)
    filteredTokens = list(filteredTokens)




    output = []

    # for word in filteredTokens:
    #
    #
    #     output.append(w2v.word_vec(word))

    output = list(map(lambda word: w2v.word_vec(word), filteredTokens))

    if len(output) > timesteps:
        output = output[:timesteps]
    else:

        while len(output) < timesteps:
            output.append([0.0] * 300)



    return np.array(output)
from numpy.random import shuffle, seed
def shuffle_data(inputs, outputs, s=1):
    seed(s)
    shuffle(inputs)
    seed(s)
    shuffle(outputs)


def load(index, amt):
    pos_paths, neg_paths = get_data_file_list()

    inputs = []
    outputs = []



    pos_index = randint(0, len(pos_paths)-amt)

    pos_amt = amt if pos_index + amt < len(pos_paths) else len(pos_paths) - 1


    for path in pos_paths[pos_index:pos_index+pos_amt]:

        inputs.append(load_file(path))
        outputs.append(np.array([1, 0]))

    neg_index = randint(0, len(neg_paths)-amt)

    neg_amt = amt if neg_index + amt < len(neg_paths) else len(neg_paths) - 1

    for path in neg_paths[neg_index:neg_index+neg_amt]:

        inputs.append(load_file(path))
        outputs.append(np.array([0, 1]))

    inputs, outputs = np.array(inputs), np.array(outputs)

    shuffle_data(inputs, outputs)
    return inputs, outputs


class DataIterator:
    def __init__(self, batch_size = 1000):
        pos_files, neg_files = get_data_file_list()
        self.pos_iter = iter(pos_files)
        self.neg_iter = iter(neg_files)
        self.batchSize = batch_size

    def get_next(self):
        vectors = []
        values = []
        while (len(vectors) < self.batchSize):

            file = next(self.pos_iter, None)
            if file == None:
                break
            vec = load_file(file)
            vectors.append(vec)
            values.append([1,0])

            file = next(self.neg_iter, None)
            if file == None:
                break
            vec = load_file(file)
            vectors.append(vec)
            values.append([0,1])
        return np.array(vectors), np.array(values)



# def train():
#     timesteps = 100
#     dimensions = 300
#     batch_size = 64
#     epochs_number = 40
#     model = Sequential()
#     model.add(LSTM(200,  input_shape=(timesteps, dimensions),  return_sequences=False))
#     model.add(Dropout(0.2))
#     model.add(Dense(2, input_dim=200, activation='softmax'))
#     model.compile(loss="categorical_crossentropy", optimizer='rmsprop')
#     fname = 'weights/keras-lstm.h5'
#     model.load_weights(fname)
#     cbks = [callbacks.ModelCheckpoint(filepath=fname, monitor='val_loss', save_best_only=True),
#             callbacks.EarlyStopping(monitor='val_loss', patience=3)]
#     train_iterator = DataIterator(train_data_path, sys.maxint)
#     test_iterator = DataIterator(test_data_path, sys.maxint)
#     train_X, train_Y = train_iterator.get_next()
#     test_X, test_Y = test_iterator.get_next()
#     model.fit(train_X, train_Y, batch_size=batch_size, callbacks=cbks, nb_epoch=epochs_number,
#               show_accuracy=True, validation_split=0.25, shuffle=True)
#     loss, acc = model.evaluate(test_X, test_Y, batch_size, show_accuracy=True)
#     print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

timesteps = 250
neurons = 300
batch_size = 64
epochs = 30

model = Sequential()

model.add(LSTM(200, input_shape=(timesteps, neurons), return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(200))
model.add(Dropout(.2))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])





print(model.summary())
#model.load_weights('weights2.h5')

from random import randint
for i in range(0, 50):

    print('\n\n\n\n\nCurrent Iteration\n\n\n\n\n', i)


    data_in, data_out = load(0, 600)

    data_in = data_in.reshape(len(data_in), timesteps, neurons)
    data_out = data_out.reshape(len(data_out), 2)

    model.fit(data_in, data_out, batch_size=batch_size, nb_epoch=epochs, shuffle=True, validation_split=.25)

    model.save('weights4.h5')














