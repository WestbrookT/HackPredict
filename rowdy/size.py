from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

from nltk import word_tokenize as tokenize

from gensim.models import KeyedVectors

import numpy as np
def load_model(path):
    timesteps = 200
    neurons = 300
    batch_size = 64
    epochs = 25

    model = Sequential()

    model.add(LSTM(200, input_shape=(timesteps, neurons), return_sequences=True))
    model.add(Dropout(.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model.load_weights(path)
    return model


def load_overall_model(path):
    timesteps = 200
    neurons = 300
    batch_size = 64
    epochs = 25

    model = Sequential()

    model.add(LSTM(200, input_shape=(timesteps, neurons), return_sequences=False))
    model.add(Dropout(.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model.load_weights(path)
    return model

a = """
Play Penalty Shootout and feel the thrill at the mark!Are you an ice cold goal getter and anticipating goalkeeper? Find out in this thrilling game of penalty shootouts, where you can become the matchwinner or the big loser!Right, left or straight throught the middle - it's your decision and your chanceto win the duel. It's not just a random corner, it's your opponents decision where to shoot or where to dive for the ball. Take up the challenge and play against real opponents! Winning games frequently? Get it on and show everybody that you are the best penalty scorer and the quickest goalkeeper of all times! Eternalize your name in the highscores! Play Penalty Shootout by dutyfarm.com
"""


def load_w2v():
    _fname = "GoogleNews-vectors-negative300.bin"
    w2vModel = KeyedVectors.load_word2vec_format(_fname, binary=True)
    return w2vModel

w2v = load_w2v()

def prepare_text(text):
    tokens = tokenize(text)

    original = []

    filteredTokens = []

    for x in tokens:
        if x in w2v.vocab:
            filteredTokens.append(x)
            original.append(0.0)
        else:
            original.append(None)





    # for word in filteredTokens:
    #
    #
    #     output.append(w2v.word_vec(word))

    output = list(map(lambda word: w2v.word_vec(word), filteredTokens))


    if len(output) > 200:
        output = output[:200]
    else:

        while len(output) < 200:
            output.append([0.0] * 300)

    return np.array([output]), original

def fix_holes(original, predictions):
    predictions = predictions[0]

    pred_index = 0

    for i, spot in enumerate(original):

        if spot is None:
            original[i] = float((predictions[pred_index-1][0] + predictions[pred_index][0])/2)
        else:
            original[i] = float(predictions[pred_index][0])
            pred_index += 1

    return original



outs = []

model = load_model('weights3.h5')



def prepare_tokens(tokens):



    original = []

    filteredTokens = []

    for x in tokens:
        if x in w2v.vocab:
            filteredTokens.append(x)
            original.append(0.0)
        else:
            original.append(None)

    # for word in filteredTokens:
    #
    #
    #     output.append(w2v.word_vec(word))

    output = list(map(lambda word: w2v.word_vec(word), filteredTokens))

    if len(output) > 200:
        output = output[:200]
    else:

        while len(output) < 200:
            output.append([0.0] * 300)

    return np.array([output]), original

overall_model = load_overall_model('weights3.h5')


def predict(text):

    tokens = tokenize(text)

    i = 0
    prep, origin = prepare_tokens(tokens[i:i+200])

    pred = model.predict(prep)

    origin = fix_holes(origin, pred)

    score = overall_model.predict(prep)[0][0]
    cnt = 1

    while i*200 < len(origin):

        p, o = prepare_tokens(tokens[i:i+200])
        i += 1
        pred = model.predict(p)

        o = fix_holes(o, pred)

        origin = origin + o
        score += overall_model.predict(prep)[0][0]
        cnt += 1


    return origin, score/cnt





#print(prepare_text(a), model.predict(pa))