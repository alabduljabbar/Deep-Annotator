import pickle
from keras.models import Sequential
from keras.models import model_from_json
import keras
import numpy as np
import tensorflow as tf
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

f = open("../Pickles/labels.pkl","rb")
labels = pickle.load(f)
# f = open("../Pickles/bagOfWords.pkl","rb")
# f = open("../Pickles/bagOfWords_u.pkl","rb")
# f = open("../Pickles/ngram1_4.pkl","rb")
# f = open("../Pickles/ngram2_4.pkl","rb")
# f = open("../Pickles/ngram2_4_u.pkl","rb")
# f = open("../Pickles/wordEmbeddings.pkl","rb")
# f = open("../Pickles/wordEmbeddingsLink1.pkl","rb")
# f = open("../Pickles/wordEmbeddingsLink2.pkl","rb")
# f = open("../Pickles/wordEmbeddingsLink3.pkl","rb")
f = open("../Pickles/wordEmbeddingsLink4.pkl","rb")


BoW = pickle.load(f)

count = [0,0,0,0,0,0,0,0,0]

for i in range(len(labels)):
    for j in range(len(labels[i])):
        if labels[i][j] ==1:
            count[j] += 1
print(len(labels))
print(count)


results = []
for i in range(9):

    print("BoW Category",i,"Classification:")
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    countT = 0
    countF = 0
    for j in range(len(labels)):
        if labels[j][i] == 1 :
            countT += 1
            if countT < 0.8*count[i]:
                y_train.append([0,1])
                x_train.append(BoW[j])
            else :
                y_test.append([0,1])
                x_test.append(BoW[j])
        else :
            countF += 1
            if countF < 0.8*(len(labels)-count[i]):
                y_train.append([1,0])
                x_train.append(BoW[j])
            else :
                y_test.append([1,0])
                x_test.append(BoW[j])

    x_train = np.asarray(x_train)
    x_train = x_train.reshape((len(x_train),len(x_train[0]),1))
    x_test = np.asarray(x_test)
    x_test = x_test.reshape((len(x_test),len(x_test[0]),1))
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    filter = 32
    # create model
    model = Sequential()
    model.add(keras.layers.Conv1D(filter,3,padding="same",activation="relu"))
    model.add(keras.layers.Conv1D(filter,3,padding="valid",activation="relu"))
    # model.add(keras.layers.Dense(256, activation='relu'))
    # model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.MaxPooling1D())
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Conv1D(filter*2,3,padding="same",activation="relu"))
    model.add(keras.layers.Conv1D(filter*2,3,padding="valid",activation="relu"))
    # model.add(keras.layers.Dense(256, activation='relu'))
    # model.add(keras.layers.Dense(256, activation='relu'))
    # model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.MaxPooling1D())
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(2, activation="softmax"))
    # model.add(keras.layers.Dense(2, activation="sigmoid"))

    # Compile model
    # model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])

    # Fit the model
    model.fit(x_train,y_train, epochs=30, batch_size=64)
    print('Evaluating model.')
    eval_results = model.evaluate(x_test,
                                  y_test)
    print(eval_results)
    print("______________________________________")
    tp = 0
    allp = 0
    tn = 0
    alln = 0
    preds = model.predict(x_test)
    for j in range(len(preds)):
        if y_test[j][1] == 1 :
            allp += 1
            if np.argmax(preds[j]) == 1:
                tp +=1
        if y_test[j][0] == 1 :
            alln+=1
            if np.argmax(preds[j]) == 0:
                tn += 1

    results.append([eval_results[1],tp/allp,tn/alln])
    print(results[-1])
for i in range(len(results)):
    print(i,": Accuracy:",results[i][0]," TPR: ",results[i][1]," TNR: ",results[i][2])
