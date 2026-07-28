#coding=utf-8
from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, MaxPooling2D, Flatten, Activation, Embedding, Dropout, LayerNormalization, Conv1D, MaxPooling1D, GlobalAveragePooling1D, InputLayer
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import LSTM
import tensorflow.keras.backend as K
import math
from nns import focal_loss,multi_category_focal_loss2,multi_category_focal_loss2_fixed,weight_variable,bias_variable


'''
[COOKING] functions
'''


'''
others
'''


def mean_pred(y_true, y_pred):
    # metrics
    return K.mean(y_pred)


'''
convlution-design
[weight]-specification
assign different weights to different features, so as making their contributions to the classification distributed
'''


def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Varialbe(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiplay(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var


'''
[tested-out] --> good to go (97%) ---  pool_size=(2,1);(1,1)
'''
# model就是Q
# model for 'image-classification'
def get_image_model(in_shape, feaNum, output):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding='Same',input_shape=in_shape, input_dim=feaNum))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))  # orginal: pool_size=(2,2)
    model.add(Conv2D(32, (5, 5), padding='Same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))  # orginal: pool_size=(2,2)
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(output))
    return model



def get_image_model2(in_shape, feaNum, output):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding='Same',input_shape=in_shape, input_dim=feaNum))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(output))
    return model


'''
[tested-out] --> good to go (100%)
'''
def get_classify_modelTest(in_shape, feaNum, num_classes):
    model = Sequential()
    model.add(Dense(units=round(feaNum), input_dim=feaNum,
                    input_shape=in_shape, activation='selu'))
    #model.add(Dense(units=feaNum,activation='relu'))
    #model.add(keras.layers.LeakyReLU(alpha=0.9))
    model.add(Dense(units=num_classes, activation='softmax'))
    return model


def get_simplePred(in_shape, feaNum, num_classes):
    model = Sequential()
    model.add(Dense(units=32, input_dim=feaNum,
              input_shape=in_shape, activation='selu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    return model


def get_simplePred1(in_shape, feaNum, num_classes):
    model = Sequential()
    model.add(Dense(units=64, input_dim=feaNum,
              input_shape=in_shape, activation='linear'))
    model.add(Dense(units=num_classes, activation='softmax'))

    return model


def get_simplePred2(in_shape, feaNum, num_classes):
    model = Sequential()
    model.add(Dense(units=64, input_dim=feaNum,
              input_shape=in_shape, activation='tanh'))
    model.add(Dense(units=num_classes, activation='softmax'))

    return model


'''
basic MLP example
[tested-out] --> good to go (100%)
'''
def get_simpleMLP(in_shape, feaNum, num_classes):
    model = Sequential()
    model.add(Dense(units=40, input_dim=feaNum,
              input_shape=in_shape, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))

    return model


def get_simpleMLP3D(in_shape, feaNum, num_classes):
    model = Sequential()
    model.add(Flatten(input_shape=in_shape))
    model.add(Dense(units=40, input_dim=feaNum, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(units=num_classes))

    return model


'''
another basic MLP example
[tested-out] --> good to go (97%)
'''


def get_simpleMLP_2(in_shape, feaNum, num_classes):
    model = Sequential()
    model.add(InputLayer(input_shape=in_shape))
    model.add(Flatten())
    model.add(Dense(units=100, activation='relu'))
    model.add(Dense(10))
    model.add(Dense(units=num_classes))

    return model


'''
lstm-layer trial
'''


def get_simpleLSTM(in_shape, feaNum, num_classes):
    model = Sequential()
    model.add(LSTM(128, input_shape=in_shape))
    model.add(Dense(units=num_classes))

    return model


'''
lstm: human activity recognition
https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/
'''


def get_LSTM_har(in_shape, feaNum, num_classes):
    model = Sequential()
    model.add(LSTM(100, input_shape=in_shape))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    return model


'''
lstm1: simple time-series prediction
https://blog.csdn.net/qq_35649669/article/details/84990183
'''


def get_LSTM_simple1(in_shape, feaNum, num_classes):
    model = Sequential()
    model.add(LSTM(32, input_shape=in_shape))
    model.add(Dense(num_classes, activation='softmax'))
    return model


'''
lstm2: simple time-series prediction
'''


def get_LSTM_simple2(in_shape, feaNum, num_classes):
    model = Sequential()
    model.add(LSTM(64, input_shape=in_shape))
    model.add(Dense(num_classes, activation='softmax'))
    return model


'''
dqn-DRL associated
https://tf.wiki/en/basic/models.html#deep-reinforcement-learning-drl
'''


def simple_drlTrial(in_shape, feaNum, num_classes):
    # Q-network associated
    model = Sequential()
    model.add(Dense(units=24, activation='relu',
              input_dim=feaNum, input_shape=in_shape))
    model.add(Dense(units=24, activation='relu'))
    model.add(Dense(units=2))
    model.add(Dense(units=num_classes, activation='softmax'))

    return model


'''
simple classification model
'''


def simple_classTrial(in_shape, feaNum, num_classes):
    '''
    dense = tf.keras.Sequential([
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=1)
    ])
    '''
    model = Sequential()
    model.add(Dense(units=64, input_dim=feaNum,
              input_shape=in_shape, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))

    return model


'''
[tested-out] --> BAD -- chance level
'''


def get_classify_model1(in_shape, feaNum, num_classes):
    model = Sequential()
    model.add(Dense(units=round(feaNum),
                    input_dim=feaNum, input_shape=in_shape))
    # ,activation='relu'
    model.add(Dense(units=round(feaNum*0.5), activation='relu'))
    model.add(keras.layers.LeakyReLU(alpha=0.3))
    #model.add(Dense(units=round(feaNum),activation='relu')) #
    model.add(Dropout(0.1))
    model.add(Dense(units=round(feaNum*0.3), activation='relu'))
    model.add(keras.layers.LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    #model.add(Dropout(0.2))
    model.add(Dense(units=round(feaNum*0.1), activation='relu'))
    model.add(keras.layers.LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    # model.add(Dense(units=round(feaNum*0.2),activation='relu')) #
    # model.add(Dense(units=round(feaNum*0.2),activation='relu'))
    #model.add(Dropout(0.2))
    #model.add(MaxPooling1D(pool_size=2))
    #model.add(Flatten())
    #model.add(keras.layers.LeakyReLU(alpha=0.3))
    #model.add(MaxPooling1D(pool_size=2))
    #model.add(Dropout(0.1))
    #model.add(Flatten())
    #model.add(Dense(units=10,activation='relu')) #round(feaNum/2)
    #model.add(Dropout(0.1))
    #model.add(keras.layers.LeakyReLU(alpha=0.3))
    #model.add(Dropout(0.25))
    model.add(Dense(units=num_classes, activation='softmax'))
    # ADAM-optimizer
    opt = keras.optimizers.Adam(learning_rate=0.0001)

    return model


'''
[tested-out] --> good to go (90%/93%)
'''


def get_classify_model3(in_shape, feaNum, num_classes):
    model = Sequential()
    model.add(Dense(units=feaNum, input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Flatten())
    #model.add(MaxPooling1D(pool_size=2))
    #model.add(Flatten())
    model.add(keras.layers.LeakyReLU(alpha=0.3))
    #model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.1))
    #model.add(Flatten())
    model.add(Dense(units=10, activation='relu'))  # round(feaNum/2)
    model.add(Dropout(0.1))
    model.add(keras.layers.LeakyReLU(alpha=0.3))
    #model.add(Dropout(0.25))
    model.add(Dense(units=num_classes, activation='softmax'))
    # RMS-prop
    tf.keras.optimizers.RMSprop(
        learning_rate=0.001,
        rho=0.9,
        momentum=0.0,
        epsilon=1e-07,
        centered=False,
        name="RMSprop",
        #**kwargs
    )
    '''
    # SGD-optimizer
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.9)
    optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
    # ADAM-optimizer
    #opt = keras.optimizers.Adam(learning_rate=0.0001)
    # compile the associated testing
    model.compile(
        optimizer=optimizer,
        loss=[focal_loss]  # 'categorical_crossentropy',
        #metrics=[keras.metrics.SparseCategoricalCrossentropy(name="cross")]
        # tf.keras.metrics.CategoricalAccuracy(name='acc'),
        #keras.metrics.SparseCategoricalAccuracy(name="accuracy"),

        #'accuracy']
    )
    '''

    return model


# modify the self-defined output -- XXL
def get_classify_model5(in_shape, feaNum, num_classes):
    model = Sequential()
    model.add(Dense(units=feaNum*2, input_shape=in_shape, input_dim=feaNum))
    model.add(Flatten())
    model.add(Dense(units=feaNum, activation='relu'))
    model.add(keras.layers.LeakyReLU(alpha=0.1))
    model.add(keras.layers.Dropout(0.25))
    model.add(Dense(units=10, activation='relu'))
    model.add(keras.layers.LeakyReLU(alpha=0.1))
    model.add(keras.layers.Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(units=num_classes, activation='softmax'))
    #model.add(Conv2D(32, (5, 5), padding='Same', input_shape=(28,28,1)))

    return model


'''
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
'''
'''
with-[CONVOLUTED COMPONENTS]
# Neural Net for Deep Q Learning
# Sequential() creates the foundation of the layers.
model = Sequential()
# Dense is the basic form of a neural network layer
# Input Layer 4 and Hidden Layer with 128 nodes
model.add(Dense(64, input_dim=4, activation='tanh'))
# Hidden layer with 128 nodes
model.add(Dense(128, activation='tanh'))
# Hidden layer with 128 nodes
model.add(Dense(128, activation='tanh'))
# Output Layer with 2 nodes
model.add(Dense(2, activation='linear'))

# Create the model based on the information above
model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate))
'''

'''
[convolution] associated contents
[conv1d]: https://keras.io/api/layers/convolution_layers/convolution1d/
Conv1D(filters=256, kernel_size=3, strides=1, activation='relu', input_shape=(99, 40), name='block1_conv1'))
[filter]
[kernel_size]
[strides]
[padding]
'''

'''
[tested-out] --> good to go (93%)
'''


def basicCNN(in_shape, feaNum, num_classes):
    model = Sequential()
    model.add(Dense(40, input_shape=in_shape, input_dim=feaNum))
    model.add(Reshape((40, 1)))
    model.add(Conv1D(16, 3))
    model.add(Flatten())
    model.add(Dense(units=num_classes, activation='softmax'))

    return model


'''
[https://tf.wiki/en/basic/models.html#convolutional-neural-network-cnn]
[tested-out] --> good to go (100%)
'''


def CNN1(in_shape, feaNum, num_classes):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, padding='Same',
              activation='relu', input_shape=in_shape, input_dim=feaNum))
    #model.add(MaxPooling2D(pool_size=[2,2],strides=2))
    #model.add(MaxPooling1D(pool_size=2,strides=2))
    model.add(MaxPooling1D(pool_size=1, strides=1))
    ####model.add(Conv2D(filters=64,kernel_size=[5,5],padding='Same',activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=5,
              padding='Same', activation='relu'))
    ####model.add(MaxPooling1D(pool_size=2,strides=2))
    model.add(MaxPooling1D(pool_size=1, strides=1))
    #model.add(MaxPooling2D(pool_size=[2,2],strides=2))
    model.add(Flatten())
    #model.add(Reshape(target_shape=(7 * 7 * 64,)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10))
    model.add(Dense(units=num_classes, activation='softmax'))

    return model


'''
[tested-out] --> good to go (97%/93%) -- filter number = 32
100% -- filter number=64
['relu'] added over convolution: --- 93%
'''


def get_convClassify_modelDraft1D_1(in_shape, feaNum, num_classes):
    model = Sequential()
    model.add(Conv1D(64, 4, input_shape=in_shape, input_dim=feaNum,
              padding='Same'))  # activation='softmax',
    model.add(Activation('relu'))
    model.add(MaxPooling1D(1))
    model.add(LayerNormalization())
    model.add(Conv1D(128, 4, padding='Same', activation='relu'))
    model.add(LayerNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(1))
    #model.add(GlobalAveragePooling1D())
    #model.add(Dropout(0.5))
    # model.add(Conv1D(4,3, activation='softmax', input_shape=in_shape,padding='same'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling1D(3))
    # model.add(LayerNormalization())
    # model.add(Conv1D(4, 3, activation='relu',padding='same'))
    # model.add(Activation('relu'))
    # model.add(LayerNormalization())
    # model.add(MaxPooling1D(3))
    #model.add(GlobalAveragePooling1D())
    model.add(Flatten())
    model.add(Dropout(0.1))
    #model.add(Dense(feaNum))
    model.add(Dense(64))
    model.add(keras.layers.LeakyReLU(alpha=0.3))
    # model.add(keras.layers.LeakyReLU(alpha=0.3))
    model.add(Dense(units=num_classes, activation='softmax'))
    #model.add(Dense(output))

    return model


'''
[tested-out] --> good to go (100%)
'''


def get_convClassify_modelDraft1D(in_shape, feaNum, num_classes):
    model = Sequential()
    model.add(Conv1D(128, 3, input_shape=in_shape,
              input_dim=feaNum, activation='softmax'))
    #model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(2))
    # model.add(LayerNormalization())
    # #model.add(Conv1D(128, 3))
    # model.add(Conv1D(32, 3, activation='relu'))
    # model.add(GlobalAveragePooling1D())
    # model.add(MaxPooling1D(3))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Activation('relu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    #model.add(Dense(output))
    return model


'''
[tested-out] --> good to go (100%)
'''


def get_convClassify_modelDraft1D_pastSave(in_shape, feaNum, num_classes):
    model = Sequential()
    #model.add(Conv2D(128,(2,2),padding='Same',input_shape=in_shape))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dense(units=num_classes,activation='softmax'))
    model.add(Conv1D(32, 3, activation='relu', padding='Same',
              input_shape=in_shape, input_dim=feaNum))
    #model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3))
    #model.add(LayerNormalization(axis=-1,epsilon=0.001,center=True,scale=True,beta_initializer="zeros",gamma_initializer="ones",beta_regularizer=None,gamma_regularizer=None,beta_constraint=None,gamma_constraint=None))
    model.add(LayerNormalization())
    # model.add(Conv2D(128, (9, 9), padding='Same'))
    # model.add(LayerNormalization())
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2,1)))
    # model.add(Conv1D(32, 3,padding='Same', activation='relu'))
    # model.add(LayerNormalization())
    # model.add(MaxPooling1D(pool_size=3))

    # model.add(Flatten())
    # model.add(Dense(256))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.2))
    model.add(Flatten())
    # model.add(Dense(64))
    model.add(Activation('relu'))
    #model.add(Dropout(0.2))
    #model.add(Flatten())
    #model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Activation('relu'))
    #model.add(Dropout(0.2))
    #model.add(Flatten())
    #model.add(Dense(1))
    model.add(Dense(units=num_classes, activation='softmax'))
    #model.add(Dense(output))

    return model


'''
[tested-out] --> good to go (90%)
'''


def get_convClassify_modelDraft(in_shape, feaNum, num_classes):
    model = Sequential()
    #model.add(Conv2D(128,(2,2),padding='Same',input_shape=in_shape))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dense(units=num_classes,activation='softmax'))
    model.add(Conv2D(128, (5, 5), padding='Same',
              input_shape=in_shape, input_dim=feaNum))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    # model.add(LayerNormalization(axis=-1,epsilon=0.001,center=True,scale=True,beta_initializer="zeros",gamma_initializer="ones",beta_regularizer=None,gamma_regularizer=None,beta_constraint=None,gamma_constraint=None))
    # model.add(Conv2D(128, (9, 9), padding='Same'))
    model.add(LayerNormalization())
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2,1)))
    model.add(Conv2D(64, (5, 5), padding='Same'))
    model.add(LayerNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    '''
    model.add(Conv2D(32, (5, 5), padding='Same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1,1)))
    '''
    # model.add(Flatten())
    # model.add(Dense(256))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.2))
    # model.add(Flatten())
    # model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Flatten())
    #model.add(Dense(1))
    model.add(Dense(units=num_classes, activation='softmax'))
    #model.add(Dense(output))

    return model


'''
[tested-out] --> good to go (93%) ... (100%/97%)
'''


def get_convClassify_modelDraft(in_shape, feaNum, num_classes):
    model = Sequential()
    #model.add(Conv2D(128,(2,2),padding='Same',input_shape=in_shape))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dense(units=num_classes,activation='softmax'))
    model.add(Conv2D(32, (5, 5), padding='Same',
              input_shape=in_shape, input_dim=feaNum))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(LayerNormalization(axis=-1, epsilon=0.001, center=True, scale=True, beta_initializer="zeros",
              gamma_initializer="ones", beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    model.add(Conv2D(32, (5, 5), padding='Same'))
    model.add(LayerNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    #model.add(Conv2D(32, (5, 5), padding='Same'))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(1,1)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    #model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    #model.add(Flatten())
    model.add(Dense(16))
    model.add(Activation('relu'))
    #model.add(Dropout(0.2))
    #model.add(Flatten())
    #model.add(Dense(1))
    model.add(Dense(units=num_classes, activation='softmax'))
    #model.add(Dense(output))

    return model


'''
[tested-out] --> good to go (98%/97%)
'''
def get_classify_model0(INPUT_SHAPE, feaNum, num_classes):
    model = Sequential()
    model.add(Dense(units=feaNum*2, input_shape=INPUT_SHAPE))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    #model.add(MaxPooling1D(pool_size=2))
    #model.add(Flatten())
    model.add(keras.layers.LeakyReLU(alpha=0.03))
    #model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    #model.add(Flatten())
    model.add(Dense(units=10, activation='relu'))
    model.add(Dropout(0.2))
    model.add(keras.layers.LeakyReLU(alpha=0.03))
    #model.add(Dropout(0.25))
    model.add(Dense(units=num_classes, activation='softmax'))

    return model
    '''
    model=Sequential()
    model.add(Dense(feaNum*2, input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    '''
    '''
    https://blog.csdn.net/JJBOOM425/article/details/100177838
    '''

    '''

    '''


'''
[tested-out] --> good to go (90%) acc: test>train
'''
def get_classify_model4(in_shape, feaNum, num_classes):
    model = Sequential()
    model.add(Dense(units=feaNum-1, input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    #model.add(MaxPooling1D(pool_size=2))
    #model.add(Flatten())
    model.add(keras.layers.LeakyReLU(alpha=0.1))
    #model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    #model.add(Flatten())
    model.add(Dense(units=10, activation='relu'))
    model.add(Dropout(0.2))
    model.add(keras.layers.LeakyReLU(alpha=0.1))
    #model.add(Dropout(0.25))
    model.add(Dense(units=num_classes, activation='softmax'))
    # compile the associated testing

    return model


def get_classify_modelConv2(in_shape, feaNum, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='Same',
              input_shape=in_shape, input_dim=feaNum))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='Same'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='Same'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def get_simpleActionPred(in_shape, feaNum, num_classes):

    model = Sequential()
    model.add(Dense(units=64, input_dim=feaNum, activation="relu"))
    model.add(Dense(units=32, activation="relu"))
    model.add(Dense(units=8, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    return model


def get_simpleActionPred3D(in_shape, feaNum, num_classes):

    model = Sequential()
    model.add(Flatten(input_shape=in_shape))
    model.add(Dense(units=64, input_dim=feaNum, activation="relu"))
    model.add(Dense(units=32, activation="relu"))
    model.add(Dense(units=8, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    return model

'''
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
'''


'''
seems NOT to be runnable in continuous dataset
'''
# model for 'text-classification'


def get_text_model(input_shape, feaNum, output):
    top_words = input_shape
    max_words = feaNum
    model = Sequential()
    model.add(Embedding(top_words, 32, input_length=max_words))
    model.add(Flatten())
    model.add(Dense(250))
    model.add(Activation('relu'))
    model.add(Dense(output))
    return model


'''
model=Sequential()
model.add(Dense(feaNum*2, input_shape=in_shape))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Activation('relu'))
model.add(Dense(num_classes))
'''
