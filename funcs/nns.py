import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, MaxPooling2D, Flatten, Activation, Embedding, Dropout, LayerNormalization, Conv1D, MaxPooling1D, GlobalAveragePooling1D, InputLayer, LSTM
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import LSTM, BatchNormalization, GlobalMaxPooling1D
from tensorflow.keras import layers, models
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import math


'''
Multiple class change record:
Output Layer: Dense(num_classes, activation='softmax') instead of Dense(1, activation='sigmoid')

Loss Function: sparse_categorical_crossentropy for integer labels or categorical_crossentropy for one-hot encoded labels

Metrics: Use accuracy and consider per-class metrics

Class Weights: Handle potential class imbalance with class_weight parameter

Predictions: Use argmax to get predicted class from probability distribution

'''




'''
for [2-D] structured data  ###################################################################
'''


def get_simpleMLP(in_shape, feaNum, outNum):
    model = Sequential()
    model.add(Dense(units=128, input_dim=feaNum,
              input_shape=in_shape, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(units=outNum))  # , activation='softmax'
    ''''''
    model.compile(
            optimizer=keras.optimizers.Adam(),
            # multi_category_focal_loss2_fixed,focal_loss
            loss=['mse', 'categorical_crossentropy'],
            metrics=['accuracy']
        )
    return model


def get_simpleMLP2(in_shape, feaNum, outNum):
    model = Sequential()
    model.add(Flatten(input_shape=in_shape))
    model.add(Dense(units=64, input_dim=feaNum, activation='softmax'))
    model.add(Dense(32, activation='softmax'))
    model.add(Dense(units=outNum))  # , activation='softmax'
    ''''''
    model.compile(
            optimizer=keras.optimizers.Adam(),
            # multi_category_focal_loss2_fixed,focal_loss
            loss=['mse', 'categorical_crossentropy'],
            metrics=['accuracy']
        )
    return model


def get_simpleMLP3(in_shape, feaNum, outNum):
    model = Sequential()
    model.add(Flatten(input_shape=in_shape))
    model.add(Dense(units=32, input_dim=feaNum, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(units=outNum))  # , activation='softmax'
    ''''''
    model.compile(
            optimizer=keras.optimizers.Adam(),
            # multi_category_focal_loss2_fixed,focal_loss
            loss=['mse', 'categorical_crossentropy'],
            metrics=['accuracy']
        )
    return model


def get_simpleMLP3D1(in_shape, feaNum, outNum):
    model = Sequential()
    model.add(Flatten(input_shape=in_shape))
    model.add(Dense(units=40, input_dim=feaNum, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(units=outNum))  # , activation='softmax'
    ''''''
    model.compile(
        optimizer=keras.optimizers.Adam(),
        # multi_category_focal_loss2_fixed
        loss=['mse', 'categorical_crossentropy'],
        metrics=['accuracy']
    )

    return model


def CNN11(in_shape, feaNum, classNum):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=5, padding='Same',
              activation='relu', input_shape=in_shape, input_dim=feaNum))
    model.add(MaxPooling1D(pool_size=1, strides=1))
    model.add(Conv1D(filters=64, kernel_size=5,
              padding='Same', activation='relu'))
    model.add(MaxPooling1D(pool_size=1, strides=1))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dense(10))
    model.add(Dense(1, activation='sigmoid'))  # , activation='softmax'
    model.compile(
        loss=['mse', 'categorical_crossentropy'],
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["accuracy"],
    )
    return model


'''
for [3-D] structured data  ###################################################################
'''

def get_simpleMLP3D(in_shape, feaNum):
    model = Sequential()
    model.add(Flatten(input_shape=in_shape))
    model.add(Dense(units=40, input_dim=feaNum, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(units=feaNum))
    ''''''
    model.compile(
        optimizer=keras.optimizers.Adam(),
        # multi_category_focal_loss2_fixed
        loss=['mse', 'categorical_crossentropy'],
        metrics=['mae']
    )

    return model


def get_simpleMLP3D_advanced0(in_shape, feaNum):
    model = Sequential()
    model.add(Flatten(input_shape=in_shape))
    model.add(Dense(units=(feaNum*3), activation='relu'))
    model.add(Dense(units=1,activation='sigmoid'))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=["binary_crossentropy"],  # focal_loss,
        metrics=["mae","accuracy"]
    )

    return model


def get_simpleLSTM(in_shape, feaNum):
    model = Sequential()
    model.add(LSTM(64, input_shape=in_shape))
    model.add(Dense(feaNum))
    model.compile(
        optimizer=keras.optimizers.Adam(),
        # 'sparse_categorical_crossentropy','mse', 'categorical_crossentropy'
        loss=['mse', 'categorical_crossentropy'],
        metrics=['mae']
    )
    return model


def get_simpleLSTM2(in_shape, feaNum):
    model = Sequential()
    model.add(InputLayer(input_shape=in_shape, activation='relu'))
    model.add(LSTM(128))
    model.add(Dense(feaNum))
    model.compile(
        optimizer=keras.optimizers.Adam(),
        # 'sparse_categorical_crossentropy','mse', 'categorical_crossentropy'
        loss=['mse', 'categorical_crossentropy'],
        metrics=['mae']
    )
    return model


# image-agent processing structure
def CNN2(in_shape, feaNum):
    model = Sequential()
    model.add(Conv1D(100, 5, activation='relu', padding='Same',
              input_dim=feaNum, input_shape=in_shape))
    model.add(MaxPooling1D(pool_size=1, strides=1))
    model.add(Conv1D(64, 10, activation='relu', padding='Same'))
    model.add(MaxPooling1D(pool_size=1, strides=1))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(28, activation='relu'))
    model.add(Dense(feaNum, activation='softmax'))
    model.compile(
        loss=["mse", "categorical_crossentropy"],
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["mae"],
    )
    return model


def CNN1(in_shape, feaNum):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=5, padding='Same',
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
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(10))
    model.add(Dense(units=feaNum))
    # compile model
    model.compile(
        loss=["mse", "categorical_crossentropy"],
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["mae"],
    )
    return model


def CNN(in_shape, feaNum):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=5, padding='same',
              activation='relu', input_shape=in_shape)) # , input_dim=feaNum
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=5,
              padding='Same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10))
    model.add(Dense(units=1,activation='sigmoid'))
    model.compile(
        loss=["binary_crossentropy"],
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["mae","accuracy"],
    )
    return model

def CNN3(in_shape, feaNum):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, padding='same',
              activation='relu', input_shape=in_shape)) # , input_dim=feaNum
    model.add(BatchNormalization())
    #model.add(Dropout(0.1))

    model.add(Conv1D(filters=128, kernel_size=3,
              padding='Same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    model.add(Conv1D(filters=256, kernel_size=5,
              padding='Same', activation='relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.1))

    model.add(GlobalMaxPooling1D())

    model.add(Dense(units=128,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=64,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=1,activation='sigmoid'))
    model.compile(
        loss=["binary_crossentropy"],
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["mae","accuracy"],
    )
    return model

def basicCNN(in_shape, feaNum):
    model = Sequential()
    model.add(Dense(40, input_shape=in_shape, input_dim=feaNum))
    #model.add(Reshape((40, 1)))
    model.add(Conv1D(16, 3))
    model.add(Flatten())
    model.add(Dense(units=feaNum))
    model.compile(
        loss=["mse", "categorical_crossentropy"],
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["mae"],
    )
    return model


def transformer_model1(in_shape, feaNum, outNum):
    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
        # Normalization and Attention
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        x = layers.Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res

    def build_model(
        input_shape,
        head_size,
        num_heads,
        ff_dim,
        num_transformer_blocks,
        mlp_units,
        dropout=0,
        mlp_dropout=0,
    ):
        inputs = keras.Input(shape=input_shape)
        x = inputs
        for _ in range(num_transformer_blocks):
            x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(mlp_dropout)(x)
        outputs = layers.Dense(feaNum, activation="relu")(x)
        return keras.Model(inputs, outputs)

    input_shape = in_shape

    # important parametic-specifications here
    model = build_model(
        input_shape,
        head_size=64,  # 256
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[128],
        mlp_dropout=0.4,
        dropout=0.25,
    )

    model.compile(
        loss=["mse", "categorical_crossentropy"],
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["mae"],
    )

    return model


def CNN4(in_shape, feaNum):
    input_layer = layers.Input(shape=in_shape)

    x = layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(input_layer)
    x = layers.BatchNormalization()(x)
    #x = layers.Dropout(0.05)(x)

    x = layers.Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    #x = layers.Dropout(0.05)(x)

    # x = layers.Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(x)
    # x = layers.BatchNormalization()(x)
    #x = layers.Dropout(0.05)(x)

    # Global feature summary over regions
    x = layers.GlobalAveragePooling1D()(x)
    #x = layers.Flatten()(x)

    # Final classification head
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.01)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dense(8,activation='relu')(x)
    
    #output = layers.Dense(1, activation='sigmoid')(x)
    outputs = layers.Dense(2, activation='softmax')(x)

    model = models.Model(inputs=input_layer, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-2),
                #loss=["mse",'binary_crossentropy'],
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.summary()

    return model



def CNN5(in_shape, feaNum):
    input_layer = layers.Input(shape=in_shape)

    # Convolutional feature extraction
    x = layers.Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(input_layer)
    #x = layers.BatchNormalization()(x)

    x = layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    #x = layers.BatchNormalization()(x)

    x = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    #x = layers.BatchNormalization()(x)

    # Attention-based aggregation (learns which positions matter)
    attention = layers.Dense(1, activation='tanh')(x)
    attention = layers.Softmax(axis=1)(attention)
    x = layers.Multiply()([x, attention])
    x = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)  # Weighted sum

    # Classification head
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.05)(x)
    x = layers.Dense(16, activation='relu')(x)

    output = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=input_layer, outputs=output)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                loss=["mse",'binary_crossentropy'],
                metrics=['accuracy'])

    model.summary()

    return model





def CNN6(in_shape, feaNum):
    input_layer = layers.Input(shape=in_shape)

    # Parallel multi-scale processing
    conv1 = layers.Conv1D(256, kernel_size=3, padding='same', activation='relu')(input_layer)
    conv2 = layers.Conv1D(128, kernel_size=5, padding='same', activation='relu')(input_layer)
    conv3 = layers.Conv1D(64, kernel_size=7, padding='same', activation='relu')(input_layer)

    x = layers.Concatenate()([conv1, conv2, conv3])
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Max over sequence (learns most discriminative position)
    x = layers.Lambda(lambda t: tf.reduce_max(t, axis=1))(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.05)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=input_layer, outputs=output)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                loss=["mse",'binary_crossentropy'],
                metrics=['accuracy'])

    model.summary()

    return model



def brainClassifier0(in_shape, feaNum):
    inputs = layers.Input(shape=in_shape, name='brain_regions')
    dropout_rate=0.05
    
    # Stage 1: Per-region feature extraction using 1D convolutions
    # Think of this as learning local patterns within each region's features
    x = layers.Conv1D(32, kernel_size=3, padding='same', activation='relu',
                      name='region_conv1')(inputs)
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu',
                      name='region_conv2')(x)
    
    # Stage 2: Cross-region interaction with attention mechanism
    # This captures relationships between different brain regions
    attention = layers.MultiHeadAttention(
        num_heads=4, 
        key_dim=64,
        dropout=dropout_rate,
        name='region_attention'
    )(x, x)
    
    x = layers.Add()([x, attention])  # Residual connection
    x = layers.LayerNormalization()(x)
    
    # Stage 3: Hierarchical pooling - aggregate information efficiently
    # Global pooling captures overall brain patterns
    global_max = layers.GlobalMaxPooling1D()(x)
    global_avg = layers.GlobalAveragePooling1D()(x)
    
    # Combine both pooling strategies
    x = layers.Concatenate()([global_max, global_avg])
    
    # Stage 4: Classification head with regularization
    x = layers.Dense(128, activation='relu', name='dense1')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation='relu', name='dense2')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Binary output
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                loss=["mse",'binary_crossentropy'],
                metrics=['accuracy'])

    model.summary()

    return model




def brainClassifier1(in_shape, feaNum):
    """
    Minimalist architecture optimized for small datasets.
    Philosophy: Let regularization do the heavy lifting, not model complexity.
    """
    inputs = layers.Input(shape=in_shape, name='brain_regions')
    dropout_rate=0.02
    
    # Single conv layer to learn regional patterns
    x = layers.Conv1D(16, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.Dropout(dropout_rate)(x)  # Heavy dropout early
    
    # Simple pooling - no fancy attention needed
    x_max = layers.GlobalMaxPooling1D()(x)
    x_avg = layers.GlobalAveragePooling1D()(x)
    x = layers.Concatenate()([x_max, x_avg])  # Just 32 features total
    
    # Minimal dense layers
    x = layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    #x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Use strong regularization in compilation too
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),  # Lower LR
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )

    model.summary()


    return model





def brainSpatialAwareBinary(in_shape,feaNum):
    """Alternative with more explicit spatial modeling"""
    inputs = layers.Input(shape=in_shape)
    
    # Expand dimensions for 2D convolution treatment
    x = layers.Reshape((in_shape[0], in_shape[1], 1))(inputs)
    
    # 2D convolutions to capture feature-region interactions
    x = layers.Conv2D(32, (5, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 1))(x)
    
    x = layers.Conv2D(128, (3, 2), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                loss=["mse",'binary_crossentropy'],
                metrics=['accuracy'])

    model.summary()

    return model






'''
for [4-D] structured Data ###################################################################
'''


def get_image_model(in_shape, feaNum):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding='Same',
              input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # orginal: pool_size=(2,2)
    model.add(Conv2D(32, (5, 5), padding='Same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # orginal: pool_size=(2,2)
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dropout(0.1))
    model.add(Dense(128))
    model.add(Dropout(0.1))
    model.add(Dense(64))
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(units=1,activation='sigmoid'))
    # model.add(Dense(units=num_classes, activation='softmax'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=["binary_crossentropy"],  # focal_loss,
        metrics=["mae","accuracy"]
    )
    '''

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
            loss=["mse",'binary_crossentropy'],
            metrics=['accuracy'])
    '''

    model.summary()

    return model


# STC-DCNN
def get_stcdcnn_model(in_shape, feaNum):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding='Same',
              input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))  # orginal: pool_size=(2,2)
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dropout(0.1))
    model.add(Dense(128))
    model.add(Dropout(0.1))
    model.add(Dense(64))
    model.add(Dense(32))
    model.add(Dense(16))
    #model.add(Activation('relu'))
    model.add(Dense(units=1,activation='sigmoid'))
    # model.add(Dense(units=num_classes, activation='softmax'))
    ''''''
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=["binary_crossentropy"],  # focal_loss,
        metrics=["mae","accuracy"]
    )

    model.summary()

    return model


'''
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Belows are specifications for [loss functions]
'''
# customized focal-loss


def focal_loss(y_true, y_pred):
   # loss
   gamma = 2
   alpha = 0.25
   pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
   pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
   return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


def multi_category_focal_loss2(gamma=2, alpha=.25):
    gamma = 2
    alpha = .25
    epsilon = 1.e-7
    gamma = float(gamma)
    alpha = tf.constant(alpha, dtype=tf.float32)

    def multi_category_focal_loss2_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        alpha_t = y_true*alpha + (tf.ones_like(y_true)-y_true)*(1-alpha)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
        ce = -ts.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
        loss = tf.reduce_mean(fl)
        return loss
    return multi_category_focal_loss2_fixed


def multi_category_focal_loss2_fixed(y_true, y_pred):
    gamma = 2
    alpha = .25
    epsilon = 1.e-7
    gamma = float(gamma)
    alpha = tf.constant(alpha, dtype=tf.float32)

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

    alpha_t = y_true*alpha + (tf.ones_like(y_true)-y_true)*(1-alpha)
    y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
    ce = -math.log(y_t)
    weight = tf.pow(tf.subtract(1., y_t), gamma)
    fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
    loss = tf.reduce_mean(fl)
    return loss


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
[variable] associated cookings
'''


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Varialbe(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


'''
[some results save]:

--------------------[test-sample-size]:  107520

--------------------[access-model]:  get_simpleMLP3D




================================== classification accuracy:  83.06547619047619 %
'''
