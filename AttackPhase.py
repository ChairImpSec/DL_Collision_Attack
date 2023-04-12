import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape, Conv1D, Flatten, Dense, BatchNormalization, Dropout, AveragePooling1D
import numpy
from sklearn.preprocessing import StandardScaler
import h5py
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# TODO: Add path to ASCAD traces
TracePath = "ATMega8515_raw_traces.h5"

# Attack target
range_start = 35060
range_end = 35860
attack_byte = 7
train_byte = 10

# Additional parameters
train_traces = 20000
num_predictions = 3


def calc_key(predictions):
    keyprob = numpy.zeros(256)
    for idx, plaintext in enumerate(train_y):
        for i in range(256):
            keyprob[i] += predictions[idx][plaintext ^ i]
    key_guess = keyprob.argmax()
    return key_guess


# Read traces and metadata
f = h5py.File(TracePath, "r")
traces = f["traces"]
plains = f["metadata"]["plaintext"]
realKey = f["metadata"]["key"][0]
numTraces = traces.shape[0]
samplesPerFile = traces.shape[1]

# Sanity check
assert train_traces <= numTraces
assert range_end <= samplesPerFile
assert range_start < range_end

# Select relevant data
train_y = plains[:train_traces, attack_byte]
train_x = traces[:train_traces, range_start:range_end]

# Normalize input data
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)

# Same network as for TrainingPhase but with additional Dropout layers
initializer_other = tf.keras.initializers.HeUniform()
initializer_SM = tf.keras.initializers.GlorotUniform()
model = Sequential(
    [Reshape((len(train_x[0]), 1), input_shape=(len(train_x[0]),)),
     Conv1D(filters=2, kernel_size=75, strides=1, padding='same'),
     AveragePooling1D(pool_size=25, strides=25),

     Conv1D(filters=2, kernel_size=3, strides=1, padding='same'),
     BatchNormalization(),
     AveragePooling1D(pool_size=4, strides=4),
     Conv1D(filters=8, kernel_size=2, strides=1, padding='same'),
     AveragePooling1D(pool_size=2, strides=2),

     Flatten(),
     Dense(10, kernel_initializer=initializer_other, activation='selu'),
     Dropout(0.2),
     Dense(4, kernel_initializer=initializer_other, activation='selu'),
     Dropout(0.2),
     Dense(2, kernel_initializer=initializer_other, activation='selu'),
     Dropout(0.2),
     Dense(256, kernel_initializer=initializer_SM, activation='softmax')])
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Correct key to validate correctness of attack
CorrKey = realKey[train_byte] ^ realKey[attack_byte]
print(f'Correct Key: {CorrKey}')

# Load weights from TrainingPhase and predict keys
model.load_weights(f'trainedByte_{train_byte}.h5')
for it in range(num_predictions):
    predictions = model(train_x, training=True)
    key = calc_key(predictions.numpy())
    print(f'Calculated Key: {key}')
