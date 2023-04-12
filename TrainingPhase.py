import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape, Conv1D, AveragePooling1D, Flatten, Dense, BatchNormalization
from sklearn.preprocessing import StandardScaler
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import h5py
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# TODO: Add path to ASCAD traces
TracePath = "ATMega8515_raw_traces.h5"

# Training target
range_start = 28814
range_end = 29614
train_byte = 10

# Additional parameters
train_traces = 20000
max_epochs = 500

# Read traces and metadata
f = h5py.File(TracePath, "r")
traces = f["traces"]
plains = f["metadata"]["plaintext"]
numTraces = traces.shape[0]
samplesPerFile = traces.shape[1]

# Sanity check
assert train_traces <= numTraces
assert range_end <= samplesPerFile
assert range_start < range_end

# Select relevant data
data_y = plains[:train_traces, train_byte]
data_x = traces[:train_traces, range_start:range_end]

# Normalize input data
scaler = StandardScaler()
data_x = scaler.fit_transform(data_x)

# Randomly create train + validation set
train_x, val_x, train_y, val_y = train_test_split(data_x, data_y, test_size=0.2)

# Neural Network from Rijsdijk et al.
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
     Dense(4, kernel_initializer=initializer_other, activation='selu'),
     Dense(2, kernel_initializer=initializer_other, activation='selu'),
     Dense(256, kernel_initializer=initializer_SM, activation='softmax')])
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Create One-Hot encoded labels
train_trues = to_categorical(train_y)
val_trues = to_categorical(val_y)

# start training
model.fit(train_x, train_trues,
                epochs=max_epochs,
                batch_size=50,
                validation_data=(val_x, val_trues),
                verbose=1)

model.save_weights(f'trainedByte_{train_byte}.h5')
