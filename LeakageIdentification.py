import keras.utils.np_utils
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
import numpy
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.utils.np_utils import to_categorical
import h5py
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# TODO: Add path to ASCAD traces
TracePath = "ATMega8515_raw_traces.h5"
# Range to analyze
start_sample = 30000
end_sample = 40000

# Network parameters
nrepochs = 20
batchsize= 1000
target_byte = 7

# Additional parameters
train_traces = 20000
gradCalc = 10000

# Read traces and metadata
f = h5py.File(TracePath, "r")
traces = f["traces"]
plains = f["metadata"]["plaintext"]
numTraces = traces.shape[0]
samplesPerFile = traces.shape[1]

# Sanity check
assert train_traces <= numTraces
assert end_sample <= samplesPerFile
assert start_sample < end_sample

# Select relevant data
train_x = traces[:, start_sample:end_sample]
train_y = plains[:,target_byte]
x = numpy.arange(start_sample, end_sample, 1)

# Normalize input data
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)


# Callback to perform sensitivity analysis after each epoch
class SACallback(tf.keras.callbacks.Callback):
    def __init__(self, globSum):
        super().__init__()
        self.globSum = globSum
        self.inp = tf.Variable(train_x[:gradCalc], dtype=tf.float32)
        self.trues = to_categorical(train_y[:gradCalc])
        self.trues_tf = tf.Variable(self.trues, dtype=tf.float32)

    def on_epoch_end(self, epoch, logs=None):
        with tf.GradientTape() as tape:
            tape.watch(self.inp)
            preds = self.model(self.inp)
            loss = tf.keras.losses.mean_squared_error(self.trues_tf, preds)
            grads = tape.gradient(loss, self.inp)
        self.globSum = self.globSum + numpy.sum(numpy.abs((grads * train_x[:gradCalc])), axis=0)

    def on_train_end(self, logs=None):
        resfile = open("weights_{}_{}_{}.dat".format(target_byte, start_sample, end_sample), "wb")
        resfile.write(self.globSum)
        resfile.close()
        print(f'Peak Position: {self.globSum.argmax()+start_sample}')
        fig, ax = plt.subplots(1)
        ax.set_title("Sensitivity Analysis Byte {}".format(target_byte))
        ax.plot(x, self.globSum, linewidth=0.6)
        ax.set_xlim([start_sample, end_sample])
        fig.show()
        plt.close()


grads_sum = 0
sa_callback = SACallback(grads_sum)

# Define model
model = Sequential(
    [keras.Input(shape=(len(train_x[0]),)),
     Dense(50, activation='relu'),
     Dense(1600, activation='relu'),
     Dense(1600, activation='relu'),
     Dense(100, activation='relu'),
     Dense(256, activation='softmax')]
)
model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])

# Create One-Hot encoded labels
trues = to_categorical(train_y)

# Start training
model.fit(train_x, trues,
          epochs=nrepochs,
          batch_size=batchsize,
          verbose=1,
          callbacks=[sa_callback])







