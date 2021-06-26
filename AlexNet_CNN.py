from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
import numpy as np
import scipy.io

# Define input tensor layer
input_layer = Input(shape=(59, 250, 1))

# 1st Tensorflow block
# Define the number of filter, kernel size, stride, padding, activation function
x = Conv2D(filters=8, # 96 original, best 8
           kernel_size=3, # 11
           strides=1, # 4
           padding='same',
           activation='relu')(input_layer)

# normalise and pool the output of the first conv layer
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=3, strides=1)(x)

# Define 2nd tensorflow block
x = Conv2D(filters=8, # 256
           kernel_size=3, # 5
           strides=1, # 4
           padding='same',
           activation='elu')(x)

x = BatchNormalization()(x)
x = MaxPool2D(pool_size=2, strides=2)(x)

# Define 3rd tensorflow block
x = Conv2D(filters=8, # 384
           kernel_size=3,
           padding='same',
           activation='elu')(x)

# Define 4th tensorflow block
x = Conv2D(filters=8, # 384
           kernel_size=3,
           padding='same',
           activation='elu')(x)

# Define 5th tensorflow block
x = Conv2D(filters=8, #256
           kernel_size=3,
           padding='same',
           activation='elu')(x)

x = BatchNormalization()(x)
x = MaxPool2D(pool_size=1, strides=2)(x) # originally pool_size=3

# Define Fully Connected Layers
x = Flatten()(x)

x = Dense(units=128, activation='elu')(x) # originally 4096
x = Dense(units=128, activation='elu')(x)
x = Dropout(rate=0.5)(x)

# Define Output Layer
# Equal to the number of classes
output_layer = Dense(units=2, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output_layer)

plot_model(model, show_shapes=True, to_file='./alexnet_cnn_basic.png')

# Insert code that can split data, define train and test sets

m = scipy.io.loadmat('BCICIV_calib_ds1f.mat', struct_as_record=True)

sample_rate = m['nfo']['fs'][0][0][0][0]
EEG = m['cnt'].T
nchannels, nsamples = EEG.shape

channel_names = [s[0] for s in m['nfo']['clab'][0][0][0]]
event_onsets = m['mrk'][0][0][0]
event_codes = m['mrk'][0][0][1]
labels = np.zeros((1, nsamples), int)
labels[0, event_onsets] = event_codes

cl_lab = [s[0] for s in m['nfo']['classes'][0][0][0]]
cl1 = cl_lab[0]
cl2 = cl_lab[1]
nclasses = len(cl_lab)
nevents = len(event_onsets)

# Print some information
print('Shape of EEG:', EEG.shape)
print('Sample rate:', sample_rate)
print('Number of channels:', nchannels)
print('Channel names:', channel_names)
print('Number of events:', len(event_onsets))
print('Event codes:', np.unique(event_codes))
print('Class labels:', cl_lab)
print('Number of classes:', nclasses)

# Dictionary to store the trials in, each class gets an entry
trials = {}

# The time window (in samples) to extract for each trial, here 0.5 -- 2.5 seconds
win = np.arange(int(1.0 * sample_rate), int(3.5 * sample_rate))

# Length of the time window
nsamples = len(win)

# Loop over the classes (right, foot)
for cl, code in zip(cl_lab, np.unique(event_codes)):

    # Extract the onsets for the class
    cl_onsets = event_onsets[event_codes == code]

    # Allocate memory for the trials
    trials[cl] = np.zeros((nchannels, nsamples, len(cl_onsets)))

    # Extract each trial
    for i, onset in enumerate(cl_onsets):
        trials[cl][:, :, i] = EEG[:, win + onset]


import scipy.signal


def bandpass(trials, lo, hi, sample_rate):
    '''
    Designs and applies a bandpass filter to the signal.

    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEGsignal
    lo : float
        Lower frequency bound (in Hz)
    hi : float
        Upper frequency bound (in Hz)
    sample_rate : float
        Sample rate of the signal (in Hz)

    Returns
    -------
    trials_filt : 3d-array (channels x samples x trials)
        The bandpassed signal
    '''

    # The iirfilter() function takes the filter order: higher numbers mean a sharper frequency cutoff,
    # but the resulting signal might be shifted in time, lower numbers mean a soft frequency cutoff,
    # but the resulting signal less distorted in time. It also takes the lower and upper frequency bounds
    # to pass, divided by the niquist frequency, which is the sample rate divided by 2:
    a, b = scipy.signal.iirfilter(6, [lo / (sample_rate / 2.0), hi / (sample_rate / 2.0)])

    # Applying the filter to each trial
    ntrials = trials.shape[2]
    trials_filt = np.zeros((nchannels, nsamples, ntrials))
    for i in range(ntrials):
        trials_filt[:, :, i] = scipy.signal.filtfilt(a, b, trials[:, :, i], axis=1)

    return trials_filt

# trials = split_data(EEG, nchannels, nsamples, event_onsets, win)

trials_filt = {cl1: bandpass(trials[cl1], 8, 15, sample_rate),
               cl2: bandpass(trials[cl2], 8, 15, sample_rate)}

# Percentage of trials to use for training (50-50 split here)
train_percentage = 0.5

# Calculate the number of trials for each class the above percentage boils down to
ntrain_r = int(trials_filt[cl1].shape[2] * train_percentage)
ntrain_f = int(trials_filt[cl2].shape[2] * train_percentage)
ntest_r = trials_filt[cl1].shape[2] - ntrain_r
ntest_f = trials_filt[cl2].shape[2] - ntrain_f

# Splitting the frequency filtered signal into a train and test set
train = {cl1: trials_filt[cl1][:,:,:ntrain_r],
         cl2: trials_filt[cl2][:,:,:ntrain_f]}

test = {cl1: trials_filt[cl1][:,:,ntrain_r:],
        cl2: trials_filt[cl2][:,:,ntrain_f:]}

train_samples = np.concatenate((train[cl1], train[cl2]), axis=2)
train_lab_r = np.ones(train[cl1].shape[2], dtype=int)
train_lab_l = np.zeros(train[cl2].shape[2], dtype=int)
y_train = np.concatenate((train_lab_r, train_lab_l), axis=0)

test_samples = np.concatenate((test[cl1], test[cl2]), axis=2)
test_lab_r = np.ones(test[cl1].shape[2], dtype=int)
test_lab_l = np.zeros(test[cl2].shape[2], dtype=int)
y_test = np.concatenate((test_lab_r, test_lab_l),axis=0)

x_train = np.transpose(train_samples, (2, 0, 1))
x_test = np.transpose(test_samples, (2, 0, 1))

model.compile(
    optimizer=keras.optimizers.Adam(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

print("Fit model on training data")
history = model.fit(
    x_train, # data
    y_train, # labels
    batch_size=20,
    epochs=100,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_test, y_test),
)

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=5)
print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for 3 samples")
predictions = model.predict(x_test[:3])
print("predictions shape:", predictions.shape)