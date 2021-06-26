import tensorflow as tf
import numpy as np
import scipy.io
import scipy.signal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.utils import plot_model
from keras.optimizers import SGD
# opt = SGD(lr=0.01)


def EEGNet(nb_classes, Chans=59, Samples=200,
           dropoutRate=0.5, kernLength=125, F1=8,
           D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(Chans, Samples, 1))

    ##################################################################
    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(Chans, Samples, 1),
                    use_bias=False)(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16),
                             use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name='flatten')(block2)

    dense = Dense(nb_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)
    # sigmoid = Activation('sigmoid', name='sigmoid')(dense)
    softmax = Activation('softmax', name='softmax')(dense)

    # return Model(inputs=input1, outputs=sigmoid)
    return Model(inputs=input1, outputs=softmax)

# Load data for training

# Insert code that can split data, define train and test sets

m = scipy.io.loadmat('ds3_9.mat', struct_as_record=True)

sample_rate = m['h']['SampleRate'][0][0][0][0]

# load RAW EEG signal
EEG = m['s'].T

EEG = EEG[0:3, :]

array_sum = np.sum(EEG)
array_has_nan = np.isnan(array_sum)

print(array_has_nan)

# obtain dimensions of raw EEG data
nchannels, nsamples = EEG.shape

# obtain channel names
# channel_names = m['h']['SampleRate'][0][0][0][0]

event_onsets = m['h'][0][0][38]
event_codes = m['h'][0][0][37]
labels = np.zeros((1, nsamples), int)
labels[0, event_onsets] = event_codes

cl_lab = ['left', 'right', 'foot', 'tongue']
cl1 = cl_lab[0]
cl2 = cl_lab[1]
cl3 = cl_lab[2]
cl4 = cl_lab[3]
nclasses = len(cl_lab)
nevents = len(event_onsets)

# Print some information
print('Shape of EEG:', EEG.shape)
print('Sample rate:', sample_rate)
print('Number of channels:', nchannels)
# print('Channel names:', channel_names)
print('Number of events:', len(event_onsets))
print('Event codes:', np.unique(event_codes))
print('Class labels:', cl_lab)
print('Number of classes:', nclasses)

# Dictionary to store the trials in, each class gets an entry
trials = {}

# The time window (in samples) to extract for each trial, here 0.5 -- 2.5 seconds
# win = np.arange(int(0.5 * sample_rate), int(2.5 * sample_rate))

win = np.arange(int(2 * sample_rate), int(5.5 * sample_rate))  # 2.1 -- 6.5, 1.5 -- 5.9
# Length of the time window
nsamples = len(win)

print("Using a window length of ", nsamples, " samples")

# Loop over the classes (right, foot)
for cl, code in zip(cl_lab, np.unique(event_codes)):

    # Extract the onsets for the class
    cl_onsets = event_onsets[event_codes == code]

    # Allocate memory for the trials
    trials[cl] = np.zeros((nchannels, nsamples, len(cl_onsets)))

    # Extract each trial
    for i, onset in enumerate(cl_onsets):
        trials[cl][:, :, i] = EEG[:, win + onset]

        array_sum = np.sum(trials[cl][:, :, i])
        array_has_nan = np.isnan(array_sum)

        if array_has_nan == True:
            print(i)
            error = trials[cl][:, :, i]
            trials[cl][:, :, i] = trials[cl][:, :, i - 1]

        # trials[cl][:, :, i] = trials[cl][:, :, i][np.logical_not(np.isnan(trials[cl][:, :, i]))]

# Some information about the dimensionality of the data (channels x time x trials)

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
train_labels = np.concatenate((train_lab_r, train_lab_l), axis=0)
x_train = np.transpose(train_samples, (2, 0, 1))


test_samples = np.concatenate((test[cl1], test[cl2]), axis=2)
test_lab_r = np.ones(test[cl1].shape[2], dtype=int)
test_lab_l = np.zeros(test[cl2].shape[2], dtype=int)
test_labels = np.concatenate((test_lab_r, test_lab_l),axis=0)

train_trans = np.transpose(train_samples, (2, 0, 1))
test_trans = np.transpose(test_samples, (2, 0, 1))

print('input shape', test_samples.shape)
print('transposed', x_train.shape)

model = EEGNet(nb_classes=2, Chans=3, Samples=875)

plot_model(model, show_shapes=True, to_file='./EEGNet_model.png')

model.compile(optimizer='Adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

fitted = model.fit(train_trans, train_labels, epochs=500,
                   validation_data=(test_trans, test_labels))

# predicted = model.predict()

test_loss, test_acc = model.evaluate(test_trans,  test_labels, verbose=2)