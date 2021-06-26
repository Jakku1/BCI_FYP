import numpy as np
import scipy.io
import scipy.linalg as splg
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
           dropoutRate=0.5, kernLength=100, F1=16,
           D=4, F2=64, norm_rate=0.25, dropoutType='Dropout'):


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


def load_data(filename):

    m = scipy.io.loadmat(filename, struct_as_record=True)

    sample_rate = m['h']['SampleRate'][0][0][0][0]

    # load RAW EEG signal
    EEG = m['s'].T

    EEG = EEG[0:22, :]

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

    # Dictionary to store the trials in, each class gets an entry
    trials = {}

    # The time window (in samples) to extract for each trial, here 0.5 -- 2.5 seconds
    # win = np.arange(int(0.5 * sample_rate), int(2.5 * sample_rate))

    win = np.arange(int(2.5 * sample_rate), int(6 * sample_rate)) # 2.1 -- 6.5, 1.5 -- 5.9
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
                trials[cl][:, :, i] = trials[cl][:, :, i-1]

            # trials[cl][:, :, i] = trials[cl][:, :, i][np.logical_not(np.isnan(trials[cl][:, :, i]))]

    # Some information about the dimensionality of the data (channels x time x trials)

    return trials, cl1, cl2, sample_rate, nchannels, nsamples

import matplotlib.pyplot as plt

import scipy.signal


def bandpass(trials, lo, hi, sample_rate):

    a, b = scipy.signal.iirfilter(6, [lo / (sample_rate / 2.0), hi / (sample_rate / 2.0)])

    # Applying the filter to each trial
    ntrials = trials.shape[2]
    nchannels = trials.shape[0]
    nsamples = trials.shape[1]
    trials_filt = np.zeros((nchannels, nsamples, ntrials))
    for i in range(ntrials):
        trials_filt[:, :, i] = scipy.signal.filtfilt(a, b, trials[:, :, i], axis=1)

    return trials_filt

from numpy import linalg

# load data for all subjects
"""
subject1 = 'BCICIV_calib_ds1a.mat'
subject2 = 'BCICIV_calib_ds1b.mat'
subject3 = 'BCICIV_calib_ds1c.mat'
subject4 = 'BCICIV_calib_ds1d.mat'
subject5 = 'BCICIV_calib_ds1e.mat'
subject6 = 'BCICIV_calib_ds1f.mat'
subject7 = 'BCICIV_calib_ds1g.mat'
"""


# load data for all subjects
subject1 = 'A01T.mat'
subject2 = 'A02T.mat'
subject3 = 'A03T.mat'
subject4 = 'A04T.mat'
subject5 = 'A05T.mat'
subject6 = 'A06T.mat'
subject7 = 'A07T.mat'
subject8 = 'A08T.mat'
subject9 = 'A09T.mat'

# define library function keys
cl1 = 'left'
cl2 = 'right'

# trials_1 = load_data(subject1)
trials_1, cl1_1, cl2_1, sample_rate, nchannels, nsamples = load_data(subject1)
trials_2, cl1_2, cl2_2, sample_rate, nchannels, nsamples = load_data(subject2)
trials_3, cl1_3, cl2_3, sample_rate, nchannels, nsamples = load_data(subject3)
trials_4, cl1_4, cl2_4, sample_rate, nchannels, nsamples = load_data(subject4)
trials_5, cl1_5, cl2_5, sample_rate, nchannels, nsamples = load_data(subject5)
trials_6, cl1_6, cl2_6, sample_rate, nchannels, nsamples = load_data(subject6)
trials_7, cl1_7, cl2_7, sample_rate, nchannels, nsamples = load_data(subject7)
trials_8, cl1_8, cl2_8, sample_rate, nchannels, nsamples = load_data(subject8)
trials_9, cl1_9, cl2_9, sample_rate, nchannels, nsamples = load_data(subject9)
print(cl1_2, cl2_2)
print(cl1_3, cl2_3)
print(cl1_4, cl2_4)
print(cl1_7, cl2_7)

# add all trials to the same library
trials = [trials_1, trials_2, trials_3, trials_4, trials_5, trials_6, trials_7, trials_8, trials_9]
all_trials = {}
for k in trials_2.keys():
    all_trials[k] = np.concatenate(list(all_trials[k] for all_trials in trials), axis=2)

sample_rate = 250

# Apply a band pass filter to the data
trials_filt = {cl1: bandpass(all_trials[cl1], 8, 15, sample_rate),
               cl2: bandpass(all_trials[cl2], 8, 15, sample_rate)}

print('Shape of all_trials[cl1]:', all_trials['left'].shape)
print('Shape of all_trials[cl2]:', all_trials['right'].shape)

# K cross fold validation

from sklearn.model_selection import KFold

folds = KFold(n_splits=7, random_state=None, shuffle=True)

scores = []

index_list = list(range(0, all_trials[cl1].shape[2]))

for train_index, test_index in folds.split(index_list):
    train = {cl1: trials_filt[cl1][:, :, train_index],
            cl2: trials_filt[cl2][:, :, train_index]}

    test = {cl1: trials_filt[cl1][:, :, test_index],
            cl2: trials_filt[cl2][:, :, test_index]}

    print(train[cl1].shape, train[cl2].shape, test[cl1].shape, test[cl2].shape)
    print(train[cl1].shape[2] + train[cl2].shape[2] + test[cl1].shape[2] + test[cl2].shape[2])

    # train model for CSP Projection

    train_samples = np.concatenate((train[cl1], train[cl2]), axis=2)
    train_lab_r = np.ones(train[cl1].shape[2], dtype=int)
    train_lab_l = np.zeros(train[cl2].shape[2], dtype=int)
    train_labels = np.concatenate((train_lab_r, train_lab_l), axis=0)
    x_train = np.transpose(train_samples, (2, 0, 1))

    test_samples = np.concatenate((test[cl1], test[cl2]), axis=2)
    test_lab_r = np.ones(test[cl1].shape[2], dtype=int)
    test_lab_l = np.zeros(test[cl2].shape[2], dtype=int)
    test_labels = np.concatenate((test_lab_r, test_lab_l), axis=0)

    train_trans = np.transpose(train_samples, (2, 0, 1))
    test_trans = np.transpose(test_samples, (2, 0, 1))

    print('input shape', train_samples.shape)
    print('transposed', x_train.shape)

    model = EEGNet(nb_classes=2, Chans=22, Samples=875)

    # plot_model(model, show_shapes=True, to_file='./EEGNet_model.png')

    model.compile(optimizer='Adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    fitted = model.fit(train_trans, train_labels, epochs=30,
                       validation_data=(test_trans, test_labels))