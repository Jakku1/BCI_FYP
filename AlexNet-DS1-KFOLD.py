import numpy as np
import scipy.io
import scipy.linalg as splg
import tensorflow as tf
import scipy.io
import scipy.signal
from keras.optimizers import SGD
# opt = SGD(lr=0.01)
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
import numpy as np
import scipy.io
from keras.optimizers import SGD
opt = SGD(lr=0.0001)


def AlexNet():
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
               activation='relu')(x)

    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=2, strides=2)(x)

    # Define 3rd tensorflow block
    x = Conv2D(filters=8, # 384
               kernel_size=3,
               padding='same',
               activation='relu')(x)

    # Define 4th tensorflow block
    x = Conv2D(filters=8, # 384
               kernel_size=3,
               padding='same',
               activation='relu')(x)

    # Define 5th tensorflow block
    x = Conv2D(filters=8, #256
               kernel_size=3,
               padding='same',
               activation='relu')(x)

    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=1, strides=2)(x) # originally pool_size=3

    # Define Fully Connected Layers
    x = Flatten()(x)

    x = Dense(units=128, activation='relu')(x) # originally 4096
    x = Dense(units=128, activation='relu')(x)
    x = Dropout(rate=0.5)(x)

    # Define Output Layer
    # Equal to the number of classes
    output_layer = Dense(units=2, activation='sigmoid')(x)

    return Model(inputs=input_layer, outputs=output_layer)


# Load data for training

# Insert code that can split data, define train and test sets

def load_data(file):

    m = scipy.io.loadmat(file, struct_as_record=True)

    # SciPy.io.loadmat does not deal well with Matlab structures, resulting in lots of
    # extra dimensions in the arrays. This makes the code a bit more cluttered

    print("Loading Data...")

    sample_rate = m['nfo']['fs'][0][0][0][0]
    EEG = m['cnt'].T
    nchannels, nsamples = EEG.shape

    event_onsets = m['mrk'][0][0][0]
    event_codes = m['mrk'][0][0][1]
    labels = np.zeros((1, nsamples), int)
    labels[0, event_onsets] = event_codes

    cl_lab = [s[0] for s in m['nfo']['classes'][0][0][0]]
    cl1 = cl_lab[0]
    cl2 = cl_lab[1]

    # Dictionary to store the trials in, each class gets an entry
    trials = {}

    win = np.arange(int(1 * sample_rate), int(3.5* sample_rate))

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

    return trials, cl1, cl2


import scipy.signal


def bandpass(trials, lo, hi, sample_rate):

    a, b = scipy.signal.iirfilter(6, [lo / (sample_rate / 2.0), hi / (sample_rate / 2.0)])

    # Applying the filter to each trial
    ntrials = trials.shape[2]
    nchannels = trials.shape[0]
    nsamples = trials.shape[1]
    ntrials = trials.shape[2]
    trials_filt = np.zeros((nchannels, nsamples, ntrials))
    for i in range(ntrials):
        trials_filt[:, :, i] = scipy.signal.filtfilt(a, b, trials[:, :, i], axis=1)

    return trials_filt


# load all data

# load data for all subjects
subject1 = 'BCICIV_calib_ds1a.mat'
subject2 = 'BCICIV_calib_ds1b.mat'
subject3 = 'BCICIV_calib_ds1c.mat'
subject4 = 'BCICIV_calib_ds1d.mat'
subject5 = 'BCICIV_calib_ds1e.mat'
subject6 = 'BCICIV_calib_ds1f.mat'
subject7 = 'BCICIV_calib_ds1g.mat'

# define library function keys
cl1 = 'left'
cl2 = 'right'

# trials_1 = load_data(subject1)
trials_2, cl1_2, cl2_2 = load_data(subject2)
trials_3, cl1_3, cl2_3 = load_data(subject3)
trials_4, cl1_4, cl2_4 = load_data(subject4)
# trials_5 = load_data(subject5)
trials_6, cl1_6, cl2_6 = load_data(subject6)
trials_7, cl1_7, cl2_7 = load_data(subject7)
print(cl1_2, cl2_2)
print(cl1_3, cl2_3)
print(cl1_4, cl2_4)
print(cl1_7, cl2_7)

# add all trials to the same library
trials = [trials_2, trials_3, trials_4, trials_7]
all_trials = {}
for k in trials_2.keys():
    all_trials[k] = np.concatenate(list(all_trials[k] for all_trials in trials), axis=2)

sample_rate = 100

# trials = split_data(EEG, nchannels, nsamples, event_onsets, win)

trials_filt = {cl1: bandpass(all_trials[cl1], 8, 15, sample_rate),
               cl2: bandpass(all_trials[cl2], 8, 15, sample_rate)}

from sklearn.model_selection import KFold

# folds = StratifiedKFold(n_splits=10)
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

    # Splitting the frequency filtered signal into a train and test set

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

    print('input shape', test_samples.shape)
    print('transposed', x_train.shape)

    model = AlexNet()

    # plot_model(model, show_shapes=True, to_file='./EEGNet_model.png')

    model.compile(optimizer='Adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    fitted = model.fit(train_trans, train_labels, epochs=100,
                       validation_data=(test_trans, test_labels))

    # predicted = model.predict()

    test_loss, test_acc = model.evaluate(test_trans, test_labels, verbose=2)