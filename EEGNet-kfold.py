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
           dropoutRate=0.2, kernLength=100, F1=8,
           D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
    """ Keras Implementation of EEGNet
    http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta
    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version. For example:

        1. Depthwise Convolutions to learn spatial filters within a
        temporal convolution. The use of the depth_multiplier option maps
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn
        spatial filters within each filter in a filter-bank. This also limits
        the number of free parameters to fit when compared to a fully-connected
        convolution.

        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions.


    While the original paper used Dropout, we found that SpatialDropout2D
    sometimes produced slightly better results for classification of ERP
    signals. However, SpatialDropout2D significantly reduced performance
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.

    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the
    kernel lengths for double the sampling rate, etc). Note that we haven't
    tested the model performance with this rule so this may not work well.

    The model with default parameters gives the EEGNet-8,2 model as discussed
    in the paper. This model should do pretty well in general, although it is
	advised to do some model searching to get optimal performance on your
	particular dataset.
    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of this
    parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D for
    overcomplete). We believe the main parameters to focus on are F1 and D.
    Inputs:

      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D.
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.
    """

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

def load_data(file):

    m = scipy.io.loadmat(file, struct_as_record=True)

    # SciPy.io.loadmat does not deal well with Matlab structures, resulting in lots of
    # extra dimensions in the arrays. This makes the code a bit more cluttered

    print("Loading Data...")

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
    # win = np.arange(int(0.5 * sample_rate), int(2.5 * sample_rate))

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

    # Some information about the dimensionality of the data (channels x time x trials)
    print('Shape of trials[cl1]:', trials[cl1].shape)
    print('Shape of trials[cl2]:', trials[cl2].shape)

    return trials, cl1, cl2


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

    model = EEGNet(nb_classes=2, Chans=59, Samples=250)

    # plot_model(model, show_shapes=True, to_file='./EEGNet_model.png')

    model.compile(optimizer='Adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    fitted = model.fit(train_trans, train_labels, epochs=100,
                       validation_data=(test_trans, test_labels))

    # predicted = model.predict()

    test_loss, test_acc = model.evaluate(test_trans,  test_labels, verbose=2)