###############################################################
# Copyright (C) 2022 by Emmy S. Wei
# A shift-invariant denoiser using an autoencoder adapted from:
# Copyright (C) 2021 by Santiago L. Valdarrama
# https://keras.io/examples/vision/autoencoder/
###############################################################

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

import LibShiftInvar as libinv
import PlotTools as PT

from LibShiftInvar import REAL_TYPE

##########################
# Model control parameters
epochs_denoiser = 10
n_code_filters = 32
random_seed = 23
##########################

np.random.seed(random_seed)
tf.keras.utils.set_random_seed(random_seed)
tf.config.experimental.enable_op_determinism()

(train_data, train_read), (test_data, test_read) = mnist.load_data()

train_data = train_data.astype(REAL_TYPE) / 255.0
train_data = np.reshape(train_data, (len(train_data), 28, 28, 1))
noisy_train = libinv.AddNoise(train_data)

test_data = test_data.astype(REAL_TYPE) / 255.0
test_data = np.reshape(test_data, (len(test_data), 28, 28, 1))
noisy_test = libinv.AddNoise(test_data)

train_read = keras.utils.to_categorical(train_read, 10)
test_read = keras.utils.to_categorical(test_read, 10)

# Examples of original and noise-corrupted images
PT.DisplaySamples2(train_data, noisy_train, 5, 'NoisyDataSamples.png')

# relu_like_act = 'relu' or 'swish'
def CreateDenoiserModel(relu_like_act):
    inputs = layers.Input(shape=(28, 28, 1), dtype=REAL_TYPE)
    # Encoder
    pool_stages = (0, 1)
    symm_copies = (1, 2)
    [x, flags_axis1, flags_axis2] = libinv.OptimalShiftFront2D(inputs, pool_stages, (2, 2))
    x = libinv.CircConv2D(n_code_filters, (3, 3), activation=relu_like_act)(x)
    x = libinv.DirectDownSample2D(x)
    x = libinv.SymmComboFront(x, symm_copies)
    x = libinv.CircConv2D(n_code_filters, (3, 3), activation=relu_like_act)(x)
    x = libinv.DirectDownSample2D(x)
    # Decoder
    x = libinv.CircConv2DTrans(n_code_filters, (3, 3), strides=2, activation=relu_like_act)(x)
    x = libinv.SymmComboBack(x, symm_copies)
    x = libinv.CircConv2DTrans(n_code_filters, (3, 3), strides=2, activation=relu_like_act)(x)
    x = libinv.CircConv2D(1, (3, 3))(x)
    x = libinv.OptimalShiftBack2D(x, pool_stages, flags_axis1, flags_axis2)
    # Final sigmoid
    outputs = layers.Activation('sigmoid', dtype=REAL_TYPE)(x)
    return keras.Model(inputs, outputs)

denoiser = CreateDenoiserModel('swish')
# denoiser.compile(optimizer="adam", loss="binary_crossentropy")
denoiser.compile(optimizer='adam', loss='mean_squared_error')
denoiser.summary()
keras.utils.plot_model(denoiser, to_file='Denoiser.png', show_shapes=True)

history1 = denoiser.fit( x=train_data, y=train_data,
                         epochs=epochs_denoiser,
                         batch_size=128, shuffle=True,
                         validation_data=(test_data, test_data)
                       )
denoiser.save('Denoiser1')

decoded_test = denoiser.predict(test_data)
PT.DisplaySamples2(test_data, decoded_test, 5, 'EnDecoderSamples.png')
MAE = tf.keras.losses.MeanAbsoluteError()(test_data, decoded_test).numpy()
print('EnDecoder MeanAbsoluteError =', MAE)

history2 = denoiser.fit( x=noisy_train, y=train_data,
                         epochs=epochs_denoiser * 2,
                         batch_size=128, shuffle=True,
                         validation_data=(noisy_test, test_data),
                       )
denoiser.save('Denoiser2')

denoised_test = denoiser.predict(noisy_test)
PT.DisplaySamples3(test_data, noisy_test, denoised_test, 5, 'DenoiserSamples.png')
MAE = tf.keras.losses.MeanAbsoluteError()(test_data, denoised_test).numpy()
print('Denoiser MeanAbsoluteError =', MAE)

denoiser_loss_list = history1.history['loss'] + history2.history['loss']
denoiser_val_loss_list = history1.history['val_loss'] + history2.history['val_loss']

denoiser_loss_file = open('LossInvar.txt', 'w')
for loss in denoiser_loss_list:
    denoiser_loss_file.write(str(loss) + '\n')
denoiser_loss_file.close()
denoiser_val_loss_file = open('ValLossInvar.txt', 'w')
for loss in denoiser_val_loss_list:
    denoiser_val_loss_file.write(str(loss) + '\n')
denoiser_val_loss_file.close()

plt.plot(range(1, 1+len(denoiser_loss_list)), denoiser_loss_list, label='loss')
plt.plot(range(1, 1+len(denoiser_loss_list)), denoiser_val_loss_list, label='val_loss')
plt.legend(fontsize=14)
plt.xlabel('epochs', fontsize=14)
plt.ylabel('MSE losses', fontsize=14)
plt.grid()
plt.savefig('DenoiserLosses.png')
plt.show()

MAE_denoiser = []
RSV_L1_denoiser = []
RSV_Linf_denoiser = []
denoised_test_zero = denoiser.predict(noisy_test)
for shift in range(-14, 14 + 1):
    noisy_test_shift = tf.roll(noisy_test, shift, axis=2)
    test_data_shift = tf.roll(test_data, shift, axis=2)
    denoised_test_shift = denoiser.predict(noisy_test_shift)
    fMAE = tf.keras.losses.MeanAbsoluteError()
    MAE = fMAE(denoised_test_shift, test_data_shift).numpy()
    MAE_denoiser.append(MAE)
    denoised_test_zero_shift = tf.roll(denoised_test_zero, shift, axis=2)
    RSV_L1 = fMAE(denoised_test_shift, denoised_test_zero_shift).numpy()
    RSV_L1_denoiser.append(RSV_L1)
    result_diff = denoised_test_shift - denoised_test_zero_shift
    RSV_Linf = np.max(np.abs(result_diff))
    RSV_Linf_denoiser.append(RSV_Linf)
    print('shift =', shift, ', MAE =', MAE, ', RSV_L1 =', RSV_L1, ', RSV_Linf =', RSV_Linf)
    if shift == -11:
        iBadRSV = libinv.ArgMaxBatchIndex(np.abs(result_diff))
        print('iBadRSV =', iBadRSV)
        PT.DisplayShiftVariance( 'Denoiser', test_data, noisy_test,
                                 denoised_test_zero, denoised_test_shift,
                                 shift, iBadRSV
                               )
MAE_denoiser = np.array(MAE_denoiser)
RSV_L1_denoiser = np.array(RSV_L1_denoiser)
RSV_Linf_denoiser = np.array(RSV_Linf_denoiser)
PT.PlotShiftVariance('Denoiser', MAE_denoiser, 'MAE')
PT.PlotShiftVariance('Denoiser', RSV_L1_denoiser, 'RSV_L1')
PT.PlotShiftVariance('Denoiser', RSV_Linf_denoiser, 'RSV_Linf')
