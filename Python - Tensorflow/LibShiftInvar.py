# Copyright (C) 2022 by Emmy S. Wei

import keras.backend as KB
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

USE_HIGH_PRECISION = False

if USE_HIGH_PRECISION:
    COMPLEX_TYPE = 'complex128'
    REAL_TYPE = 'float64'
    SMALL_ENOUGH = 1e-9
else:
    COMPLEX_TYPE = 'complex64'
    REAL_TYPE = 'float32'
    SMALL_ENOUGH = 1e-5

####################################################################
# AddNoise is adapted from:
# Copyright (C) 2021 by Santiago L. Valdarrama
# https://keras.io/examples/vision/autoencoder/
####################################################################
# Add random noise to array of images
def AddNoise(array, noise_factor=0.4):
    noisy_array = array + noise_factor * np.random.normal(
                  loc=0.0, scale=1.0, size=array.shape)
    return np.clip(noisy_array, 0.0, 1.0)

def ArgMaxBatchIndex(tensor):
    max3 = np.amax(tensor, 3)
    max2 = np.amax(max3, 2)
    max1 = np.amax(max2, 1)
    return tf.argmax(max1)

def ArgMinBatchIndex(tensor):
    min3 = np.amin(tensor, 3)
    min2 = np.amin(min3, 2)
    min1 = np.amin(min2, 1)
    return tf.argmin(min1)

####################################################################
# CircConv2D is adapted from:
# Copyright (C) 2019 by Stefan Schubert
# https://www.tu-chemnitz.de/etit/proaut/en/team/stefanSchubert.html
####################################################################
def CircConv2D(filters, kernel_size, strides=(1, 1), activation='linear',
               kernel_initializer='glorot_uniform', kernel_regularizer=None):
    def CircConv2D_inner(x):
        in_height = x.shape[1]
        in_width = x.shape[2]

        # left and right paddings
        num_left = (kernel_size[1] - 1) // 2
        num_right = kernel_size[1] - 1 - num_left
        if num_left > 0:
            pad_left = x[:, :, (in_width - num_left):, :]
        if num_right > 0:
            pad_right = x[:, :, :num_right, :]
        # add padding to incoming image
        if num_left > 0 and num_right < 1:
            x = tf.concat([pad_left, x], axis=2)
        elif num_left < 1 and num_right > 0:
            x = tf.concat([x, pad_right], axis=2)
        elif num_left > 0 and num_right > 0:
            x = tf.concat([pad_left, x, pad_right], axis=2)

        # top and bottom paddings
        num_top = (kernel_size[0] - 1) // 2
        num_bottom = kernel_size[0] - 1 - num_top
        if num_top > 0:
            pad_top = x[:, (in_height - num_top):, :, :]
        if num_bottom > 0:
            pad_bottom = x[:, :num_bottom, :, :]
        # add padding to incoming image
        if num_top > 0 and num_bottom < 1:
            x = tf.concat([pad_top, x], axis=1)
        elif num_top < 1 and num_bottom > 0:
            x = tf.concat([x, pad_bottom], axis=1)
        elif num_top > 0 and num_bottom > 0:
            x = tf.concat([pad_top, x, pad_bottom], axis=1)

        x = layers.Conv2D(
            filters=filters, kernel_size=kernel_size,
            strides=strides, activation=activation,
            padding='same',
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            dtype=REAL_TYPE)(x)

        if type(strides) is tuple:
            stride_tuple = strides
        else:
            stride_tuple = (strides, strides)
        new_top = max(num_top // stride_tuple[0], min(1, num_top))
        new_left = max(num_left // stride_tuple[1], min(1, num_left))
        out_height = in_height // stride_tuple[0]
        out_width = in_width // stride_tuple[1]

        x = x[:, new_top:(new_top + out_height), new_left:(new_left + out_width), :]
        return x
    return CircConv2D_inner

####################################################################
# CircConv2DTrans is adapted from:
# Copyright (C) 2019 by Stefan Schubert
# https://www.tu-chemnitz.de/etit/proaut/en/team/stefanSchubert.html
####################################################################
def CircConv2DTrans(filters, kernel_size, strides=(1, 1), activation='linear',
                    kernel_initializer='glorot_uniform', kernel_regularizer=None):
    def CircConv2DTrans_inner(x):
        in_height = x.shape[1]
        in_width = x.shape[2]

        # left and right paddings
        num_right = (kernel_size[1] - 1) // 2
        num_left = kernel_size[1] - 1 - num_right
        if num_left > 0:
            pad_left = x[:, :, (in_width - num_left):, :]
        if num_right > 0:
            pad_right = x[:, :, :num_right, :]
        # add padding to incoming image
        if num_left > 0 and num_right < 1:
            x = tf.concat([pad_left, x], axis=2)
        elif num_left < 1 and num_right > 0:
            x = tf.concat([x, pad_right], axis=2)
        elif num_left > 0 and num_right > 0:
            x = tf.concat([pad_left, x, pad_right], axis=2)

        # top and bottom paddings
        num_bottom = (kernel_size[0] - 1) // 2
        num_top = kernel_size[0] - 1 - num_bottom
        if num_top > 0:
            pad_top = x[:, (in_height - num_top):, :, :]
        if num_bottom > 0:
            pad_bottom = x[:, :num_bottom, :, :]
        # add padding to incoming image
        if num_top > 0 and num_bottom < 1:
            x = tf.concat([pad_top, x], axis=1)
        elif num_top < 1 and num_bottom > 0:
            x = tf.concat([x, pad_bottom], axis=1)
        elif num_top > 0 and num_bottom > 0:
            x = tf.concat([pad_top, x, pad_bottom], axis=1)

        x = layers.Conv2DTranspose(
            filters=filters, kernel_size=kernel_size,
            strides=strides, activation=activation,
            padding='same',
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            dtype=REAL_TYPE)(x)

        if type(strides) is tuple:
            stride_tuple = strides
        else:
            stride_tuple = (strides, strides)
        new_top = num_top * stride_tuple[0]
        new_left = num_left * stride_tuple[1]
        out_height = in_height * stride_tuple[0]
        out_width = in_width * stride_tuple[1]

        x = x[:, new_top:(new_top + out_height), new_left:(new_left + out_width), :]
        return x
    return CircConv2DTrans_inner

# axis = 1 or 2, stride is fixed to 2
# dir = +1/-1 for right/left-looking
def CircMaxCalc1D(x, axis, relu_like_act='relu', dir=+1):
    if (dir != +1) and (dir != -1):
        print('ERROR: dir =', dir, 'is illegal!')
        return x
    y = tf.roll(x, shift=-dir, axis=axis)
    y = tf.math.subtract(y, x)
    if relu_like_act == 'relu':
        y = tf.nn.relu(y)
    elif relu_like_act == 'swish':
        y = tf.nn.swish(y)
    else:
        print('ERROR:', relu_like_act, 'is not supported, default to relu')
        y = tf.nn.relu(y)
    return tf.math.add(x, y)

def CircMaxCalc2D(x, relu_like_act):
    x = CircMaxCalc1D(x, 1, relu_like_act, +1)
    x = CircMaxCalc1D(x, 2, relu_like_act, +1)
    return x

def CircMaxPool2D(x, relu_like_act):
    x = CircMaxCalc2D(x, relu_like_act)
    x = DirectDownSample2D(x)
    return x

# axis = 1 or 2
def DirectBiPart1D(x, axis, stride=2):
    if axis == 1 and (stride % 2) == 0 and (x.shape[1] % stride) == 0:
        y = KB.reshape(x, (-1, x.shape[1] // stride, 2, stride // 2, x.shape[2], x.shape[3]))
        z = y[:, :, 0, :, :, :]
        y = KB.reshape(z, (-1, x.shape[1] // 2, x.shape[2], x.shape[3]))
    elif axis == 2 and (stride % 2) == 0 and (x.shape[2] % stride) == 0:
        y = KB.reshape(x, (-1, x.shape[1], x.shape[2] // stride, 2, stride // 2, x.shape[3]))
        z = y[:, :, :, 0, :, :]
        y = KB.reshape(z, (-1, x.shape[1], x.shape[2] // 2, x.shape[3]))
    else:
        print("ERROR: axis must be either 1 or 2,")
        print("       stride must be even, and")
        print("       stride must divide image size!")
        return x
    return y

def DirectBiPart2D(x, strides=(2, 2)):
    x = DirectBiPart1D(x, 1, strides[0])
    x = DirectBiPart1D(x, 2, strides[1])
    return x

# axis = 1 or 2
def DirectDownSample1D(x, axis, stride=2):
    if axis == 1 and (x.shape[1] % stride) == 0:
        y = KB.reshape(x, (-1, x.shape[1] // stride, stride, x.shape[2], x.shape[3]))
        z = y[:, :, 0, :, :]
    elif axis == 2 and (x.shape[2] % stride) == 0:
        y = KB.reshape(x, (-1, x.shape[1], x.shape[2] // stride, stride, x.shape[3]))
        z = y[:, :, :, 0, :]
    else:
        print("ERROR: axis must be either 1 or 2, and")
        print("       stride must divide image size!")
        return x
    return z

def DirectDownSample2D(x, strides=(2, 2)):
    x = DirectDownSample1D(x, 1, strides[0])
    x = DirectDownSample1D(x, 2, strides[1])
    return x

# axis = 1 or 2
def FlagLargerTensor(x, y, axis, spec_points):
    xx = tf.transpose(x, perm=[3, 0, 1, 2])
    yy = tf.transpose(y, perm=[3, 0, 1, 2])
    xSpec = tf.abs(tf.signal.fft2d(tf.cast(xx, COMPLEX_TYPE)))
    ySpec = tf.abs(tf.signal.fft2d(tf.cast(yy, COMPLEX_TYPE)))
    for h in range(0, min(x.shape[1], spec_points[0])):
        for w in range(0, min(x.shape[2], spec_points[1])):
            xNorm = tf.norm(xSpec[:, :, h, w], ord=1, axis=0)
            yNorm = tf.norm(ySpec[:, :, h, w], ord=1, axis=0)
            bool_pos0 = tf.math.greater(yNorm, SMALL_ENOUGH)
            bool_pos1 = tf.math.greater(yNorm, xNorm * (1 + SMALL_ENOUGH))
            flags_pos = tf.math.logical_and(bool_pos0, bool_pos1)
            flags_pos = tf.cast(flags_pos, 'int32')
            bool_neg0 = tf.math.greater(xNorm, SMALL_ENOUGH)
            bool_neg1 = tf.math.greater(xNorm, yNorm * (1 + SMALL_ENOUGH))
            flags_neg = tf.math.logical_and(bool_neg0, bool_neg1)
            flags_neg = tf.cast(flags_neg, 'int32')
            flags_new = flags_pos - flags_neg
            glafs_new = 1 - tf.abs(flags_new)
            if h == 0 and w == 0:
                flags = flags_new
                glafs = glafs_new
            else:
                flags = flags * (1 - glafs) + flags_new * glafs
                glafs = glafs * glafs_new
    return tf.cast(tf.math.greater(flags, 0), 'int32')

def OptimalShiftBack2D(x, pool_stages, flags_axis1, flags_axis2):
    y = x
    if pool_stages[1] > 0:
        y = OptimalShiftBiDir1D(y, 2, pool_stages[1], flags_axis2, +1)
    z = y
    if pool_stages[0] > 0:
        z = OptimalShiftBiDir1D(z, 1, pool_stages[0], flags_axis1, +1)
    return z

# axis = 1 or 2, flags are 0 or 1-valued
# direction = -1 / +1 for front and back
def OptimalShiftBiDir1D(x, axis, pool_stages, flags, direction):
    y = x
    if pool_stages > 0:
        for stage in range(0, pool_stages):
            z = tf.roll(y, shift=direction*(2**stage), axis=axis)
            yy = tf.transpose(y, perm=[3, 1, 2, 0])
            zz = tf.transpose(z, perm=[3, 1, 2, 0])
            ff = tf.cast(flags[:, stage], REAL_TYPE)
            yy = tf.multiply(yy, 1 - ff) + tf.multiply(zz, ff)
            y = tf.transpose(yy, perm=[3, 1, 2, 0])
    return y

# stride is fixed to 2, pool_stages=(m0, m1), spec_points=(n0, n1)
def OptimalShiftFront2D(x, pool_stages, spec_points):
    [flags_axis1, flags_axis2] = OptimalShiftPrep2D(x, pool_stages, spec_points)
    y = x
    if pool_stages[0] > 0:
        y = OptimalShiftBiDir1D(y, 1, pool_stages[0], flags_axis1, -1)
    z = y
    if pool_stages[1] > 0:
        z = OptimalShiftBiDir1D(z, 2, pool_stages[1], flags_axis2, -1)
    return [z, flags_axis1, flags_axis2]

# axis = 1 or 2, stride is fixed to 2
def OptimalShiftPrep1D(x, axis, pool_stages, spec_points):
    [y, flags] = [x, []]
    if pool_stages > 0:
        for stage in range(0, pool_stages):
            z = tf.roll(y, shift=-(2**stage), axis=axis)
            y2 = DirectBiPart1D(y, axis, stride=2**(stage+1))
            z2 = DirectBiPart1D(z, axis, stride=2**(stage+1))
            ff = FlagLargerTensor(y2, z2, axis, spec_points)
            if stage == 0:
                flags = KB.reshape(ff, (-1, 1))
            else:
                flags = tf.concat([flags, KB.reshape(ff, (-1, 1))], axis=1)
            gg = tf.cast(ff, REAL_TYPE)
            yy = tf.transpose(y, perm=[3, 1, 2, 0])
            zz = tf.transpose(z, perm=[3, 1, 2, 0])
            yy = yy * (1 - gg) + zz * gg
            y = tf.transpose(yy, perm=[3, 1, 2, 0])
    return [y, flags]

def OptimalShiftPrep2D(x, pool_stages, spec_points):
    [y, flags_axis1] = OptimalShiftPrep1D(x, 1, pool_stages[0], spec_points)
    [z, flags_axis2] = OptimalShiftPrep1D(y, 2, pool_stages[1], spec_points)
    return [flags_axis1, flags_axis2]

def SymmComboBack(x, shifts):
    y = KB.reshape(x, (shifts[0], shifts[1], -1, x.shape[1], x.shape[2], x.shape[3]))
    z = y[0, :]
    for shift0 in range(1, shifts[0]):
        w = y[shift0, :]
        w = tf.roll(w, -shift0, axis=2)
        z = tf.math.add(z, w)
    y = z[0, :]
    for shift1 in range(1, shifts[1]):
        w = z[shift1, :]
        w = tf.roll(w, -shift1, axis=2)
        y = tf.math.add(y, w)
    return y

def SymmComboFront(x, shifts):
    z = z0 = tf.expand_dims(x, 0)
    for shift1 in range(1, shifts[1]):
        y = tf.roll(z0, shift1, axis=3)
        z = tf.concat([z, y], axis=0)
    y = y0 = tf.expand_dims(z, 0)
    for shift0 in range(1, shifts[0]):
        z = tf.roll(y0, shift0, axis=3)
        y = tf.concat([y, z], axis=0)
    y = KB.reshape(y, (-1, x.shape[1], x.shape[2], x.shape[3]))
    return y

###############################################################################
if __name__ == '__main__':
    in_x = layers.Input((28, 28, 1), dtype=REAL_TYPE)
    out_std = layers.Conv2D(1, (3, 3), strides=(1, 1), padding='same', dtype=REAL_TYPE)(in_x)
    out_circ = CircConv2D(1, (3, 3), strides=(1, 1))(in_x)
    conv_std = keras.models.Model(in_x, out_std)
    conv_circ = keras.models.Model(in_x, out_circ)
    print('image_data_format =', keras.backend.image_data_format())

    v = [0.5, 1, 0.5]
    w = conv_std.get_weights()
    w0 = w[0]
    for i in range(0, 3):
        for j in range(0, 3):
            w0[i, j, 0, 0] = v[i] * v[j]
    # shape of weight = (rows, cols, input_depth, output_depth)
    print('Shape of w0 =', tf.shape(w0))
    print('w0 =', w0)
    print('w =', w)
    conv_circ.set_weights(w)

    test_image = np.loadtxt('image3221.txt', dtype=REAL_TYPE)
    x0 = AddNoise(test_image)
    print('Input x:')
    plt.imshow(x0, cmap='bwr') # cmap='bwr', 'cool', 'gray'
    plt.colorbar()

    max_diff_std = 0
    max_diff_circ = 0
    x0 = x0.reshape(1, 28, 28, 1)
    y0_std = conv_std.predict(x0)
    y0_circ = conv_circ.predict(x0)
    for shift in range(-14, 14 + 1):
        x = tf.roll(x0, shift, axis=2)
        y_std = conv_std.predict(x)
        y_std = tf.roll(y_std, -shift, axis=2)
        y_circ = conv_circ.predict(x)
        y_circ = tf.roll(y_circ, -shift, axis=2)
        max_diff_std = max(max_diff_std, np.max(np.abs(y_std - y0_std)))
        max_diff_circ = max(max_diff_circ, np.max(np.abs(y_circ - y0_circ)))
    print('max_diff_std =', max_diff_std)
    print('max_diff_circ =', max_diff_circ)