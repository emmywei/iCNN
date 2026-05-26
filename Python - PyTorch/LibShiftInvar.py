# Copyright (C) 2026 by Emmy S. Wei

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

USE_HIGH_PRECISION = False

if USE_HIGH_PRECISION:
    COMPLEX_TYPE = torch.complex128
    REAL_TYPE = torch.float64
    SMALL_ENOUGH = 1e-9
else:
    COMPLEX_TYPE = torch.complex64
    REAL_TYPE = torch.float32
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
    if isinstance(tensor, torch.Tensor):
        max3 = torch.amax(tensor, dim=3)
        max2 = torch.amax(max3, dim=2)
        max1 = torch.amax(max2, dim=1)
        return torch.argmax(max1)
    max3 = np.amax(tensor, 3)
    max2 = np.amax(max3, 2)
    max1 = np.amax(max2, 1)
    return np.argmax(max1)

def ArgMinBatchIndex(tensor):
    if isinstance(tensor, torch.Tensor):
        min3 = torch.amin(tensor, dim=3)
        min2 = torch.amin(min3, dim=2)
        min1 = torch.amin(min2, dim=1)
        return torch.argmin(min1)
    min3 = np.amin(tensor, 3)
    min2 = np.amin(min3, 2)
    min1 = np.amin(min2, 1)
    return np.argmin(min1)

####################################################################
# CircConv2D is adapted from:
# Copyright (C) 2019 by Stefan Schubert
# https://www.tu-chemnitz.de/etit/proaut/en/team/stefanSchubert.html
####################################################################
class CircConv2D(nn.Module):
    def __init__(self, filters, kernel_size, strides=(1, 1), activation='linear',
                 kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 in_channels=None):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.in_channels = in_channels
        self.conv = None
        if in_channels is not None:
            self._build(in_channels)

    def _build(self, in_channels):
        stride_tuple = self.strides if type(self.strides) is tuple else (self.strides, self.strides)
        self.conv = nn.Conv2d(in_channels, self.filters, self.kernel_size,
                              stride=stride_tuple, padding='same', bias=True,
                              dtype=REAL_TYPE)
        if self.kernel_initializer == 'glorot_uniform':
            nn.init.xavier_uniform_(self.conv.weight)
        self.in_channels = in_channels

    def forward(self, x):
        in_height = x.shape[1]
        in_width = x.shape[2]

        # left and right paddings
        num_left = (self.kernel_size[1] - 1) // 2
        num_right = self.kernel_size[1] - 1 - num_left
        if num_left > 0:
            pad_left = x[:, :, (in_width - num_left):, :]
        if num_right > 0:
            pad_right = x[:, :, :num_right, :]
        # add padding to incoming image
        if num_left > 0 and num_right < 1:
            x = torch.cat([pad_left, x], dim=2)
        elif num_left < 1 and num_right > 0:
            x = torch.cat([x, pad_right], dim=2)
        elif num_left > 0 and num_right > 0:
            x = torch.cat([pad_left, x, pad_right], dim=2)

        # top and bottom paddings
        num_top = (self.kernel_size[0] - 1) // 2
        num_bottom = self.kernel_size[0] - 1 - num_top
        if num_top > 0:
            pad_top = x[:, (in_height - num_top):, :, :]
        if num_bottom > 0:
            pad_bottom = x[:, :num_bottom, :, :]
        # add padding to incoming image
        if num_top > 0 and num_bottom < 1:
            x = torch.cat([pad_top, x], dim=1)
        elif num_top < 1 and num_bottom > 0:
            x = torch.cat([x, pad_bottom], dim=1)
        elif num_top > 0 and num_bottom > 0:
            x = torch.cat([pad_top, x, pad_bottom], dim=1)

        if self.conv is None:
            self._build(x.shape[3])
            self.conv = self.conv.to(device=x.device, dtype=x.dtype)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).contiguous()

        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'swish':
            x = F.silu(x)
        elif self.activation == 'linear':
            pass
        else:
            raise ValueError(f'Unsupported activation: {self.activation}')

        if type(self.strides) is tuple:
            stride_tuple = self.strides
        else:
            stride_tuple = (self.strides, self.strides)
        new_top = max(num_top // stride_tuple[0], min(1, num_top))
        new_left = max(num_left // stride_tuple[1], min(1, num_left))
        out_height = in_height // stride_tuple[0]
        out_width = in_width // stride_tuple[1]

        x = x[:, new_top:(new_top + out_height), new_left:(new_left + out_width), :]
        return x

####################################################################
# CircConv2DTrans is adapted from:
# Copyright (C) 2019 by Stefan Schubert
# https://www.tu-chemnitz.de/etit/proaut/en/team/stefanSchubert.html
####################################################################
class CircConv2DTrans(nn.Module):
    def __init__(self, filters, kernel_size, strides=(1, 1), activation='linear',
                 kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 in_channels=None):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.in_channels = in_channels
        self.conv = None
        if in_channels is not None:
            self._build(in_channels)

    def _build(self, in_channels):
        stride_tuple = self.strides if type(self.strides) is tuple else (self.strides, self.strides)
        if isinstance(self.kernel_size, tuple):
            padding_tuple = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        else:
            padding_tuple = self.kernel_size // 2

        if isinstance(stride_tuple, tuple):
            output_padding_tuple = (stride_tuple[0] - 1, stride_tuple[1] - 1)
        else:
            output_padding_tuple = stride_tuple - 1

        self.conv = nn.ConvTranspose2d(in_channels, self.filters, self.kernel_size,
                                       stride=stride_tuple, padding=padding_tuple,
                                       output_padding=output_padding_tuple, bias=True,
                                       dtype=REAL_TYPE)
        if self.kernel_initializer == 'glorot_uniform':
            nn.init.xavier_uniform_(self.conv.weight)
        self.in_channels = in_channels

    def forward(self, x):
        in_height = x.shape[1]
        in_width = x.shape[2]

        # left and right paddings
        num_right = (self.kernel_size[1] - 1) // 2
        num_left = self.kernel_size[1] - 1 - num_right
        if num_left > 0:
            pad_left = x[:, :, (in_width - num_left):, :]
        if num_right > 0:
            pad_right = x[:, :, :num_right, :]
        # add padding to incoming image
        if num_left > 0 and num_right < 1:
            x = torch.cat([pad_left, x], dim=2)
        elif num_left < 1 and num_right > 0:
            x = torch.cat([x, pad_right], dim=2)
        elif num_left > 0 and num_right > 0:
            x = torch.cat([pad_left, x, pad_right], dim=2)

        # top and bottom paddings
        num_bottom = (self.kernel_size[0] - 1) // 2
        num_top = self.kernel_size[0] - 1 - num_bottom
        if num_top > 0:
            pad_top = x[:, (in_height - num_top):, :, :]
        if num_bottom > 0:
            pad_bottom = x[:, :num_bottom, :, :]
        # add padding to incoming image
        if num_top > 0 and num_bottom < 1:
            x = torch.cat([pad_top, x], dim=1)
        elif num_top < 1 and num_bottom > 0:
            x = torch.cat([x, pad_bottom], dim=1)
        elif num_top > 0 and num_bottom > 0:
            x = torch.cat([pad_top, x, pad_bottom], dim=1)

        if self.conv is None:
            self._build(x.shape[3])
            self.conv = self.conv.to(device=x.device, dtype=x.dtype)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).contiguous()

        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'swish':
            x = F.silu(x)
        elif self.activation == 'linear':
            pass
        else:
            raise ValueError(f'Unsupported activation: {self.activation}')

        if type(self.strides) is tuple:
            stride_tuple = self.strides
        else:
            stride_tuple = (self.strides, self.strides)
        new_top = num_top * stride_tuple[0]
        new_left = num_left * stride_tuple[1]
        out_height = in_height * stride_tuple[0]
        out_width = in_width * stride_tuple[1]

        x = x[:, new_top:(new_top + out_height), new_left:(new_left + out_width), :]
        return x

# axis = 1 or 2, stride is fixed to 2
# dir = +1/-1 for right/left-looking
def CircMaxCalc1D(x, axis, relu_like_act='relu', dir=+1):
    if (dir != +1) and (dir != -1):
        print('ERROR: dir =', dir, 'is illegal!')
        return x
    y = torch.roll(x, shifts=-dir, dims=axis)
    y = torch.subtract(y, x)
    if relu_like_act == 'relu':
        y = F.relu(y)
    elif relu_like_act == 'swish':
        y = F.silu(y)
    else:
        print('ERROR:', relu_like_act, 'is not supported, default to relu')
        y = F.relu(y)
    return torch.add(x, y)

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
        y = torch.reshape(x, (-1, x.shape[1] // stride, 2, stride // 2, x.shape[2], x.shape[3]))
        z = y[:, :, 0, :, :, :]
        y = torch.reshape(z, (-1, x.shape[1] // 2, x.shape[2], x.shape[3]))
    elif axis == 2 and (stride % 2) == 0 and (x.shape[2] % stride) == 0:
        y = torch.reshape(x, (-1, x.shape[1], x.shape[2] // stride, 2, stride // 2, x.shape[3]))
        z = y[:, :, :, 0, :, :]
        y = torch.reshape(z, (-1, x.shape[1], x.shape[2] // 2, x.shape[3]))
    else:
        print('ERROR: axis must be either 1 or 2,')
        print('       stride must be even, and')
        print('       stride must divide image size!')
        return x
    return y

def DirectBiPart2D(x, strides=(2, 2)):
    x = DirectBiPart1D(x, 1, strides[0])
    x = DirectBiPart1D(x, 2, strides[1])
    return x

# axis = 1 or 2
def DirectDownSample1D(x, axis, stride=2):
    if axis == 1 and (x.shape[1] % stride) == 0:
        y = torch.reshape(x, (-1, x.shape[1] // stride, stride, x.shape[2], x.shape[3]))
        z = y[:, :, 0, :, :]
    elif axis == 2 and (x.shape[2] % stride) == 0:
        y = torch.reshape(x, (-1, x.shape[1], x.shape[2] // stride, stride, x.shape[3]))
        z = y[:, :, :, 0, :]
    else:
        print('ERROR: axis must be either 1 or 2, and')
        print('       stride must divide image size!')
        return x
    return z

def DirectDownSample2D(x, strides=(2, 2)):
    x = DirectDownSample1D(x, 1, strides[0])
    x = DirectDownSample1D(x, 2, strides[1])
    return x

# axis = 1 or 2
def FlagLargerTensor(x, y, axis, spec_points):
    xx = torch.permute(x, (3, 0, 1, 2))
    yy = torch.permute(y, (3, 0, 1, 2))
    xSpec = torch.abs(torch.fft.fft2(xx.to(COMPLEX_TYPE)))
    ySpec = torch.abs(torch.fft.fft2(yy.to(COMPLEX_TYPE)))
    for h in range(0, min(x.shape[1], spec_points[0])):
        for w in range(0, min(x.shape[2], spec_points[1])):
            xNorm = torch.linalg.vector_norm(xSpec[:, :, h, w], ord=1, dim=0)
            yNorm = torch.linalg.vector_norm(ySpec[:, :, h, w], ord=1, dim=0)
            bool_pos0 = torch.greater(yNorm, torch.tensor(SMALL_ENOUGH, device=yNorm.device, dtype=yNorm.dtype))
            bool_pos1 = torch.greater(yNorm, xNorm * (1 + SMALL_ENOUGH))
            flags_pos = torch.logical_and(bool_pos0, bool_pos1)
            flags_pos = flags_pos.to(torch.int32)
            bool_neg0 = torch.greater(xNorm, torch.tensor(SMALL_ENOUGH, device=xNorm.device, dtype=xNorm.dtype))
            bool_neg1 = torch.greater(xNorm, yNorm * (1 + SMALL_ENOUGH))
            flags_neg = torch.logical_and(bool_neg0, bool_neg1)
            flags_neg = flags_neg.to(torch.int32)
            flags_new = flags_pos - flags_neg
            glafs_new = 1 - torch.abs(flags_new)
            if h == 0 and w == 0:
                flags = flags_new
                glafs = glafs_new
            else:
                flags = flags * (1 - glafs) + flags_new * glafs
                glafs = glafs * glafs_new
    return torch.greater(flags, 0).to(torch.int32)

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
            z = torch.roll(y, shifts=direction*(2**stage), dims=axis)
            yy = torch.permute(y, (3, 1, 2, 0))
            zz = torch.permute(z, (3, 1, 2, 0))
            ff = flags[:, stage].to(dtype=y.dtype, device=y.device)
            yy = torch.multiply(yy, 1 - ff) + torch.multiply(zz, ff)
            y = torch.permute(yy, (3, 1, 2, 0))
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
            z = torch.roll(y, shifts=-(2**stage), dims=axis)
            y2 = DirectBiPart1D(y, axis, stride=2**(stage+1))
            z2 = DirectBiPart1D(z, axis, stride=2**(stage+1))
            ff = FlagLargerTensor(y2, z2, axis, spec_points)
            if stage == 0:
                flags = torch.reshape(ff, (-1, 1))
            else:
                flags = torch.cat([flags, torch.reshape(ff, (-1, 1))], dim=1)
            gg = ff.to(dtype=y.dtype, device=y.device)
            yy = torch.permute(y, (3, 1, 2, 0))
            zz = torch.permute(z, (3, 1, 2, 0))
            yy = yy * (1 - gg) + zz * gg
            y = torch.permute(yy, (3, 1, 2, 0))
    return [y, flags]

def OptimalShiftPrep2D(x, pool_stages, spec_points):
    [y, flags_axis1] = OptimalShiftPrep1D(x, 1, pool_stages[0], spec_points)
    [z, flags_axis2] = OptimalShiftPrep1D(y, 2, pool_stages[1], spec_points)
    return [flags_axis1, flags_axis2]

def SymmComboBack(x, shifts):
    y = torch.reshape(x, (shifts[0], shifts[1], -1, x.shape[1], x.shape[2], x.shape[3]))
    z = y[0, :]
    for shift0 in range(1, shifts[0]):
        w = y[shift0, :]
        w = torch.roll(w, -shift0, dims=2)
        z = torch.add(z, w)
    y = z[0, :]
    for shift1 in range(1, shifts[1]):
        w = z[shift1, :]
        w = torch.roll(w, -shift1, dims=2)
        y = torch.add(y, w)
    return y

def SymmComboFront(x, shifts):
    z = z0 = torch.unsqueeze(x, 0)
    for shift1 in range(1, shifts[1]):
        y = torch.roll(z0, shift1, dims=3)
        z = torch.cat([z, y], dim=0)
    y = y0 = torch.unsqueeze(z, 0)
    for shift0 in range(1, shifts[0]):
        z = torch.roll(y0, shift0, dims=3)
        y = torch.cat([y, z], dim=0)
    y = torch.reshape(y, (-1, x.shape[1], x.shape[2], x.shape[3]))
    return y

###############################################################################
if __name__ == '__main__':
    conv_std = nn.Conv2d(1, 1, (3, 3), stride=(1, 1), padding='same', dtype=REAL_TYPE)
    conv_circ = CircConv2D(1, (3, 3), strides=(1, 1), in_channels=1)
    print('image_data_format = NHWC tensors are used at function boundaries')

    v = [0.5, 1, 0.5]
    with torch.no_grad():
        conv_std.weight.zero_()
        for i in range(0, 3):
            for j in range(0, 3):
                conv_std.weight[0, 0, i, j] = v[i] * v[j]
        conv_std.bias.zero_()
        conv_circ.conv.weight.copy_(conv_std.weight)
        conv_circ.conv.bias.copy_(conv_std.bias)

    print('Shape of w0 =', conv_std.weight.shape)
    print('w0 =', conv_std.weight)

    test_image = np.loadtxt('image3221.txt', dtype=np.float64 if USE_HIGH_PRECISION else np.float32)
    x0 = AddNoise(test_image)
    print('Input x:')
    plt.imshow(x0, cmap='bwr')
    plt.colorbar()

    max_diff_std = 0
    max_diff_circ = 0
    x0 = torch.tensor(x0.reshape(1, 28, 28, 1), dtype=REAL_TYPE)
    y0_std = conv_std(x0.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
    y0_circ = conv_circ(x0)
    for shift in range(-14, 14 + 1):
        x = torch.roll(x0, shift, dims=2)
        y_std = conv_std(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        y_std = torch.roll(y_std, -shift, dims=2)
        y_circ = conv_circ(x)
        y_circ = torch.roll(y_circ, -shift, dims=2)
        max_diff_std = max(max_diff_std, torch.max(torch.abs(y_std - y0_std)).item())
        max_diff_circ = max(max_diff_circ, torch.max(torch.abs(y_circ - y0_circ)).item())
    print('max_diff_std =', max_diff_std)
    print('max_diff_circ =', max_diff_circ)
