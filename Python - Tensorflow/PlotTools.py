import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

import numpy as np
import tensorflow as tf

def DisplaySamples2(array1, array2, n=5, file_name=''):
    indices = np.random.randint(len(array1), size=n)
    print("indices =", indices)
    images1 = array1[indices, :]
    images2 = array2[indices, :]
    # figsize=(width_in_inch, height_in_inch)
    fig2n = plt.figure(figsize=(n*2, 2*2))
    grid2n = AxesGrid(fig2n, 111,
                      nrows_ncols=(2, n),
                      axes_pad=0.05,
                      cbar_mode='single',
                      cbar_location='right',
                      cbar_pad=0.1
                     )
    for i, ax in enumerate(grid2n):
        ax.set_axis_off()
        if i < n:
            img = images1[i]
        else:
            img = images2[i - n]
        # cmap='bwr', 'cool', 'gray', 'Spectral', ...
        im = ax.imshow(img.reshape(28, 28), cmap='bwr', vmin=0, vmax=1)
    cbar = ax.cax.colorbar(im)
    cbar = grid2n.cbar_axes[0].colorbar(im)
    cbar.ax.set_yticks(np.arange(0, 1.1, 0.5))
    cbar.ax.set_yticklabels(['0.0', '0.5', '1.0'])
    if len(file_name) > 0:
        plt.savefig(file_name, bbox_inches='tight')
    plt.show()

def DisplaySamples3(array1, array2, array3, n=5, file_name=''):
    indices = np.random.randint(len(array1), size=n)
    print("indices =", indices)
    images1 = array1[indices, :]
    images2 = array2[indices, :]
    images3 = array3[indices, :]
    # figsize=(width_in_inch, height_in_inch)
    fig3n = plt.figure(figsize=(n*2, 3*2))
    grid3n = AxesGrid(fig3n, 111,
                      nrows_ncols=(3, n),
                      axes_pad=0.05,
                      cbar_mode='single',
                      cbar_location='right',
                      cbar_pad=0.1
                     )
    for i, ax in enumerate(grid3n):
        ax.set_axis_off()
        if i / n < 1:
            img = images1[i]
        elif i / n < 2:
            img = images2[i - n]
        else:
            img = images3[i - n - n]
        # cmap='bwr', 'cool', 'gray', 'Spectral', ...
        im = ax.imshow(img.reshape(28, 28), cmap='bwr', vmin=0, vmax=1)
    cbar = ax.cax.colorbar(im)
    cbar = grid3n.cbar_axes[0].colorbar(im)
    cbar.ax.set_yticks(np.arange(0, 1.1, 0.5))
    cbar.ax.set_yticklabels(['0.0', '0.5', '1.0'])
    if len(file_name) > 0:
        plt.savefig(file_name, bbox_inches='tight')
    plt.show()

def DisplayShiftVariance(test_model, test_data, noise_data,
                         predictions_zero, predictions_shift,
                         shift, iBadRSV):
    test_zero = test_data[iBadRSV, :, :, 0]
    noisy_zero = noise_data[iBadRSV, :, :, 0]
    decoded_zero = predictions_zero[iBadRSV, :, :, 0]
    test_shift = tf.roll(test_zero, shift, axis=1)
    noisy_shift = tf.roll(noisy_zero, shift, axis=1)
    decoded_shift = predictions_shift[iBadRSV, :, :, 0]
    images23 = [test_zero, noisy_zero, decoded_zero,
                test_shift, noisy_shift, decoded_shift]
    # figsize=(width_in_inch, height_in_inch)
    fig23 = plt.figure(figsize=(3*2, 2*2))
    grid23 = AxesGrid(fig23, 111,
                      nrows_ncols=(2, 3),
                      axes_pad=0.05,
                      cbar_mode='single',
                      cbar_location='right',
                      cbar_pad=0.1
                     )
    for i, ax in enumerate(grid23):
        ax.set_axis_off()
        # cmap='bwr', 'cool', 'gray', 'Spectral', ...
        im = ax.imshow(images23[i], cmap='bwr', vmin=0, vmax=1)
    cbar = ax.cax.colorbar(im)
    cbar = grid23.cbar_axes[0].colorbar(im)
    cbar.ax.set_yticks(np.arange(0, 1.1, 0.5))
    cbar.ax.set_yticklabels(['0.0', '0.5', '1.0'])

    decoded_back = tf.roll(decoded_shift, -shift, axis=1)
    shift_variance = decoded_back - decoded_zero
    print('Max shift variance =', np.max(shift_variance))
    print('Min shift variance =', np.min(shift_variance))
    # figsize=(width_in_inch, height_in_inch)
    fig13 = plt.figure(figsize=(3*2, 1*2))
    for i, img in enumerate([decoded_zero, decoded_back, shift_variance]):
        ax = plt.subplot(1, 3, i + 1)
        ax.set_axis_off()
        aa = plt.imshow(img, cmap='bwr')
        fig13.colorbar(aa, shrink=0.7)
    plt.savefig(test_model + 'ImageVariance.png')
    plt.show()

def PlotShiftVariance(test_model, test_result, y_label):
    plt.figure()
    plt.plot(np.array(range(-14, 14 + 1)), test_result)
    plt.xlim([-14, 14])
    plt.grid()
    plt.xlabel('horizontal shift [pixels]', fontname='Times New Roman', fontsize=14)
    plt.ylabel(y_label, fontname='Times New Roman', fontsize=14)
    plt.title(y_label + ' vs. shift', fontname='Times New Roman', fontsize=14)
    plt.savefig(test_model + y_label + '.png')
    plt.show()