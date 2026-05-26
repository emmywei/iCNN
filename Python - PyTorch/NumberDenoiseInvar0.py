###############################################################
# Copyright (C) 2026 by Emmy S. Wei
# A shift-invariant denoiser using an autoencoder adapted from:
# Copyright (C) 2021 by Santiago L. Valdarrama
# https://keras.io/examples/vision/autoencoder/
###############################################################

import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets

import LibShiftInvar as libinv
import PlotTools as PT
from LibShiftInvar import REAL_TYPE

##########################
# Model control parameters
epochs_denoiser = 10
epochs_reader = 60
n_code_filters = 32
random_seed = 23
batch_size = 128
##########################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_set = datasets.MNIST("./data", train=True, download=True)
test_set = datasets.MNIST("./data", train=False, download=True)

train_data = train_set.data.numpy().astype(np.float32) / 255.0
train_data = np.reshape(train_data, (len(train_data), 28, 28, 1))
noisy_train = libinv.AddNoise(train_data)

test_data = test_set.data.numpy().astype(np.float32) / 255.0
test_data = np.reshape(test_data, (len(test_data), 28, 28, 1))
noisy_test = libinv.AddNoise(test_data)

train_data = torch.tensor(train_data, dtype=REAL_TYPE)
test_data = torch.tensor(test_data, dtype=REAL_TYPE)
noisy_train = torch.tensor(noisy_train, dtype=REAL_TYPE)
noisy_test = torch.tensor(noisy_test, dtype=REAL_TYPE)

train_read = train_set.targets.clone().detach().long()
test_read = test_set.targets.clone().detach().long()

# Examples of original and noise-corrupted images
PT.DisplaySamples2(train_data, noisy_train, 5, 'NoisyDataSamples.png')

# relu_like_act = 'relu' or 'swish'
class Denoiser(nn.Module):
    def __init__(self, relu_like_act):
        super().__init__()
        self.pool_stages = (0, 2)
        self.conv1 = libinv.CircConv2D(n_code_filters, (3, 3), activation=relu_like_act, in_channels=1)
        self.conv2 = libinv.CircConv2D(n_code_filters, (3, 3), activation=relu_like_act, in_channels=n_code_filters)
        self.deconv1 = libinv.CircConv2DTrans( n_code_filters, (3, 3), strides=2, activation=relu_like_act, in_channels=n_code_filters)
        self.deconv2 = libinv.CircConv2DTrans(n_code_filters, (3, 3), strides=2, activation=relu_like_act, in_channels=n_code_filters)
        self.conv3 = libinv.CircConv2D(1, (3, 3),in_channels=n_code_filters)

    def forward(self, x):
        x, flags_axis1, flags_axis2 = libinv.OptimalShiftFront2D(x, self.pool_stages, (2, 2))
        x = self.conv1(x)
        x = libinv.DirectDownSample2D(x)
        x = self.conv2(x)
        x = libinv.DirectDownSample2D(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.conv3(x)
        x = libinv.OptimalShiftBack2D(x, self.pool_stages, flags_axis1, flags_axis2)
        x = torch.sigmoid(x)
        return x

# relu_like_act = 'relu' or 'swish'
class Reader(nn.Module):
    def __init__(self, relu_like_act):
        super().__init__()
        self.pool_stages = (0, 2)
        self.conv1 = libinv.CircConv2D(5, (3, 3), activation=relu_like_act, in_channels=1)
        self.conv2 = libinv.CircConv2D(5, (3, 3), activation=relu_like_act, in_channels=5)
        self.conv3 = libinv.CircConv2D(5, (3, 3), activation=relu_like_act, in_channels=5)
        self.conv4 = libinv.CircConv2D(5, (3, 3), activation=relu_like_act, in_channels=5)
        self.conv5 = libinv.CircConv2D(10, (3, 3), activation=relu_like_act, in_channels=5)

    def forward(self, x):
        x, flags_axis1, flags_axis2 = libinv.OptimalShiftFront2D(x, self.pool_stages, (2, 2))
        x = self.conv1(x)
        x = self.conv2(x)
        x = libinv.DirectDownSample2D(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = libinv.DirectDownSample2D(x)
        x = self.conv5(x)
        x = torch.mean(x, dim=(1, 2))
        return x


def make_loader(x, y, shuffle=True):
    dataset = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs):
    loss_list = []
    val_loss_list = []

    for epoch in range(epochs):
        model.train()
        total = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            total += loss.item() * x.shape[0]

        train_loss = total / len(train_loader.dataset)
        loss_list.append(train_loss)

        model.eval()
        total = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                pred = model(x)
                loss = loss_fn(pred, y)

                total += loss.item() * x.shape[0]

        val_loss = total / len(val_loader.dataset)
        val_loss_list.append(val_loss)

        print(f"epoch {epoch + 1}/{epochs}: loss = {train_loss}, val_loss = {val_loss}")

    return loss_list, val_loss_list

def predict(model, x):
    model.eval()
    outputs = []

    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = x[i:i + batch_size].to(device)
            outputs.append(model(xb).cpu())

    return torch.cat(outputs, dim=0)

def evaluate_reader(model, x, y):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    loader = make_loader(x, y, shuffle=False)

    total = 0.0
    correct = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = loss_fn(logits, yb)

            total += loss.item() * xb.shape[0]
            correct += (torch.argmax(logits, dim=1) == yb).sum().item()

    return total / len(loader.dataset), correct / len(loader.dataset)

denoiser = Denoiser('swish').to(device)
optimizer = optim.Adam(denoiser.parameters())
mse = nn.MSELoss()

print(denoiser)

history1_loss, history1_val_loss = train_model(
    model=denoiser,
    train_loader=make_loader(train_data, train_data, shuffle=True),
    val_loader=make_loader(test_data, test_data, shuffle=False),
    loss_fn=mse,
    optimizer=optimizer,
    epochs=epochs_denoiser
)

torch.save(denoiser.state_dict(), 'Denoiser1.pt')

decoded_test = predict(denoiser, test_data)
PT.DisplaySamples2(test_data, decoded_test, 5, 'EnDecoderSamples.png')

MAE = torch.mean(torch.abs(test_data - decoded_test)).item()
print('EnDecoder MeanAbsoluteError =', MAE)

history2_loss, history2_val_loss = train_model(
    model=denoiser,
    train_loader=make_loader(noisy_train, train_data, shuffle=True),
    val_loader=make_loader(noisy_test, test_data, shuffle=False),
    loss_fn=mse,
    optimizer=optimizer,
    epochs=epochs_denoiser * 2
)

torch.save(denoiser.state_dict(), 'Denoiser2.pt')

denoised_test = predict(denoiser, noisy_test)
PT.DisplaySamples3(test_data, noisy_test, denoised_test, 5, 'DenoiserSamples.png')

MAE = torch.mean(torch.abs(test_data - denoised_test)).item()
print('Denoiser MeanAbsoluteError =', MAE)

denoiser_loss_list = history1_loss + history2_loss
denoiser_val_loss_list = history1_val_loss + history2_val_loss

denoiser_loss_file = open('LossInvar.txt', 'w')
for loss in denoiser_loss_list:
    denoiser_loss_file.write(str(loss) + '\n')
denoiser_loss_file.close()

denoiser_val_loss_file = open('ValLossInvar.txt', 'w')
for loss in denoiser_val_loss_list:
    denoiser_val_loss_file.write(str(loss) + '\n')
denoiser_val_loss_file.close()

plt.plot(range(1, 1 + len(denoiser_loss_list)), denoiser_loss_list, label='loss')
plt.plot(range(1, 1 + len(denoiser_loss_list)), denoiser_val_loss_list, label='val_loss')
plt.legend(fontsize=14)
plt.xlabel('epochs', fontsize=14)
plt.ylabel('MSE losses', fontsize=14)
plt.grid()
plt.savefig('DenoiserLosses.png')
plt.show()

MAE_denoiser = []
RSV_L1_denoiser = []
RSV_Linf_denoiser = []

denoised_test_zero = predict(denoiser, noisy_test)

for shift in range(-14, 14 + 1):
    noisy_test_shift = torch.roll(noisy_test, shifts=shift, dims=2)
    test_data_shift = torch.roll(test_data, shifts=shift, dims=2)

    denoised_test_shift = predict(denoiser, noisy_test_shift)

    MAE = torch.mean(torch.abs(denoised_test_shift - test_data_shift)).item()
    MAE_denoiser.append(MAE)

    denoised_test_zero_shift = torch.roll(denoised_test_zero, shifts=shift, dims=2)
    result_diff = denoised_test_shift - denoised_test_zero_shift

    RSV_L1 = torch.mean(torch.abs(result_diff)).item()
    RSV_L1_denoiser.append(RSV_L1)

    RSV_Linf = torch.max(torch.abs(result_diff)).item()
    RSV_Linf_denoiser.append(RSV_Linf)

    print('shift =', shift, ', MAE =', MAE, ', RSV_L1 =', RSV_L1, ', RSV_Linf =', RSV_Linf)

    if shift == -11:
        iBadRSV = libinv.ArgMaxBatchIndex(torch.abs(result_diff))
        print('iBadRSV =', iBadRSV.item())

        PT.DisplayShiftVariance(
            'Denoiser',
            test_data,
            noisy_test,
            denoised_test_zero,
            denoised_test_shift,
            shift,
            iBadRSV
        )

MAE_denoiser = np.array(MAE_denoiser)
RSV_L1_denoiser = np.array(RSV_L1_denoiser)
RSV_Linf_denoiser = np.array(RSV_Linf_denoiser)

PT.PlotShiftVariance('Denoiser', MAE_denoiser, 'MAE')
PT.PlotShiftVariance('Denoiser', RSV_L1_denoiser, 'RSV_L1')
PT.PlotShiftVariance('Denoiser', RSV_Linf_denoiser, 'RSV_Linf')

denoised_train = predict(denoiser, noisy_train)
denoised_test = predict(denoiser, noisy_test)

reader = Reader('swish').to(device)
reader_optimizer = optim.Adam(reader.parameters())
ce = nn.CrossEntropyLoss()

print(reader)

history3_loss, history3_val_loss = train_model(
    model=reader,
    train_loader=make_loader(denoised_train, train_read, shuffle=True),
    val_loader=make_loader(denoised_test, test_read, shuffle=False),
    loss_fn=ce,
    optimizer=reader_optimizer,
    epochs=epochs_reader
)

torch.save(reader.state_dict(), 'Reader.pt')

denoised_reader_loss_list = history3_loss
denoised_reader_val_loss_list = history3_val_loss

denoised_reader_loss_file = open('DenoisedReaderLoss.txt', 'w')
for loss in denoised_reader_loss_list:
    denoised_reader_loss_file.write(str(loss) + '\n')
denoised_reader_loss_file.close()

denoised_reader_val_loss_file = open('DenoisedReaderValLoss.txt', 'w')
for loss in denoised_reader_val_loss_list:
    denoised_reader_val_loss_file.write(str(loss) + '\n')
denoised_reader_val_loss_file.close()

accuracy_file = open('AccuracyCombo.txt', 'w')

print('Test reader on denoised_test:')
score = evaluate_reader(reader, denoised_test, test_read)
print('loss =', score[0], 'accuracy =', score[1])
accuracy_file.write('reader on denoised_test: loss = ' + str(score[0]) + '\n')
accuracy_file.write('reader on denoised_test: accuracy = ' + str(score[1]) + '\n')
accuracy_file.write('\n')

print('Test reader on noisy_test:')
score = evaluate_reader(reader, noisy_test, test_read)
print('loss =', score[0], 'accuracy =', score[1])
accuracy_file.write('reader on noisy_test: loss = ' + str(score[0]) + '\n')
accuracy_file.write('reader on noisy_test: accuracy = ' + str(score[1]) + '\n')
accuracy_file.close()

history4_loss, history4_val_loss = train_model(
    model=reader,
    train_loader=make_loader(noisy_train, train_read, shuffle=True),
    val_loader=make_loader(noisy_test, test_read, shuffle=False),
    loss_fn=ce,
    optimizer=reader_optimizer,
    epochs=epochs_reader
)

noisy_reader_loss_list = history4_loss
noisy_reader_val_loss_list = history4_val_loss

noisy_reader_loss_file = open('NoisyReaderLoss.txt', 'w')
for loss in noisy_reader_loss_list:
    noisy_reader_loss_file.write(str(loss) + '\n')
noisy_reader_loss_file.close()

noisy_reader_val_loss_file = open('NoisyReaderValLoss.txt', 'w')
for loss in noisy_reader_val_loss_list:
    noisy_reader_val_loss_file.write(str(loss) + '\n')
noisy_reader_val_loss_file.close()

accuracy_file = open('AccuracyDirect.txt', 'w')

print('Test reader on noisy_test:')
score = evaluate_reader(reader, noisy_test, test_read)
print('loss =', score[0], 'accuracy =', score[1])
accuracy_file.write('reader on noisy_test: loss = ' + str(score[0]) + '\n')
accuracy_file.write('reader on noisy_test: accuracy = ' + str(score[1]) + '\n')
accuracy_file.close()