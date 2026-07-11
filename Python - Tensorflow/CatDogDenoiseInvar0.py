###############################################################
# Copyright (C) 2026 by Emmy S. Wei
# Cat/dog Invar U-Net denoiser with an EfficientNet reader.
###############################################################

import gc
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

import LibShiftInvar as libinv
import PlotToolsCatDog as PT
from LibShiftInvar import REAL_TYPE


program_start = time.time()

##########################
# Model control parameters
epochs_denoiser = 10
epochs_reader = 10
reader_batch_size = 4
run_shift_analysis = True
random_seed = 23
batch_size = 16
image_size = 128
max_images = None
data_dir = './PetImages'
max_shift = image_size // 2
##########################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


###############################################################
# Dataset
###############################################################

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

full_set = datasets.ImageFolder(data_dir, transform=transform)

if max_images is not None:
    keep = torch.randperm(
        len(full_set),
        generator=torch.Generator().manual_seed(random_seed)
    )[:max_images]

    full_set = torch.utils.data.Subset(
        full_set,
        keep.tolist()
    )

train_size = int(0.8 * len(full_set))
test_size = len(full_set) - train_size

train_set, test_set = torch.utils.data.random_split(
    full_set,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(random_seed)
)


def dataset_to_tensors(dataset):
    images = []
    labels = []

    for image, label in dataset:
        images.append(image.permute(1, 2, 0).numpy())
        labels.append(label)

    x = torch.tensor(
        np.stack(images).astype(np.float32),
        dtype=REAL_TYPE
    )

    y = torch.tensor(
        labels,
        dtype=torch.long
    )

    return x, y


train_data, train_read = dataset_to_tensors(train_set)
test_data, test_read = dataset_to_tensors(test_set)

noisy_train = torch.tensor(
    libinv.AddNoise(train_data.numpy()),
    dtype=REAL_TYPE
)

noisy_test = torch.tensor(
    libinv.AddNoise(test_data.numpy()),
    dtype=REAL_TYPE
)

classes = (
    full_set.dataset.classes
    if isinstance(full_set, torch.utils.data.Subset)
    else full_set.classes
)

print('classes =', classes)
print('train_data shape =', train_data.shape)
print('test_data shape =', test_data.shape)

train_indices = np.array([100, 101, 102, 103, 104])

PT.DisplaySamples2(
    train_data,
    noisy_train,
    file_name='NoisyCatDogSamplesInvar.png',
    indices=train_indices
)


###############################################################
# Invar U-Net denoiser
###############################################################

class Denoiser(nn.Module):
    def __init__(self, relu_like_act='swish'):
        super().__init__()

        self.relu_like_act = relu_like_act

        # There are two factor-2 downsampling stages, so the total
        # downsampling factor is 4.
        self.pool_stages = (0, 2)

        self.enc1 = nn.Sequential(
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1,
                padding_mode='circular'
            ),
            nn.SiLU(),

            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                padding=1,
                padding_mode='circular'
            ),
            nn.SiLU()
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
                padding_mode='circular'
            ),
            nn.SiLU(),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1,
                padding_mode='circular'
            ),
            nn.SiLU()
        )

        self.middle = nn.Sequential(
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1,
                padding_mode='circular'
            ),
            nn.SiLU(),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1,
                padding_mode='circular'
            ),
            nn.SiLU()
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec2 = nn.Sequential(
            nn.Conv2d(
                128,
                64,
                kernel_size=3,
                padding=1,
                padding_mode='circular'
            ),
            nn.SiLU(),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1,
                padding_mode='circular'
            ),
            nn.SiLU()
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec1 = nn.Sequential(
            nn.Conv2d(
                64,
                32,
                kernel_size=3,
                padding=1,
                padding_mode='circular'
            ),
            nn.SiLU(),

            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                padding=1,
                padding_mode='circular'
            ),
            nn.SiLU()
        )

        self.out = nn.Conv2d(
            32,
            3,
            kernel_size=1
        )

    @staticmethod
    def direct_downsample(x):
        # LibShiftInvar downsampling expects channel-last tensors.
        x = x.permute(0, 2, 3, 1).contiguous()
        x = libinv.DirectDownSample2D(x)
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        # OptimalShiftFront2D receives [N, H, W, C].
        x, flags_axis1, flags_axis2 = libinv.OptimalShiftFront2D(
            x,
            self.pool_stages,
            (2, 2)
        )

        # The U-Net body uses [N, C, H, W].
        x = x.permute(0, 3, 1, 2).contiguous()

        e1 = self.enc1(x)
        p1 = self.direct_downsample(e1)

        e2 = self.enc2(p1)
        p2 = self.direct_downsample(e2)

        middle = self.middle(p2)

        u2 = self.up2(middle)
        d2 = self.dec2(
            torch.cat([u2, e2], dim=1)
        )

        u1 = self.up1(d2)
        d1 = self.dec1(
            torch.cat([u1, e1], dim=1)
        )

        out = self.out(d1)

        # OptimalShiftBack2D expects channel-last tensors.
        out = out.permute(0, 2, 3, 1).contiguous()

        out = libinv.OptimalShiftBack2D(
            out,
            self.pool_stages,
            flags_axis1,
            flags_axis2
        )

        return torch.sigmoid(out)


###############################################################
# Shared training and evaluation functions
###############################################################

def make_loader(
    x,
    y,
    shuffle=True,
    loader_batch_size=None
):
    if loader_batch_size is None:
        loader_batch_size = batch_size

    dataset = torch.utils.data.TensorDataset(x, y)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=loader_batch_size,
        shuffle=shuffle
    )


def format_time(total_seconds):
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60

    return f'{hours:02d}:{minutes:02d}:{seconds:05.2f}'


def train_model(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    epochs,
    block_name='Training'
):
    loss_list = []
    val_loss_list = []
    epoch_times = []

    training_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        model.train()
        total = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            pred = model(xb)
            loss = loss_fn(pred, yb)

            loss.backward()
            optimizer.step()

            total += loss.item() * xb.shape[0]

        train_loss = total / len(train_loader.dataset)
        loss_list.append(train_loss)

        model.eval()
        total = 0.0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb)
                loss = loss_fn(pred, yb)

                total += loss.item() * xb.shape[0]

        val_loss = total / len(val_loader.dataset)
        val_loss_list.append(val_loss)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        print(
            f'epoch {epoch + 1}/{epochs}: '
            f'loss = {train_loss}, '
            f'val_loss = {val_loss}, '
            f'epoch_time = {epoch_time:.2f} sec'
        )

    total_epoch_time = time.time() - training_start
    avg_epoch_time = float(np.mean(epoch_times))

    print(
        f'\n{block_name} average epoch time = '
        f'{avg_epoch_time:.2f} sec'
    )

    print(
        f'{block_name} total epoch time = '
        f'{total_epoch_time:.2f} sec '
        f'({total_epoch_time / 60:.2f} min)'
    )

    print(
        f'{block_name} total epoch time formatted = '
        f'{format_time(total_epoch_time)}\n'
    )

    return loss_list, val_loss_list, epoch_times


def predict(model, x):
    model.eval()
    outputs = []

    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = x[i:i + batch_size].to(device)
            outputs.append(
                model(xb).cpu()
            )

    return torch.cat(outputs, dim=0)


###############################################################
# EfficientNet reader
###############################################################

class EfficientNetReader(nn.Module):
    def __init__(self):
        super().__init__()

        self.reader = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )

        self.reader.classifier[1] = nn.Linear(
            self.reader.classifier[1].in_features,
            2
        )

        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=torch.float32
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=torch.float32
        ).view(1, 3, 1, 1)

        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        # Input: [N, H, W, C], values in [0, 1]
        x = x.permute(0, 3, 1, 2).contiguous().float()
        x = (x - self.mean) / self.std

        return self.reader(x)


def make_pretrained_reader():
    return EfficientNetReader()


def evaluate_accuracy(model, x, y):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        loader = make_loader(
            x,
            y,
            shuffle=False,
            loader_batch_size=reader_batch_size
        )

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            predicted_labels = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                predicted_labels == yb
            ).sum().item()

            total += yb.shape[0]

    return correct / total


###############################################################
# Train Invar denoiser
###############################################################

denoiser = Denoiser('swish').to(device)

optimizer = optim.Adam(
    denoiser.parameters(),
    lr=1e-3
)

print(denoiser)


def denoise_loss(pred, target):
    mse_loss = nn.functional.mse_loss(
        pred,
        target
    )

    l1_loss = nn.functional.l1_loss(
        pred,
        target
    )

    return 0.5 * mse_loss + 0.5 * l1_loss


history_loss, history_val_loss, denoiser_epoch_times = train_model(
    model=denoiser,

    train_loader=make_loader(
        noisy_train,
        train_data,
        shuffle=True
    ),

    val_loader=make_loader(
        noisy_test,
        test_data,
        shuffle=False
    ),

    loss_fn=denoise_loss,
    optimizer=optimizer,
    epochs=epochs_denoiser * 3,
    block_name='CatDogUNetInvarDenoiser training'
)

torch.save(
    denoiser.state_dict(),
    'CatDogUNetDenoiserInvar.pt'
)

denoised_test = predict(
    denoiser,
    noisy_test
)

test_indices = np.array(
    [100, 101, 102, 103, 104]
)

PT.DisplaySamples3(
    test_data,
    noisy_test,
    denoised_test,
    file_name='CatDogDenoiserSamplesInvar.png',
    indices=test_indices
)

mae = torch.mean(
    torch.abs(test_data - denoised_test)
).item()

print(
    'Denoiser MeanAbsoluteError =',
    mae
)

denoiser_loss_list = history_loss
denoiser_val_loss_list = history_val_loss

with open('CatDogLossInvar.txt', 'w') as file:
    for loss in denoiser_loss_list:
        file.write(str(loss) + '\n')

with open('CatDogValLossInvar.txt', 'w') as file:
    for loss in denoiser_val_loss_list:
        file.write(str(loss) + '\n')

with open('CatDogEpochTimesInvar.txt', 'w') as file:
    for epoch_time in denoiser_epoch_times:
        file.write(str(epoch_time) + '\n')

    file.write(
        'average_epoch_time_sec = '
        + str(float(np.mean(denoiser_epoch_times)))
        + '\n'
    )

    file.write(
        'total_epoch_time_sec = '
        + str(float(np.sum(denoiser_epoch_times)))
        + '\n'
    )

plt.plot(
    range(1, 1 + len(denoiser_loss_list)),
    denoiser_loss_list,
    label='loss'
)

plt.plot(
    range(1, 1 + len(denoiser_val_loss_list)),
    denoiser_val_loss_list,
    label='val_loss'
)

plt.legend(fontsize=14)
plt.xlabel('epochs', fontsize=14)
plt.ylabel('0.5 MSE + 0.5 L1 loss', fontsize=14)
plt.grid()
plt.savefig('CatDogDenoiserLossesInvar.png')
plt.show()


###############################################################
# Shift-variance analysis
###############################################################

if run_shift_analysis:
    mae_denoiser = []
    rsv_l1_denoiser = []
    rsv_linf_denoiser = []
    shift_times = []

    denoised_test_zero = predict(
        denoiser,
        noisy_test
    )

    shift_analysis_start = time.time()

    for shift in range(
        -max_shift,
        max_shift + 1
    ):
        shift_start = time.time()

        noisy_test_shift = torch.roll(
            noisy_test,
            shifts=shift,
            dims=2
        )

        test_data_shift = torch.roll(
            test_data,
            shifts=shift,
            dims=2
        )

        denoised_test_shift = predict(
            denoiser,
            noisy_test_shift
        )

        mae_shift = torch.mean(
            torch.abs(
                denoised_test_shift
                - test_data_shift
            )
        ).item()

        mae_denoiser.append(mae_shift)

        denoised_test_zero_shift = torch.roll(
            denoised_test_zero,
            shifts=shift,
            dims=2
        )

        result_diff = (
            denoised_test_shift
            - denoised_test_zero_shift
        )

        rsv_l1 = torch.mean(
            torch.abs(result_diff)
        ).item()

        rsv_l1_denoiser.append(rsv_l1)

        rsv_linf = torch.max(
            torch.abs(result_diff)
        ).item()

        rsv_linf_denoiser.append(rsv_linf)

        shift_time = time.time() - shift_start
        shift_times.append(shift_time)

        print(
            'shift =',
            shift,
            ', MAE =',
            mae_shift,
            ', RSV_L1 =',
            rsv_l1,
            ', RSV_Linf =',
            rsv_linf,
            ', shift_time =',
            f'{shift_time:.2f} sec'
        )

        if shift == -min(11, max_shift):
            i_bad_rsv = libinv.ArgMaxBatchIndex(
                torch.abs(result_diff)
            )

            print(
                'iBadRSV =',
                i_bad_rsv.item()
            )

            PT.DisplayShiftVariance(
                'CatDogDenoiserInvar',
                test_data,
                noisy_test,
                denoised_test_zero,
                denoised_test_shift,
                shift,
                i_bad_rsv
            )

    shift_analysis_total = (
        time.time() - shift_analysis_start
    )

    avg_shift_time = float(
        np.mean(shift_times)
    )

    print(
        f'\nAverage shift evaluation time = '
        f'{avg_shift_time:.2f} sec'
    )

    print(
        f'Total shift evaluation time = '
        f'{shift_analysis_total:.2f} sec '
        f'({shift_analysis_total / 60:.2f} min)'
    )

    print(
        f'Total shift evaluation time formatted = '
        f'{format_time(shift_analysis_total)}'
    )

    with open('CatDogShiftTimesInvar.txt', 'w') as file:
        for shift_time in shift_times:
            file.write(str(shift_time) + '\n')

        file.write(
            'average_shift_time_sec = '
            + str(avg_shift_time)
            + '\n'
        )

        file.write(
            'total_shift_time_sec = '
            + str(shift_analysis_total)
            + '\n'
        )

    mae_denoiser = np.array(
        mae_denoiser
    )

    rsv_l1_denoiser = np.array(
        rsv_l1_denoiser
    )

    rsv_linf_denoiser = np.array(
        rsv_linf_denoiser
    )

    PT.PlotShiftVariance(
        'CatDogDenoiserInvar',
        mae_denoiser,
        'MAE',
        max_shift
    )

    PT.PlotShiftVariance(
        'CatDogDenoiserInvar',
        rsv_l1_denoiser,
        'RSV_L1',
        max_shift
    )

    PT.PlotShiftVariance(
        'CatDogDenoiserInvar',
        rsv_linf_denoiser,
        'RSV_Linf',
        max_shift
    )

    del (
        mae_denoiser,
        rsv_l1_denoiser,
        rsv_linf_denoiser,
        denoised_test_zero,
        shift_times
    )

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

else:
    print(
        'Skipping shift analysis because '
        'run_shift_analysis = False'
    )


###############################################################
# Train the reader only on clean cat/dog images
###############################################################

# Recompute the final denoised test images after denoiser training.
denoised_test = predict(
    denoiser,
    noisy_test
)

# Move the denoiser off the GPU before reader training.
denoiser = denoiser.cpu()

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

reader = make_pretrained_reader().to(device)

reader_optimizer = optim.Adam(
    reader.parameters(),
    lr=1e-4
)

reader_loss_fn = nn.CrossEntropyLoss()

reader_loss, reader_val_loss, reader_epoch_times = train_model(
    model=reader,

    # Train only on clean training images.
    train_loader=make_loader(
        train_data,
        train_read,
        shuffle=True,
        loader_batch_size=reader_batch_size
    ),

    # Validate only on clean test images.
    val_loader=make_loader(
        test_data,
        test_read,
        shuffle=False,
        loader_batch_size=reader_batch_size
    ),

    loss_fn=reader_loss_fn,
    optimizer=reader_optimizer,
    epochs=epochs_reader,
    block_name='EfficientNet reader training on clean images'
)

torch.save(
    reader.state_dict(),
    'CatDogCleanTrainedEfficientNetReaderInvar.pt'
)


###############################################################
# Evaluate the same reader on all test sets
###############################################################

reader_clean_acc = evaluate_accuracy(
    reader,
    test_data,
    test_read
)

reader_noisy_acc = evaluate_accuracy(
    reader,
    noisy_test,
    test_read
)

reader_denoised_acc = evaluate_accuracy(
    reader,
    denoised_test,
    test_read
)

print(
    'Clean-trained EfficientNet reader accuracy '
    'on clean_test =',
    reader_clean_acc
)

print(
    'Clean-trained EfficientNet reader accuracy '
    'on noisy_test =',
    reader_noisy_acc
)

print(
    'Clean-trained EfficientNet reader accuracy '
    'on denoised_test =',
    reader_denoised_acc
)


###############################################################
# Save reader losses, times, and accuracies
###############################################################

with open('CatDogCleanReaderLossInvar.txt', 'w') as file:
    for loss in reader_loss:
        file.write(str(loss) + '\n')

with open('CatDogCleanReaderValLossInvar.txt', 'w') as file:
    for loss in reader_val_loss:
        file.write(str(loss) + '\n')

with open('CatDogCleanReaderEpochTimesInvar.txt', 'w') as file:
    for epoch_time in reader_epoch_times:
        file.write(str(epoch_time) + '\n')

    file.write(
        'average_epoch_time_sec = '
        + str(float(np.mean(reader_epoch_times)))
        + '\n'
    )

    file.write(
        'total_epoch_time_sec = '
        + str(float(np.sum(reader_epoch_times)))
        + '\n'
    )

with open('CatDogCleanReaderAccuraciesInvar.txt', 'w') as file:
    file.write(
        'clean_test_accuracy = '
        + str(reader_clean_acc)
        + '\n'
    )

    file.write(
        'noisy_test_accuracy = '
        + str(reader_noisy_acc)
        + '\n'
    )

    file.write(
        'denoised_test_accuracy = '
        + str(reader_denoised_acc)
        + '\n'
    )

plt.figure()

plt.plot(
    range(1, 1 + len(reader_loss)),
    reader_loss,
    label='reader_loss'
)

plt.plot(
    range(1, 1 + len(reader_val_loss)),
    reader_val_loss,
    label='reader_val_loss'
)

plt.legend(fontsize=14)
plt.xlabel('epochs', fontsize=14)
plt.ylabel('Cross-entropy loss', fontsize=14)
plt.grid()
plt.savefig('CatDogCleanReaderLossesInvar.png')
plt.show()


###############################################################
# Total runtime
###############################################################

total = time.time() - program_start

hours = int(total // 3600)
minutes = int((total % 3600) // 60)
seconds = total % 60

print(
    f'\nTotal Runtime: '
    f'{hours:02d}:{minutes:02d}:{seconds:05.2f}'
)