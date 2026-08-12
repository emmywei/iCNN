# iCNN

This repository provides Octave code for demonstrating and evaluating aliasing-free nonlinear signal-processing methods, together with TensorFlow and PyTorch implementations of CNNs designed to be invariant to integer-pixel shifts with minimal computational overhead. The instructions below reproduce the PyTorch experiments on MNIST and cat/dog images.

## Shift-Invariant CNN Experiments in PyTorch

This is for the PyTorch code accompanying the manuscript **“Solving the Problem of Shift Variance in Convolutional Neural Networks.”** The experiments compare conventional CNN downsampling with optimal image positioning (OIP), which selects a canonical sampling phase before the network and restores the original phase after image reconstruction.

This guide reproduces both PyTorch experiment families:

- **MNIST:** shallow denoising autoencoders and a digit reader.
- **Cat/dog:** 128 x 128 U-Net denoisers and an EfficientNet-B0 reader.

### Repository layout

The repository has the following layout:

```text
iCNN/
├── Python - PyTorch/
│   ├── LibShiftInvar.py
│   ├── NumberDenoiseStill.py
│   ├── NumberDenoiseInvar0.py
│   ├── CatDogDenoiseStill.py
│   ├── CatDogDenoiseInvar0.py
│   ├── PlotTools.py
│   └── PlotToolsCatDog.py
├── Python - Tensorflow/
└── LICENSE.txt
```

| Script | Experiment |
| --- | --- |
| `NumberDenoiseStill.py` | Four-branch shift-and-combine MNIST comparison |
| `NumberDenoiseInvar0.py` | OIP MNIST denoiser and digit reader |
| `CatDogDenoiseStill.py` | Conventional circular U-Net and cat/dog reader |
| `CatDogDenoiseInvar0.py` | OIP U-Net and cat/dog reader |


### Installation

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/emmywei/iCNN.git
cd iCNN
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision numpy matplotlib pillow
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS, activate with:

```bash
source .venv/bin/activate
```

If `python` is unavailable on macOS, use `python3` when creating the environment:

```bash
python3 -m venv .venv
```


### Data

#### MNIST configuration

No manual download is needed. Torchvision downloads MNIST to `./data` on the first run. Images are converted to `[0,1]` tensors. The scripts use the standard 60,000-image training set and 10,000-image test set.

#### Cat/dog configuration

Place the cleaned dataset outside the repository as follows:

```text
/absolute/path/to/PetImages/
├── Cat/
│   └── *.jpg
└── Dog/
    └── *.jpg
```

The reference collection contains 23,412 images. Images are resized directly to 128 x 128 and loaded into memory as float32 tensors. Allow at least 16 GB of RAM for the full workflow. A CUDA GPU is strongly recommended. The first reader run also downloads pretrained EfficientNet-B0 weights.

### Reproduce the experiments

Each script writes checkpoints, text files, and figures to the current working directory, which can be done using the following commands.

#### 1. MNIST comparison

```bash
mkdir -p results/mnist-still
cd results/mnist-still
python "../../Python - PyTorch/NumberDenoiseStill.py"
```

#### 2. MNIST OIP

```bash
mkdir -p results/mnist-oip
cd results/mnist-oip
python "../../Python - PyTorch/NumberDenoiseInvar0.py"
```

#### 3. Cat/dog comparison

```bash
mkdir -p results/catdog-still
ln -s /absolute/path/to/PetImages results/catdog-still/PetImages
cd results/catdog-still
python "../../Python - PyTorch/CatDogDenoiseStill.py"
```

#### 4. Cat/dog OIP

```bash
mkdir -p results/catdog-oip
ln -s /absolute/path/to/PetImages results/catdog-oip/PetImages
cd results/catdog-oip
python "../../Python - PyTorch/CatDogDenoiseInvar0.py"
```

The scripts currently run every training and shift-analysis stage by default, and no command-line arguments are required.

### Exact configurations

#### MNIST

| Setting | Comparison | OIP |
| --- | --- | --- |
| Seed / batch size | 23 / 128 | Same |
| Noise | Gaussian, factor 0.4, clipped to `[0,1]` | Same |
| Denoiser training | 10 clean-to-clean + 20 noisy-to-clean epochs | Same |
| Denoiser loss / optimizer | MSE / Adam | Same |
| Downsampling | Four shifted branches with circular max pooling | OIP + two direct-downsampling stages |
| Approximate parameters | 29,060 | 28,353 |
| Shift sweep | Horizontal shifts -14 to +14 | Same |
| Reader | None | About 1,200 parameters; 60 epochs on denoised images, then the same reader continues for 60 epochs on noisy images |

The two stride-2 reductions give a four-pixel sampling period. The OIP model
uses `pool_stages=(0,2)` and `spec_points=(2,2)` and restores the selected
horizontal phase after decoding.

#### Cat/dog

| Setting | Comparison | OIP |
| --- | --- | --- |
| Seed / image batch size | 23 / 16 | Same |
| Input | RGB, resized to 128 x 128 | Same |
| Noise in current code | Gaussian, factor 0.4, clipped to `[0,1]` | Same |
| Denoiser training | 30 noisy-to-clean epochs | Same |
| Loss / optimizer | `0.5*MSE + 0.5*MAE` / Adam, learning rate `1e-3` | Same |
| Architecture | Circular U-Net with max pooling | OIP circular U-Net with direct downsampling |
| Approximate parameters | 466,595 | Same |
| Shift sweep | Horizontal shifts -64 to +64 | Same |
| Reader | Pretrained EfficientNet-B0, fine-tuned end to end for 10 epochs on clean images | Same |
| Reader batch / optimizer | 4 / Adam, learning rate `1e-4` | Same |

Both U-Nets use encoder widths 32 and 64, a 128-channel bottleneck, skip connections, transposed-convolution upsampling, and a sigmoid RGB output. The same clean-trained reader is evaluated on clean, noisy, and denoised test images.

### Evaluation and outputs

Shift variance is evaluated by comparing the unshifted output with the corresponding input-forward, output-backward shifted result. For each MNIST shift, the maximum absolute pixel-to-pixel difference across the test images is recorded. The experiments also report denoising losses, reader accuracy, and runtime.

Essential generated files include:

| Experiment | Checkpoint and numeric outputs |
| --- | --- |
| MNIST comparison | `Denoiser1.pt`, `Denoiser2.pt`, `LossStill.txt`, `ValLossStill.txt` |
| MNIST OIP | `Denoiser1.pt`, `Denoiser2.pt`, `Reader.pt`, `LossInvar.txt`, `ValLossInvar.txt`, `AccuracyCombo.txt`, `AccuracyDirect.txt` |
| Cat/dog comparison | `CatDogUNetDenoiser.pt`, `CatDogLossStill.txt`, `CatDogValLossStill.txt`, `CatDogCleanReaderAccuracies.txt` |
| Cat/dog OIP | `CatDogUNetDenoiserInvar.pt`, `CatDogLossInvar.txt`, `CatDogValLossInvar.txt`, `CatDogCleanReaderAccuraciesInvar.txt` |


### License

This repository is distributed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](LICENSE.txt). See `LICENSE.txt` for the complete terms.

### Contact

Emmy Wei — `emwei@ucsd.edu`
