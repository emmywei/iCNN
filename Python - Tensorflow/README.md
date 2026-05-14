# iCNN/Python
This folder contains TensorFlow Python codes showcasing implementations of integral shift-invariant CNNs with a negligible computational overhead.

Usage:

1) Run "NumberDenoiseStill.py" to get a baseline using a bad shift-variant denoiser, which opens "LossStill.txt" and "ValLossStill.txt" to write.

2) Run "NumberDenoiseInvar(0-3).py" (with "NumberDenoiseInvar2.py" being the most preferred) to get good shift-invariant results, which also opens "LossInvar.txt" and "ValLossInvar.txt" to write.

3) Use "PlotLosses.py" to show the results, which opens "LossStill.txt" or "LossInvar.txt" and "ValLossStill.txt" or "ValLossInvar.txt" to read.
