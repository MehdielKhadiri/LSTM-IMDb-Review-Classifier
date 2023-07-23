# LSTM IMDb Review Classifier

This repository contains Python scripts for a movie review sentiment classifier. The model is trained on the IMDb dataset using an LSTM (Long Short-Term Memory) network implemented in PyTorch.

## Setup and Requirements

The application requires the following software and libraries:

- Python 3.6 or later
- PyTorch 1.x
- TorchText

To install the necessary libraries, you can use pip:

```bash
pip install torch torchtext
```

Please note: If you plan to use a CUDA-enabled GPU, make sure to install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn) according to the instructions provided by NVIDIA.

## Usage

After installing the dependencies, you can run the training script:

```bash
python lstmIMDb.py
```

After training, you can run the testing script:

```bash
python lstmIMDbtest.py
```

## Contributing

Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.
