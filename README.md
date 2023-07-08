# MAMMO-NET

This code repository aims to facilitate reproducibility of research results involving deep learning models for breast cancer detection in digital mammography. The code demonstrates the training and testing of state-of-the-art convolutional neural networks which build the core component of most commercially available breast cancer AI systems. The default model in the code base is a ResNet-18, but this can be easily adjusted to higher capacity models (e.g., ResNet-50). The training routine implements state-of-the-art data augmentation techniques including photometric and geometric transformations that are randomly applied to the training images to improve the model robustness and generalization to unseen data.

## Data

For demonstration purposes, the repository uses the publicly available [VinDr-Mammo dataset](https://physionet.org/content/vindr-mammo/1.0.0/). Note, the dataset is somewhat limited in terms of number of samples (14,000 images for training, 6,000 for testing). State-of-the-art breast cancer detection AI systems are commonly trained on several hundreds of thousands of mammograms. Also note that the VinDr-Mammo dataset has only cases with BI-RADS labels 1 to 5, but no confirmed negative cases with long-term follow-up and no biopsy-confirmed positive cases.

## Code

For running the code, we recommend setting up a dedicated Python environment.

### Requirements

The code has been tested on Windows 11 and Ubuntu 22.04 operating systems. The training and testing of the breast cancer detection model requires a high-end GPU workstation. For our experiments with the code base, we used an NVIDIA Titan X RTX 24 GB. If less memory is available, the default value for the variable `batch_size` in script [mammo-net.py](mammo-net.py) may need to be decreased.

We recommend the use of [Visual Studio Code](https://code.visualstudio.com/) for experimenting with the code base.

### Setup Python environment using conda

Create and activate a Python 3 conda environment:

   ```shell
   conda create -n mammo python=3
   conda activate mammo
   ```
   
### Install PyTorch

Check out instructions for [how to install PyTorch](https://pytorch.org/get-started/locally/) using conda or pip.
   
### Install additional Python packages:
   
   ```shell
   pip install pandas pydicom pytorch-lightning torchsampler scikit-learn scikit-image seaborn jupyter tensorboard tqdm
   ```

### How to use

In order to train and test a breast cancer detection model, please follow these steps:

1. Download the [VinDr-Mammo dataset](https://physionet.org/content/vindr-mammo/1.0.0/) (requires 350 GB of free space on the hard disk).

2. Adjust the variable `data_dir` both in the notebook [data_processing.ipynb](data_processing.ipynb) and the script [mammo-net.py](mammo-net.py) to point to the path where the dataset was downloaded and extracted.

2. Run the notebook [data_processing.ipynb](data_processing.ipynb) in order to convert and pre-process the mammographic images.

3. Run the script [mammo-net.py](mammo-net.py) to automatically train and test a CNN model for breast cancer detection.

The training progress can be monitored with TensorBoard. From the parent directory of the code repository, run:
   ```shell
   tensorboard --logdir output
   ```
Then open a web browser with URL http://localhost:6006/ for a dashboard with training metrics and visualisation of example training images.

After training, the script [evaluation.ipynb](evaluation.ipynb) can be used to plot the ROC-AUC for the test set.

### Expected outputs and runtimes

The script [mammo-net.py](mammo-net.py) will produce a trained model and outputs for the test data in csv format containing predictions for the individual test images. Training the model may take several hours on a high-end GPU workstation.

## Copyright

(c) 2023. All rights reserved.