
# CUDA-GHR: Controllable Unsupervised Domain Adaptation for Gaze and Head Redirection

This repository is the official implementation of [CUDA-GHR: Controllable Unsupervised Domain Adaptation for Gaze and Head Redirection](). 

## Requirements
We used Python 3.7.10 and torch 1.18.1 to test our experiments. We ran our codebase on Ubuntu 20.04.

To install all the packages:

```setup
pip install -r requirements.txt
```



## Usage

#### Data
Download the three datasets: [GazeCapture](https://gazecapture.csail.mit.edu/), [MPIIFaceGaze](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation/), 
[Columbia](https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/).

To pre-process the datasets, please use [this repository](https://github.com/swook/faze_preprocess) and 
follow instructions provided to generate eye-strip images for FAZE. Put the h5 files in the data folder.


#### Train

Create a config json file similar to [configs/config_gc_to_mpii.json](configs/config_gc_to_mpii.json) describing 
all the training parameters and paths to the input files.

To train the task network, run this command:

```
python train_tasknet.py --config-json configs/config_tasknet.json
```

To train and evaluate the CUDA-GHR model in the paper, run this command:

GazeCapture &rarr; MPIIGaze: 

```
python train_cudaghr.py --config_json configs/config_gc_to_mpii.json
```

GazeCapture &rarr; Columbia: 

```
python train_cudaghr.py --config_json configs/config_gc_to_col.json --columbia
```

The training images, losses and evaluation metrics will be loggged in Tensorboard. We also save generated images 
in the save folder.


#### Evaluate

To evaluate CUDA-GHR model, run this command:

```
python eval_cudaghr.py --model_path <path to model> --config_json <path to config file> --test_people <subset to test>
```

Add '--columbia' option to test on Columbia dataset.

## Pre-trained Models

You can download pretrained models here:

- [GazeCapture &rarr; MPIIGaze](https://drive.google.com/file/d/1XeJ16XHghjnr5O4V34AUEE6eoaKODsZK/view?usp=sharing)
- [GazeCapture &rarr; Columbia](https://drive.google.com/file/d/1AIWplDl0790YG9jGK0vgp82ypLwM_XLe/view?usp=sharing)
- [ResNet-50 task model](https://drive.google.com/file/d/1iS1faPkrIe8W1LiAcpvotYPhXa5RGmI-/view?usp=sharing)

## Contributing

For any inquiries, please contact us.