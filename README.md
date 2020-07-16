# *LAP-Net*: Deep learning-based non-rigid registration in k-space for MR imaging

This repository contains the TensorFlow implementation of the paper

[LAP-Net: Deep learning-based non-rigid registration in k-space for MR imaging](https://arxiv.org/abs/...) (Paper link & Journal name)

[Thomas Kuestner](https://www.medizin.uni-tuebingen.de/de/das-klinikum/mitarbeiter/profil/252),
Jiazhen Pan, and
[...](...) (* Authors contributed equally).

### Slides 

[Download slides](...)

### Citation

If you find *LAP-Net* useful in your research, please consider citing:

	@article{LAPNET2020,
	 title={LAP-Net: Deep learning-based non-rigid registration in k-space for MR imaging},
	 author={Kuestner, Thomas and Pan, Jiazhen and ...},
	 journal={IEEE Access},
	 year={2020}
	 }

### License

LAP-Net is released under the MIT License (refer to the LICENSE file for details).

## Contents
1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Replicating our Models](#replicating-our-models)
4. [Improvement and outlook](#improvement-and-outlook)

## Introduction
In the field of medical imaging processing, fast and accurate motion estimation is always an integral part of 
prospective and retrospective motion correction methods. This task has remained one of the key challenges 
for the respiratory and cardiac motion correction because these data are usually acquired with acceleration by 
compressed sensing and are therefore highly undersampled in k-space. As a result, a blurred and aliasing-included 
MR image is constructed. Thus, using the conventional image space-based approaches to solve the non-rigid 
motion estimation and further image registration problems could be difficult and error-propagated. 

Therefore, in this work we introduce a novel deep learning-based method - LAP-Net - to solve the non-rigid 
flow estimation problem for respiratory images. The major contributions of our work can be summarized as 
the following three points: First, it is a neural network-based approach which can be trained end-to-end. 
Second, it can estimate the non-rigid deformation field directly from k-space. 
Third, it can deal with the highly undersampled k-space data with a very stable performance. 
We compared this network with the flow estimation network baseline - FlowNet and our network 
surpasses FlowNet in almost all aspects.

### Data
The respiratory and cardiac data which are applied in this work for training and testing are provided by 
King's College London and St Thomas' Hospital in London. 

Since the real flow are very hard or even impossible to capture, we employed simulated flow in this work. 
There are flow type "C" (Constant flow), flow type "S" (random smoothed flow), flow type "R" (generated from
[LAP algorithm](link...)) and flow tyoe "X" (flow "S" x "R"). Figure 1 reveals the details of flow generation. 

![Simulated Flow](https://user-images.githubusercontent.com/46929357/87416750-70d30080-c5cf-11ea-8751-1a382d95b86b.png)
Figure 1: Applied simulated flow in LAP-Net for the supervised learning

### Network Architecture
![Network Architecture](https://user-images.githubusercontent.com/46929357/87417780-f4d9b800-c5d0-11ea-98c0-276d61aa89be.png)
Figure 2: Network Architecture of LAP-Net

### Other Supported Architectures
There are also some network architectures which are supported in this work: 
- [FlowNetS](https://arxiv.org/abs/1504.06852),
- [FlowNetC](https://arxiv.org/abs/1504.06852)*, 
- [FlowNet 2.0](https://arxiv.org/abs/1612.01925)*,
- [Automap](https://www.nature.com/articles/nature25988)*.

Architectures marked with * need to be modified further in the script and cannot be used directly.

### Undersampling Mask
There are in total three undersampling masks which are applied in this work:
- [variable-density CASPR](paper link) (for 3D respiratory images)
- variable-density CASPR in central sampling version (for 3D respiratory images)
- [radial subsampling](paper link) (for 2D cardiac images)

### Example of test results in comparison with FlowNet-S
![Results resp comparison](https://user-images.githubusercontent.com/46929357/87658754-e91afc80-c75c-11ea-8f30-2720c12ef7d8.png)
Figure 3: Prediction respiratory results of LAP-Net in comparison with FLowNet-S

![Results comparison](https://user-images.githubusercontent.com/46929357/87418732-7bdb6000-c5d2-11ea-883f-5ad1e5a6dc8a.png)
Figure 4: Prediction cardiac results of LAP-Net in comparison with FLowNet-S

## Usage

Please refer to the configuration file template (`config_template/config.ini`) for a detailed description
of the different operating modes.

### Hardware requirements
- at least one NVIDIA GPU (multi-GPU training is supported).
- for best performance, at least 8GB of GPU memory is recommended.
- to run the code with less GPU memory, take a look at https://github.com/openai/gradient-checkpointing. 
[One user reported](https://github.com/simonmeister/UnFlow/issues/54) 
successfully running the full CSS model with a GTX 960 with 4GB memory. Note that training will take longer in that case.

### Software requirements
- python 3
- gcc4
- RAR backend tool for `rarfile` (see https://pypi.python.org/pypi/rarfile/)
- python packages: 
    - `matplotlib`
    - `pypng` 
    - `rarfile` 
    - `pillow`
    - `numpy`
    - `scipy`
    - `pylab`
    - `skimage`
    - `multiprocessing`
    - `h5py`
    - `tensorflow-gpu` and `tensorflow 1.x` (at least version 1.7)
- CUDA and CuDNN. You should use the installer downloaded from the NVIDIA website and install to /usr/local/cuda. Please make sure that the versions you install are compatible with the version of `tensorflow-gpu` you are using (see e.g. https://github.com/tensorflow/tensorflow/releases).

### Prepare environment
- make a `data` and an `output` dir, copy data and its corresponding `slice_to_take` file to `./data` and modify settings 
in the `[dir]`, `[run]` and `[compile]` sections for your environment (see comments in the file).

### Train & validate experiments
- adapt settings in `./config.ini` depending on your experiment (which dataset and which network), 
the corresponding parameters should be set as the following:

Model & data| batch size | save interval | divisor | network | dataset
-------|:--------:|:--------:|:--------:|:--------:|:--------:|
LAP-Net & resp |64 |1000 | 20|lapnet |resp_2D
FlowNet-S & resp|16 |100 | 5|flownet|card_2D
LAP-Net & card|64 |1000| 10|lapnet|resp_2D
FlowNet-S & card|16 |100 | 3|flownet|card_2D

Tabel 1: Basic parameters to be set for training 

- The crucial parameters in `./config.ini` which should be checked or modified are:
    - `manual_decay_iters`
    - `manual_decay_lrs`
    - `augment_type_percent`: the proportion of generated flow type for the training 
    - `total_data_num`: how many data should be generated for the training. Here should be noticed that the 
    amount of motion type R cannot surpass the amount of the available data
    - `mask_type`: radial, crUS or drUS  
- `cd src`
- train with `python run.py --ex my_experiment`. Validation is run during training as specified
in `config.ini`.

- If you want to use transfer learning, then the experiment should already exist in `./log/ex` dir and the
experiment's name should align with this experiment. Don't forget to modify the parameters in `config.ini` 
in the corresponding experiment's dir.  

### Evaluation of experiments
- adapt setting and flags in `eval_lapnet.py` (for LAP-Net )or `eval_flownet.py` (for FlowNet-S) 
The crucial parameters which should be checked or modified are:
    - `slice_info_file`: there are two files (for resp and card data) which documents which slices to take for 
    each subject
    - `test_dir`: the path to the testing subjects
    - `test_type`: 0: constant generated flow, 1: smooth generated flow, 2: matlab simulated test data 
    3: simulated x smooth   
    - `US_acc`: undersampling rate for the test
    - `mask_type`: which undersampling mask to apply for the test
    - `selected_frames`: if it is deactivated, then all slices from `slice_info_file` will be taken
    - `cropped_image_size`: this configuration is only for cardiac data, choosing a proper size for cropping
    the image. If it is deactivated, then no cropping is carried out.
    - the save setting: save the loss and the results in png, pdf, mat, npz...
    
- evaluate (multiple) experiments with `python eval_lapnet.py --ex experiment_name_1[, experiment_name_2 [...]]`. 
The name of the experiments should align with the experiments which have already be taken and trained. The 
network weights should be available in the corresponding experiment's dir. 

You can use the --help flag with `run.py` or `eval_lapnet.py` to view all available flags.

### View tensorboard logs
- view logs for all experiments with `tensorboard --logdir=<log_dir>/ex`

### Plot a line diagram 
plot a line diagram of EPE or EAE loss in relation to undersampling rate
1. Evaluate and generate the selected subjects as described in [Evaluation of experiments](#evaluation-of-experiments), 
don't forget to activate the option of `save_loss`.
2. Go to `./src/e2eflow/line_plot.py`, there is an example below. Type the path of calculated losses for the comparison, 
update the name of labels, titel, save path and etc. 
3. run the script `python3 line_plot.py` and check the plot results.

## Replicating our models

In the following, we list the training strategy and the parameters to set to replicate our models.
The basic parameters should be set as in table 1.
### For respiratory data

### For LAP-Net
- *random undersampling*: training with random undersampling strategy (drUS, randomly selected the US rate form 
{1, 7, 13, 19, 26, 31}), using mask variable-density CASPR. The generated data proportion of S:R:X = 4:2:4. 
An available model of this version is `srx424_drUS_1603` and is saved under ... 

To replicate this model, in `config.ini`:
```
flow_amplitude = 10
manual_decay_iters = 50000,50000,50000,50000
manual_decay_lrs = 2.5e-4,1e-4,5e-5,2.5e-5
automap = False
random_crop = True
crop_size = 33
crop_box_num = 200
padding = True
augment_type_percent = 0,0.4,0.2,0.4
network = lapnet
long_term_train = True
total_data_num = 7500
mask_type = drUS
us_rate = random
``` 

- *center undersampling*: training with random undersampling strategy (crUS, randomly selected the US rate form 
{1, 5, 9, 13}), using mask variable-density CASPR in central sampling version. 
The generated data proportion of S:R:X = 4:2:4. 
An available model of this version is `srx424_crUS13_test_1104` and is saved under ... 

The parameters are same as above, only one modification:
```
mask_type = crUS
```
#### For FlowNet-S
- *no undersampling*: Since the random undersampling strategy doesn't work for FlowNet-S (loss curve didn't converge),
we applied no undersampling training strategy for FlowNet-S. The generated data proportion of S:R:X is  4:2:4 as well.
An available model of this version is `flown_srx424_noUS_2003` and is saved under ... 

To replicate this model, the parameters in `config.ini` should be modified as:
```
flow_amplitude = 10
manual_decay_iters = 5000,5000,5000,5000
manual_decay_lrs = 1e-4,0.5e-4,0.25e-4,1e-5
automap = False
augment_type_percent = 0,0.4,0.2,0.4
network = flownet
long_term_train = True
total_data_num = 7500
mask_type = crUS
us_rate = False
```

- *center undersampling*: by using crUS training strategy, the loss curve during the training can be converged.
All the setting are the same as above for FlowNet-S just one single update:
```
us_rate = random
```
An available example for this model is `flown_srx424_crUS_2105` and is saved under ...

### For cardiac data
#### For LAP-Net

- First no undersampling training and then random undersampling training strategy. Since we use 2D radial undersampling 
for the cardiac case which is totally different as variable-density CASPR and might introduce more difficulty for
LAP-Net, the random training strategy for the respiratory case doesn't work anymore. Therefore, we introduce 
"transfer learning" for the cardiac case. First, we train the network with the parameters as the following (the basic parameters
should be same as in table 1):
 ```
flow_amplitude = 10
manual_decay_iters = 40000,40000,40000,40000
manual_decay_lrs = 2.5e-4,1e-4,5e-5,2.5e-5
data_per_interval = 500
automap = False
random_crop = True
crop_size = 33
crop_box_num = 200
padding = True
augment_type_percent = 0,0.5,0.1,0.4
network = lapnet
long_term_train = True
total_data_num = 4750
mask_type = radial
us_rate = False
```

Note that we have less cardiac data than the respiratory case. Thus, the data proportion is set as 5:1:4 and only
4750 data will be created. After the training of the first round, go to the corresponding experiment dir. 
The following parameters should be updated  in its `config.ini` file:
 ```
manual_decay_iters = 30000,30000,30000,30000
manual_decay_lrs = 2e-4,1e-4,5e-5,2.5e-5
us_rate = random
```
Then run the experiment with the same name as before to carry out the "transfer learning". An available model from this 
training strategy is `card_LAP_noUS_drUS17_2906` and is saved under ...

#### For FlowNet-S
- *no undersampling*: the idea is the same as the training of FlowNet-S with no undersampling training strategy for 
the respiratory case, the parameters should be set as the following:
```
manual_decay_iters = 4000,4000,4000,4000
manual_decay_lrs = 1e-4,0.5e-4,0.25e-4,1e-5
automap = False
augment_type_percent = 0,0.5,0.1,0.4
network = flownet
long_term_train = True
total_data_num = 4750
epoch = 18
us_rate = False
```
An available model with this training strategy for FlowNet-S is `card_FN_noUS_srx514_2606` and is saved under ...


## Improvement and Outlook 
There still some room for improvement of the script framework and the network architecture. 

### Potential improvement of the script:
- The current implementation is still based on `tensorflow 1.x`. Many applications and functions are out of date and
it is not convenient for data-feeding, debugging and visualization. It is not necessary but will be beneficial if we
can upgrade it to `tensorflow 2.x`.
- Because the framework of this project is inherited from the project [UnFlow](https://github.com/simonmeister/UnFlow), some 
classes and functions in this project are redundant. They may be useful for further extension, so we still keep them in
this project.
- Because of the limitation of the time, the interface of this project is not perfect and some part of the code might be
hard-coded. They should be removed in the further version.

### Potential improvement of the network and the methodology:
- Some hyperparameters like kernel size, parameters number haven't been tested completely. Those experiments should be 
done in further work.
- Regarding to the cardiac data, LAP-Net has the problem at the margin of the image by estimating the undersampled data.
- LAP-Net still use the cropping and sliding window mechanism and didn't start from the full-resolution image. 
Therefore, the estimation time is not short (for a 2D image with the size of 256 x 256), it need approx. 16s on an NVIDIA 
TITAN RTX. Regarding to the next step of the improvement, we should start directly from the full resolution MR images
without applying any cropping. 
