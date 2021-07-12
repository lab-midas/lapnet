# *LAPNet*: Deep learning-based non-rigid registration derived in k-space for Magnetic Resonance Imaging

This repository contains the TensorFlow implementation of the paper

[Thomas Kuestner](https://www.medizin.uni-tuebingen.de/de/das-klinikum/mitarbeiter/profil/252),
[Jiazhen Pan](https://aim-lab.io/author/jiazhen-pan/), 
[Haikun Qi](https://scholar.google.com/citations?user=AWI7KUsAAAAJ&hl=zh-CN), 
[Gastao Cruz](https://kclpure.kcl.ac.uk/portal/gastao.cruz.html), 
[Christopher Gilliam](https://www.rmit.edu.au/contact/staff-contacts/academic-staff/g/gilliam-dr-christopher), 
[Thierry Blu](https://www.ee.cuhk.edu.hk/~tblu/monsite/phps/), 
[Bin Yang](https://www.iss.uni-stuttgart.de/institut/team/Yang-00004/), 
[Sergios Gatidis](https://www.medizin.uni-tuebingen.de/de/das-klinikum/mitarbeiter/profil/1479), 
[Rene Botnar](https://kclpure.kcl.ac.uk/portal/rene.botnar.html), 
[Claudia Prieto](https://kclpure.kcl.ac.uk/portal/claudia.prieto.html)<br/>
**[LAPNet: Non-rigid registration derived in k-space for Magnetic Resonance Imaging](https://ieeexplore.ieee.org/document/9478906)** <br/>
*IEEE Transactions on Medical Imaging* 2021.

If you find *LAPNet* useful in your research, please consider citing:

	@article{KuestnerLAPNet2021,
	 title={LAPNet: Non-rigid registration derived in k-space for Magnetic Resonance Imaging},
	 author={Kuestner, Thomas and Pan, Jiazhen and Qi, Haikun and Cruz, Gastao and Gilliam, Christopher and Blu, Thierry and Yang, Bin and Gatidis, Sergios and Botnar, Rene and Prieto, Claudia},
	 journal={IEEE Transactions on Medical Imaging},
	 year={2021},
	 doi = {10.1109/TMI.2021.3096131},
	 }

### License
LAPNet is released under the MIT License.

## Introduction
In the field of medical imaging processing, fast and accurate motion estimation is an integral part of 
prospective and retrospective motion correction methods. This task has remained one of the key challenges 
for respiratory and cardiac motion because acquired and motion-resolved data are usually highly undersampled in k-space. 
As a result, blurring and aliasing induces artifacts in the image which deteriorate registration performance. 
Thus, solving the non-rigid motion estimation with conventional image space-based approaches could be difficult and 
errors can propagate in the motion fields. 

Therefore, in this work we introduced a novel deep learning-based registration method - LAPNet - to solve the non-rigid 
motion estimation problem directly in the acquired (and undersampled) k-space. 

## LAPNet
![LAPNet](https://user-images.githubusercontent.com/15344655/119488462-df89fa00-bd5a-11eb-95ef-8a8ad89a2e38.png)
Figure 1: Network architecture of LAPNet to perform non-rigid registration in k-space.

![RespMotion](https://user-images.githubusercontent.com/15344655/121178314-a5425180-c85e-11eb-9de7-1e74da42a672.png)
Figure 2: Respiratory non-rigid motion estimation in a patient with neuroendocrine tumor in the liver by the proposed 
LAPNet in k-space in comparison to image-based non-rigid registration by FlowNet-S (deep learning), and image-based NiftyReg (cubic B-Splines). Estimated flow displacement are depicted in coronal and sagittal orientation. Reference flows are obtained from imageLAP (optical flow) on fully-sampled images. Undersampling was performed prospectively with a vdPD undersampling for 8x and 30x acceleration.

![CardMotion](https://user-images.githubusercontent.com/15344655/121178587-ed617400-c85e-11eb-8669-840c145a3387.png)
Figure 3: Cardiac non-rigid motion estimation in a patient with myocarditis. by the proposed 
LAPNet in k-space in comparison to image-based non-rigid registration by FlowNet-S (deep learning), and image-based NiftyReg (cubic B-Splines). Estimated flow displacement are depicted in coronal and sagittal orientation. Reference flows are obtained from imageLAP (optical flow) on fully-sampled images. Undersampling was performed prospectively with a golden angle radial undersampling for 8x and 16x acceleration.

### Other Supported Architectures
There are also some network architectures which are supported in this work: 
- [FlowNetS](https://arxiv.org/abs/1504.06852),
- [FlowNetC](https://arxiv.org/abs/1504.06852)*, 
- [FlowNet 2.0](https://arxiv.org/abs/1612.01925)*,
- [Automap](https://www.nature.com/articles/nature25988)*.

Architectures marked with * need to be modified further in the script and cannot be used directly.

## Data
Since real reference flows are very hard or even impossible to capture, we augmented the captured flows by simulation. 
The flow augmentation types are: "C" (Constant flow), "S" (random smoothed flow), "R" (generated from
[Local All-Pass](https://ieeexplore.ieee.org/document/7493264)) and "X" (flow "S" x "R"). Figure 4 shows an example of the augmented flows and the deformed image. 

![Simulated Flow](https://user-images.githubusercontent.com/46929357/87416750-70d30080-c5cf-11ea-8751-1a382d95b86b.png)
Figure 4: Applied simulated flow in LAP-Net for the supervised learning

### Undersampling
There are in total three undersampling masks which are applied in this work:
- [3D variable-density Poisson-Disc](https://ieeexplore.ieee.org/document/7486011)
- 3D central sampling
- [2D golden angle radial undersampling](https://ieeexplore.ieee.org/abstract/document/4039540)

## Usage

Please refer to the configuration file for a detailed description of the different operating modes.
```
TF1: config.ini
TF2: config.yaml
```

### Hardware requirements
- at least one NVIDIA GPU (multi-GPU training is supported).
- for best performance, at least 8GB of GPU memory is recommended.
- to run the code with less GPU memory, take a look at https://github.com/openai/gradient-checkpointing. 

### Software requirements
- python 3
- gcc4
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
    - `tensorflow-gpu` (TF1: at least version 1.7, TF2: at least version 2.0)
- CUDA and CuDNN. You should use the installer downloaded from the NVIDIA website and install to /usr/local/cuda. Please make sure that the versions you install are compatible with the version of `tensorflow-gpu` you are using (see e.g. https://github.com/tensorflow/tensorflow/releases).


## Replicating our models

In the following, we list the training strategy and the parameters to set to replicate our models (TF1: `config.ini`, TF2: `config.yaml`).
The basic parameters should be set as in Table 1.

Tabel 1: Basic parameters to be set for training 

Model & data| batch size | save interval | divisor | network | dataset
-------|:--------:|:--------:|:--------:|:--------:|:--------:|
LAPNet & resp+card | 32 | 1000 | 10 | lapnet/lapnet3D | resp/card 
LAPNet & resp |64 |1000 | 20|lapnet |resp_2D
FlowNet-S & resp|16 |100 | 5|flownet|card_2D
LAPNet & card|64 |1000| 10|lapnet|resp_2D
FlowNet-S & card|16 |100 | 3|flownet|card_2D

#### LAPNet
- *random undersampling* (`mask_type = drUS`): Training with variable-density Poisson Disc undersampling (drUS, randomly selected the US rate from 1:30).
- *center undersampling* (`mask_type = crUS`): Training with central sampling strategy (crUS, randomly selected the US rate from 1:30). 
The generated data proportion of S:R:X = 4:2:4. 

`config.ini` | `config.yaml`: 
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
mask_type = drUS | crUS
us_rate = random
``` 

#### FlowNet-S
- *random undersampling* (`mask_type = drUS`): Training with variable-density Poisson Disc undersampling (drUS, randomly selected the US rate from 1:8). Best combination for random undersampling strategy was selected for FlowNet-S.
- *center undersampling* (`mask_type = crUS`): Training with central sampling strategy (crUS, randomly selected the US rate from 1:30).
The generated data proportion of S:R:X is 4:2:4 as well.

`config.ini` | `config.yaml`: 
```
flow_amplitude = 10
manual_decay_iters = 5000,5000,5000,5000
manual_decay_lrs = 1e-4,0.5e-4,0.25e-4,1e-5
automap = False
augment_type_percent = 0,0.4,0.2,0.4
network = flownet
long_term_train = True
total_data_num = 7500
mask_type = drUS | crUS
us_rate = random
```
