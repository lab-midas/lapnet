## Usage

Please refer to the configuration file template (`config_template/config.ini`) for a detailed description
of the different operating modes.

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

### Prepare environment
- make a `data` and an `output` dir, copy data and its corresponding `slice_to_take` file to `./data` and modify settings 
in the `[dir]`, `[run]` and `[compile]` sections for your environment (see comments in the file).

### Train & validate experiments
- `mkdir data`, copy the data and its corresponding `slice_to_take.ods` file to this dir 
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
- `mkdir output`, the results of evaluation are saved in this dir
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