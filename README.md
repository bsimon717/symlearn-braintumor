# Introduction

This is a project meant to prototype the *symbiotic learning* paradigm using a brain tumor dataset.

All details regarding the dataset can be found [here](https://www.kaggle.com/datasets/mylee77/brain-tumor-mri-deduplicated-clean-version).

# Instructions

First, run the following to retrieve the symbiotic-learning package and create the logs directory:
`git clone https://github.com/bsimon717/symlearn.git`

`mkdir sym_logs`

Hyperparameters can be specified using command-line arguments (run `python train.py --help` to see the full list) or with a `config.json` file.

To generate a configuration file, run the following:

`python main.py -n 3 -c 0.5 0.5 0.5 -o True --comment='default'`

This will generate a config.json file in the relevant log directory, the path to which will be printed to the terminal.

To start a training with a configuration file, run:

`python main.py --config_path=/PATH/TO/CONFIG`

To save the state directories of pre-Readout models BEFORE uplift, use the `--save_before_uplift` flag.

To save all models at the end of training, use the `--save_end` flag.

To start a training AT uplift, use both of the `--load_at_uplift` and `--load_path` arguments.

To save the best-performing epoch (after uplift), use the `--save_best` flag.