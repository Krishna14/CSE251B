# Image Captioning

* The pretrained Resnet50 encoder, as well as LSTM and RNN decoder models are implemented in model_factory.py
* The training, validation, and testing procedures are implemented in experiment.py
* Caption generation procedures are implemented in model_factory.py and caption_utils.py
* bleu1 and bleu4 scoring functions are implemented in caption_utils.py
* To run the model with RNN, specify the model_type variable in default.json as "vanilla_rnn". 
* To run the model with LSTM, specify the model_type variable in default.json as "lstm". 
* main.py driver class works as expected to run the experiments. 

## Usage

* Define the configuration for your experiment. See `default.json` to see the structure and available options. You are free to modify and restructure the configuration as per your needs.
* After defining the configuration (say `my_exp.json`) - simply run `python3 main.py my_exp` to start the experiment
* The logs, stats, plots and saved models would be stored in `./experiment_data/my_exp` dir. This can be configured in `contants.py`
* To resume an ongoing experiment, simply run the same command again. It will load the latest stats and models and resume training pr evaluate performance.

## Files
- main.py: Main driver class
- experiment.py: Main experiment class. Initialized based on config - takes care of training, saving stats and plots, logging and resuming experiments.
- dataset_factory: Factory to build datasets based on config
- model_factory.py: Factory to build models based on config
- constants.py: constants used across the project
- file_utils.py: utility functions for handling files 
- caption_utils.py: utility functions to generate bleu scores
- vocab.py: A simple Vocabulary wrapper
- coco_dataset: A simple implementation of `torch.utils.data.Dataset` the Coco Dataset
- get_datasets.ipynb: A helper notebook to set up the dataset in your workspace