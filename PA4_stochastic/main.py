################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

from experiment import Experiment
import sys

# Main Driver for your code. Either run `python main.py` which will run the experiment with default config
# or specify the configuration by running `python main.py custom`
if __name__ == "__main__":
    exp_name = 'default'

    if len(sys.argv) > 1:
        exp_name = sys.argv[1]

    print("Running Experiment: ", exp_name)
    exp = Experiment(exp_name)
    exp.run()
    #exp.test()
