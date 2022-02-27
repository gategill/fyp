"""

"""

import argparse
from icecream import ic
from run_experiment import run_experiment

parser = argparse.ArgumentParser(description = "Run User and Item KNN")
parser.add_argument("--config", type = str, default = "simple")
args = parser.parse_args()

ic(args.config)

run_experiment(f"config_files/{args.config}.yml")
