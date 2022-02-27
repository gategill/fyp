"""

"""

import argparse
from icecream import ic
import run

parser = argparse.ArgumentParser(description = "Run User and Item KNN")
parser.add_argument("--config", type = str, default = "simple")
args = parser.parse_args()

ic(args.config)

run.run_experiment(f"config_files/{args.config}.yml")
