"""

"""

import argparse
from icecream import ic

from run import run_experiment

parser = argparse.ArgumentParser(description = "Run User and Item KNN")
parser.add_argument("--which", type = str, default = "ui")
parser.add_argument("--k", type = int, default = 3)
args = parser.parse_args()

ic(args.which)
ic(args.k)

run_experiment(k = args.k,which = args.which)
