"""

"""

import argparse
from icecream import ic
from run_experiment import run_experiment


parser = argparse.ArgumentParser(description = "Run User and Item KNN")
parser.add_argument("--w", type = str, default = "ui")
parser.add_argument("--k", type = int, default = 3)
args = parser.parse_args()


ic(args.w)
ic(args.k)


run_experiment(k = args.k, which = args.w)
