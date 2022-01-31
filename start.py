"""

"""

import argparse
from icecream import ic
from run_experiment import run_experiment


parser = argparse.ArgumentParser(description = "Run User and Item KNN")
parser.add_argument("--w", type = str, default = "ui")
parser.add_argument("--k", type = int, default = 3)
parser.add_argument("--s", type = bool, default = True)
args = parser.parse_args()


ic(args.w)
ic(args.k)
ic(args.s)


run_experiment(k = args.k, which = args.w, save_results = args.s)
