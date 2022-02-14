"""

"""

import argparse
from icecream import ic
from run_experiment import run_experiment


parser = argparse.ArgumentParser(description = "Run User and Item KNN")
parser.add_argument("--w", type = str, default = "u")
parser.add_argument("--k", type = int, default = 3)
parser.add_argument("--s", type = bool, default = True)
parser.add_argument("--s3", type = bool, default = True)
parser.add_argument("--kfolds", type = int, default = 5)
args = parser.parse_args()


ic(args.w)
ic(args.k)
ic(args.s)
ic(args.s3)
ic(args.kfolds)


run_experiment(k = args.k, which = args.w, save_results = args.s, save_in_s3=args.s3, kfolds = args.kfolds)
