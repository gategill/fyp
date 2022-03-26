"""

"""

import argparse
import run

parser = argparse.ArgumentParser(description = "Run User and Item KNN")
parser.add_argument("--config", type = str, default = "simple")
args = parser.parse_args()

run.run_experiment("config_files/{}.yml".format(args.config))
