import argparse
from dataloader import DataLoader

import pdb

if __name__=="__main__":

  # parser = argparse.ArgumentParser() 
  # parser.add_argument("--langs", "-l", type=str, help="Lang iso ids, comma separated")
  # # parser.add_argument("--k", default=3,type=int, help="Number of folds")
  # args = parser.parse_args()

  loader = DataLoader()

  loader.dump_sopa_format()
