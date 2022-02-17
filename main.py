"""
hlo graph에 있는 각 fusion 별로 어떠한 computation인지 알려준다.
"""
import csv
import argparse
from hlo_parser import HloTable

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--csv', type=str, required=True, help="result csv file (.csv)")
  parser.add_argument('--hlo', type=str, required=True, help="xla hlo graph path (.txt)")
  args = parser.parse_args()
  result_file = open(args.csv, "r")
  hlo_file = open(args.hlo, "r")
  hlo_table = HloTable(hlo_file.read()).hlo_table

  # csv file을 읽고, kernel name을 바꿔준다.
  