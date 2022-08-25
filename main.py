"""
Breakdowns all the kernels, especially fused computation.
"""
import csv
import argparse
import pathlib
from hlo_parser import HloTable
from rewrite_fusion_kernel import rewrite_fusion_kernel

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
# parser.add_argument('--csv', type=str, required=True, help="result csv file (.csv)")
# parser.add_argument('--hlo', type=str, required=True, help="xla hlo graph path (.txt)")

def rewrite_bert_kernel(row):
  if (row[4].find("nn") != -1 and row[4].find("gemm") != -1):
    row.append("GEMM")
  elif (row[4].find("tn") != -1 and row[4].find("gemm") != -1):
    row.append("dxFC")
  elif (row[4].find("nt") != -1 and row[4].find("gemm") != -1):
    row.append("dwFC")
  elif (row[4].find("cudnn") != -1):
    row.append("cudnn")
  else:
    row.append("others")
  return row
  
def rewrite_resnet_kernel(row):
  if (row[4].find("Transpose") != -1):
    row.append("Transpose")
  elif (row[4].find("Wgrad") != -1):
    row.append("wgrad")
  elif (row[4].find("wgrad") != -1):
    row.append("wgrad")
  elif (row[4].find("dgrad") != -1):
    row.append("dgrad")
  else:
    row.append("others")
  return row

def rewrite_mobilenet_kernel(row):
  if (row[4].find("Transpose") != -1):
    row.append("Transpose")
  elif (row[4].find("Wgrad") != -1):
    row.append("wgrad")
  elif (row[4].find("wgrad") != -1):
    row.append("wgrad")
  elif (row[4].find("dgrad") != -1):
    row.append("dgrad")
  else:
    row.append("others")
  return row

def rewrite_transformer_kernel(row):
  if (row[4].find("Transpose") != -1):
    row.append("Transpose")
  elif (row[4].find("Wgrad") != -1):
    row.append("wgrad")
  elif (row[4].find("wgrad") != -1):
    row.append("wgrad")
  elif (row[4].find("dgrad") != -1):
    row.append("dgrad")
  else:
    row.append("others")
  return row

if __name__ == "__main__":
  args = parser.parse_args()
  model = ""
  if args.model.find("bert"):
    model = "bert"
  elif args.model.find("resnet"):
    model = "resnet"
  elif args.model.find("mobilenet"):
    model = "mobilenet"
  elif args.model.find("transformer"):
    model = "transformer"
  else:
    model = args.model

  csv_path = f'/home/jueonpark/csv_files/{args.model}-NDPX_baseline_64-1-nosync.csv'
  xla_hlo_path_str = f'/home/jueonpark/tracegen/traces/{args.model}/xla_hlo'
  xla_hlo_path = pathlib.Path(xla_hlo_path_str)
  graph_paths = list(xla_hlo_path.glob("*after_optimizations.txt"))
  result_file = open(csv_path, "r")
  hlo_tables = []
  for graph_path in graph_paths:
    hlo_file = open(graph_path, "r")
    hlo_tables.append(HloTable(hlo_file.read()).hlo_table)
  
  # Read the csv file and change the kernel name.
  breakdown = {}
  for hlo_table in hlo_tables:
    breakdown = breakdown.merge(breakdown, rewrite_fusion_kernel(hlo_table))
  
  # NDP
  breakdown["NDP_OP"] = "NDP_OP"

  csv_result = csv.reader(result_file)
  header = next(csv_result)
  header.append("BREAKDOWN")
  csv_to_rewrite = csv.writer(open(csv_path[:-4] + "_breakdown.csv", "w"))
  csv_to_rewrite.writerow(header)

  for row in csv_result:
    try:
      row.append(breakdown[row[4]])
    except:
      if model == "bert":
        row = rewrite_bert_kernel(row)
      elif model == "resnet":
        row = rewrite_resnet_kernel(row)
      elif model == "mobilenet":
        row = rewrite_mobilenet_kernel(row)
      elif model == "transformer":
        row = rewrite_transformer_kernel(row)
      else:
        pass
    csv_to_rewrite.writerow(row)