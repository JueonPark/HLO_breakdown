"""
Breakdowns all the kernels, especially fused computation.
"""
import csv
import argparse
import pathlib
from hlo_parser import HloTable
from kernel_metadata import *
from rewrite_fusion_kernel import rewrite_fusion_kernel

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)

if __name__ == "__main__":
  args = parser.parse_args()
  model = ""
  if args.model.find("bert") != -1:
    model = "bert"
  elif args.model.find("resnet") != -1:
    model = "resnet"
  elif args.model.find("mobilenet") != -1:
    model = "mobilenet"
  elif args.model.find("transformer") != -1:
    model = "transformer"
  elif args.model.find("dlrm") != -1:
    model = "dlrm"
  elif args.model.find("vit") != -1:
    model = "vit"
  else:
    model = args.model
  print(model)

  csv_path = f'/home/jueonpark/tracegen/csv_files/{args.model}-NDPX_baseline_64-1-nosync.csv'
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
    breakdown = {**breakdown, **rewrite_fusion_kernel(hlo_table, model)}
  
  # NDP
  breakdown["NDP_OP"] = "NDP_OP"

  csv_result = csv.reader(result_file)
  header = next(csv_result)
  header.append("BREAKDOWN")
  output_path = f'/home/jueonpark/tracegen/experiments_results/{args.model}/{args.model}-NDPX_baseline_64-1-nosync_breakdown.csv'
  csv_to_rewrite = csv.writer(open(output_path, "w+"))
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
      elif model == "dlrm":
        row = rewrite_dlrm_kernel(row)
      elif model == "vit":
        row = rewrite_vit_kernel(row)
      else:
        pass
    csv_to_rewrite.writerow(row)