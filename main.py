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
  if (row[4].find("dgemm") != -1):
    row.append("dxFC")
  elif (row[4].find("wgemm") != -1):
    row.append("dwFC")
  elif (row[4].find("MetaKernel") != -1):
    row.append("elementwise")
  elif (row[4].find("add") != -1):
    row.append("elementwise")
  elif (row[4].find("mul") != -1):
    row.append("elementwise")
  elif (row[4].find("log") != -1):
    row.append("elementwise")
  elif (row[4].find("tn") != -1 and row[4].find("gemm") != -1):
    row.append("dxFC")
  elif (row[4].find("nt") != -1 and row[4].find("gemm") != -1):
    row.append("dwFC")
  elif (row[4].find("1688cudnn") != -1):
    row.append("CONV")
  elif (row[4].find("gemm") != -1):
    row.append("FC")
  elif (row[4].find("convol") != -1):
    row.append("CONV")
  elif (row[4].find("conv2d") != -1):
    row.append("CONV")
  elif (row[4].find("conv") != -1):
    row.append("CONV")
  elif (row[4].find("first_layer") != -1):
    row.append("CONV")
  elif (row[4].find("pool") != -1):
    row.append("POOLING")
  elif (row[4].find("c1_k1") != -1):
    row.append("CONV")
  elif (row[4].find("bn") != -1):
    row.append("BN+ELEMWISE")
  elif (row[4].find("adam") != -1):
    row.append("OPT")
  else:
    row.append("others")
  return row

def rewrite_mobilenet_kernel(row):
  if (row[4].find("dgemm") != -1):
    row.append("dxFC")
  elif (row[4].find("wgemm") != -1):
    row.append("dwFC")
  elif (row[4].find("MetaKernel") != -1):
    row.append("elementwise")
  elif (row[4].find("add") != -1):
    row.append("elementwise")
  elif (row[4].find("mul") != -1):
    row.append("elementwise")
  elif (row[4].find("log") != -1):
    row.append("elementwise")
  elif (row[4].find("tn") != -1 and row[4].find("gemm") != -1):
    row.append("dxFC")
  elif (row[4].find("nt") != -1 and row[4].find("gemm") != -1):
    row.append("dwFC")
  elif (row[4].find("1688cudnn") != -1):
    row.append("CONV")
  elif (row[4].find("gemm") != -1):
    row.append("FC")
  elif (row[4].find("convol") != -1):
    row.append("CONV")
  elif (row[4].find("conv2d") != -1):
    row.append("CONV")
  elif (row[4].find("conv") != -1):
    row.append("CONV")
  elif (row[4].find("first_layer") != -1):
    row.append("CONV")
  elif (row[4].find("pool") != -1):
    row.append("POOLING")
  elif (row[4].find("c1_k1") != -1):
    row.append("CONV")
  elif (row[4].find("bn") != -1):
    row.append("BN+ELEMWISE")
  elif (row[4].find("adam") != -1):
    row.append("OPT")
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
  if args.model.find("bert") != -1:
    model = "bert"
  elif args.model.find("resnet") != -1:
    model = "resnet"
  elif args.model.find("mobilenet") != -1:
    model = "mobilenet"
  elif args.model.find("transformer") != -1:
    model = "transformer"
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
      else:
        pass
    csv_to_rewrite.writerow(row)