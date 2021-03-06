"""
Does the breakdown of all the kernels, especially fused computation.
"""
import csv
import argparse
from hlo_parser import HloTable
from rewrite_fusion_kernel import rewrite_fusion_kernel

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--csv', type=str, required=True, help="result csv file (.csv)")
  parser.add_argument('--hlo', type=str, required=True, help="xla hlo graph path (.txt)")
  parser.add_argument("--baseline", help="baseline or not")
  args = parser.parse_args()
  result_file = open(args.csv, "r")
  hlo_file = open(args.hlo, "r")
  hlo_table = HloTable(hlo_file.read()).hlo_table

  # Read the csv file and change the kernel name.
  breakdown = rewrite_fusion_kernel(hlo_table)
  # NDP
  breakdown["NDP_OP"] = "NDP_OP"
  # forward
  # breakdown['turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_nn'] = 'GEMM'
  # breakdown['turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_nn'] = 'GEMM'
  # breakdown['turing_fp16_s1688gemm_fp16_256x64_ldg8_f2f_nn'] = 'GEMM'
  # breakdown['turing_fp16_s884gemm_fp16_64x64_ldg8_f2f_nn'] = 'GEMM'
  # breakdown['turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_nn'] = 'GEMM'
  # breakdown['turing_fp16_s884gemm_fp16_256x128_ldg8_f2f_nn'] = 'GEMM'
  # breakdown['turing_fp16_s884gemm_fp16_128x64_ldg8_f2f_nn'] = 'GEMM'
  # breakdown["_ZN5cudnn6detail25bn_fw_tr_1C11_kernel_NCHWIffLi512ELb1ELi1EEEv17cudnnTensorStructPKT_S2_PS3_PKT0_S9_S7_S7_PS7_SA_SA_SA_S7_S7_"] = "cudnn_fw"
  # backward
  # breakdown['turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_tn'] = 'dxFC'
  # breakdown['turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_tn'] = 'dxFC'
  # breakdown['turing_fp16_s1688gemm_fp16_256x64_ldg8_f2f_tn'] = 'dxFC'
  # breakdown['turing_fp16_s884gemm_fp16_64x64_ldg8_f2f_tn'] = 'dxFC'
  # breakdown['turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_nt'] = 'dwFC'
  # breakdown['turing_fp16_s884gemm_fp16_256x128_ldg8_f2f_nt'] = 'dwFC'
  # breakdown['turing_fp16_s884gemm_fp16_128x64_ldg8_f2f_nt'] = 'dwFC'
  # breakdown["_ZN5cudnn6detail21bn_bw_1C11_kernel_newIff6float2Li512ELb1ELi1EEEvT0_S3_S3_S3_17cudnnTensorStructPKT_S4_S7_S4_PS5_PKS3_PS3_SB_SA_SA_S3_"] = "cudnn_bw"

  csv_result = csv.reader(result_file)
  header = next(csv_result)
  header.append("BREAKDOWN")
  csv_to_rewrite = csv.writer(open(args.csv[:-4] + "_breakdown.csv", "w"))
  csv_to_rewrite.writerow(header)
  if args.baseline is None:
    # ndp
    for row in csv_result:
      try:
        row.append(breakdown[row[5]])
      except:
        if (row[5].find("nn") != -1 and row[5].find("gemm") != -1):
          row.append("GEMM")
        elif (row[5].find("tn") != -1 and row[5].find("gemm") != -1):
          if (row[4] == "forward"):
            row.append("GEMM")
          else:
            row.append("dxFC")
        elif (row[5].find("nt") != -1 and row[5].find("gemm") != -1):
          if (row[4] == "forward"):
            row.append("GEMM")
          else:
            row.append("dwFC")
        elif (row[5].find("cudnn") != -1):
          row.append("cudnn")
        else:
          row.append("others")
      csv_to_rewrite.writerow(row)
  else:
    # baseline
    for row in csv_result:
      try:
        row.append(breakdown[row[4]])
      except:
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
      csv_to_rewrite.writerow(row)
