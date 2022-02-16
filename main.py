"""
hlo graph에 있는 각 fusion 별로 어떠한 computation인지 알려준다.
result: array of fused computations:
 - [(fusion.181, fused$softmax$dropout$...), (fusion.180, fused$layernorm$gelu), ...]
"""
import csv
import argparse
from hlo_parser import HloTable
from bert_metadata import parse_metadata

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--hlo', type=str, required=True, help="xla hlo graph path (text)")
  args = parser.parse_args()
  hlo_file = open(args.hlo, "r")
  hlo_table = HloTable(hlo_file.read()).hlo_table
  hlo_breakdown = []

  # for each fusion:
  for hlo_fusion in hlo_table:
    # for each fusion's metadata:
    metadata_list = []
    for metadata in hlo_table[hlo_fusion]["data"]:
      parsed_metadata = parse_metadata(metadata)
      if parsed_metadata in metadata_list:
        continue
      elif parsed_metadata is None:
        continue
      else:
        metadata_list.append(parse_metadata(metadata))
    # now, with metadata_list, append hlo_fusion's computation property into hlo_breakdown
    output_name = hlo_fusion
    output_name = "Fusion"
    if len(metadata_list) != 0:
      output_name += "("
    for metadata in metadata_list:
      output_name += metadata
      if metadata != metadata_list[-1]:
        output_name += "+"
    if len(metadata_list) != 0:
      output_name += ")"

    hlo_breakdown.append(output_name)
  
  # write result to file
  with open(args.hlo[:-4] + ".breakdown", "w") as writefile:
    for element in hlo_breakdown:
      writefile.write(element + "\n")