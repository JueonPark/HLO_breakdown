"""
notify which fusion kernel does which computation in hlo_table
"""
import argparse
from hlo_parser import HloTable
from hlo_metadata import *

def rewrite_fusion_kernel(hlo_table, model):
  """
  generate dictionary of original fusion kernel's name and its new name
  """
  hlo_breakdown = {}
  # for each fusion:
  for hlo_fusion in hlo_table:
    # for each fusion's metadata:
    metadata_list = []
    for metadata in hlo_table[hlo_fusion]["data"]:
      if model == "bert":
        parsed_metadata = parse_bert_metadata(metadata)
        if parsed_metadata in metadata_list:
          continue
        elif parsed_metadata is None:
          continue
        else:
          metadata_list.append(parse_bert_metadata(metadata))
      elif model == "vit":
        parsed_metadata = parse_vit_metadata(metadata)
        if parsed_metadata in metadata_list:
          continue
        elif parsed_metadata is None:
          continue
        else:
          metadata_list.append(parse_vit_metadata(metadata))
      elif model == "transformer":
        parsed_metadata = parse_transformer_metadata(metadata)
        if parsed_metadata in metadata_list:
          continue
        elif parsed_metadata is None:
          continue
        else:
          metadata_list.append(parse_transformer_metadata(metadata))
      else:
        parsed_metadata = parse_cnn_metadata(metadata)
        if parsed_metadata in metadata_list:
          continue
        elif parsed_metadata is None:
          continue
        else:
          metadata_list.append(parse_cnn_metadata(metadata))

    # now, with metadata_list, append hlo_fusion's computation property into hlo_breakdown
    metadata_list.sort()
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
    hlo_breakdown[hlo_fusion] = output_name
  return hlo_breakdown

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--hlo', type=str, required=True, help="xla hlo graph path (text)")
  args = parser.parse_args()
  hlo_file = open(args.hlo, "r")
  hlo_table = HloTable(hlo_file.read()).hlo_table
  
  # write result to file
  hlo_breakdown = rewrite_fusion_kernel(hlo_table)
  for name in hlo_breakdown:
    print(name + ": " + hlo_breakdown[name])