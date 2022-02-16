"""
trace.json을 parsing한다.
 - traceEvents에 있는 dur을 가지고 있는 애들만 다 취합해서 data로 만들어 버린다.
 - 만든 data들을 가지고서 breakdown을 한다.
"""
import re
import csv
import json
from tqdm import tqdm
from hlo_parser import HloTable

# kernel에 필요한 data: kernel name, duration
class Kernel:
  def __init__(self, name, dur):
    self.name = name
    self.dur = dur

# get metadata from hlo graph
hlo_file = open("after_optimizations.txt")
hlo_metadata_table = HloTable(hlo_file.read()).hlo_table

def is_xla_kernel(kernel_name):
  pattern = re.compile('[a-z]+(_[0-9][0-9]*)?$')
  if pattern.match(kernel_name):
    return True
  elif (kernel_name.find("fusion") != -1):
    return True
  else:
    return False

def map_fusion(input):
  try:
    kernel_name = "fusion_" + input.split("_")[1]
  except:
    kernel_name = "fusion"
  if not kernel_name in hlo_metadata_table:
    return "fusion"
  for instr in hlo_metadata_table[kernel_name]["instr"]:
    if (instr.find("reduce-window") != -1):
      return "pool"  
  for data in hlo_metadata_table[kernel_name]["data"]:
    if (data.find("layer_norm") != -1):
      return "layernorm"
    elif (data.find("LayerNorm") != -1):
      return "layernorm"
    elif (data.find("dropout") != -1):
      return "fused_dropout"
    elif (data.find("Softmax") != -1):
      return "fused_softmax"
  return "fusion"

def pattern(input):
  if (input.find("fusion") != -1):
    return map_fusion(input)
  elif (is_xla_kernel(input)):
    # return input.split("_")[0]
    return "not_fused_xla_kernels"
  else:
    return "others"

# parse trace.json and get specific information from 
tracefile = open("trace.json")
trace = json.load(tracefile)

total_duration = 0
kernel_data = []
to_dump = []
compute_stream = []
trace_events = trace["traceEvents"]
for data in trace_events:
  if ("ph" in data) and (data["ph"] == 'M'):
    continue
  elif "dur" in data:
    total_duration += data["dur"]
    kernel = Kernel(data["name"], data["dur"])
    kernel_data.append(kernel)

print("total duration: {}".format(total_duration))

# to check kernel name
kernel_name_out = open("kernel_names.csv", "w+")
kernel_name_writer = csv.writer(kernel_name_out)
kernel_name_list = []
for kernel in kernel_data:
  if pattern(kernel.name) in kernel_name_list:
    continue
  else:
    kernel_name_list.append(pattern(kernel.name))
for kernel_name in kernel_name_list:
  kernel_name_writer.writerow([kernel.name])

# finally, write result to csv file
output = open("parsed_output_fusion.csv", "w+")
writer = csv.writer(output)
header = ["kernel_name", "duration"]
writer.writerow(header)
for kernel in kernel_data:
  row = [pattern(kernel.name), kernel.dur]
  writer.writerow(row)