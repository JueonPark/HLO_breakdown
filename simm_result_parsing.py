import pandas as pd
import gzip
import json
import csv
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
def parse_name(name):
  name = name.lower()
  name = re.sub(r'[0-9]*', '', name)
  if 'loss' in name or 'entropy' in name:
    return 'Fusion(Loss)'
  elif 'optimizer' in name or 'update' in name or 'adam' in name:
    return 'Fusion(Optimizer)'
  elif 'layernorm' in name or 'layer_norm' in name:
    return 'Fusion(LayerNorm)'
  elif 'batchnorm' in name or 'batch_norm' in name or 'relu' in name:
    return 'Fusion(BN+ReLU)'
  # elif 'cast' in name:
  #   return 'Fusion(Cast)'
  elif 'dropout' in name:
    return 'Fusion(Dropout)'
  elif 'einsum' in name:
    return 'Fusion(Einsum)'
  elif 'biasadd' in name:
    return 'Fusion(BiasAdd)'
  elif 'gelu' in name:
  #   return 'Fusion(Gelu)'
  # elif 'relu' in name:
    return 'Fusion(ReLU)'
  elif 'self_attention' in name:
    return 'Fusion(SelfAttention)'
  return 'Fusion'
def find_kernel_metadata(name, hlo_file_name):
  if len(name.split('_')) == 1:
    return name
  name = 'fusion.' + name.split('_')[1]
  with open(hlo_file_name, 'r') as f:
    for line in f.readlines():
      if name in line:
        find_meta = line.split('metadata')
        if len(find_meta) > 1 and len(find_meta[1].split('\"')) > 3:
          print(name, find_meta[1])
          name1 =find_meta[1].split('\"')[3]
          name1 = parse_name(name1)
          return name1
    return 'Fusion'
namespace = ['CONV+BN+ELEMWISE', 'NDP_OP', 'BN+ELEMWISE', 'CONV', 'FC', 'POOLING', 'dxCONV', 'dwCONV', 'dxFC', 'dwFC', 'OPT',\
        'Fusion(Loss)', 'Fusion(Optimizer)', 'Fusion(LayerNorm)','Fusion(BatchNorm)', 'Fusion(Cast)', 'Fusion(Dropout)',\
        'Fusion(Einsum)', 'Fusion(BiasAdd)', 'Fusion(Gelu)', 'Fusion(ReLU)', 'Fusion(SelfAttention)', 'Fusion', 'Fusion(BN+ReLU)']
def setup_dataframe(df):
   # df.to_csv('./before.csv')   
   # namespace = ['fusion', 'elementwise', 'GEMM', 'CONV', 'Dgrad', 'Wgrad' ,'others']
   compute_intensives = ['GEMM', 'CONV', 'Wgrad', 'Dgrad']
   df['NAME'] = df['NAME'].replace('fusion.*', 'fusion', regex=True)\
                                       .replace('dgemm', 'dxFC', regex=True)\
                                       .replace('wgemm', 'dwFC', regex=True)\
                                       .replace('.*MetaKernel.*', 'elementwise', regex=True)\
                                       .replace('.*add.*', 'elementwise', regex=True)\
                                       .replace('.*mul.*', 'elementwise', regex=True)\
                                       .replace('.*log.*', 'elementwise', regex=True)\
                                       .replace('.*gemm_.*_tn', 'dxFC', regex=True)\
                                       .replace('.*gemm_.*_nt', 'dwFC', regex=True)\
                                       .replace('.*dgrad.*', 'dxCONV', regex=True)\
                                       .replace('.*wgrad.*', 'dwCONV', regex=True)\
                                       .replace('.*1688cudnn.*', 'CONV+BN+ELEMWISE', regex=True)\
                                       .replace('.*convol.*', 'CONV+BN+ELEMWISE', regex=True)\
                                       .replace('.*gemm_.*', 'FC', regex=True)\
                                       .replace('gemm', 'FC', regex=True)\
                                       .replace('conv', 'CONV', regex=True)\
                                       .replace('.*conv2d.*', 'CONV+BN+ELEMWISE', regex=True)\
                                       .replace('.*first_layer.*', "CONV+BN+ELEMWISE", regex=True)\
                                       .replace('pool', "POOLING", regex=True)\
                                       .replace('.*c1_k1.*', "CONV+BN+ELEMWISE", regex=True)\
                                       .replace('.*bn.*', "BN+ELEMWISE", regex=True)\
                                       .replace('.*adam*', "OPT", regex=True)
                                       # .replace('turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_tn', 'dxFC', regex=True)\
                                       # .replace('turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_tn', 'dxFC', regex=True)\
                                       # .replace('turing_fp16_s1688gemm_fp16_256x64_ldg8_f2f_tn', 'dxFC', regex=True)\
                                       # .replace('turing_fp16_s884gemm_fp16_64x64_ldg8_f2f_tn', 'dxFC', regex=True)\
                                       # .replace('turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_nt', 'dwFC', regex=True)\
                                       # .replace('turing_fp16_s884gemm_fp16_256x128_ldg8_f2f_nt', 'dwFC', regex=True)\
                                       # .replace('turing_fp16_s884gemm_fp16_128x64_ldg8_f2f_nt', 'dwFC', regex=True)\
   # df.to_csv('./test2.csv')
   df['CONFIG'] = df['CONFIG'].replace('V100_downscaled_HBM2_PCI6x1_write_prio_debug', 'NDPX_debug')
   df['CONFIG'] = df['CONFIG'].replace('V100_downscaled_HBM2_PCI6x1_write_prio', 'NDPX')
   df['CONFIG'] = df['CONFIG'].replace('V100_downscaled_HBM2_PCI6x1', 'NDPX')
   df['CONFIG'] = df['CONFIG'].replace('V100_downscaled_HBM2_PCI6x1_baseline', 'NDPX')
   df.loc[~df['NAME'].isin(namespace), 'NAME'] = 'Others'
   df.loc[df['CYCLE'] == '  NOT FOUND', 'CYCLE'] = '0'
   
   return df
def setup_dataframe_baseline(df, hlo_path):
   df.to_csv('./before.csv')   
    # namespace = ['fusion', 'elementwise', 'GEMM', 'CONV', 'Dgrad', 'Wgrad' ,'others']
   compute_intensives = ['GEMM', 'CONV', 'Wgrad', 'Dgrad']
   fusions = df.loc[df['NAME'].str.contains('fusion')]['NAME']
   for fusion in fusions:
      opname = find_kernel_metadata(fusion, hlo_path)
      print(fusion, opname)
      df.loc[df['NAME'] == fusion, 'NAME'] = opname
   df['NAME'] = df['NAME'].replace('dgemm', 'dxFC', regex=True)\
                                       .replace('wgemm', 'dwFC', regex=True)\
                                       .replace('.*MetaKernel.*', 'elementwise', regex=True)\
                                       .replace('.*add.*', 'elementwise', regex=True)\
                                       .replace('.*mul.*', 'elementwise', regex=True)\
                                       .replace('.*log.*', 'elementwise', regex=True)\
                                       .replace('.*gemm_.*_tn', 'dxFC', regex=True)\
                                       .replace('.*gemm_.*_nt', 'dwFC', regex=True)\
                                       .replace('.*dgrad.*', 'dxCONV', regex=True)\
                                       .replace('.*wgrad.*', 'dwCONV', regex=True)\
                                       .replace('.*1688cudnn.*', 'CONV', regex=True)\
                                       .replace('.*convol.*', 'CONV', regex=True)\
                                       .replace('.*gemm_.*', 'FC', regex=True)\
                                       .replace('gemm', 'FC', regex=True)\
                                       .replace('conv', 'CONV', regex=True)\
                                       .replace('.*conv2d.*', 'CONV', regex=True)\
                                       .replace('.*first_layer.*', "CONV", regex=True)\
                                       .replace('pool', "POOLING", regex=True)\
                                       .replace('.*c1_k1.*', "CONV", regex=True)\
                                       .replace('.*bn.*', "BN+ELEMWISE", regex=True)\
                                       .replace('.*adam*', "OPT", regex=True)
                                       # .replace('turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_tn', 'dxFC', regex=True)\
                                       # .replace('turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_tn', 'dxFC', regex=True)\
                                       # .replace('turing_fp16_s1688gemm_fp16_256x64_ldg8_f2f_tn', 'dxFC', regex=True)\
                                       # .replace('turing_fp16_s884gemm_fp16_64x64_ldg8_f2f_tn', 'dxFC', regex=True)\
                                       # .replace('turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_nt', 'dwFC', regex=True)\
                                       # .replace('turing_fp16_s884gemm_fp16_256x128_ldg8_f2f_nt', 'dwFC', regex=True)\
                                       # .replace('turing_fp16_s884gemm_fp16_128x64_ldg8_f2f_nt', 'dwFC', regex=True)\
   # df.to_csv('./test2.csv')
   df['CONFIG'] = df['CONFIG'].replace('V100_downscaled_HBM2_PCI6x1', 'NDPX')
   df.loc[~df['NAME'].isin(namespace), 'NAME'] = 'Others'
   df.loc[df['CYCLE'] == '  NOT FOUND', 'CYCLE'] = '0'
   df.to_csv('./after.csv')
   return df