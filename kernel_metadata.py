def rewrite_bert_kernel(row):
  if (row[4].find("nn") != -1 and row[4].find("gemm") != -1):
    row.append("GEMM")
  elif (row[4].find("tn") != -1 and row[4].find("gemm") != -1):
    row.append("dxFC")
  elif (row[4].find("nt") != -1 and row[4].find("gemm") != -1):
    row.append("dwFC")
  elif (row[4].find("cudnn") != -1):
    row.append("cudnn")
  elif (row[4].find("convert") != -1):
    row.append("Convert")
  elif (row[4].find("reduce") != -1):
    row.append("Reduction")
  elif (row[4].find("pad") != -1):
    row.append("Pad")
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

def rewrite_vit_kernel(row):
  if (row[4].find("gemm") != -1):
    row.append("GEMM")
  elif (row[4].find("Transpose") != -1):
    row.append("Transpose")
  elif (row[4].find("broadcast") != -1):
    row.append("Broadcast")
  elif (row[4].find("wgrad") != -1):
    row.append("wgrad")
  elif (row[4].find("dgrad") != -1):
    row.append("dgrad")
  elif (row[4].find("add") != -1):
    row.append("Elementwise")
  elif (row[4].find("mul") != -1):
    row.append("Elementwise")
  elif (row[4].find("div") != -1):
    row.append("Elementwise")
  else:
    row.append("others")
  return row

def rewrite_transformer_kernel(row):
  if (row[4].find("gemm") != -1):
    row.append("GEMM")
  elif (row[4].find("Transpose") != -1):
    row.append("Transpose")
  elif (row[4].find("broadcast") != -1):
    row.append("Broadcast")
  elif (row[4].find("add") != -1):
    row.append("Elementwise")
  elif (row[4].find("mul") != -1):
    row.append("Elementwise")
  elif (row[4].find("div") != -1):
    row.append("Elementwise")
  elif (row[4].find("reduce") != -1):
    row.append("Reduction")
  elif (row[4].find("log") != -1):
    row.append("Log")
  elif (row[4].find("RandomKernel") != -1):
    row.append("RandomKernel")
  elif (row[4].find("cudnn") != -1 and (row[4].find("interior") != -1)):
    row.append("Conv")
  else:
    row.append("others")
  return row

def rewrite_dlrm_kernel(row):
  if (row[4].find("gemm") != -1):
    row.append("GEMM")
  elif (row[4].find("Transpose") != -1):
    row.append("Transpose")
  elif (row[4].find("broadcast") != -1):
    row.append("Broadcast")
  elif (row[4].find("add") != -1):
    row.append("Elementwise")
  elif (row[4].find("reduce") != -1):
    row.append("Reduction")
  elif (row[4].find("log") != -1):
    row.append("Log")
  else:
    row.append("others")
  return row

def rewrite_lstm_kernel(row):
  if (row[4].find("gemm") != -1):
    row.append("GEMM")
  elif (row[4].find("Transpose") != -1):
    row.append("Transpose")
  elif (row[4].find("add") != -1):
    row.append("Elementwise")
  elif (row[4].find("mul") != -1):
    row.append("Elementwise")
  elif (row[4].find("div") != -1):
    row.append("Elementwise")
  elif (row[4].find("reduce") != -1):
    row.append("Reduction")
  elif (row[4].find("log") != -1):
    row.append("Log")
  else:
    row.append("others")
  return row