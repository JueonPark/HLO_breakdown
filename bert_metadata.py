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

def parse_bert_metadata(input):
  # if (input.find("gradient_tape") != -1):
  #   return "backward"
  if (input.find("SparseSoftmaxCrossEntropyWithLogits") != -1):
    return "Loss"
  elif (input.find("bert_pretrain_loss_and_metric_layer") != -1):
    return "Loss"
  elif (input.find("layer_norm") != -1):
    return "LN"
  elif (input.find("LayerNorm") != -1):
    return "LN"
  elif (input.find("layernorm") != -1):
    return "LN"
  elif (input.find("batch_norm") != -1):
    return "LN"
  elif (input.find("BatchNorm") != -1):
    return "LN"
  elif (input.find("batchnorm") != -1):
    return "LN"
  elif (input.find("dropout") != -1):
    return "Dropout"
  elif (input.find("Softmax") != -1):
    return "Softmax"
  elif (input.find("BiasAdd") != -1):
    return "Biasadd"
  elif (input.find("Gelu") != -1):
    return "Gelu"
  else:
    return None