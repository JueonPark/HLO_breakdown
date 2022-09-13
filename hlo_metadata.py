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

def parse_cnn_metadata(input):
  if (input.find("SparseSoftmaxCrossEntropyWithLogits") != -1):
    return "Loss"
  elif (input.find("bert_pretrain_loss_and_metric_layer") != -1):
    return "Loss"
  elif (input.find("FusedBatchNormV3") != -1):
    return "BN"
  elif (input.find("batch_norm") != -1):
    return "BN"
  elif (input.find("BatchNorm") != -1):
    return "BN"
  elif (input.find("batchnorm") != -1):
    return "BN"
  elif (input.find("BiasAdd") != -1):
    return "Biasadd"
  elif (input.find("Relu6") != -1):
    return "Relu6"
  elif (input.find("Relu") != -1):
    return "Relu"
  elif (input.find("Adam") != -1):
    return "Adam"
  else:
    return None

def parse_vit_metadata(input):
  # if (input.find("gradient_tape") != -1):
  #   return "backward"
  if (input.find("SparseSoftmaxCrossEntropyWithLogits") != -1):
    return "Loss"
  elif (input.find("multi_head_self_attention") != -1):
    return "MultiHeadSelfAttention"
  elif (input.find("layer_norm") != -1):
    return "LN"
  elif (input.find("layer_normalization") != -1):
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
  elif (input.find("transpose") != -1):
    return "Transpose"
  elif (input.find("dropout") != -1):
    return "Dropout"
  elif (input.find("Gelu") != -1):
    return "Gelu"
  # elif (input.find("add") != -1):
  #   return "Elementwise"
  # elif (input.find("Mul") != -1):
  #   return "Elementwise"
  # elif (input.find("Softmax") != -1):
  #   return "Softmax"
  # elif (input.find("BiasAdd") != -1):
  #   return "Biasadd"
  else:
    return None