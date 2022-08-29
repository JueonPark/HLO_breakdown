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
  # elif (input.find("pooling") != -1):
  #   return "Pooling"
  # elif (input.find("pad") != -1):
  #   return "Padding"
  else:
    return None