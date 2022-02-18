def parse_metadata(input):
  # if (input.find("gradient_tape") != -1):
  #   return "backward"
  if (input.find("SparseSoftmaxCrossEntropyWithLogits") != -1):
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