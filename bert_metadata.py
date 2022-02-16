def parse_metadata(input):
  # if (input.find("gradient_tape") != -1):
  #   return "backward"
  if (input.find("SparseSoftmaxCrossEntropyWithLogits") != -1):
    return "SoftmaxCrossEntropy"
  elif (input.find("layer_norm") != -1):
    return "layernorm"
  elif (input.find("LayerNorm") != -1):
    return "layernorm"
  elif (input.find("dropout") != -1):
    return "dropout"
  elif (input.find("Softmax") != -1):
    return "softmax"
  elif (input.find("BiasAdd") != -1):
    return "biasadd"
  elif (input.find("Gelu") != -1):
    return "gelu"
  else:
    return None