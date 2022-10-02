from models.list_models import *

class Model:
  def get(params):
    model = None
    if params["name"] == "MLP":
      model = FCN(params["MLP"]["input_dim"], params["MLP"]["num_layers"], params["MLP"]["hidden_size"], params["MLP"]["num_class"])
    elif params["name"] == "ROASTMLP":
      model = ROASTFCN(params["ROASTMLP"]["input_dim"], params["ROASTMLP"]["num_layers"], params["ROASTMLP"]["hidden_size"], params["ROASTMLP"]["num_class"], params["ROASTMLP"]["compression"], params["ROASTMLP"]["seed"])
    elif params["name"] == "MLP_REG":
      model = FCN_REG(params["MLP_REG"]["input_dim"], params["MLP_REG"]["num_layers"], params["MLP_REG"]["hidden_size"])
    elif params["name"] == "LIN_REG":
      model = LIN_REG(params["LIN_REG"]["input_dim"])
    elif params["name"] == "MLPSG":
      model = FCNSG(params["MLPSG"]["input_dim"], params["MLPSG"]["num_layers"], params["MLPSG"]["hidden_size"])
    elif params["name"] == "MLPSGDLN":
      model = FCNSGDLN(params["MLPSGDLN"]["input_dim"], params["MLPSGDLN"]["num_layers"], params["MLPSGDLN"]["hidden_size"], params["MLPSGDLN"]["num_linear_layers"], params["MLPSGDLN"]["dropout"])
    elif params["name"] == "PLearn1HSG":
      model = PLearn1HSG(params["PLearn1HSG"]["input_dim"], params["PLearn1HSG"]["small_hidden_size"], params["PLearn1HSG"]["num_components"])
    elif params["name"] == "SRP_REG":
      model = SRPSingleModelRegression(params["SRP_REG"]["input_dim"], params["SRP_REG"]["hidden_size"], params["SRP_REG"]["k"], params["SRP_REG"]["l"])
    elif params["name"] == "RELUHNN":
      model = ReLUHNN1(params["RELUHNN"]["input_dim"], params["RELUHNN"]["hidden_size"], params["RELUHNN"]["mode"], params["RELUHNN"])
    elif params["name"] == "MLP-ROBEZ":
      model = FCN_robez(params["MLP-ROBEZ"]["input_dim"], params["MLP-ROBEZ"]["num_layers"], params["MLP-ROBEZ"]["hidden_size"] ,params["MLP-ROBEZ"]["weight_sizes"], params["MLP-ROBEZ"]["chunk_size"], params["MLP-ROBEZ"]["num_class"])
    elif params["name"] == "MLP-LS":
      model = FCN_firstLS(params["MLP-LS"]["input_dim"], params["MLP-LS"]["num_layers"], params["MLP-LS"]["hidden_size"], params["MLP-LS"]["LS"], params["MLP-LS"]["num_class"])
    elif params["name"] == "MLPSG-LS":
      model = FCNSG_firstLS(params["MLPSG-LS"]["input_dim"], params["MLPSG-LS"]["num_layers"], params["MLPSG-LS"]["hidden_size"], params["MLPSG-LS"]["LS"])
    elif params["name"] == "1HNN_LGD":
      model = FCN_REG_LGD(params["1HNN_LGD"]["input_dim"], params["1HNN_LGD"]["hidden_size"], params["1HNN_LGD"]["si_params"])
    elif params["name"] == "1HNN_REG_S":
      model = FCN_HNN_REG_S(params["1HNN_REG_S"]["input_dim"], params["1HNN_REG_S"]["hidden_size"], params["1HNN_REG_S"])
    else:
      raise NotImplementedError
    return model


