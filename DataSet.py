#from dataparsers.list_dataparsers import GenSVMFormatParser, CSVParser, RaceSampler, RaceSamplerPreProc, RaceGenSamplerPreProc, RunningRaceParser
from dataparsers.list_dataparsers import *


class DataSet:
    ''' specifies the parsing for the data
    '''
    def get(data_file, name, params):
        if name == "gensvm":
          dataset = GenSVMFormatParser(data_file, params)
        elif name == "csv":
          dataset = CSVParser(data_file, params)
        elif name == "csv_1hnn":
          dataset = CSV1hnnParser(data_file, params)
        elif name == "bin":
          dataset = BinParser(data_file, params)
        elif name == "race":
          dataset = RaceSampler(data_file, params)
        elif name == "race_pp":
          dataset = RaceSamplerPreProc(data_file, params)
        elif name == "race_ppd":
          dataset = RaceSamplePPD(data_file, params)
        elif name == "race_hist":
          dataset = RaceSampleHist(data_file, params)
        elif name == "race_gen_pp":
          dataset = RaceGenSamplerPreProc(data_file, params)
        elif name == "running_race":
          dataset = RunningRaceParser(data_file, params)
        elif name == "gaussian_kde":
          dataset = GaussianKDE(data_file, params)
        else:
          raise NotImplementedError
        return dataset



