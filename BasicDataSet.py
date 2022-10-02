from dataparsers.list_primary_dataparsers import *


class BasicDataSet:
    ''' specifies the parsing for the data
    '''
    def get(data_file, name, params):
        if name == "gensvm":
          dataset = GenSVMFormatParser(data_file, params)
        elif name == "csv":
          dataset = CSVParser(data_file, params)
        elif name == "race":
          dataset = RaceSampler(data_file, params)
        elif name == "race_pp":
          dataset = RaceSamplerPreProc(data_file, params)
        elif name == "race_gen_pp":
          dataset = RaceGenSamplerPreProc(data_file, params)
        else:
          raise NotImplementedError
        return dataset


