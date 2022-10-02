from samplers.list_samplers import *

class Sampler:
    def get(dataset, name, params):
        print(name, flush=True)
        if name == "simple":
            sampler = SimpleSampler(dataset, params)
        elif name == "simple_1hnn":
            sampler = SimpleSampler1hnn(dataset, params)
        elif name == "subsimple":
            sampler = SubSimpleSampler(dataset, params)
        else:
            raise NotImplementedError
        return sampler

        
      
