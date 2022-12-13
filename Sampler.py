from samplers.list_samplers import *

class Sampler:
    def get(dataset, name, params, model=None, race=None, device_id=None, loss=None):
        print(name, flush=True)
        if name == "simple":
            sampler = SimpleSampler(dataset, params)
        elif name == "simple_1hnn":
            sampler = SimpleSampler1hnn(dataset, params)
        elif name == "subsimple":
            sampler = SubSimpleSampler(dataset, params)
        elif name == "race":
            sampler = RaceSampler(dataset, params, model, race, device_id, loss)
        else:
            raise NotImplementedError
        return sampler

        
      
