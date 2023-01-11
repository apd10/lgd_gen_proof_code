import torch
from torch import nn
from torch.autograd import Variable
from Data import Data
from Model import Model
from Optimizer import Optimizer
from Loss import Loss
from ProgressEvaluator import ProgressEvaluator
import pdb
import numpy as np
from tqdm import tqdm
import pandas as pd
from race.race_classwise_torch import Race_Classwise_torch

class Loop_withRace:
    def __init__(self, params):
        self.device_id = params["device_id"]
        self.epochs = params["epochs"]
        # data
       # self.train_data = Data(params["train_data"])
        self.progress_test_data = Data(params["progress_test_data"], loss=params["loss"])
        datas = {"valid" : self.progress_test_data}
        self.test_data = None
        if "test_data" in params:
            self.test_data = Data(params["test_data"], loss=params["loss"])
            datas['test'] = self.test_data
        self.progress_train_data = None
        if "progress_train_data" in params:
            self.progress_train_data = Data(params["progress_train_data"], loss=params["loss"])
            datas['train'] = self.progress_train_data
        # model
        self.model = Model.get(params["model"])
        print(self.model)
        #self.race_weighted = RaceGen(params["race"])
        #self.race_unweighted = RaceGen(params["race"])
        #pdb.set_trace()
        self.race = Race_Classwise_torch(params["race"])
        self.train_data = Data(params["train_data"],model=self.model, race=self.race, device_id=self.device_id, loss=params["loss"])

        if self.device_id != -1:
          self.model = self.model.cuda(self.device_id)
        # optimizer
        self.optimizer = Optimizer.get(self.model, params["optimizer"])
        # loss
        self.loss_func = Loss.get(params["loss"])
        #if self.device_id != -1:
        #  self.loss_func = self.loss_func.cuda(self.device_id)
        # progress evaluator
        self.progress_evaluator = ProgressEvaluator.get(params["progress_evaluator"], datas, self.device_id)
        self.metrics = []
        if "metrics" in params:
            self.metrics = params["metrics"].split(",")
        self.binary = False
        if "binary" in params:
            self.binary = params["binary"]

        self.regression = False
        if "regression" in params:
            self.regression = params["regression"]

        self.quiet = False
        if "quiet" in params:
            self.quiet = params["quiet"]

        self.model_internal_logging_itr = -1
        if "model_internal_logging_itr" in params:
            self.model_internal_logging_itr = params["model_internal_logging_itr"]
        self.model_log_file = "./model_log.csv"
        if "model_log_file" in params:
            self.model_log_file = params["model_log_file"]
        self.set_full_data_in_model = False
        if "set_full_data_in_model" in params:
            self.set_full_data_in_model = params["set_full_data_in_model"]

        #np.random.seed(self.np_seed)
        #torch.manual_seed(self.np_seed)

    def get_complete_data(self):
        self.train_data.reset()
        num_samples = self.train_data.len()
        batch_size = self.train_data.batch_size()
        num_batches = int(np.ceil(num_samples/batch_size))
        xs = []
        ys = []
        for i in tqdm(range(num_batches), disable=self.quiet):
            if self.train_data.end():
              break
            x, y = self.train_data.next()
            xs.append(x)
            ys.append(y)

        return torch.cat(xs, dim=0).cuda(self.device_id), torch.cat(ys, dim=0).cuda(self.device_id)
    

    def loop(self):
        epoch = 0
        iteration = 0
        if self.set_full_data_in_model:
            x_data, y_data = self.get_complete_data()
            self.model.set_data(x_data, y_data)

        while epoch < self.epochs :
            self.train_data.reset()
            num_samples = self.train_data.len()
            batch_size = self.train_data.batch_size()
            num_batches = int(np.ceil(num_samples/batch_size))
            loc_itr = 0
            for i in tqdm(range(num_batches), disable=self.quiet):
                if self.train_data.end_filter():
                  break
                self.model.train()
                self.optimizer.zero_grad()
                x_cpu, y_cpu, w_cpu = self.train_data.next_filter()
                x = Variable(x_cpu).cuda(self.device_id) if self.device_id!=-1 else Variable(x_cpu)
                y = Variable(y_cpu).cuda(self.device_id) if self.device_id!=-1 else Variable(y_cpu)
                wght = Variable(w_cpu).cuda(self.device_id) if self.device_id!=-1 else Variable(w_cpu)
               # pdb.set_trace()
                output = self.model(x)
              #  self.race.sketch(x_cpu, y_cpu)
                if (self.set_full_data_in_model):
                    y = self.model.y_data()
                if self.binary or self.regression:
                    loss = self.loss_func(output.view(-1), y.float(), reduction='none')
                else:
                    loss = self.loss_func(output, y, reduction='none')

               # self.race.sketch(x_cpu, y_cpu, loss.to(device=x_cpu.device))    
              #  pdb.set_trace()
                #self.race.query_values(x_cpu)
                loss = torch.mul(loss, wght) # weighted_loss
                loss.mean().backward()
                self.optimizer.step()
                self.progress_evaluator.evaluate(epoch, loc_itr, iteration, self.model, self.loss_func, self.train_data.num_points/self.train_data.unfiltered_num_points, metrics=self.metrics, binary=self.binary, regression=self.regression)
                #pdb.set_trace()
                if self.model_internal_logging_itr > 0 and iteration % self.model_internal_logging_itr == 0:
                   # pdb.set_trace()
                    self.model.logger(iteration, True)
                    logdata = self.model.get_logged_data(True)
                    if len(logdata['iterations']) > 0:
                        df = pd.DataFrame(logdata)
                        df.to_csv(self.model_log_file, index=False)
                iteration = iteration + 1
                loc_itr = loc_itr + 1
                
               # print('********iteration******', iteration) # iteration over the filtered data
               # print('*********data iter**********',self.train_data.sampler.iter_idx) # iteraion over the filtered data
               # print('*********data iter interim**********',self.train_data.sampler.iter_interim_idx) # iteration over the whole data
               # print()
                #print("Loss", loss)
        #    pdb.set_trace()
           # sketch = self.race.get_dictionary()
            loc_itr = 0
       #     self.progress_evaluator.evaluate(epoch, loc_itr, iteration, self.model, self.loss_func, self.train_data.num_points, metrics=self.metrics, binary=self.binary, regression=self.regression)
            epoch = epoch + 1 
            #loc_itr = 0
            print('****num whole data****',self.train_data.unfiltered_num_points)
            print('****num filtered data****',self.train_data.num_points)
            assert self.train_data.unfiltered_num_points==num_samples

        self.progress_evaluator.evaluate(epoch-1, loc_itr, iteration, self.model, self.loss_func, self.train_data.num_points/self.train_data.unfiltered_num_points, metrics=self.metrics, binary=self.binary, regression=self.regression)
        if self.model_internal_logging_itr > 0:
            logdata = self.model.get_logged_data(True)
            if logdata is not None:
                df = pd.DataFrame(logdata)
                df.to_csv(self.model_log_file)
