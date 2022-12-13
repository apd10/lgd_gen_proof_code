import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import pdb
from Loss import Loss
from special_modules.race.RaceGen import *

class RaceSampler:
    def __init__(self, dataset, params, model, race, device_id, loss):
        self.dataset = dataset
        self.total_size = dataset.__len__()
        self.batch_size = params["batch_size"]
        self.sampling_rate = params["sampling_rate"]
        self.current_idx = 0
        self.model = model
        self.race = race
        self.x_leftovers = np.array(())
        self.y_leftovers = np.array(())
        self.warmup_niter = 10 # TO DO: add it to the config
        self.iter_interim_idx = 0
        self.iter_idx = 0
        self.race = race
       # self.loss_func = Loss.get({"name": "BCE"})
        self.loss_func = Loss.get(loss)
        # TO DO: add train_flag to the config rather than extracting it from X_file!!!
        self.train_flag = 'train' in self.dataset.X_file # if 'train' is in the name of data, train_flag is True
        self.device_id = device_id
        
    def reset(self):
        # ??????
        self.current_idx = 0
        self.x_leftovers = np.array(())
        self.y_leftovers = np.array(())

    def next(self):
        if self.train_flag: # train data
            if self.iter_idx < self.warmup_niter:
                # Warmup
                X,y = self.dataset[self.current_idx:self.current_idx+self.batch_size]
                self.current_idx = self.current_idx + self.batch_size
                # update race sketch
                output = self.model(Variable(torch.FloatTensor(X)).cuda(self.device_id))
                loss = self.loss_func(output.view(-1), Variable(torch.FloatTensor(y)).cuda(self.device_id).float(), reduction='none')
    # #             self.race.sketch(torch.FloatTensor(X), torch.LongTensor(y), loss.to(device=X.device))
                self.race.sketch(torch.FloatTensor(X), torch.LongTensor(y), loss)
                # increment interation indecies
                self.iter_idx+=1
                self.iter_interim_idx+=1
            else:    
                xs = []
                ys = []
                if self.x_leftovers.shape[0]>0:
                    xs.append(self.x_leftovers)
                    ys.append(self.y_leftovers)
                    # update current batchsize
                    curr_batchsize = self.x_leftovers.shape[0]
                else:
                    curr_batchsize = 0

                while (curr_batchsize < self.batch_size) and (self.current_idx < self.total_size):
                   # pdb.set_trace()
                    xb_interim,yb_interim = self.dataset[self.current_idx:self.current_idx+self.batch_size]
                    self.current_idx = self.current_idx + self.batch_size
                    self.iter_interim_idx+=1
                    race_scores = self.race.query_values(torch.FloatTensor(xb_interim)) # b x num_classes
                    race_scores_np = race_scores.detach().numpy()
                    race_scores_class_specific = race_scores_np[np.arange(race_scores_np.shape[0]),yb_interim.astype(int)] # b x 1
                    sampling_rate = self.sampling_rate
                    threshold = np.percentile(race_scores_class_specific,100*(1-sampling_rate))

                   # race_scores = self.race.query_values_classspecific(torch.FloatTensor(xb_interim),torch.FloatTensor(yb_interim))
                   # accp_inds = np.where(race_scores.numpy()[:,0]>0)[0]
                   # pdb.set_trace()
                    accp_inds = np.where(race_scores_class_specific>threshold)[0]
                   # accp_inds = np.where(race_scores.detach().numpy()[:,0]>-0.01)[0]
                   # accp_inds = np.random.choice(1000,600) 
                    diff_batchsize = self.batch_size-curr_batchsize
                    if diff_batchsize < len(accp_inds):
                        accp_len = diff_batchsize
                    else:
                        accp_len = len(accp_inds)

                    xs.append(xb_interim[accp_inds[:accp_len]])
                    ys.append(yb_interim[accp_inds[:accp_len]])
                    # update current batchsize
                    curr_batchsize+=accp_len
                    if accp_len < len(accp_inds): 
                        self.x_leftovers = xb_interim[accp_inds[accp_len:]]
                        self.y_leftovers = yb_interim[accp_inds[accp_len:]]

    #             pdb.set_trace()
                self.iter_idx+=1
                X = np.concatenate(xs,axis=0)
                y = np.concatenate(ys,axis=0)
        else: # test and valid data
            X,y = self.dataset[self.current_idx:min(self.current_idx+self.batch_size, self.total_size)]
            self.current_idx = self.current_idx + self.batch_size


        if self.dataset.regression:
            return X.shape[0], self.current_idx, (torch.FloatTensor(X), torch.FloatTensor(y))
        else:
            return X.shape[0], self.current_idx, (torch.FloatTensor(X), torch.LongTensor(y))
