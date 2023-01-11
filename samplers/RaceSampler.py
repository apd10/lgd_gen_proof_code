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
        self.race_update_freq = params["race_update_freq"]
        self.accept_prob = params["accept_prob"]
        self.warmup_niter = params["warmup_niter"]
        self.current_idx = 0
        self.model = model
        self.race = race
        self.x_leftovers = np.array(())
        self.y_leftovers = np.array(())
        self.iter_interim_idx = 0
        self.iter_idx = 0
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
        self.w_leftovers = np.array(())

    def update_race(self,x,y):
       # pdb.set_trace()
        output = self.model(Variable(torch.FloatTensor(x)).cuda(self.device_id))
        loss = self.loss_func(output.view(-1), Variable(torch.FloatTensor(y)).cuda(self.device_id).float(), reduction='none')
        self.race.add_class(torch.FloatTensor(x), torch.LongTensor(y), loss.to(device=torch.FloatTensor(x).device))

    def next(self):
        if self.train_flag: # train data
            if self.iter_idx < self.warmup_niter:
                # Warmup
                X,y = self.dataset[self.current_idx:self.current_idx+self.batch_size]
                wght = np.ones_like(y)
                self.current_idx = self.current_idx + self.batch_size
                # update race sketch
             #   output = self.model(Variable(torch.FloatTensor(X)).cuda(self.device_id))
             #   loss = self.loss_func(output.view(-1), Variable(torch.FloatTensor(y)).cuda(self.device_id).float(), reduction='none')
    # #             self.race.sketch(torch.FloatTensor(X), torch.LongTensor(y), loss.to(device=X.device))
             #   self.race.sketch(torch.FloatTensor(X), torch.LongTensor(y), loss)
                self.update_race(X,y)
                # increment interation indecies
                self.iter_idx+=1
                self.iter_interim_idx+=1
            else:    
                xs = []
                ys = []
                ws = []
                if self.x_leftovers.shape[0]>0:
                    xs.append(self.x_leftovers)
                    ys.append(self.y_leftovers)
                    ws.append(self.w_leftovers)
                    # update current batchsize
                    curr_batchsize = self.x_leftovers.shape[0]
                else:
                    curr_batchsize = 0

                while (curr_batchsize < self.batch_size) and (self.current_idx < self.total_size):
                    #pdb.set_trace()
                    xb_interim,yb_interim = self.dataset[self.current_idx:min(self.current_idx+self.batch_size, self.total_size)]
                    self.current_idx = min(self.current_idx + self.batch_size, self.total_size)
                    self.iter_interim_idx+=1 # iter over whole data
        
                    race_w_scores, race_uw_scores = self.race.query_class(torch.FloatTensor(xb_interim)) # b x num_classes
                    # weighted race scores
                    race_w_scores_np = race_w_scores.detach().numpy()
                    race_w_scores_class_specific = race_w_scores_np[np.arange(race_w_scores_np.shape[0]),yb_interim.astype(int)] # b x 1
                    # unweighted race scores
                    race_uw_scores_np = race_uw_scores.detach().numpy()
                    race_uw_scores_class_specific = race_uw_scores_np[np.arange(race_uw_scores_np.shape[0]),yb_interim.astype(int)] # b x 1

                    race_scores_class_specific = np.divide(race_w_scores_class_specific,race_uw_scores_class_specific) # estimated loss
                   # inds_nan = np.where(np.isnan(race_scores_class_specific))[0] 
                   # num_nan = len(inds_nan) # number of nans
                   # if num_nan>0: # TO DO: compelete
                   #     sorted_race_scores_class_specific = np.sort(race_scores_class_specific)
                   #     sorted_scores_without_nan = sorted_race_scores_class_specific[:-num_nan]
                   #     adjusted_sampling_rate = (sampling_rate*(race_uw_scores_class_specific.shape[0])-num_nan)/(race_uw_scores_class_specific.shape[0]-num_nan)
                   #     raise NotImplementedError

                    #pdb.set_trace()
                    per = ((1-self.sampling_rate)/(1-self.accept_prob)*100.)
                    self.threshold = np.percentile(race_scores_class_specific,per)
                   # self.threshold = np.percentile(race_scores_class_specific,100*(1-sampling_rate))

                    
                   # pdb.set_trace()
                   # accp_inds = np.where(race_scores_class_specific>self.threshold)[0]
                   # accp_inds = np.where(race_scores.detach().numpy()[:,0]>-0.01)[0]
                   # accp_inds = np.random.choice(1000,600) 
                    accp_inds, weights = self.weighted_sampling(race_scores_class_specific)

                    diff_batchsize = self.batch_size-curr_batchsize
                    if diff_batchsize < len(accp_inds):
                        accp_len = diff_batchsize
                    else:
                        accp_len = len(accp_inds)

                    xs.append(xb_interim[accp_inds[:accp_len]])
                    ys.append(yb_interim[accp_inds[:accp_len]])
                    ws.append(weights[:accp_len])
                    # update current batchsize
                    curr_batchsize+=accp_len
                    if accp_len < len(accp_inds): 
                        self.x_leftovers = xb_interim[accp_inds[accp_len:]]
                        self.y_leftovers = yb_interim[accp_inds[accp_len:]]
                        self.w_leftovers = weights[accp_len:]
                    else:
                        self.x_leftovers = np.array(())
                        self.y_leftovers = np.array(())
                        self.w_leftovers = np.array(())


                    # update race sketch
                    if self.iter_idx % self.race_update_freq == 0:
                        print('****update race sketch*****')
                        self.race_update(xb_interim, yb_interim)

    #             pdb.set_trace()
                self.iter_idx+=1 # iter over filtered data
                X = np.concatenate(xs,axis=0)
                y = np.concatenate(ys,axis=0)
                wght = np.concatenate(ws,axis=0)
        else: # test and valid data
            X,y = self.dataset[self.current_idx:min(self.current_idx+self.batch_size, self.total_size)]
            self.current_idx = self.current_idx + self.batch_size


        if self.dataset.regression:
            return X.shape[0], self.current_idx, (torch.FloatTensor(X), torch.FloatTensor(y), torch.FloatTensor(wght))
        else:
            return X.shape[0], self.current_idx, (torch.FloatTensor(X), torch.LongTensor(y), torch.FloatTensor(wght))



    def weighted_sampling(self, scores):
       #pdb.set_trace()
       inds_accept_by_score = np.where(scores>self.threshold)[0] # pick samples with estimated loss of greater than a threshold
       weight_accept_by_score = np.ones_like(inds_accept_by_score).astype(float)

       num_accept_by_chance = int(np.ceil((scores.shape[0]-inds_accept_by_score.shape[0])*self.accept_prob))
       inds_accept_by_chance = np.random.choice(np.where(scores<=self.threshold)[0],num_accept_by_chance, replace=False)
       weight_accept_by_chance = np.ones_like(inds_accept_by_chance)*(1/self.accept_prob)
       
       weights = np.concatenate((weight_accept_by_score,weight_accept_by_chance))
       accepted_inds = np.concatenate((inds_accept_by_score,inds_accept_by_chance))

       return accepted_inds, weights
