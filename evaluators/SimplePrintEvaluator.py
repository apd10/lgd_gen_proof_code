import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import accuracy_score,roc_auc_score
import pdb

class SimplePrintEvaluator:
    def __init__(self, params, datas, device_id):
      self.datas = datas
      self.device_id = device_id
      self.eval_itr = params["eval_itr"]
      self.eval_epoch = params["eval_epoch"]
      self.skip_0 = False
      if "skip_0" in params:
          self.skip_0  = params["skip_0"]
      self.csv_dump = None
      if "csv_dump" in params:
          self.csv_dump = params["csv_dump"]
      self.header_written = False

    def evaluate(self, epoch, loc_itr, iteration, model, loss_func, metrics=None, binary=False, regression=False, split_label=False): # also logs
        if self.skip_0 :
            if epoch == 0 and loc_itr == 0:
                return
        if iteration % self.eval_itr == 0 or (epoch % self.eval_epoch == 0 and loc_itr == 0):
            #self._eval(epoch, loc_itr, iteration, model, loss_func, self.train_data, "TRAIN")
            for key in self.datas.keys():
                self._eval(epoch, loc_itr, iteration, model, loss_func, self.datas[key], key, metrics, binary, regression, split_label)

    def _eval(self, epoch, loc_itr, iteration, model, loss_func, data, key, metrics, binary, regression, split_label):
        count = 0
        total_loss = 0.
        model.eval()
        correct = 0.0
        y_true = np.array([])
        y_pred = np.array([])
        y_score = np.array([])
        
        with torch.no_grad():
            data.reset()
            while not data.end():
                x, y = data.next()
                x = Variable(x).cuda(self.device_id) if self.device_id!=-1 else Variable(x)
                y = Variable(y).cuda(self.device_id) if self.device_id!=-1 else Variable(y)
                output = model(x)

                y_true = np.concatenate([y_true, np.array(y.detach().cpu())])
                if not regression:
                    if binary:
                        y_score = np.concatenate([y_score, np.array(output.detach().cpu()).reshape(-1)])
                        pred = (output > 0.5).view(-1).float()
                    else:
                        pred = output.data.max(1, keepdim=True)[1].view(-1)
    
                    y_pred = np.concatenate([y_pred, np.array(pred.cpu())])
                    correct += torch.sum(pred == y)
                if binary or regression:
                    loss = loss_func(output.view(-1), y.float(), size_average=False).item()
                else:
                    loss = loss_func(output, y, size_average=False).item()
                total_loss += loss
                count += 1
        valid_loss = total_loss / data.len()
        acc = correct / data.len()
        aux_print_str = ""
        aux_header = ""
        aux_data = ""

        accs = []
        for key in np.unique(y_true):
            yp1 = y_pred[y_true == key]
            yt1 = y_true[y_true == key]
            ct = np.sum(yp1 == yt1)
            acc1 = ct / len(yp1)
            accs.append((key,acc1))
            
        for metric in metrics:
            if split_label:
                for key in np.unique(y_true):
                    yp1 = y_pred[y_true == key]
                    yt1 = y_true[y_true == key]
                    
                    if metric == 'acc':
                          met_acc = accuracy_score(y_true, y_pred)
                          aux_print_str += " {}-{}: {:.4f}".format("ACC", key, met_acc)
                          aux_header += ",ACC"
                          aux_data += "," + str(met_acc)
                    else:
                          print("metric: ", metric, "undefined. add here")
                          aux_print_str += "{}: {}".format(metric, "[undefined-see-SimplePrintEvaluator]")
            else:
                if metric == 'acc':
                      met_acc = accuracy_score(y_true, y_pred)
                      aux_print_str += " {}: {:.4f}".format("ACC", met_acc)
                      aux_header += ",ACC"
                      aux_data += "," + str(met_acc)
                elif metric == 'auc':
                      if binary:
                          met_auc = roc_auc_score(y_true, y_score)
                          aux_print_str += " {}: {:.4f}".format("AUC", met_auc)
                          aux_header += ",AUC"
                          aux_data += "," + str(met_auc)
                else:
                      print("metric: ", metric, "undefined. add here")
                      aux_print_str += "{}: {}".format(metric, "[undefined-see-SimplePrintEvaluator]")
                
        if split_label:
            print('{} : Epoch : {} Loc_itr: {} Iteration: {} Loss: {:.4f} Accuracy: {:.4f} LabelAcc: {}'.format(key, epoch, loc_itr, iteration, valid_loss, acc, accs) + aux_print_str)
        else:    
            print('{} : Epoch : {} Loc_itr: {} Iteration: {} Loss: {:.4f} Accuracy: {:.4f}'.format(key, epoch, loc_itr, iteration, valid_loss, acc) + aux_print_str)
        if self.csv_dump is not None:
            f = open(self.csv_dump, "a")
            if  not self.header_written:
                self.header_written = True
                f.write('{},{},{},{},{},{}{}\n'.format("key", "epoch", "loc_itr", "iteration", "loss", "acc",aux_header))
            f.write('{},{},{},{},{},{}{}\n'.format(key, epoch, loc_itr, iteration, valid_loss, acc,aux_data))
            f.close()
            
