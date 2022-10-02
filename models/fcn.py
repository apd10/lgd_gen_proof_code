import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb
import numpy as np
from math import sqrt

class BaseModel(nn.Module):
    def __init__(self): 
        super(BaseModel, self).__init__()
        self.initial_values = None
        self.logdata = None

    def logger(self, itr, enabled):
        if not enabled :
            return
        pdb.set_trace()
        if self.initial_values is None:
            self.initial_values = {}
            self.logdata = {}
            self.logdata['iterations'] = []
            for name, value in list(self.named_parameters()):
                self.initial_values[name] = value.data.detach().clone()
                self.logdata[name] = []
        else:
            self.logdata['iterations'].append(itr)
            for name, value in list(self.named_parameters()):
                val = value.data.detach().clone()
                norm = np.float(torch.norm(val - self.initial_values[name])) / np.sqrt(torch.numel(val))
                self.logdata[name].append(norm)

    def get_logged_data(self, enabled):
        return self.logdata
        

class RoastLinear(nn.Module):
    def __init__(self, input, output, compression):
        super(RoastLinear, self).__init__()
        self.idim = input
        self.odim = output
        self.wsize = int(self.idim * self.odim * compression)
        self.weight = nn.Parameter(torch.zeros(self.wsize, dtype=torch.float), requires_grad=True)
        nn.init.uniform_(self.weight.data, a=-1/sqrt(self.idim), b = 1/sqrt(self.idim))
        
        self.IDX = nn.Parameter(torch.randint(0, self.wsize, size=(self.idim, self.odim), dtype=torch.int64), requires_grad=False)
        self.G = nn.Parameter(torch.randint(0, 2, size=(self.idim, self.odim), dtype=torch.float)*2 - 1, requires_grad=False)

        self.bias = nn.Parameter(torch.zeros(self.odim, dtype=torch.float), requires_grad = True)

    def forward(self, x):
        W = torch.mul(self.weight[self.IDX], self.G)
        x = torch.mm(x, W) + self.bias
        return x

    def grad_comp_to_orig(self, grad):
        return torch.mul(grad[self.IDX],self.G)

    def grad_orig_to_comp(self, grad):
        out_grad = torch.zeros(self.wsize, dtype=torch.float, device=grad.device)
        out_grad.scatter_add_(0, self.IDX.view(-1), (torch.mul(grad, self.G)).view(-1))

        count = torch.zeros(self.wsize, dtype=torch.float, device=grad.device)
        count.scatter_add_(0, self.IDX.view(-1), torch.ones_like(self.IDX, device=grad.device, dtype=torch.float).view(-1))
        return torch.div(out_grad, count + 1e-3)

    def wt_comp_to_orig(self, wt):
        return torch.mul(wt[self.IDX],self.G)

    def wt_orig_to_comp(self, wt):
        out_wt = torch.zeros(self.wsize, dtype=torch.float, device=wt.device)
        out_wt.scatter_add_(0, self.IDX.view(-1), (torch.mul(wt, self.G)).view(-1))

        count = torch.zeros(self.wsize, dtype=torch.float, device=wt.device)
        count.scatter_add_(0, self.IDX.view(-1), torch.ones_like(self.IDX, device=wt.device, dtype=torch.float).view(-1)) + 1e-3
        return torch.div(out_wt, count + 1e-3)
       
class ROASTFCN(BaseModel):
    def __init__(self, dimension, num_layers, hidden_size, num_class, compression, seed):
        super(ROASTFCN, self).__init__()
        torch.manual_seed(seed)

        self.first_layer = RoastLinear(dimension, hidden_size, compression)
        self.first_layer_relu = nn.ReLU()
        mid_layers = []
        for i in range(num_layers - 2):
            mid_layers.append(RoastLinear(hidden_size, hidden_size, compression))
            mid_layers.append(nn.ReLU())
        self.mid_layers = nn.Sequential(*mid_layers)

        self.last_layer = RoastLinear(hidden_size, num_class, compression)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.first_layer(x)
        x = self.first_layer_relu(x)
        for layer in self.mid_layers:
            x = layer(x)
        x = self.last_layer(x)
        x = F.log_softmax(x, dim=1)
        return x
        
            
            


class FCN(BaseModel):
    def __init__(self, dimension, num_layers, hidden_size, num_class=2):
        super(FCN, self).__init__()

        self.first_layer = nn.Linear(dimension, hidden_size)
        self.first_layer_relu = nn.ReLU()
        mid_layers = []
        for i in range(num_layers - 2):
            mid_layers.append(nn.Linear(hidden_size, hidden_size))
            mid_layers.append(nn.ReLU())
        self.mid_layers = nn.Sequential(*mid_layers)

        self.last_layer = nn.Linear(hidden_size, num_class)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.first_layer(x)
        x = self.first_layer_relu(x)
        for layer in self.mid_layers:
            x = layer(x)
        x = self.last_layer(x)
        x = F.log_softmax(x, dim=1)
        return x



class FCNSG(BaseModel):
    def __init__(self, dimension, num_layers, hidden_size):
        super(FCNSG, self).__init__()

        self.first_layer = nn.Linear(dimension, hidden_size)
        self.first_layer_relu = nn.ReLU()
        mid_layers = []
        for i in range(num_layers - 2):
            mid_layers.append(nn.Linear(hidden_size, hidden_size))
            mid_layers.append(nn.ReLU())
        self.mid_layers = nn.Sequential(*mid_layers)

        self.last_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.first_layer(x)
        x = self.first_layer_relu(x)
        for layer in self.mid_layers:
            x = layer(x)
        x = self.last_layer(x)
        x = torch.sigmoid(x)
        return x


class PLearn1HSG(BaseModel):
    def __init__(self, dimension, small_hidden_size, num_components):
        super(PLearn1HSG, self).__init__()
        assert(num_components >=1)
        self.num_components  = num_components
        self.active = []
        self.first_layer = []
        self.second_layer = []
        self.relus = []
        for i in range(num_components):
            self.active.append(False)
            self.first_layer.append(nn.Linear(dimension, small_hidden_size))
            self.relus.append(nn.ReLU())
            self.second_layer.append(nn.Linear(small_hidden_size, 1))
        self.first_layer = nn.ModuleList(self.first_layer)
        self.second_layer = nn.ModuleList(self.second_layer)
        self.relus = nn.ModuleList(self.relus)
        self.active[0] = True
    def set_active(self, component):
        ''' set the component active for computation '''
        if component < self.num_components:
            self.active[component] = True

    def unfreeze_all(self):
        for component in range(self.num_components):
            self.first_layer[component].weight.requires_grad = True
            self.first_layer[component].bias.requires_grad = True

            self.second_layer[component].weight.requires_grad = True
            self.second_layer[component].bias.requires_grad = True
            
    def freeze_parms(self, component):
        if component < self.num_components:
            self.first_layer[component].weight.requires_grad = False
            self.first_layer[component].bias.requires_grad = False

            self.second_layer[component].weight.requires_grad = False
            self.second_layer[component].bias.requires_grad = False

    def print_state(self):
        print("State")
        for i in range(self.num_components):
            print("component: ",i, "active: ",self.active[i], "requires_grad: (",
                  self.first_layer[i].weight.requires_grad,self.first_layer[i].bias.requires_grad,
                  self.second_layer[i].weight.requires_grad,self.second_layer[i].bias.requires_grad,")", "weight sum:(",
                  torch.sum(self.first_layer[i].weight).detach().cpu(), torch.sum(self.second_layer[i].weight).detach().cpu(),")",
                  "grad:(", self.first_layer[i].weight.grad.sum().detach().cpu() if (self.first_layer[i].weight.grad is not None) else  None,
                  self.second_layer[i].weight.grad.sum().detach().cpu() if (self.second_layer[i].weight.grad is not None) else  None,")",
                  self.first_layer[i].weight[0,0].detach().cpu(), self.second_layer[i].weight[0,0].detach().cpu())
            

    def forward(self, x):
        first_layer_outputs = []
        for i in range(self.num_components):
            if self.active[i]:
                first_layer_outputs.append(self.relus[i](self.first_layer[i](x)))
            else:
                first_layer_outputs.append(None)
        
        second_layer_outputs = []
        for i in range(self.num_components):
            if self.active[i]:
                second_layer_outputs.append(self.second_layer[i](first_layer_outputs[i]))
        x = torch.stack(second_layer_outputs).sum(dim=0)
        x = torch.sigmoid(x)
        return x


class FCNSGDLN(BaseModel):
    def __init__(self, dimension, num_layers, hidden_size, num_linear_layers, dropout):
        super(FCNSGDLN, self).__init__()
        self.add_dropout = dropout > 0

        layers = []
        for i in range(num_layers - 1 ):
            if i == 0:
                layers.append(nn.Linear(dimension, hidden_size))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
            if self.add_dropout:
                layers.append(nn.Dropout(dropout))

            for j in range(num_linear_layers):
                layers.append(nn.Linear(hidden_size, hidden_size))
                if self.add_dropout and j!=num_linear_layers-1:
                    layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())
            if self.add_dropout:
                layers.append(nn.Dropout(dropout))


        self.initial_layers = nn.Sequential(*layers)
        self.last_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.initial_layers(x)
        x = self.last_layer(x)
        x = torch.sigmoid(x)
        return x
