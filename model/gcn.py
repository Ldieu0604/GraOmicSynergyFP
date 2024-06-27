#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir("../")


# In[2]:


ls


# In[3]:


import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from utils import *
import datetime
import matplotlib.pyplot as plt
import pickle
from tqdm.notebook import tqdm


# In[4]:


data_path = "data"
data_processed_path = "data/processed/"


# #Define model

# In[5]:


from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GCNConv, global_max_pool as gmp

# GINConv model
class GCNConvNet(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2, out_tissue_d=13):

        super(GCNConvNet, self).__init__()

        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        # convolution layers
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.fc_g1 = torch.nn.Linear(num_features_xd*4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # cell line feature
        self.target_cnv_block = Sequential(
            nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8, stride=2),
            nn.MaxPool1d(3),
            nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=8, stride=2),
            nn.MaxPool1d(3),
            nn.Conv1d(in_channels=n_filters, out_channels=n_filters*2, kernel_size=4),
            nn.MaxPool1d(3),
            nn.Conv1d(in_channels=n_filters*2, out_channels=n_filters*2, kernel_size=4),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=n_filters*2, out_channels=n_filters*4, kernel_size=4),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=n_filters*4, out_channels=n_filters*4, kernel_size=2),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=n_filters*4, out_channels=n_filters*2, kernel_size=2),
            nn.MaxPool1d(2), 
        )
        self.fc1_xt = nn.Linear(512, output_dim)

        # combined layers
        self.fc1 = nn.Linear(3*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
#         Batch(c_size_1=[1024], c_size_2=[1024], edge_index_1=[2, 67864], edge_index_2=[2, 68244],
#               target=[1024, 735], x_1=[31406, 78], x_1_batch=[31406], x_2=[31559, 78], x_2_batch=[31559], y=[1024])
        x_1_batch = data.x_1_batch
        x_1, edge_index_1,  = data.x_1, data.edge_index_1
        x_2_batch = data.x_2_batch
        x_2, edge_index_2,  = data.x_2, data.edge_index_2
        
#         drug 1
        x_1 = self.conv1(x_1, edge_index_1)
        x_1 = self.relu(x_1)

        x_1 = self.conv2(x_1, edge_index_1)
        x_1 = self.relu(x_1)

        x_1 = self.conv3(x_1, edge_index_1)
        x_1 = self.relu(x_1)
        x_1 = gmp(x_1, x_1_batch)       # global max pooling

        # flatten
        x_1 = self.relu(self.fc_g1(x_1))
        x_1 = self.dropout(x_1)
        x_1 = self.fc_g2(x_1)
        x_1 = self.dropout(x_1)
#         drug 2
        x_2 = self.conv1(x_2, edge_index_2)
        x_2 = self.relu(x_2)

        x_2 = self.conv2(x_2, edge_index_2)
        x_2 = self.relu(x_2)

        x_2 = self.conv3(x_2, edge_index_2)
        x_2 = self.relu(x_2)
        x_2 = gmp(x_2, x_2_batch)       # global max pooling

        # flatten
        x_2 = self.relu(self.fc_g1(x_2))
        x_2 = self.dropout(x_2)
        x_2 = self.fc_g2(x_2)
        x_2 = self.dropout(x_2)
        
        # protein input feed-forward:
        target = data.target
        target = target[:,None,:]
#         print(target.shape)

        # 1d conv layers
        conv_xt = self.target_cnv_block(target)
#         print(conv_xt.shape)
        # flatten
        xt = conv_xt.view(-1, conv_xt.shape[1] * conv_xt.shape[2])
        xt = self.fc1_xt(xt)
        
        # concat
        xc = torch.cat((x_1, x_2, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        
        return out


# In[6]:


# training function at each epoch
import timeit

def train(model, device, train_loader, optimizer, epoch, log_interval):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    loss_fn = nn.MSELoss()
    avg_loss = []
    loop = tqdm(enumerate(train_loader))
    for batch_idx, data in loop:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())
       
        text = 'Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x_1),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item())
        
        loop.set_description(text)
        loop.refresh() # to show immediately the update
    return sum(avg_loss)/len(avg_loss)


# In[7]:


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()


# In[8]:


def draw(list_, label, y_label, title):
    plt.figure()
    plt.plot(list_, label=label)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.legend()
    # save image
    plt.savefig(title+".png")  # should before show method


# #Load data

# In[9]:


model_st = "GCNConvNet"
dataset = 'GDSC'
train_batch = 512
val_batch = 512
test_batch = 512
log_interval = 20


# In[10]:


print('\nrunning on ', model_st + '_' + dataset )
train_data = TestbedDataset(root=data_path, dataset=dataset+"_"+'train_dc')
val_data = TestbedDataset(root=data_path, dataset=dataset+"_"+'val_dc')
test_data = TestbedDataset(root=data_path, dataset=dataset+"_"+'test_dc')

val_mix_dc = TestbedDataset(root=data_path, dataset=dataset+"_"+'mix_val')
val_blind_cell = TestbedDataset(root=data_path, dataset=dataset+"_"+'blind_cell_val')
val_blind_1_drug = TestbedDataset(root=data_path, dataset=dataset+"_"+'blind_1_drug_val')
val_blind_1_drug_cell = TestbedDataset(root=data_path, dataset=dataset+"_"+'blind_1_drug_cell_val')
val_blind_2_drug = TestbedDataset(root=data_path, dataset=dataset+"_"+'blind_2_drug_val')
val_blind_all = TestbedDataset(root=data_path, dataset=dataset+"_"+'blind_all_val')


test_mix_dc = TestbedDataset(root=data_path, dataset=dataset+"_"+'mix_test')
test_blind_cell = TestbedDataset(root=data_path, dataset=dataset+"_"+'blind_cell_test')
test_blind_1_drug = TestbedDataset(root=data_path, dataset=dataset+"_"+'blind_1_drug_test')
test_blind_1_drug_cell = TestbedDataset(root=data_path, dataset=dataset+"_"+'blind_1_drug_cell_test')
test_blind_2_drug = TestbedDataset(root=data_path, dataset=dataset+"_"+'blind_2_drug_test')
test_blind_all = TestbedDataset(root=data_path, dataset=dataset+"_"+'blind_all_test')


# In[11]:


# make data PyTorch
# mini-batch processing ready
train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=True, follow_batch=['x_1', 'x_2'])
val_loader = DataLoader(val_data, batch_size=val_batch, shuffle=False, follow_batch=['x_1', 'x_2'])
test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False, follow_batch=['x_1', 'x_2'])

val_mix_dc_loader = DataLoader(val_mix_dc, batch_size=test_batch, shuffle=False, follow_batch=['x_1', 'x_2'])
val_blind_cell_loader = DataLoader(val_blind_cell, batch_size=test_batch, shuffle=False, follow_batch=['x_1', 'x_2'])
val_blind_1_drug_loader = DataLoader(val_blind_1_drug, batch_size=test_batch, shuffle=False, follow_batch=['x_1', 'x_2'])
val_blind_1_drug_cell_loader = DataLoader(val_blind_1_drug_cell, batch_size=test_batch, shuffle=False, follow_batch=['x_1', 'x_2'])
val_blind_2_drug_loader = DataLoader(val_blind_2_drug, batch_size=test_batch, shuffle=False, follow_batch=['x_1', 'x_2'])
val_blind_all_loader = DataLoader(val_blind_all, batch_size=test_batch, shuffle=False, follow_batch=['x_1', 'x_2'])

test_mix_dc_loader = DataLoader(test_mix_dc, batch_size=test_batch, shuffle=False, follow_batch=['x_1', 'x_2'])
test_blind_cell_loader = DataLoader(test_blind_cell, batch_size=test_batch, shuffle=False, follow_batch=['x_1', 'x_2'])
test_blind_1_drug_loader = DataLoader(test_blind_1_drug, batch_size=test_batch, shuffle=False, follow_batch=['x_1', 'x_2'])
test_blind_1_drug_cell_loader = DataLoader(test_blind_1_drug_cell, batch_size=test_batch, shuffle=False, follow_batch=['x_1', 'x_2'])
test_blind_2_drug_loader = DataLoader(test_blind_2_drug, batch_size=test_batch, shuffle=False, follow_batch=['x_1', 'x_2'])
test_blind_all_loader = DataLoader(test_blind_all, batch_size=test_batch, shuffle=False, follow_batch=['x_1', 'x_2'])


print("CPU/GPU: ", torch.cuda.is_available())


# #Training

# ##Init model

# In[12]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
modeling = GCNConvNet
model = modeling().to(device)


# ##Define paramters

# In[13]:


lr = 0.001
num_epoch = 300
best_ret_test = None
print('Learning rate: ', lr)
print('Epochs: ', num_epoch)


# In[14]:


train_losses = []
val_losses = []
val_pearsons = []

val_mix_dc_losses = []
val_blind_cell_losses = []
val_blind_1_drug_losses = []
val_blind_1_drug_cell_losses = []
val_blind_2_drug_losses = []
val_blind_all_losses = []

val_mix_dc_pearsons = []
val_blind_cell_pearsons = []
val_blind_1_drug_pearsons = []
val_blind_1_drug_cell_pearsons = []
val_blind_2_drug_pearsons = []
val_blind_all_pearsons = []


# ##Train model

# In[15]:


save_path = "model/save_model/" + "GCN" + "/"

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
best_mse = 1000
best_pearson = 1
best_epoch = -1

best_val_losses = 1000
best_val_mix_dc_losses = 1000
best_val_blind_cell_losses = 1000
best_val_blind_1_drug_losses = 1000
best_val_blind_1_drug_cell_losses = 1000
best_val_blind_2_drug_losses = 1000
best_val_blind_all_losses = 1000


ret_test_save = None
ret_test_mix_dc_save = None 
ret_test_blind_cell_save = None
ret_test_blind_1_drug_save = None
ret_test_blind_1_drug_cell_save = None
ret_test_blind_2_drug_save = None
ret_test_blind_all_save = None


model_file_name = save_path + 'model_' + model_st + '_' + dataset +  '.model'
current_file_model = save_path + 'current_model_' + model_st + '_' + dataset +  '.model'

result_file_name = save_path + 'result_' + model_st + '_' + dataset +  '.csv'

loss_fig_name = save_path + 'model_' + model_st + '_' + dataset + '_loss'
pearson_fig_name = save_path + 'model_' + model_st + '_' + dataset + '_pearson'

loss_fig_name_mix_dc = save_path + 'model_' + model_st + '_' + dataset + '_loss_mix_dc'
pearson_fig_name_mix_dc = save_path + 'model_' + model_st + '_' + dataset + '_pearson_mix_dc'

loss_fig_name_blind_cell = save_path + 'model_' + model_st + '_' + dataset + '_loss_blind_cell'
pearson_fig_name_blind_cell = save_path + 'model_' + model_st + '_' + dataset + '_pearson_blind_cell'

loss_fig_name_blind_1_drug = save_path + 'model_' + model_st + '_' + dataset + '_loss_blind_1_drug'
pearson_fig_name_blind_1_drug = save_path + 'model_' + model_st + '_' + dataset + '_pearson_blind_1_drug'

loss_fig_name_blind_1_drug_cell = save_path + 'model_' + model_st + '_' + dataset + '_loss_blind_1_drug_cell'
pearson_fig_name_blind_1_drug_cell = save_path + 'model_' + model_st + '_' + dataset + '_pearson_blind_1_drug_cell'

loss_fig_name_blind_2_drug = save_path + 'model_' + model_st + '_' + dataset + '_loss_blind_2_drug'
pearson_fig_name_blind_2_drug = save_path + 'model_' + model_st + '_' + dataset + '_pearson_blind_2_drug'

loss_fig_name_blind_all = save_path + 'model_' + model_st + '_' + dataset + '_loss_blind_all'
pearson_fig_name_blind_all = save_path + 'model_' + model_st + '_' + dataset + '_pearson_blind_all'


# In[16]:


val_mix_dc_losses = []
val_blind_cell_losses = []
val_blind_1_drug_losses = []
val_blind_1_drug_cell_losses = []
val_blind_2_drug_losses = []
val_blind_all_losses = []

val_mix_dc_pearsons = []
val_blind_cell_pearsons = []
val_blind_1_drug_pearsons = []
val_blind_1_drug_cell_pearsons = []
val_blind_2_drug_pearsons = []
val_blind_all_pearsons = []


# 
# #Continue Training

# In[17]:


#load model
if os.path.exists(current_file_model):
  model.load_state_dict(torch.load(current_file_model))
  with open(save_path+ "log.pickle", "rb") as f:
    log = pickle.load(f)
  train_losses, val_losses,\
  val_mix_dc_losses, val_blind_cell_losses,\
  val_blind_1_drug_losses, val_blind_1_drug_cell_losses,\
  val_blind_2_drug_losses, val_blind_all_losses,\
  val_mix_dc_pearsons, val_blind_cell_pearsons,\
  val_blind_1_drug_pearsons, val_blind_1_drug_cell_pearsons,\
  val_blind_2_drug_pearsons, val_blind_all_pearsons = log
  best_mse = min(val_losses)
  best_pearson = 1
  best_epoch = -1
  print("load previous model")


# In[18]:


def return_ret(G, P):
    return [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P)]


# In[19]:


from IPython.display import clear_output 
model.train()
loss_fn = nn.MSELoss()
for epoch in range(num_epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    avg_loss = []    
    train_loss = train(model, device, train_loader, optimizer, epoch+1, log_interval)
    #VAL:
    G,P = predicting(model, device, val_loader)
    G_mix_dc, P_mix_dc = predicting(model, device, val_mix_dc_loader)
    G_blind_cell, P_blind_cell = predicting(model, device, val_blind_cell_loader)
    G_blind_1_drug, P_blind_1_drug = predicting(model, device, val_blind_1_drug_loader)
    G_blind_1_drug_cell, P_blind_1_drug_cell = predicting(model, device, val_blind_1_drug_cell_loader)
    G_blind_2_drug, P_blind_2_drug = predicting(model, device, val_blind_2_drug_loader)
    G_blind_all, P_blind_all = predicting(model, device, val_blind_all_loader)
    

    ret = return_ret(G, P)
    ret_mix_dc = return_ret(G_mix_dc, P_mix_dc)
    ret_blind_cell = return_ret(G_blind_cell, P_blind_cell)
    ret_blind_1_drug = return_ret(G_blind_1_drug, P_blind_1_drug)
    ret_blind_1_drug_cell = return_ret(G_blind_1_drug_cell, P_blind_1_drug_cell)
    ret_blind_2_drug = return_ret(G_blind_2_drug, P_blind_2_drug)
    ret_blind_all = return_ret(G_blind_all, P_blind_all)

    


    train_losses.append(train_loss)
    val_losses.append(ret[1])
    val_pearsons.append(ret[2])

    val_mix_dc_losses.append(ret_mix_dc[1])
    val_blind_cell_losses.append(ret_blind_cell[1])
    val_blind_1_drug_losses.append(ret_blind_1_drug[1])
    val_blind_1_drug_cell_losses.append(ret_blind_1_drug_cell[1])
    val_blind_2_drug_losses.append(ret_blind_2_drug[1])
    val_blind_all_losses.append(ret_blind_all[1])
    

    val_mix_dc_pearsons.append(ret_mix_dc[2])
    val_blind_cell_pearsons.append(ret_blind_cell[2])
    val_blind_1_drug_pearsons.append(ret_blind_1_drug[2])
    val_blind_1_drug_cell_pearsons.append(ret_blind_1_drug_cell[2])
    val_blind_2_drug_pearsons.append(ret_blind_2_drug[2])
    val_blind_all_pearsons.append(ret_blind_all[2])
    
    if ret[1]<best_val_losses:
        best_val_losses = ret[1]
        G_test,P_test = predicting(model, device, test_loader)
        ret_test_save = return_ret(G_test, P_test)
        print("RMSE all test improved")
    if ret_mix_dc[1]<best_val_mix_dc_losses:
        best_val_mix_dc_losses = ret_mix_dc[1]
        G_mix_dc_test, P_mix_dc_test = predicting(model, device, test_mix_dc_loader)
        ret_test_mix_dc_save = return_ret(G_mix_dc_test, P_mix_dc_test)
        print("RMSE mix test improved")
    if ret_blind_cell[1]<best_val_blind_cell_losses:
        best_val_blind_cell_losses = ret_blind_cell[1]
        G_blind_cell_test, P_blind_cell_test = predicting(model, device, test_blind_cell_loader)
        ret_test_blind_cell_save = return_ret(G_blind_cell_test, P_blind_cell_test)
        print("RMSE blind cell test improved")
    if ret_blind_1_drug[1]<best_val_blind_1_drug_losses:
        best_val_blind_1_drug_losses = ret_blind_1_drug[1]
        G_blind_1_drug_test, P_blind_1_drug_test = predicting(model, device, test_blind_1_drug_loader)
        ret_test_blind_1_drug_save = return_ret(G_blind_1_drug_test, P_blind_1_drug_test)
        print("RMSE blind 1 drug improved")
    if ret_blind_1_drug_cell[1]<best_val_blind_1_drug_cell_losses:
        best_val_blind_1_drug_cell_losses = ret_blind_1_drug_cell[1]
        G_blind_1_drug_cell_test, P_blind_1_drug_cell_test = predicting(model, device, test_blind_1_drug_cell_loader)
        ret_test_blind_1_drug_cell_save = return_ret(G_blind_1_drug_cell_test, P_blind_1_drug_cell_test)
        print("RMSE blind 1 drug cell improved")
    if ret_blind_2_drug[1]<best_val_blind_2_drug_losses:
        best_val_blind_2_drug_losses = ret_blind_2_drug[1]
        G_blind_2_drug_test, P_blind_2_drug_test = predicting(model, device, test_blind_2_drug_loader)
        ret_test_blind_2_drug_save = return_ret(G_blind_2_drug_test, P_blind_2_drug_test)
        print("RMSE blind 2 drug improved")
    if ret_blind_all[1]<best_val_blind_all_losses:
        best_val_blind_all_losses = ret_blind_all[1]
        G_blind_all_test, P_blind_all_test = predicting(model, device, test_blind_all_loader)
        ret_test_blind_all_save = return_ret(G_blind_all_test, P_blind_all_test)
        print("RMSE blind all improved")
        
    torch.save(model.state_dict(), model_file_name)
    with open(result_file_name,'w') as f:
        f.write('ret_test: '+','.join(map(str,ret_test_save)) +"\n")
        f.write('ret_mix_dc: '+','.join(map(str,ret_test_mix_dc_save)) +"\n")
        f.write('ret_blind_cell: '+','.join(map(str,ret_test_blind_cell_save)) +"\n")
        f.write('ret_blind_1_drug: '+','.join(map(str,ret_test_blind_1_drug_save)) +"\n")
        f.write('ret_blind_1_drug_cell: '+','.join(map(str,ret_test_blind_1_drug_cell_save)) +"\n")
        f.write('ret_blind_2_drug: '+','.join(map(str,ret_test_blind_2_drug_save)) +"\n")
        f.write('ret_blind_all: '+','.join(map(str,ret_test_blind_all_save)) +"\n")
        
    draw_loss(train_losses, val_losses, loss_fig_name)
    draw_pearson(val_pearsons, pearson_fig_name)

    draw_loss(train_losses, val_mix_dc_losses, loss_fig_name_mix_dc)
    draw_pearson(val_mix_dc_pearsons, pearson_fig_name_mix_dc)

    draw_loss(train_losses, val_blind_cell_losses, loss_fig_name_blind_cell)
    draw_pearson(val_blind_cell_pearsons, pearson_fig_name_blind_cell)

    draw_loss(train_losses, val_blind_1_drug_losses, loss_fig_name_blind_1_drug)
    draw_pearson(val_blind_1_drug_pearsons, pearson_fig_name_blind_1_drug)
    
    draw_loss(train_losses, val_blind_1_drug_cell_losses, loss_fig_name_blind_1_drug_cell)
    draw_pearson(val_blind_1_drug_cell_pearsons, pearson_fig_name_blind_1_drug_cell)
    
    draw_loss(train_losses, val_blind_2_drug_losses, loss_fig_name_blind_2_drug)
    draw_pearson(val_blind_2_drug_pearsons, pearson_fig_name_blind_2_drug)
    
    draw_loss(train_losses, val_blind_all_losses, loss_fig_name_blind_all)
    draw_pearson(val_blind_all_pearsons, pearson_fig_name_blind_all)

    torch.save(model.state_dict(), current_file_model)
    log = [
          train_losses, val_losses,\
          val_mix_dc_losses, val_blind_cell_losses,\
          val_blind_1_drug_losses, val_blind_1_drug_cell_losses,\
          val_blind_2_drug_losses, val_blind_all_losses,\
          val_mix_dc_pearsons, val_blind_cell_pearsons,\
          val_blind_1_drug_pearsons, val_blind_1_drug_cell_pearsons,\
          val_blind_2_drug_pearsons, val_blind_all_pearsons
          ]

    with open(save_path+ "/log.pickle", "wb") as f:
      pickle.dump(log, f)


# In[ ]:




