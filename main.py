
from scipy.io import loadmat, savemat
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from scipy.io import loadmat
import CTRNN
from sklearn.preprocessing import MinMaxScaler

# Hyperparameters
supercomputer = True
batch_size = 75
sequence_length = 50
num_iter = 25
learning_rate = 0.00001
epochs = 1000
num_nodes = 200 #number of nodes in the network
neurons_per_node = 15
hidden_size = num_nodes*neurons_per_node

#Add model
dt = 100
#net = CTRNN.CTRNN(input_size, hidden_size, dt=dt)

net = CTRNN.RNNNet(input_size=num_nodes, hidden_size=hidden_size,
             output_size=num_nodes, dt=dt)

# open data
if supercomputer == True:
    data = loadmat('/N/project/networkRNNs/schaefer200_HCP7t_movie_rest_struct.mat')
else:
    data = loadmat('/Users/jacobtanner/Brain networks lab Dropbox/Jacob Tanner/jacobcolbytanner/schaefer200_HCP7t_movie_rest_struct.mat')
brain_data = data['HCP_7t_movie_rest']



def get_batch(brain_data, sequence_length, batch_size,num_nodes):
  
    # Create the scaler with the desired range
    scaler = MinMaxScaler(feature_range=(-1, 1))
    inputs = torch.zeros(sequence_length,batch_size,num_nodes)
    
    for i in range(batch_size):
        subject = np.random.randint(0,high=129)
        scan = np.random.randint(0,high=3)
        
        TT = brain_data[0,subject]
        ts_unscaled = TT['rest'][0,scan][0]  #time by nodes
        ts = torch.from_numpy(scaler.fit_transform(ts_unscaled))

        start = np.random.randint(0,high= ts.shape[0]-sequence_length)
        stop = start+sequence_length
        inputs[:,i,:] = ts[start:stop,:]
        

    return inputs






# Loss function
criterion = nn.MSELoss()  

# Optimizer
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
print("training started...")
# Training Loop
for epoch in range(epochs):
    print("epoch")
    print(epoch)
    total_loss = 0
    start_time = time.time()
    r = []
    for j in range(num_iter):
        #create batches
        real_hidden = get_batch(brain_data, sequence_length, batch_size, num_nodes)
        
        #network inputs and starting state
        hidden_starting_states = real_hidden[0,:,:].squeeze()
        
        #inputs = torch.randn(sequence_length,batch_size,input_size)
        inputs = torch.zeros(sequence_length,batch_size,num_nodes)
        inputs[0,:,:] = hidden_starting_states
        # Forward pass
        outputs, hidden = net(inputs)
     
        loss = criterion(outputs, real_hidden)

        O = outputs.detach().numpy().flatten()
        T = real_hidden.detach().numpy().flatten()
  

        corr = np.corrcoef(O,T)
        r.append(corr[0,1])
        # Backward pass and optimization
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update weights

        total_loss += loss.item()


    end_time = time.time()-start_time
    # Print average loss for the epoch
    print("Epoch: ", epoch," Loss: ", total_loss/num_iter,"r: ",np.nanmean(r), "time: ", end_time, flush=True)

if supercomputer == True:
    torch.save(net.state_dict(), '/N/project/networkRNNs/CTRNN_fMRI_no_gaus.pth')
else:
    torch.save(net.state_dict(), 'models/CTRNN_fMRI_no_gaus.pth')
