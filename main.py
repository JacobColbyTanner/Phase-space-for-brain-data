
from scipy.io import loadmat, savemat
import numpy as np
import model as net
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from scipy.io import loadmat
import CTRNN

# Hyperparameters

batch_size = 25
sequence_length = 5
num_iter = 10
learning_rate = 0.000001
epochs = 1000
input_size = 10
hidden_size = 200

#Add model
dt = 100
net = CTRNN.CTRNN(input_size, hidden_size, dt=dt)



# open data
data = loadmat('/Users/jacobtanner/Brain networks lab Dropbox/Jacob Tanner/jacobcolbytanner/schaefer200_HCP7t_movie_rest_struct.mat')
brain_data = data['HCP_7t_movie_rest']



def get_batch(brain_data, sequence_length, batch_size,hidden_size):
  

    subject = np.random.randint(0,high=129)
    scan = np.random.randint(0,high=3)
    
    TT = brain_data[0,subject]
    ts = torch.from_numpy(TT['rest'][0,scan][0])  #time by nodes
    


    inputs = torch.zeros(sequence_length,batch_size,hidden_size)
    
    for i in range(batch_size):
        start = np.random.randint(0,high= ts.shape[0]-sequence_length)
        stop = start+sequence_length
        inputs[:,i,:] = ts[start:stop,:]
        

    return inputs






# Loss function
criterion = nn.MSELoss()  

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    total_loss = 0
    start_time = time.time()
    r = []
    for j in range(num_iter):
        #create batches
        real_hidden = get_batch(brain_data, sequence_length, batch_size,ntokens,out_size)
        
        #network inputs and starting state
        hidden_starting_states = torch.from_numpy(real_hidden[0,:,:].squeeze())
        inputs = torch.randn(sequence_length,batch_size,input_size)
        # Forward pass
        outputs, hidden = net(inputs,hidden_starting_states)
     
        loss = criterion(outputs, real_hidden)

        O = outputs.detach().numpy().squeeze()
        T = targets.detach().numpy().squeeze()

        corr = np.corrcoef(O,T)
        r.append(corr[0,1])
        # Backward pass and optimization
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update weights

        total_loss += loss.item()
    #plt.plot(outputs[-1,0,:].detach().numpy())
    #plt.plot(targets[-1,0,:].detach().numpy())
    #plt.show()

    

    end_time = time.time()-start_time
    # Print average loss for the epoch
    print("Epoch: ", epoch," Loss: ", total_loss/num_iter,"r: ",np.nanmean(r), "time: ", end_time)


torch.save(model.state_dict(), 'models/CTRNN_fMRI.pth')
