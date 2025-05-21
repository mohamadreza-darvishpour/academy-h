

'''
in complicated env we use nerual network in order to model 
different types of non-linear and complicated states.
in last trains we handle update parameter by our serlves 
but 
when when we use neural network the graph model will very complex and 
calculating the gradian for any weight will complex too caused backpropagation rules and chain rules.  
so we use the library to auto calculate gradian in neural netweork
and we just specify the objective function and use auto calculate weight with libs

one lib : torch 

'''

from torch import nn 
import numpy as np 
import torch 
from matplotlib import pyplot as plt 


# define tensor variables 
# torch tensors are like numpy arrays could use in torch and 
#do backpropagination



#  torch.Size([6, 1])
state = torch.tensor([
    [-1.0] , 
    [0.0] ,
    [1.0] ,
    [2.0] ,
    [3.0] ,
    [4.0] ,
                ],dtype=torch.float)


#  torch.Size([6, 1])
q_target = torch.tensor([
    [-3] , 
    [-1.5] ,
    [2.0] ,
    [3] ,
    [4] ,
    [7] ,
                ],dtype=torch.float)



print('\n > ', state.shape , q_target.shape)


'''
we want to give a state to a model 
and 
the model shold regres the state to predict the 'q' values 
that are close to 'target' value.
here the train problem is 1 dim 
the dim of featrues is 1 so we use linear 
'''


#make a simple linear layer with 'nn'
#layers : l1,l2 ,...
#in_feature input features dim
#out_feature output features dim
# add an amout to w_T*x why ? maybe for bias? 
l1 = nn.Linear(in_features=1 , out_features=1 , bias=True)
l2 = nn.Linear(in_features=1 , out_features=1 , bias=True)

#activation function
relu = nn.ReLU()

# how to make sequatial of made layers?
model = nn.Sequential(l1,relu , l2)

#optimization need loss function
loss_fn = nn.MSELoss()

#optimizer 
#optimizer make last gradient zero and 
optimizer = torch.optim.SGD(model.parameters() , lr=0.01)

# q_pred = model(state)

for i in range(100):
    q_pred = model(state)
    loss_val = loss_fn(q_pred , q_target)

    optimizer.zero_grad()   #make last gradient zero
    loss_val.backward()     #then with backwarrd calculate the gradient with given loss 
    optimizer.step()        #to enable changes

    #see the loss val value 
    print('\nloss : ' , loss_val.item())





plt.plot(state.detach().numpy() , q_pred.detach().numpy() , color='red' )
plt.plot(state.detach().numpy() , q_target.detach().numpy() , color='blue' )
plt.show()




