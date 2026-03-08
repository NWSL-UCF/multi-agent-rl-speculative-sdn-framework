#!/usr/bin/env python
# coding: utf-8

import argparse
import wandb
import numpy as np
import pandas as pd
import random
import statistics
import csv
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from collections import deque
import sys
import os
import wandb



parser = argparse.ArgumentParser()
parser.add_argument('--numberofFlowsPerAgent', type=int)
parser.add_argument('--tablesize', type=int)
parser.add_argument('--LFUTimeInterval', type=int)
parser.add_argument('--agingfactor', type=float)
parser.add_argument('--speculativesdn', type=str)
parser.add_argument('--seed', type=int)
parser.add_argument('--epsilon', type=float)
parser.add_argument('--gamma', type=float)
parser.add_argument('--target_replace_iter', type=int)
parser.add_argument('--memory_capacity', type=int)
parser.add_argument('--LR', type=float)
parser.add_argument('--rewardAgingFactor', type=float)
parser.add_argument('--spatialReward', type=float)
parser.add_argument('--sdn', type=str)
parser.add_argument('--dataset', type=int)
parser.add_argument('--trace', type=int)
parser.add_argument('--NN', type=int)
parser.add_argument('--LTI', type=float)
parser.add_argument('--RTI', type=float)

#os.environ['WANDB_API_KEY'] = "enter your key"
#wandb.login()
#wandb.init(project="sdn_", entity="nwsl", group="sdn", job_type="speculativeSDN_")
#wandb.config = {"numberofFlowsPerAgent":1 ,"tablesize":1 ,"LFUTimeInterval":1 ,"agingfactor":1 ,"speculativesdn":"1" ,"seed": 1, "epsilon":1, "gamma":1, "target_replace_iter":1, "memory_capacity":1, "LR": 1, "rewardAgingFactor": 1, "spatialReward": 1, "sdn": "1", "dataset": 1, "trace": 1, "NN": 1, "LTI": 1, "RTI": 1}
args = parser.parse_args()
#wandb.config=args
#wandb.log({"numberofFlowsPerAgent":wandb.config.numberofFlowsPerAgent, "tablesize": wandb.config.tablesize, "LFUTimeInterval": wandb.config.LFUTimeInterval, "agingfactor": wandb.config.agingfactor, "speculativesdn": wandb.config.speculativesdn, "seed": wandb.config.seed, "epsilon": wandb.config.epsilon, "gamma": wandb.config.gamma, "target_replace_iter": wandb.config.target_replace_iter, "memory_capacity": wandb.config.memory_capacity, "LR": wandb.config.LR, "rewardAgingFactor": wandb.config.rewardAgingFactor, "spatialReward": wandb.config.spatialReward, "sdn": wandb.config.sdn, "dataset": wandb.config.dataset, "trace": wandb.config.trace, "NN": wandb.config.NN, "LTI": wandb.config.LTI, "RTI": wandb.config.RTI})


debug=True
source=False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# store the trace file
if args.dataset==1:
    if args.trace==1:
        dataset = pd.read_csv('pcap.csv')
    if args.trace==2:
        dataset = pd.read_csv('pcap_.csv')
    if args.trace==3:
        dataset = pd.read_csv('pcapfile_.csv')

if args.dataset==2:
    if args.trace==1:
        dataset = pd.read_csv('dataset.csv')
    if args.trace==2:
        dataset = pd.read_csv('dataset_.csv')
    if args.trace==3:
        dataset = pd.read_csv('dataset__.csv')

#remove columns
rem = ['No.','Time','Protocol','Length','Info'] 
remove= ['No.','Source','Destination','Protocol','Length','Info']

# remove the columns and get the distinct pairs
value= dataset.copy()
value.drop(remove, axis=1, inplace=True)


dataset.drop(rem, axis=1, inplace=True)


table= dataset.drop_duplicates()





# create controller table
controller= table.copy()


controller['hit']=0
controller['miss']=0



if args.sdn=="trace":
    pass
if args.sdn=="source":
    controller= controller.sort_values(by='Source', ascending=True)
if args.sdn=="destination":
    controller= controller.sort_values(by='Destination', ascending=True)
column = {'Source':[0],'Destination':[0]}
datasetcopy=pd.DataFrame(data=column)
datasetcopy=datasetcopy.iloc[1:,:]
dataset_=pd.DataFrame(data=dataset)
remove_= ['Source','Destination']

newdataset = dataset_.groupby(['Source','Destination'], sort=False).ngroup()


newdataset_=pd.DataFrame(index=newdataset, data=column)

newdataset_.drop(remove_, axis=1, inplace=True)


r, c = controller.shape

array = controller.values

X = controller.iloc[:, 0:(c - 2)]
X.insert(0, 'No.', value=np.arange(len(X)))
remove_= ['Source','Destination']
X.drop(remove_, axis=1, inplace=True)
N_flows = X.shape[0] # get the number of flow rules





# Parameters for the DQN algorithm
BATCH_SIZE = 32 # the number of data samples propagated through the network before the parameters are updated
LR = float(args.LR/10) # how much to update models parameters at each batch/epoch. Smaller values yield slow learning speed, while large values may result in unpredictable behavior during training.
EPSILON = float(args.epsilon/10) # greedy policy
GAMMA = float(args.gamma/10) # reward discount
TARGET_REPLACE_ITER = args.target_replace_iter # After how much time you refresh target network
MEMORY_CAPACITY = args.memory_capacity # The size of experience replay buffer


numberofFlowsPerAgent= args.numberofFlowsPerAgent
N_ACTIONS = pow(2,numberofFlowsPerAgent)
group=True
N_STATES = math.ceil(N_flows/numberofFlowsPerAgent)


# class for neural network
class Net(nn.Module):
    # We define our neural network by subclassing nn.Module, and initialize the neural network layers in __init__. Every nn.Module subclass implements the operations on input data in the forward method.
    def __init__(self,N_STATES,N_ACTIONS):
        super(Net,self).__init__()
        # The linear layer is a module that applies a linear transformation on the input using its stored weights and biases.
        self.fc1 = nn.Linear(N_STATES,10)# math.ceil((N_STATES+N_ACTIONS)/2)) # input the action lost and output 100 nodes
        self.fc1.weight.data.normal_(0,0.1) #initialization, set seed to ensure the same result
        self.out = nn.Linear(10,N_ACTIONS)#math.ceil((N_STATES+N_ACTIONS)/2), N_ACTIONS) # input 100 nodes and output the actions
        self.out.weight.data.normal_(0,0.1) #initialization
        
    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x) # Non-linear activations are what create the complex mappings between the model’s inputs and outputs. They are applied after linear transformations to introduce nonlinearity, helping neural networks learn a wide variety of phenomena.
        action_value = self.out(x)
        return action_value



# DQN algorithm
class DQN(object):

    def __init__(self,N_STATES,N_ACTIONS):
        self.eval_net, self.target_net = Net(N_STATES,N_ACTIONS), Net(N_STATES,N_ACTIONS)
        
        self.learn_step_counter = 0 # for target updating
        self.memory_counter = 0 # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY,N_STATES*2+2)) # intialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(),lr=LR) # Optimization is the process of adjusting model parameters to reduce model error in each training step. Optimization algorithms define how this process is performed. All optimization logic is encapsulated in the optimizer object. We initialize the optimizer by registering the model’s parameters that need to be trained, and passing in the learning rate hyperparameter. Inside the training loop (learn function), optimization happens in three steps.
        self.loss_func = nn.MSELoss() # When presented with some training data, our untrained network is likely not to give the correct answer. Loss function measures the degree of dissimilarity of obtained result to the target value, and it is the loss function that we want to minimize during training. To calculate the loss we make a prediction using the inputs of our given data sample and compare it against the true data label value. Common loss functions include nn.MSELoss (Mean Square Error) for regression tasks. We pass our model’s output logits to nn.CrossEntropyLoss, which will normalize the logits and compute the prediction error.
        
    def choose_action(self,x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON: # greedy
            action_value = self.eval_net.forward(x)

            action = torch.max(action_value,0)[1].data.numpy()

            action = action[0] # return the argmax index
        else: # random
            action = np.random.randint(0,N_ACTIONS)
        return action
 
    def store_transition(self,s,a,r,s_):
            transition = np.hstack((s,[a,r],s_))
            # replace the old memory with new memory
            index = self.memory_counter%MEMORY_CAPACITY # If full, restart from the beginning
            self.memory[index,:] = transition
            self.memory_counter +=1
 
    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter +=1
        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index,:]
        b_s = torch.FloatTensor(b_memory[:,:N_STATES])
        b_a = torch.LongTensor(b_memory[:,N_STATES:N_STATES+1])
        b_r = torch.FloatTensor(b_memory[:,N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:,-N_STATES:])
        # q_val w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1,b_a) # shape (batch, 1)
        q_next = self.target_net(b_s_).detach() # detach from graph, don't backpropagate
        q_target = b_r + GAMMA*q_next.max(1)[0].view(BATCH_SIZE,1) # bellman equation, shape (batch, 1)
        loss = self.loss_func(q_eval,q_target)
        
        self.optimizer.zero_grad() #  to reset the gradients of model parameters. Gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration.
        loss.backward() # optimize weights of parameters in the neural network, we need to compute the derivatives of our loss function with respect to parameters, Backpropagate the prediction loss with a call to loss.backward(). PyTorch deposits the gradients of the loss w.r.t. each parameter.
        self.optimizer.step() # Once we have our gradients, we call optimizer.step() to adjust the parameters by the gradients collected in the backward pass.


# reward function
# calculate the reward for each agent and return these rewards as a list
def cal_reward(X_selected,controllerTable,oldControllerTable,step,w):
 
 reward=np.zeros(N_flows)

 for i in range(len(controller)):

     controllerTable.iloc[i,4] = (float(args.rewardAgingFactor/10)) * controllerTable.iloc[i,4] # reward aging factor


     if oldControllerTable.iloc[i,2] < controllerTable.iloc[i,2]:
            reward[i]+=controllerTable.iloc[i,2]-oldControllerTable.iloc[i,2]


     if oldControllerTable.iloc[i,3] < controllerTable.iloc[i,3]:
           reward[i]+=controllerTable.iloc[i,3]-oldControllerTable.iloc[i,3]


 for i in range(len(switch)):
        flow=switch.iloc[i,0]
        flow_=switch.iloc[i,1]


        controllernumber=0
        for j in range(len(controller)):
         if flow==controller.iloc[j,0] and flow_==controller.iloc[j,1]:
            controllernumber=j
            break
     
        if controllerTable.iloc[controllernumber,5]!=0:
            if oldControllerTable.iloc[controllernumber,5]<controllerTable.iloc[controllernumber,5]:
                reward_=controllerTable.iloc[controllernumber,2]+controllerTable.iloc[controllernumber,3] 


                spatialReward = reward_

                j=0
                while j in range(0,math.ceil(w/2)):
                    if ((controllernumber+j) < len(controllerTable)):
                        reward[controllernumber+j]+=spatialReward


                    if ((number-j) >= 0):
                        reward[number-j]+=spatialReward


                    j+=1
                    spatialReward *= (float(args.spatialReward/10)) # spatial reward factor


 for i in range(len(controller)):
           controllerTable.iloc[i,4]+=reward[i]

 return reward



class Queue:
  def __init__(self,size):
    self.queue = [None]*size
    self.front = 0
    self.rear = 0
    self.size = size
    self.available = size
 
  def enqueue(self, item):
    if self.available == 0:
      print('Queue Overflow!')
    else:
      self.queue[self.rear] = item
      self.rear = (self.rear + 1) % self.size
      self.available -= 1
 
  def dequeue(self):
    if self.available == self.size:
      print('Queue Underflow!')
    else:
      self.queue[self.front] = None
      self.front = (self.front + 1) % self.size
      self.available += 1
 
  def peek(self):
    print(self.queue[self.front])
 
  def print_queue(self):
    print(self.queue)
 



action_list = np.random.randint(N_ACTIONS,
                                size=math.ceil(
                                    N_flows/numberofFlowsPerAgent))

agentAction = np.zeros(N_flows, int)
slist=np.zeros(0, int)



for i in range(len(action_list)): 
  tempAction = action_list[i]
  for j in range(numberofFlowsPerAgent):
    if (i*numberofFlowsPerAgent + j == len(agentAction)):
      break
    agentAction[i*numberofFlowsPerAgent + j] = tempAction % 2
    tempAction = tempAction / 2



X_selected = X.iloc[agentAction==1,:]


s= torch.Tensor(action_list)



slist=list(s)


# update controller table with columns
controller.insert(4, 'reward', value=0)
controller.insert(5, 'counter', value=0)
controller.insert(6, 'wasHit', value=0)
controller.insert(7,'spatialReward',value=0)
controller.insert(8,'speculatedflow',value=0)

#intialize switch table
newcolumn = {'Source': [0],'Destination': [0], 'hit': [0], 'miss': [0], 'reward': [0], 'counter': [0], 'wasHit': [0]}

newcolumn_ = {'Source': [0],'Destination': [0], 'age': [1]}

column = {'Source': [0],'Destination': [0]}

column_ = {'No.'}
newrate = pd.DataFrame(data=newcolumn)
newrate = controller.copy()
switch = pd.DataFrame(data=newcolumn_)
switch=switch.iloc[1:,:]
switchcopy=pd.DataFrame(data=newcolumn_)
switchcopy=switchcopy.iloc[1:,:]
switchcopy_=pd.DataFrame(data=newcolumn_)
switchcopy_=switchcopy_.iloc[1:,:]
switchcopy_.insert(3,'switchcopy',value=0)
controllercopy=controller.copy()
controllercopy_=controller.copy()
numberofflows=pd.DataFrame(data=newcolumn)
numberofflows=controller.copy()
numberofflows = numberofflows.drop(columns=['hit', 'miss','reward','counter','wasHit'])
numberofflows.insert(1, 'flow', value=0)
numberofflows_=pd.DataFrame(data=newcolumn)
numberofflows_=controller.copy()
datasetcopy=pd.DataFrame(data=column)
datasetcopy=datasetcopy.iloc[1:,:]


numberofflows_=[]
numberofflowscopy_=[]
new=0

# counter to update the swicth table
rule=0

# counter to stop reading in a reactive way from the file
trace=0

# counter for the least frequently used feature
least=0

# counter for the graphs
draw=0

# counter for the flow rate
rate=0

# variables for plots
plot=[]
plot1=[]
plot2=[]
plot3=[]
plt1=[]
plt2=[]
plt3=[]
plotlist=[]
pltcounter=[]
newpltcounter=[]
dqn_list = []
agentAction = np.zeros(N_flows, int)

# number of agents
for agent in range(math.ceil((N_flows/numberofFlowsPerAgent))):
    dqn_list.append(DQN(N_STATES = N_STATES,N_ACTIONS = N_ACTIONS))


result = [];

# switch table size
tablesize = args.tablesize

# queue for flow rules
queue=deque()
newlist=[]

sdnflag=True

speculationflowcount=[]
graphcounter=0

flaggedflows=[]

speculatedflows=[]

counternew=0

newflow_=[]

flowresult_=[]

speculationcounter=0

count=False

learningTimeInterval=float(args.LTI/1000)

LFUTimeInterval=float(float(args.LFUTimeInterval)*learningTimeInterval)

graphRateInterval=1

w=0

agingfactor=float(args.agingfactor/100)

speculatedflowplot_=[]

speculatedflowcounter_=0

speculative=False
reactive=False

controllercounter=0

controllerlist=[]

if args.speculativesdn=="speculative":
    speculative=True
if args.speculativesdn=="reactive":
    reactive=True
if args.speculativesdn=="speculativereactive":
    speculative=True
    reactive=True
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# start the loop for agent learning
while True:
    
    # switch first copy
    switchcopy=pd.DataFrame(data=column)
    switchcopy=switchcopy.iloc[1:,:]
    switchcopy=switch.copy()


        
        
    breaknew=False

    for k in range(len(switch)):
         for j in range(len(controller)):                
          if controller.iloc[j,0] == switch.iloc[k,0] and controller.iloc[j,1] == switch.iloc[k,1]:
            if controller.iloc[j,6]!=1:
                switch.iloc[k,2]=switch.iloc[k,2]*agingfactor
            else:
                controller.iloc[j,6]=0


    # start the loop to read from the trace file
    # This loop emulates reactive SDN as packets go through the switch
    while True:
          if reactive==True:
              processFlow=False
              queue_=False
              # check if the queue is empty
              if len(queue)==0:
                  processFlow=False
              if len(queue)!=0 and queue_==False:
                        tempqueue=queue[0]


                        queue_=True    
              if len(queue)!=0:
                        if tempqueue.iloc[0,3]<float(value.loc[new]):
                            queue.popleft() # dequeue the first flow rule
                            processFlow=True
                            queue_=False
                           


              step=float(value.loc[new])
              
              if processFlow==True:


                  
                  # check if the switch table is full to remove the least frequently used flow rule
                  if tablesize==len(switch):
                    
                        temp=10000000
                        no=-1
                        breaknew=False
                        for k in range(len(switch)):
                           
                                          flow=switch.iloc[k,0]
                                          flow_=switch.iloc[k,1]
                                          controllernumber=0
                                          for j in range(len(controller)):
                                            if flow==controller.iloc[j,0] and flow_==controller.iloc[j,1]:
                                              controllernumber=j
                                              break
  
                                          if reactive==True and speculative==False:

                                           if temp>controller.iloc[controllernumber,5]:

                                                    temp=controller.iloc[controllernumber,5]
                                                    no=switch.index[k]
                                                    location=switch.loc[no]
                                           if temp==0:

                                                    temp=controller.iloc[controllernumber,5]
                                                    no=switch.index[k]
                                                    location=switch.loc[no]
                                                    breaknew=True
                                                    break


                                          if reactive==True and speculative==True:
                                           if switch.iloc[k,2]<=0.5:
                                            if temp>controller.iloc[controllernumber,5]*switch.iloc[k,2]:

                                                    temp=controller.iloc[controllernumber,5]*switch.iloc[k,2]

                                                    no=switch.index[k]
                                                    location=switch.loc[no]
                                            if temp==0:
                                                    temp=controller.iloc[controllernumber,5]*switch.iloc[k,2]

                                                    no=switch.index[k]
                                                    location=switch.loc[no]
                                                    breaknew=True
                                                    break
                                           else:
                                                 continue


                      
                        if no==-1:
                            pass
                        else:


                            switch=switch.drop([no])
                  
                  if tablesize==len(switch):
                  
                        temp=10000000
                        no=-1
                        breaknew=False
                        for k in range(len(switch)):

                                              flow=switch.iloc[k,0]
                                              flow_=switch.iloc[k,1]
                                              controllernumber=0
                                              for j in range(len(controller)):
                                                if flow==controller.iloc[j,0] and flow_==controller.iloc[j,1]:
                                                  controllernumber=j
                                                  break
                                                  

                                              if temp>controller.iloc[controllernumber,5]:

                                                temp=controller.iloc[controllernumber,5]
                                                no=switch.index[k]
                                                location=switch.loc[no]
                                              if temp==0:

                                                temp=controller.iloc[controllernumber,5]
                                                no=switch.index[k]
                                                location=switch.loc[no]
                                                breaknew=True
                                                break
     
                       
                        if no==-1:
                            pass
                        else:

                            switch=switch.drop([no])

                            
                  tempqueue.iloc[0,2]=1
 # install the flow rule
                  switch=pd.concat([switch, tempqueue], ignore_index=True)
                  switch=switch.drop_duplicates(subset=['Source','Destination'], keep='last')

                  switch.drop(['switchcopy'], axis=1, inplace=True)

              # if the entry is in the selected set, then increment hits
              datasetcopy=pd.concat([datasetcopy, dataset_.loc[[new]]], ignore_index=True)

              found=False
              for j in range(len(switch)):
                 if switch.iloc[j,0] == datasetcopy.iloc[0,0] and switch.iloc[j,1] == datasetcopy.iloc[0,1]:

                   found=True

              if found==True:
                   for j in range(len(controller)):
                    if controller.iloc[j,0] == datasetcopy.iloc[0,0] and controller.iloc[j,1] == datasetcopy.iloc[0,1]:
                      controller.iloc[j,2]+=1
                      controller.iloc[j,6]=1
                      newrate.iloc[j,2]+=1
                      controller.iloc[j,5]+=1
                      rate+=1
                      numberofflows_.append(controller.iloc[j,0])
                      numberofflows_=list(set(numberofflows_))
                      if numberofflows.iloc[j,1]!=1:
                        numberofflows.iloc[j,1]=1
                      controller.iloc[j,8]=1


                                
              # else, increment misses
              else:

                  # if the table at the switch is not full, then add the entry to templist
                    for j in range(len(controller)):
                      if controller.iloc[j,0] == datasetcopy.iloc[0,0] and controller.iloc[j,1] == datasetcopy.iloc[0,1]:
                        controller.iloc[j,3]+=1
                        newrate.iloc[j,3]+=1
                        controller.iloc[j,5]+=1
                        rate+=1
                        if numberofflows.iloc[j,1]!=1:
                          numberofflows.iloc[j,1]=1
                        
                        numberofflows_.append(controller.iloc[j,0])
                        numberofflows_=list(set(numberofflows_))
                        controller.iloc[j,8]=1

                        switchcopy_=pd.concat([switchcopy_, datasetcopy], ignore_index=True)
                        switchcopy_['age']=1
                        switchcopy_['switchcopy']=float(value.loc[new])+float(args.RTI/1000)
                        switchcopy_=switchcopy_[['Source','Destination', 'age', 'switchcopy']]

                        queue.append(switchcopy_)

                        switchcopy_=pd.DataFrame(data=newcolumn_)
                        switchcopy_=switchcopy_.iloc[1:,:]



            # increment for the next packet
              new+=1
              datasetcopy=pd.DataFrame(data=column)
              datasetcopy=datasetcopy.iloc[1:,:]

              if debug==True:

                    step=float(value.loc[new])

          else:

              # if the entry is in the selected set, then increment hits
              datasetcopy=pd.concat([datasetcopy, dataset_.loc[[new]]], ignore_index=True)

       	      found=False
              for j in range(len(switch)):
                 if switch.iloc[j,0] == datasetcopy.iloc[0,0] and switch.iloc[j,1] == datasetcopy.iloc[0,1]:

                   found=True

              if found==True:
                    for j in range(len(controller)):
                      if controller.iloc[j,0] == datasetcopy.iloc[0,0] and controller.iloc[j,1] == datasetcopy.iloc[0,1]:
                        controller.iloc[j,2]+=1
                        controller.iloc[j,6]=1
                        newrate.iloc[j,2]+=1
                        controller.iloc[j,5]+=1
                        rate+=1
                        numberofflows_.append(controller.iloc[j,0])
                        numberofflows_=list(set(numberofflows_))
                        if numberofflows.iloc[j,1]!=1:
                          numberofflows.iloc[j,1]=1
                        controller.iloc[j,8]=1


                   
                              
              # else, increment misses
              else:

                # if the table at the switch is not full, then add the entry to templist
                  for j in range(len(controller)):
                    if controller.iloc[j,0] == datasetcopy.iloc[0,0] and controller.iloc[j,1] == datasetcopy.iloc[0,1]:
                      controller.iloc[j,3]+=1
                      newrate.iloc[j,3]+=1
                      controller.iloc[j,5]+=1
                      rate+=1
                      if numberofflows.iloc[j,1]!=1:
                        numberofflows.iloc[j,1]=1
                      
                      numberofflows_.append(controller.iloc[j,0])
                      numberofflows_=list(set(numberofflows_))
                      controller.iloc[j,8]=1

                      switchcopy_=pd.concat([switchcopy_, datasetcopy], ignore_index=True)
                      switchcopy_['age']=1
                      switchcopy_['switchcopy']=float(value.loc[new])+float(args.RTI/1000)

                      queue.append(switchcopy_)
                      switchcopy_=pd.DataFrame(data=newcolumn_)
                      switchcopy_=switchcopy_.iloc[1:,:]



          # increment for the next packet
              new+=1
              datasetcopy=pd.DataFrame(data=column)
              datasetcopy=datasetcopy.iloc[1:,:]

              if debug==True:

                  step=float(value.loc[new])


#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

          # if more than 0.9 seconds, then reset the counter for least frequently used feature
          if ((value.loc[new] - value.loc[least]) > LFUTimeInterval).bool():
            least=new
            controller['counter']=0

          # store the total number of hits and misses per second for the graph 
          if ((value.loc[new] - value.loc[draw]) > graphRateInterval).bool():
            plt1.append(((newrate['hit'].sum()/(newrate['hit'].sum()+newrate['miss'].sum()))*100,value.loc[new]))
            plt1result=((newrate['hit'].sum()/(newrate['hit'].sum()+newrate['miss'].sum()))*100)
            step=float(value.loc[new])
#            wandb.log({'hitrate': plt1result}, step=new)
#            wandb.log({'trafficrate': rate}, step=new)
            pltcounter.append((speculationcounter,value.loc[new]))
            newpltcounter.append(pltcounter)
#            wandb.log({'speculatedflowcount': speculationcounter}, step=new)
            speculativestep=float(value.loc[new])
#            wandb.log({'speculativeseconds': speculativestep}, step=new)
            speculationcounter=0
            flaggedflows.append((counternew,value.loc[new]))
            speculativestep=float(value.loc[new])
#            wandb.log({'flaggedflows': counternew}, step=new)
            counternew=0
#            flowresult=(numberofflows['flow'].sum())


#            wandb.log({'flow': flowresult}, step=new)
#            numberofflows['flow']=0
#            wandb.log({'seconds': step})#, step=new)
            plt2.append(((newrate['miss'].sum()/(newrate['hit'].sum()+newrate['miss'].sum()))*100,value.loc[new]))
            draw=new
            plot1.append((newrate['hit'].sum()/(newrate['hit'].sum()+newrate['miss'].sum()))*100)
            plot2.append((newrate['miss'].sum()/(newrate['hit'].sum()+newrate['miss'].sum()))*100)
            newrate['hit']=0
            newrate['miss']=0
            plot3.append((rate,value.loc[new]))
            plt3.append(plot3)
            rate=0


            speculatableflow=len(controllerlist)
#            wandb.log({'speculatableflow': speculatableflow}, step=new)
            controllerlist=[]
#            numberofflowslist_=list(set(numberofflows_)-set(numberofflowscopy_))
#            print("new flows")
#            print(numberofflowslist_)
#            print(len(numberofflowslist_))
#            newflow=len(numberofflowslist_)
#            wandb.log({'newflow': newflow}, step=new)
            
          # if we reach 0.3 second, then we stop the loop
          if ((value.loc[new] - value.loc[trace]) > learningTimeInterval).bool():
            newtrace=trace
            trace=new
            numberofflowslist_=list(set(numberofflows_)-set(numberofflowscopy_))


            newflow=len(numberofflowslist_)
            newflow_.append((newflow,value.loc[new]))
#            wandb.log({'newflow': newflow}, step=new)
            flowresult=(numberofflows['flow'].sum())


            flowresult_.append((flowresult,value.loc[new]))
#            wandb.log({'flow': flowresult}, step=new)
            numberofflows['flow']=0
#            flowresult=(numberofflows_['flow'].sum())


#            wandb.log({'newflow': flowresult}, step=new)
#            for i in range(len(controller)):
#             if numberofflows_.iloc[i,1]==1:
#                                 numberofflows_.iloc[i,1]=2
            numberofflowscopy_=numberofflows_.copy()

            break
          if new==len(newdataset):
            break
          if args.dataset==1:
            if ((value.loc[new]) > 200).bool():
                break
          if args.dataset==2:
            if ((value.loc[new]) > 0.1).bool():
                break
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    if speculative==True:

        # initialize the action list
        action_list = np.zeros(math.ceil((N_flows/numberofFlowsPerAgent)))

        # the agents will select the flow rules to include in the switch table
        for agent, dqn in enumerate(dqn_list):
          if group==True:
            action_list[agent] = dqn.choose_action(s)
          else:
            action_list[agent] = dqn.choose_action(s)
        newlist.append(action_list)


        for i in range(len(action_list)): 
          tempAction = action_list[i]
          for j in range(numberofFlowsPerAgent):
            if (i*numberofFlowsPerAgent + j==len(agentAction)):
              break
            agentAction[i*numberofFlowsPerAgent + j] = tempAction % 2
            tempAction = tempAction / 2
        np.set_printoptions(threshold=1000000)

        # selected set of flow rules by the agents
        X_selected = X.iloc[agentAction==1]

        np.set_printoptions(threshold=1000000)


        if group==True:
          s_= torch.FloatTensor(action_list)
          s_list= list(s_)
        else:
          s_= torch.FloatTensor(action_list)
        X_selected.insert(1, 'Source', value=0)
        X_selected.insert(2, 'Destination', value=0)

        for i in range(len(controller)):
            for j in range(len(X_selected)):
                if X_selected.iloc[j,0]==controller.index[i]:
                    X_selected.iloc[j,1]=controller.iloc[i,1]
                    X_selected.iloc[j,2]=controller.iloc[i,2]
        
        counter=0
        for i in range(len(switch)):
            for j in range(len(X_selected)):
             if X_selected.iloc[j,1]==switch.iloc[i,0] and X_selected.iloc[j,2]==switch.iloc[i,1]:
                if switch.iloc[i,2]>0.5:
                    switch.iloc[i,2]+=2
                    counter+=1


        # switch second copy
        switchcopy_=pd.DataFrame(data=column)
        switchcopy_=switchcopy_.iloc[1:,:]
        switchcopy_=switch.copy()
        switchcopynew=pd.DataFrame(data=column)
        switchcopynew=switchcopynew.iloc[1:,:]
        switchcopynew_=len(switchcopy_)
        
        for i in range(switchcopynew_):
                if switchcopy_.iloc[i,2]>0.5:
                    switchcopynew=switchcopynew.append(switchcopy_.iloc[[i]])
       
        # if the selected set is < table size, then copy the table
        # start a loop with the size of the selected flow rules set to install the flows rules
        # and check if the switch table is full to remove the least frequently used flow rule
       
        xselected=pd.DataFrame(data=newcolumn)
        xselected=xselected.iloc[1:,:]
        xselected['Source']=X_selected['Source']
        xselected['Destination']=X_selected['Destination']
        xselected.update(controller)
        xselected = xselected.sort_values(by='reward', ascending=True)
        xselectednew=len(xselected)
        xselect = xselected.copy()
        xselect = xselect.sort_values(by='reward', ascending=True)
        xselect = xselect.drop(columns=['hit','miss','reward','counter','wasHit'])
        xselect.insert(2, 'age', value=3.5)
        counternew_=0
        for k in range(len(switch)):
                if switch.iloc[k,2]>0.5:
                    counternew+=1
                    counternew_+=1


        new_=len(xselect)
        for i in range(new_):

          if len(xselect)<=(tablesize-counternew_):
                break
          xselect=xselect.iloc[1:,:]            

        copy=xselect.copy()
   
        if debug==True:


            speculatedflows.append((len(xselect),value.loc[new]))
                
            for k in range(len(xselect)):
                flow=xselect.iloc[k,0]
                flow_=xselect.iloc[k,1]
                controllernumber=0
                for j in range(len(controller)):
                  if flow==controller.iloc[j,0] and flow_==controller.iloc[j,1]:
                    controllernumber=j
                    break
                if controller.iloc[controllernumber,8]==0:
                    speculatedflowcounter_+=1
            speculatedflowplot_.append((speculatedflowcounter_,value.loc[new]))
            speculatedflowcounter_=0
            controller['speculatedflow']=0
#            wandb.log({'speculatedflows': len(xselect)}, step=new)


        for i in range(len(xselect)):
                if tablesize==len(switch):
                      temp=10000000
                      no=-1
                      breaknew=False
                      for k in range(len(switch)):
               
                                    flow=switch.iloc[k,0]
                                    flow_=switch.iloc[k,1]
                                    controllernumber=0
                                    for j in range(len(controller)):
                                      if flow==controller.iloc[j,0] and flow_==controller.iloc[j,1]:
                                        controllernumber=j
                                        break
                                        
                                    if switch.iloc[k,2]<=0.5:
                                            if temp>controller.iloc[controllernumber,5]:
                                              temp=controller.iloc[controllernumber,5]
                                              no=switch.index[k]
                                              location=switch.loc[no]
                                            if temp==0:
                                              temp=controller.iloc[controllernumber,5]
                                              no=switch.index[k]
                                              location=switch.loc[no]
                                              breaknew=True
                                              break
                                    else:
                                          continue

                      if no==-1:
                          pass
                      else:
                          switch=switch.drop([no])

              
                if tablesize==len(switch):
                      temp=10000000
                      no=-1
                      breaknew=False
                      for k in range(len(switch)):
              
                                        flow=switch.iloc[k,0]
                                        flow_=switch.iloc[k,1]
                                        controllernumber=0
                                        for j in range(len(controller)):
                                          if flow==controller.iloc[j,0] and flow_==controller.iloc[j,1]:
                                            controllernumber=j
                                            break
                                            
                                        if temp>controller.iloc[controllernumber,5]:
                                          temp=controller.iloc[controllernumber,5]
                                          no=switch.index[k]
                                          location=switch.loc[no]
                                        if temp==0:
                                          temp=controller.iloc[controllernumber,5]
                                          no=switch.index[k]
                                          location=switch.loc[no]
                                          breaknew=True
                                          break
                       
                      if no==-1:
                          pass
                      else:
                          switch=switch.drop([no])
                     
           
                temp=copy.iloc[:1]
                copy=copy.iloc[1:,:]
                          
                switch=switch.append(temp) # install the flow rule
                  


        for k in range(len(switch)):
                        if switch.iloc[k,2]>=3:
                            switch.iloc[k,2]-=3
        w=2*(math.ceil(len(controller)/flowresult))
        # calculate the reward
        r_list = cal_reward(X_selected,controller,controllercopy,step,w)#/sum(agentAction)*agentAction
        plotlist = cal_reward(X_selected,controller,controllercopy,step,w)
        i=0
        j=0

        agentresult=np.zeros(math.ceil((N_flows/numberofFlowsPerAgent)))
        while (i < len(r_list)):   
            agentresult[j]+=r_list[i]
            i+=1
            if (i%numberofFlowsPerAgent == 0):
              j+=1

        # append the reward result to the plot variable    
        plot.append((agentresult,value.loc[new]))
#        wandb.log({'reward': agentresult}, step=new)
        # store the controller values to compare for the next iteration
        controllercopy=controller.copy()

        # store the states , action , reward for the dqn algorithm for learning
        for agent, dqn in enumerate(dqn_list):
          if group==True:
            dqn.store_transition(s, action_list[agent], agentresult[agent], s_)
          else:
            dqn.store_transition(s, action_list[agent], agentresult[agent], s_)

        # dqn learning
        if dqn_list[0].memory_counter > MEMORY_CAPACITY:
         for dqn in dqn_list:
                    dqn.learn()

        # update the state
        if group==True:
          slist = s_list
        else:
          s=s_  

        # store the result
        result.append([sum(r_list), action_list])

        switchcount=switch.copy()
        switchcount = switchcount.drop(columns=['age']) 
        switchcopycount=switchcopy.copy()
        switchcopycount = switchcopycount.drop(columns=['age']) 
       

        switchcopycount_=switchcopynew.copy()
        if len(switchcopycount_)!=0:
            switchcopycount_ = switchcopycount_.drop(columns=['age']) 

        


              


        for k in range(len(switch)):
                        if switch.iloc[k,2]>=2:
                            switch.iloc[k,2]=1
                            #switch.iloc[k,1]-=2
        switch=switch[switch.groupby(['Source','Destination'])['age'].transform('max') == switch['age']]

        if new==len(newdataset):
            break                       
        
        if ((value.loc[new]) > 200).bool():
            break
    else:  
        if ((value.loc[new]) > 200).bool():
            break
        continue





# list of the selected flow rules sets
#if debug==True:
#    print(newlist)




import matplotlib.pyplot as plt

if speculative==True:
#    x, y = zip(*pltcounter)
#    new=[]
#    result,=plt.plot(y,x)
#    getxdata=result.get_xdata()
#    getxdata=list(getxdata)
#    getxdata.insert(0,"seconds")
#    getxdata = [str(i) for i in getxdata]
#    print(' '.join(getxdata))
#    getydata=result.get_ydata()
#    getydata=list(getydata)
#    getydata.insert(0,"speculatedflowcount")
#    getydata = [str(i) for i in getydata]
#    print(' '.join(getydata))
#    plt.ylabel('speculated flow count per second')
#    plt.xlabel('seconds')
#    wandb.log({'speculatedflowcountplot':plt})
    
    x, y = zip(*flaggedflows)
    result,=plt.plot(y,x)
    getxdata=result.get_xdata()
    getxdata=list(getxdata)
    getxdata.insert(0,"seconds")
    getxdata = [str(i) for i in getxdata]
#    print(' '.join(getxdata))
    getydata=result.get_ydata()
    getydata=list(getydata)
    getydata.insert(0,"flaggedflows")
    getydata = [str(i) for i in getydata]
    print(' '.join(getydata))
    plt.ylabel('flagged flows count per second')
    plt.xlabel('seconds')
#    wandb.log({'flaggedflowsplot':plt})
    

#    x, y = zip(*speculatedflows)
#    result,=plt.plot(y,x)
#    getxdata=result.get_xdata()
#    getxdata=list(getxdata)
#    getxdata.insert(0,"seconds")
#    getxdata = [str(i) for i in getxdata]
##    print(' '.join(getxdata))
#    getydata=result.get_ydata()
#    getydata=list(getydata)
#    getydata.insert(0,"speculatedflows")
#    getydata = [str(i) for i in getydata]
#    print(' '.join(getydata))
#    plt.ylabel('speculatedflows count per second')
#    plt.xlabel('seconds')
##    wandb.log({'speculatedflowsplot':plt})
    
    x, y = zip(*newflow_)
    result,=plt.plot(y,x)
    getxdata=result.get_xdata()
    getxdata=list(getxdata)
    getxdata.insert(0,"seconds")
    getxdata = [str(i) for i in getxdata]
#    print(' '.join(getxdata))
    getydata=result.get_ydata()
    getydata=list(getydata)
    getydata.insert(0,"newflow")
    getydata = [str(i) for i in getydata]
    print(' '.join(getydata))
    plt.ylabel('newflow count per second')
    plt.xlabel('seconds')
#    wandb.log({'newflowplot':plt})
    
    x, y = zip(*flowresult_)
    result,=plt.plot(y,x)
    getxdata=result.get_xdata()
    getxdata=list(getxdata)
    getxdata.insert(0,"seconds")
    getxdata = [str(i) for i in getxdata]
#    print(' '.join(getxdata))
    getydata=result.get_ydata()
    getydata=list(getydata)
    getydata.insert(0,"flowresult")
    getydata = [str(i) for i in getydata]
    print(' '.join(getydata))
    plt.ylabel('flow count per second')
    plt.xlabel('seconds')
#    wandb.log({'flowplot':plt})

    x, y = zip(*speculatedflowplot_)
    result,=plt.plot(y,x)
    getxdata=result.get_xdata()
    getxdata=list(getxdata)
    getxdata.insert(0,"seconds")
    getxdata = [str(i) for i in getxdata]
    print(' '.join(getxdata))
    getydata=result.get_ydata()
    getydata=list(getydata)
    getydata.insert(0,"speculatedflow")
    getydata = [str(i) for i in getydata]
    print(' '.join(getydata))
    plt.ylabel('speculatedflow count per second')
    plt.xlabel('seconds')
#    wandb.log({'speculatedflowsplot':plt})

x, y = zip(*plt1)
x=np.nan_to_num(x)
y=np.nan_to_num(y)
result,=plt.plot(y,x)
getxdata=result.get_xdata()
getxdata=list(getxdata)
getxdata.insert(0,"seconds")
getxdata = [str(i) for i in getxdata]
print(' '.join(getxdata))
getydata=result.get_ydata()
getydata=list(getydata)
getydata.insert(0,"hitrate")
getydata = [str(i) for i in getydata]
print(' '.join(getydata))
plt.ylabel('hit rate per second')
plt.xlabel('seconds')
#wandb.log({'hitrateplot':plt})

#plot.index(0,"reward")
#x, y = zip(*plot)
#print(x)
#print(y)
#result,=plt.plot(y,x)
#getxdata=result.get_xdata()
#getxdata=list(getxdata)
#getxdata = [str(i) for i in getxdata]
#print(' '.join(getxdata))
##print(list(getxdata))
#getydata=result.get_ydata()
#getydata=list(getydata)
#getydata = [str(i) for i in getydata]
#print(' '.join(getydata))
##print(list(getydata))
#plt.xlabel('total number of rewards')
#plt.ylabel('rewards')
##plt.xlim([0,len(plot3)])
##plt.show()
##print(plot)

#rate=[]
#print(newrate)
#print(plt1)
#for i in range(len(newrate)):
  #print(i)
  #if i == 0:
    #continue
  #if i == (len(newrate)-1):
    #break
  #rate.append(abs(newrate[i+1]-newrate[i]))
#print(rate)
#plt.plot(newrate)
#plt.ylabel('hit rate per second')
#plt.xlabel('seconds')
#plt.show()

x, y = zip(*plt2)
x=np.nan_to_num(x)
y=np.nan_to_num(y)
result,=plt.plot(y,x)
getxdata=result.get_xdata()
getxdata=list(getxdata)
getxdata = [str(i) for i in getxdata]
getydata=result.get_ydata()
getydata=list(getydata)
getydata = [str(i) for i in getydata]
plt.ylabel('miss rate per second')
plt.xlabel('seconds')

x, y = zip(*plot3)
result,=plt.plot(y,x)
getxdata=result.get_xdata()
getxdata=list(getxdata)
getxdata.insert(0,"seconds")
getxdata = [str(i) for i in getxdata]
#    print(' '.join(getxdata))
getydata=result.get_ydata()
getydata=list(getydata)
getydata.insert(0,"trafficrate")
getydata = [str(i) for i in getydata]
print(' '.join(getydata))
plt.ylabel('trafficrate per second')
plt.xlabel('seconds')
#wandb.log({'trafficrate':plt})

#x, y = zip(*plot3)
#result,=plt.plot(y,x)
#getxdata=result.get_xdata()
#getxdata=list(getxdata)
#getxdata = [str(i) for i in getxdata]
#print(' '.join(getxdata))
#getydata=result.get_ydata()
#getydata=list(getydata)
#getydata = [str(i) for i in getydata]
#plt.xlabel('seconds')
#plt.ylabel('traffic rate')
#wandb.log({'trafficrateplot':plt})

newplot1=sum(plot1)/len(plot1)
newplot2=sum(plot2)/len(plot2)



wandb.finish()
