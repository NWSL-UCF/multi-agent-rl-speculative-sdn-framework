import argparse
import numpy as np
import pandas as pd
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from collections import deque
import time
import logging
import os
import json

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--numberofFlowsPerAgent', type=int, default=10)
parser.add_argument('--tablesize', type=int, default=30)
parser.add_argument('--LFUTimeInterval', type=int, default=10)
parser.add_argument('--agingfactor', type=float, default=50)
parser.add_argument('--speculativesdn', type=str, default='speculativereactive')
parser.add_argument('--seed', type=int, default=97)
parser.add_argument('--epsilon', type=int, default=1)
parser.add_argument('--gamma', type=int, default=90)
parser.add_argument('--target_replace_iter', type=int, default=50)
parser.add_argument('--memory_capacity', type=int, default=1024 * 1024)
parser.add_argument('--LR', type=int, default=90)
parser.add_argument('--rewardAgingFactor', type=int, default=90)
parser.add_argument('--spatialReward', type=int, default=90)
parser.add_argument('--sdn', type=str, default='trace')
parser.add_argument('--dataset', type=int, default=1)
parser.add_argument('--trace', type=int, default=1)
parser.add_argument('--numberOfHiddenLayer', type=int, default=1)
parser.add_argument('--LTI', type=float, default=0.1)
parser.add_argument('--RTI', type=float, default=0.01)
parser.add_argument('--processId', type=int, default=0)
parser.add_argument('--expId', type=str, default="")
parser.add_argument('--ins', type=str, default="")
parser.add_argument('--savePath', type=str, default="data/raw")
parser.add_argument('--testMode', type=bool, default=False)
args = parser.parse_args()

# Define base directory
home_path = os.path.expanduser("~")
base_dir = os.path.join(home_path, args.savePath, str(args.expId), str(args.processId))
os.makedirs(base_dir, exist_ok=True)

logging.basicConfig(
    filename=f"{base_dir}/process.log",  # Log file name
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",  # Overwrites the file on each run. Use 'a' for append mode.
)

logger = logging.getLogger(__name__)

def printInLog(message): 
    logger.info(str(args.processId) + " " + message)

def save_args_to_json(filename=f"{base_dir}/args.json"):
    args_dict = vars(args)
    with open(filename, 'w') as f:
        json.dump(args_dict, f, indent=4)
    printInLog(f"Arguments saved to {filename}")

save_args_to_json()
debug = True
source = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.dataset == 1:
    if args.trace == 1:
        dataset = pd.read_csv('Pcap/pcap.csv')
    elif args.trace == 2:
        dataset = pd.read_csv('Pcap/pcap_.csv')
    elif args.trace == 3:
        dataset = pd.read_csv('Pcap/pcapfile_.csv')

if args.dataset == 2:
    if args.trace == 1:
        dataset = pd.read_csv('dataset.csv')
    elif args.trace == 2:
        dataset = pd.read_csv('dataset_.csv')
    elif args.trace == 3:
        dataset = pd.read_csv('dataset__.csv')

rem = ['No.', 'Time', 'Protocol', 'Length', 'Info']
remove = ['No.', 'Source', 'Destination', 'Protocol', 'Length', 'Info']
value = dataset.copy()
value.drop(remove, axis=1, inplace=True)
dataset.drop(rem, axis=1, inplace=True)
table = dataset.drop_duplicates()
controller = table.copy()
controller['hit'] = 0
controller['miss'] = 0

if args.sdn == "trace":
    pass
elif args.sdn == "source":
    controller = controller.sort_values(by='Source', ascending=True)
elif args.sdn == "destination":
    controller = controller.sort_values(by='Destination', ascending=True)

column = {'Source': [0], 'Destination': [0]}
datasetcopy = pd.DataFrame(data=column)
datasetcopy = datasetcopy.iloc[1:, :]
dataset_ = pd.DataFrame(data=dataset)
remove_ = ['Source', 'Destination']

newdataset = dataset_.groupby(['Source', 'Destination'], sort=False).ngroup()
newdataset_ = pd.DataFrame(index=newdataset, data=column)
newdataset_.drop(remove_, axis=1, inplace=True)


r, c = controller.shape
array = controller.values

X = controller.iloc[:, 0:(c - 2)]
X.insert(0, 'No.', value=np.arange(len(X)))
X.drop(remove_, axis=1, inplace=True)
N_flows = X.shape[0]

BATCH_SIZE = 32
LR = float(args.LR / 100)
EPSILON = float(args.epsilon / 10)
GAMMA = float(args.gamma / 100)
TARGET_REPLACE_ITER = args.target_replace_iter
MEMORY_CAPACITY = args.memory_capacity

numberofFlowsPerAgent = args.numberofFlowsPerAgent
N_ACTIONS = pow(2, numberofFlowsPerAgent)
group = True
N_STATES = math.ceil(N_flows / numberofFlowsPerAgent)

def save_network_shape_to_json(network_shape, filename=f"{base_dir}/network_shape.json"):
    with open(filename, 'w') as file:
        json.dump(network_shape, file, indent=4)
    printInLog(f"Network shape saved to {filename}")

def getDeepNetworkShape():
    networkShape = {
        "inputLayerSize": N_STATES,
        "outputLayerSize": N_ACTIONS,
        "hiddenLayersSize": []
    }

    for i in range(args.numberOfHiddenLayer):
        layerSize = networkShape["inputLayerSize"] + math.ceil(
            (networkShape["outputLayerSize"] - networkShape["inputLayerSize"]) / 
            (args.numberOfHiddenLayer + 1) * (i + 1)
        )
        networkShape["hiddenLayersSize"].append(layerSize)
    return networkShape

networkShape = getDeepNetworkShape()
save_network_shape_to_json(networkShape)

# class Net(nn.Module):
#     def __init__(self, N_STATES, N_ACTIONS):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(N_STATES, 10)
#         self.fc1.weight.data.normal_(0, 0.1)
#         self.out = nn.Linear(10, N_ACTIONS)
#         self.out.weight.data.normal_(0, 0.1)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = F.relu(x)
#         action_value = self.out(x)
#         return action_value

class Net(nn.Module):
    def __init__(self, networkShape):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        input_size = networkShape['inputLayerSize']
        for hidden_size in networkShape['hiddenLayersSize']:
            self.layers.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size  # Update input size for the next layer
        self.out = nn.Linear(input_size, networkShape['outputLayerSize'])

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        action_value = self.out(x)
        return action_value

# class DQN(object):
#     def __init__(self, N_STATES, N_ACTIONS):
#         self.eval_net, self.target_net = Net(N_STATES, N_ACTIONS), Net(N_STATES, N_ACTIONS)
#         self.learn_step_counter = 0
#         self.memory_counter = 0
#         self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
#         self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
#         self.loss_func = nn.MSELoss()

#     def choose_action(self, x):
#         x = torch.unsqueeze(torch.FloatTensor(x), 0)
#         if np.random.uniform() < EPSILON:
#             action_value = self.eval_net.forward(x)
#             action = torch.max(action_value, 0)[1].data.numpy()
#             action = action[0]
#         else:
#             action = np.random.randint(0, N_ACTIONS)
#         return action

#     def store_transition(self, s, a, r, s_):
#         transition = np.hstack((s, [a, r], s_))
#         index = self.memory_counter % MEMORY_CAPACITY
#         self.memory[index, :] = transition
#         self.memory_counter += 1

#     def learn(self):
#         if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
#             self.target_net.load_state_dict(self.eval_net.state_dict())
#         self.learn_step_counter += 1
#         sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
#         b_memory = self.memory[sample_index, :]
#         b_s = torch.FloatTensor(b_memory[:, :N_STATES])
#         b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1])
#         b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
#         b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
#         q_eval = self.eval_net(b_s).gather(1, b_a)
#         q_next = self.target_net(b_s_).detach()
#         q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
#         loss = self.loss_func(q_eval, q_target)
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()


class DQN(object):
    def __init__(self, networkShape):
        self.eval_net = Net(networkShape)
        self.target_net = Net(networkShape)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, networkShape['inputLayerSize'] * 2 + 2))
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.EPSILON = EPSILON
        self.GAMMA = GAMMA
        self.TARGET_REPLACE_ITER = TARGET_REPLACE_ITER
        self.BATCH_SIZE = BATCH_SIZE
        self.N_ACTIONS = networkShape['outputLayerSize']

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < self.EPSILON:
            action_value = self.eval_net(x)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, self.N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :networkShape['inputLayerSize']])
        b_a = torch.LongTensor(b_memory[:, networkShape['inputLayerSize']:networkShape['inputLayerSize'] + 1])
        b_r = torch.FloatTensor(b_memory[:, networkShape['inputLayerSize'] + 1:networkShape['inputLayerSize'] + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -networkShape['inputLayerSize']:])

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.GAMMA * q_next.max(1)[0].view(self.BATCH_SIZE, 1)
        
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def cal_reward(X_selected, controllerTable, oldControllerTable, step, w):
    reward = np.zeros(N_flows)
    for i in range(len(controller)):
        controllerTable.iloc[i, 4] = (float(args.rewardAgingFactor) / 100) * controllerTable.iloc[i, 4]
        if oldControllerTable.iloc[i, 2] < controllerTable.iloc[i, 2]:
            reward[i] += controllerTable.iloc[i, 2] - oldControllerTable.iloc[i, 2]
        if oldControllerTable.iloc[i, 3] < controllerTable.iloc[i, 3]:
            reward[i] += controllerTable.iloc[i, 3] - oldControllerTable.iloc[i, 3]
    for i in range(len(switch)):
        flow = switch.iloc[i, 0]
        flow_ = switch.iloc[i, 1]
        controllernumber = 0
        for j in range(len(controller)):
            if flow == controller.iloc[j, 0] and flow_ == controller.iloc[j, 1]:
                controllernumber = j
                break
        if controllerTable.iloc[controllernumber, 5] != 0:
            if oldControllerTable.iloc[controllernumber, 5] < controllerTable.iloc[controllernumber, 5]:
                reward_ = controllerTable.iloc[controllernumber, 2] + controllerTable.iloc[controllernumber, 3]
                spatialReward = reward_
                j = 0
                while j in range(0, math.ceil(w / 2)):
                    if (controllernumber + j) < len(controllerTable):
                        reward[controllernumber + j] += spatialReward
                    if (controllernumber - j) >= 0:
                        reward[controllernumber - j] += spatialReward
                    j += 1
                    spatialReward *= (float(args.spatialReward / 100))
    for i in range(len(controller)):
        controllerTable.iloc[i, 4] += reward[i]
    return reward

class Queue:
    def __init__(self, size):
        self.queue = [None] * size
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

action_list = np.random.randint(N_ACTIONS, size=math.ceil(N_flows / numberofFlowsPerAgent))
agentAction = np.zeros(N_flows, int)
slist = np.zeros(0, int)

for i in range(len(action_list)):
    tempAction = action_list[i]
    for j in range(numberofFlowsPerAgent):
        if (i * numberofFlowsPerAgent + j == len(agentAction)):
            break
        agentAction[i * numberofFlowsPerAgent + j] = tempAction % 2
        tempAction = tempAction / 2

X_selected = X.iloc[agentAction == 1, :]
s = torch.Tensor(action_list)
slist = list(s)

controller.insert(4, 'reward', value=0)
controller.insert(5, 'counter', value=0)
controller.insert(6, 'wasHit', value=0)
controller.insert(7, 'spatialReward', value=0)
controller.insert(8, 'speculatedflow', value=0)

newcolumn = {'Source': [0], 'Destination': [0], 'hit': [0], 'miss': [0], 'reward': [0], 'counter': [0], 'wasHit': [0]}
newcolumn_ = {'Source': [0], 'Destination': [0], 'age': [1]}
column = {'Source': [0], 'Destination': [0]}
column_ = {'No.'}
newrate = pd.DataFrame(data=newcolumn)
newrate = controller.copy()
switch = pd.DataFrame(data=newcolumn_)
switch = switch.iloc[1:, :]
switchcopy = pd.DataFrame(data=newcolumn_)
switchcopy = switchcopy.iloc[1:, :]
switchcopy_ = pd.DataFrame(data=newcolumn_)
switchcopy_ = switchcopy_.iloc[1:, :]
switchcopy_.insert(3, 'switchcopy', value=0)
controllercopy = controller.copy()
controllercopy_ = controller.copy()
numberofflows = pd.DataFrame(data=newcolumn)
numberofflows = controller.copy()
numberofflows = numberofflows.drop(columns=['hit', 'miss', 'reward', 'counter', 'wasHit'])
numberofflows.insert(1, 'flow', value=0)
numberofflows_ = pd.DataFrame(data=newcolumn)
numberofflows_ = controller.copy()
datasetcopy = pd.DataFrame(data=column)
datasetcopy = datasetcopy.iloc[1:, :]

numberofflows_ = []
numberofflowscopy_ = []
new = 0

rule = 0
trace = 0
least = 0
draw = 0
rate = 0

plot = []
plot1 = []
plot2 = []
plot3 = []
plt1 = {
    'HitRate': [],
    'New': [],
    'Time': [],
    'HitCount': []
}
plt2 = {
    'MissRate': [],
    'New': [],
    'Time': [],
    'MissCount': []
}
plt3 = {
    'Rate': [],
    'New': [],
    'Time': []
}
plotlist = []
pltcounter = []
newpltcounter = []
dqn_list = []

for agent in range(math.ceil(N_flows / numberofFlowsPerAgent)):
    dqn_list.append(DQN(networkShape))

result = []
tablesize = args.tablesize
queue = deque()
newlist = []
sdnflag = True
speculationflowcount = []
graphcounter = 0
flaggedflows = {
    'CounterNew': [],
    'New': [], 
    'Time': []
}

speculatedflows = []
counternew = 0
newflow_ = {
    'NewFlow': [],
    'New': [], 
    'Time': []
}
flowresult_ = {
    'FlowResult': [],
    'New': [], 
    'Time': []
}
speculationcounter = 0
count = False
learningTimeInterval = float(args.LTI / 1000)
LFUTimeInterval = float(float(args.LFUTimeInterval) * learningTimeInterval)
graphRateInterval = 1
w = 0
agingfactor = float(args.agingfactor / 100)
speculatedflowplot_ = {
    'SpeculatedFlowCounter_': [],
    'New': [], 
    'Time': []
}
speculatedflowcounter_ = 0
speculative = False
reactive = False
controllercounter = 0
controllerlist = []

if args.speculativesdn == "speculative":
    speculative = True
if args.speculativesdn == "reactive":
    reactive = True
if args.speculativesdn == "speculativereactive":
    speculative = True
    reactive = True

# the main loop which simulates the whole experiment
itr = 0
prev_timestamp = time.time() * 1000
while True:
    current_timestamp = time.time() * 1000
    printInLog(f"Simulation itr {itr} takes {current_timestamp - prev_timestamp} ms")
    itr += 1
    prev_timestamp = current_timestamp
    
    switchcopy = pd.DataFrame(data=column)
    switchcopy = switchcopy.iloc[1:, :]
    switchcopy = switch.copy()

    breaknew = False

    for k in range(len(switch)):
        for j in range(len(controller)):
            if controller.iloc[j, 0] == switch.iloc[k, 0] and controller.iloc[j, 1] == switch.iloc[k, 1]:
                if controller.iloc[j, 6] != 1:
                    switch.iloc[k, 2] = switch.iloc[k, 2] * agingfactor
                else:
                    controller.iloc[j, 6] = 0

    while True:
        if reactive:
            processFlow = False
            queue_ = False
            if len(queue) == 0:
                processFlow = False
            if len(queue) != 0 and not queue_:
                tempqueue = queue[0]
                queue_ = True
            if len(queue) != 0:
                if tempqueue.iloc[0, 3] < float(value.loc[new].iloc[0]):
                    queue.popleft()
                    processFlow = True
                    queue_ = False

            step = float(value.loc[new].iloc[0])

            if processFlow:
                if tablesize == len(switch):
                    temp = 10000000
                    no = -1
                    breaknew = False
                    for k in range(len(switch)):
                        flow = switch.iloc[k, 0]
                        flow_ = switch.iloc[k, 1]
                        controllernumber = 0
                        for j in range(len(controller)):
                            if flow == controller.iloc[j, 0] and flow_ == controller.iloc[j, 1]:
                                controllernumber = j
                                break

                        if reactive and not speculative:
                            if temp > controller.iloc[controllernumber, 5]:
                                temp = controller.iloc[controllernumber, 5]
                                no = switch.index[k]
                                location = switch.loc[no]
                            if temp == 0:
                                temp = controller.iloc[controllernumber, 5]
                                no = switch.index[k]
                                location = switch.loc[no]
                                breaknew = True
                                break

                        if reactive and speculative:
                            if switch.iloc[k, 2] <= 0.5:
                                if temp > controller.iloc[controllernumber, 5] * switch.iloc[k, 2]:
                                    temp = controller.iloc[controllernumber, 5] * switch.iloc[k, 2]
                                    no = switch.index[k]
                                    location = switch.loc[no]
                                if temp == 0:
                                    temp = controller.iloc[controllernumber, 5] * switch.iloc[k, 2]
                                    no = switch.index[k]
                                    location = switch.loc[no]
                                    breaknew = True
                                    break
                            else:
                                continue

                    if no == -1:
                        pass
                    else:
                        switch = switch.drop([no])

                if tablesize == len(switch):
                    temp = 10000000
                    no = -1
                    breaknew = False
                    for k in range(len(switch)):
                        flow = switch.iloc[k, 0]
                        flow_ = switch.iloc[k, 1]
                        controllernumber = 0
                        for j in range(len(controller)):
                            if flow == controller.iloc[j, 0] and flow_ == controller.iloc[j, 1]:
                                controllernumber = j
                                break

                        if temp > controller.iloc[controllernumber, 5]:
                            temp = controller.iloc[controllernumber, 5]
                            no = switch.index[k]
                            location = switch.loc[no]
                        if temp == 0:
                            temp = controller.iloc[controllernumber, 5]
                            no = switch.index[k]
                            location = switch.loc[no]
                            breaknew = True
                            break

                    if no == -1:
                        pass
                    else:
                        switch = switch.drop([no])

                tempqueue.iloc[0, 2] = 1
                switch = pd.concat([switch, tempqueue], ignore_index=True)
                switch = switch.drop_duplicates(subset=['Source', 'Destination'], keep='last')
                switch.drop(['switchcopy'], axis=1, inplace=True)

            datasetcopy = pd.concat([datasetcopy, dataset_.loc[[new]]], ignore_index=True)
            found = False
            for j in range(len(switch)):
                if switch.iloc[j, 0] == datasetcopy.iloc[0, 0] and switch.iloc[j, 1] == datasetcopy.iloc[0, 1]:
                    found = True

            if found:
                for j in range(len(controller)):
                    if controller.iloc[j, 0] == datasetcopy.iloc[0, 0] and controller.iloc[j, 1] == datasetcopy.iloc[0, 1]:
                        controller.iloc[j, 2] += 1
                        controller.iloc[j, 6] = 1
                        newrate.iloc[j, 2] += 1
                        controller.iloc[j, 5] += 1
                        rate += 1
                        numberofflows_.append(controller.iloc[j, 0])
                        numberofflows_ = list(set(numberofflows_))
                        if numberofflows.iloc[j, 1] != 1:
                            numberofflows.iloc[j, 1] = 1
                        controller.iloc[j, 8] = 1

            else:
                for j in range(len(controller)):
                    if controller.iloc[j, 0] == datasetcopy.iloc[0, 0] and controller.iloc[j, 1] == datasetcopy.iloc[0, 1]:
                        controller.iloc[j, 3] += 1
                        newrate.iloc[j, 3] += 1
                        controller.iloc[j, 5] += 1
                        rate += 1
                        if numberofflows.iloc[j, 1] != 1:
                            numberofflows.iloc[j, 1] = 1

                        numberofflows_.append(controller.iloc[j, 0])
                        numberofflows_ = list(set(numberofflows_))
                        controller.iloc[j, 8] = 1

                        switchcopy_ = pd.concat([switchcopy_, datasetcopy], ignore_index=True)
                        switchcopy_['age'] = 1
                        switchcopy_['switchcopy'] = float(value.loc[new].iloc[0]) + (float(args.RTI) / 1000)
                        switchcopy_ = switchcopy_[['Source', 'Destination', 'age', 'switchcopy']]
                        queue.append(switchcopy_)
                        switchcopy_ = pd.DataFrame(data=newcolumn_)
                        switchcopy_ = switchcopy_.iloc[1:, :]

            new += 1
            datasetcopy = pd.DataFrame(data=column)
            datasetcopy = datasetcopy.iloc[1:, :]

            if debug:
                step = float(value.loc[new].iloc[0])
        
        

        else:
            datasetcopy = pd.concat([datasetcopy, dataset_.loc[[new]]], ignore_index=True)
            found = False
            for j in range(len(switch)):
                if switch.iloc[j, 0] == datasetcopy.iloc[0, 0] and switch.iloc[j, 1] == datasetcopy.iloc[0, 1]:
                    found = True

            if found:
                for j in range(len(controller)):
                    if controller.iloc[j, 0] == datasetcopy.iloc[0, 0] and controller.iloc[j, 1] == datasetcopy.iloc[0, 1]:
                        controller.iloc[j, 2] += 1
                        controller.iloc[j, 6] = 1
                        newrate.iloc[j, 2] += 1
                        controller.iloc[j, 5] += 1
                        rate += 1
                        numberofflows_.append(controller.iloc[j, 0])
                        numberofflows_ = list(set(numberofflows_))
                        if numberofflows.iloc[j, 1] != 1:
                            numberofflows.iloc[j, 1] = 1
                        controller.iloc[j, 8] = 1

            else:
                for j in range(len(controller)):
                    if controller.iloc[j, 0] == datasetcopy.iloc[0, 0] and controller.iloc[j, 1] == datasetcopy.iloc[0, 1]:
                        controller.iloc[j, 3] += 1
                        newrate.iloc[j, 3] += 1
                        controller.iloc[j, 5] += 1
                        rate += 1
                        if numberofflows.iloc[j, 1] != 1:
                            numberofflows.iloc[j, 1] = 1

                        numberofflows_.append(controller.iloc[j, 0])
                        numberofflows_ = list(set(numberofflows_))
                        controller.iloc[j, 8] = 1

                        switchcopy_ = pd.concat([switchcopy_, datasetcopy], ignore_index=True)
                        switchcopy_['age'] = 1
                        switchcopy_['switchcopy'] = float(value.loc[new]) + float(args.RTI / 1000)
                        switchcopy_ = switchcopy_[['Source', 'Destination', 'age', 'switchcopy']]
                        queue.append(switchcopy_)
                        switchcopy_ = pd.DataFrame(data=newcolumn_)
                        switchcopy_ = switchcopy_.iloc[1:, :]

            new += 1
            datasetcopy = pd.DataFrame(data=column)
            datasetcopy = datasetcopy.iloc[1:, :]

            if debug:
                step = float(value.loc[new])
        diff = (value.loc[new] - value.loc[least]).item()
        
        
        
        
        if diff > LFUTimeInterval:
            least = new
            controller['counter'] = 0

        if (value.loc[new] - value.loc[draw]).item() > graphRateInterval:
            # plt1.append(((newrate['hit'].sum() / (newrate['hit'].sum() + newrate['miss'].sum())) * 100, value.loc[new]))
            plt1['HitCount'].append(newrate['hit'].sum())
            plt1['HitRate'].append((newrate['hit'].sum() / (newrate['hit'].sum() + newrate['miss'].sum())) * 100)
            plt1['New'].append(new)
            plt1['Time'].append(value.loc[new]['Time'].item())
            plt1result = ((newrate['hit'].sum() / (newrate['hit'].sum() + newrate['miss'].sum())) * 100)
            step = float(value.loc[new].iloc[0])
            pltcounter.append((speculationcounter, value.loc[new]))
            newpltcounter.append(pltcounter)
            speculativestep = float(value.loc[new].iloc[0])
            speculationcounter = 0
            
            #flaggedflows.append((counternew, value.loc[new])) 
            flaggedflows['CounterNew'].append(counternew)
            flaggedflows['New'].append(new)
            flaggedflows['Time'].append(value.loc[new]['Time'].item())
            
            speculativestep = float(value.loc[new].iloc[0])
            counternew = 0
            # plt2.append(((newrate['miss'].sum() / (newrate['hit'].sum() + newrate['miss'].sum())) * 100, value.loc[new]))
            plt2['MissCount'].append(newrate['miss'].sum())
            plt2['MissRate'].append((newrate['miss'].sum() / (newrate['hit'].sum() + newrate['miss'].sum())) * 100)
            plt2['New'].append(new)
            plt2['Time'].append(value.loc[new]['Time'].item()) 
            
            draw = new
            plot1.append((newrate['hit'].sum() / (newrate['hit'].sum() + newrate['miss'].sum())) * 100)
            plot2.append((newrate['miss'].sum() / (newrate['hit'].sum() + newrate['miss'].sum())) * 100)
            newrate['hit'] = 0
            newrate['miss'] = 0
            plot3.append((rate, value.loc[new]))
            # plt3.append(plot3)
            plt3['Rate'].append(rate)
            plt3['Time'].append(value.loc[new]['Time'].item())
            plt3['New'].append(new)
            rate = 0

            speculatableflow = len(controllerlist)
            controllerlist = []

        if (value.loc[new] - value.loc[trace]).item() > learningTimeInterval:
            newtrace = trace
            trace = new
            numberofflowslist_ = list(set(numberofflows_) - set(numberofflowscopy_))

            newflow = len(numberofflowslist_)
            # newflow_.append((newflow, value.loc[new]))
            newflow_['NewFlow'].append(newflow)
            newflow_['New'].append(new)
            newflow_['Time'].append(value.loc[new]['Time'].item())
            
            flowresult = (numberofflows['flow'].sum())

            # flowresult_.append((flowresult, value.loc[new]))
            flowresult_['FlowResult'].append(flowresult)
            flowresult_['New'].append(new)
            flowresult_['Time'].append(value.loc[new]['Time'].item())
            
            numberofflows['flow'] = 0
            numberofflowscopy_ = numberofflows_.copy()

            break
        if new == len(newdataset):
            break
        if args.dataset == 1:
            if (value.loc[new]).item() > 200:
                break
        if args.dataset == 2:
            if (value.loc[new]).item() > 0.1:
                break

    if speculative:
        action_list = np.zeros(math.ceil(N_flows / numberofFlowsPerAgent))

        for agent, dqn in enumerate(dqn_list):
            if group:
                action_list[agent] = dqn.choose_action(s)
            else:
                action_list[agent] = dqn.choose_action(s)
        newlist.append(action_list)

        for i in range(len(action_list)):
            tempAction = action_list[i]
            for j in range(numberofFlowsPerAgent):
                if (i * numberofFlowsPerAgent + j == len(agentAction)):
                    break
                agentAction[i * numberofFlowsPerAgent + j] = tempAction % 2
                tempAction = tempAction / 2

        X_selected = X.iloc[agentAction == 1]

        if group:
            s_ = torch.FloatTensor(action_list)
            s_list = list(s_)
        else:
            s_ = torch.FloatTensor(action_list)
        X_selected.insert(1, 'Source', value='')
        X_selected.insert(2, 'Destination', value='')

        for i in range(len(controller)):
            for j in range(len(X_selected)):
                if X_selected.iloc[j, 0] == controller.index[i]:
                    X_selected.iloc[j, 1] = controller.iloc[i, 1]
                    X_selected.iloc[j, 2] = controller.iloc[i, 2]

        counter = 0
        for i in range(len(switch)):
            for j in range(len(X_selected)):
                if X_selected.iloc[j, 1] == switch.iloc[i, 0] and X_selected.iloc[j, 2] == switch.iloc[i, 1]:
                    if switch.iloc[i, 2] > 0.5:
                        switch.iloc[i, 2] += 2
                        counter += 1

        switchcopy_ = pd.DataFrame(data=column)
        switchcopy_ = switchcopy_.iloc[1:, :]
        switchcopy_ = switch.copy()
        switchcopynew = pd.DataFrame(data=column)
        switchcopynew = switchcopynew.iloc[1:, :]
        switchcopynew_ = len(switchcopy_)

        for i in range(switchcopynew_):
            if switchcopy_.iloc[i, 2] > 0.5:
                # switchcopynew = switchcopynew.append(switchcopy_.iloc[[i]])
                switchcopynew = pd.concat([switchcopynew, switchcopy_.iloc[[i]]], ignore_index=True)


        xselected = pd.DataFrame(data=newcolumn)
        xselected = xselected.iloc[1:, :]
        xselected['Source'] = X_selected['Source']
        xselected['Destination'] = X_selected['Destination']
        xselected.update(controller)
        xselected = xselected.sort_values(by='reward', ascending=True)
        xselectednew = len(xselected)
        xselect = xselected.copy()
        xselect = xselect.sort_values(by='reward', ascending=True)
        xselect = xselect.drop(columns=['hit', 'miss', 'reward', 'counter', 'wasHit'])
        xselect.insert(2, 'age', value=3.5)
        counternew_ = 0
        for k in range(len(switch)):
            if switch.iloc[k, 2] > 0.5:
                counternew += 1
                counternew_ += 1

        new_ = len(xselect)
        for i in range(new_):
            if len(xselect) <= (tablesize - counternew_):
                break
            xselect = xselect.iloc[1:, :]

        copy = xselect.copy()

        if debug:
            speculatedflows.append((len(xselect), value.loc[new]))

            for k in range(len(xselect)):
                flow = xselect.iloc[k, 0]
                flow_ = xselect.iloc[k, 1]
                controllernumber = 0
                for j in range(len(controller)):
                    if flow == controller.iloc[j, 0] and flow_ == controller.iloc[j, 1]:
                        controllernumber = j
                        break
                if controller.iloc[controllernumber, 8] == 0:
                    speculatedflowcounter_ += 1
            # speculatedflowplot_.append((speculatedflowcounter_, value.loc[new]))
            speculatedflowplot_['SpeculatedFlowCounter_'].append(speculatedflowcounter_)
            speculatedflowplot_['New'].append(new)
            speculatedflowplot_['Time'].append(value.loc[new]['Time'].item())
            
            speculatedflowcounter_ = 0
            controller['speculatedflow'] = 0

        for i in range(len(xselect)):
            if tablesize == len(switch):
                temp = 10000000
                no = -1
                breaknew = False
                for k in range(len(switch)):
                    flow = switch.iloc[k, 0]
                    flow_ = switch.iloc[k, 1]
                    controllernumber = 0
                    for j in range(len(controller)):
                        if flow == controller.iloc[j, 0] and flow_ == controller.iloc[j, 1]:
                            controllernumber = j
                            break

                    if switch.iloc[k, 2] <= 0.5:
                        if temp > controller.iloc[controllernumber, 5]:
                            temp = controller.iloc[controllernumber, 5]
                            no = switch.index[k]
                            location = switch.loc[no]
                        if temp == 0:
                            temp = controller.iloc[controllernumber, 5]
                            no = switch.index[k]
                            location = switch.loc[no]
                            breaknew = True
                            break
                    else:
                        continue

                if no == -1:
                    pass
                else:
                    switch = switch.drop([no])

            if tablesize == len(switch):
                temp = 10000000
                no = -1
                breaknew = False
                for k in range(len(switch)):
                    flow = switch.iloc[k, 0]
                    flow_ = switch.iloc[k, 1]
                    controllernumber = 0
                    for j in range(len(controller)):
                        if flow == controller.iloc[j, 0] and flow_ == controller.iloc[j, 1]:
                            controllernumber = j
                            break

                    if temp > controller.iloc[controllernumber, 5]:
                        temp = controller.iloc[controllernumber, 5]
                        no = switch.index[k]
                        location = switch.loc[no]
                    if temp == 0:
                        temp = controller.iloc[controllernumber, 5]
                        no = switch.index[k]
                        location = switch.loc[no]
                        breaknew = True
                        break

                if no == -1:
                    pass
                else:
                    switch = switch.drop([no])

            temp = copy.iloc[:1]
            copy = copy.iloc[1:, :]

            # switch = switch.append(temp)
            switch = pd.concat([switch, temp], ignore_index=True)


        for k in range(len(switch)):
            if switch.iloc[k, 2] >= 3:
                switch.iloc[k, 2] -= 3
        w = 2 * (math.ceil(len(controller) / flowresult))
        r_list = cal_reward(X_selected, controller, controllercopy, step, w)
        plotlist = cal_reward(X_selected, controller, controllercopy, step, w)
        i = 0
        j = 0

        agentresult = np.zeros(math.ceil(N_flows / numberofFlowsPerAgent))
        while i < len(r_list):
            agentresult[j] += r_list[i]
            i += 1
            if (i % numberofFlowsPerAgent) == 0:
                j += 1

        plot.append((agentresult, value.loc[new]))

        controllercopy = controller.copy()

        for agent, dqn in enumerate(dqn_list):
            if group:
                dqn.store_transition(s, action_list[agent], agentresult[agent], s_)
            else:
                dqn.store_transition(s, action_list[agent], agentresult[agent], s_)

        if dqn_list[0].memory_counter > MEMORY_CAPACITY:
            for dqn in dqn_list:
                dqn.learn()

        if group:
            slist = s_list
        else:
            s = s_

        result.append([sum(r_list), action_list])

        switchcount = switch.copy()
        switchcount = switchcount.drop(columns=['age'])
        switchcopycount = switchcopy.copy()
        switchcopycount = switchcopycount.drop(columns=['age'])

        switchcopycount_ = switchcopynew.copy()
        if len(switchcopycount_) != 0:
            switchcopycount_ = switchcopycount_.drop(columns=['age'])

        for k in range(len(switch)):
            if switch.iloc[k, 2] >= 2:
                switch.iloc[k, 2] = 1

        switch = switch[switch.groupby(['Source', 'Destination'])['age'].transform('max') == switch['age']]

        if new == len(newdataset):
            break

        if (value.loc[new]).item() > 200:
            break
    else:
        if (value.loc[new]).item() > 200:
            break
        continue
    

end_time = time.time()

printInLog(f"Process #{args.processId} took {end_time - start_time} seconds")


def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

def save_results():
    # Log flagged flows
    logger.info("FLAGGED_FLOWS_SAVED_AS_CSV")
    save_to_csv(flaggedflows, os.path.join(base_dir, "flaggedflows.csv"))

    # Log new flows
    logger.info("NEW_FLOW_SAVED_AS_CSV")
    save_to_csv(newflow_, os.path.join(base_dir, "newflow.csv"))

    # Log flow results
    logger.info("FLOW_RESULT_SAVED_AS_CSV")
    save_to_csv(flowresult_, os.path.join(base_dir, "flowresult.csv"))

    # Log speculated flow plots
    logger.info("SPECULATED_FLOW_PLOT_SAVED_AS_CSV")
    save_to_csv(speculatedflowplot_, os.path.join(base_dir, "speculatedflowplot.csv"))

    # Log hit rate per second
    logger.info("HIT_RATE_SAVED_AS_CSV")
    save_to_csv(plt1, os.path.join(base_dir, "hit_rate.csv"))

    # Log miss rate per second
    logger.info("MISS_RATE_SAVED_AS_CSV")
    save_to_csv(plt2, os.path.join(base_dir, "miss_rate.csv"))

    # Log traffic rate per second
    logger.info("TRAFFIC_RATE_SAVED_AS_CSV")
    save_to_csv(plt3, os.path.join(base_dir, "traffic_rate.csv"))

save_results()


import requests
import sys
BASE_URL = "http://abdur-rouf.com:5000"
DONE = "DONE"
ABORTED = "ABORTED"
def update_job_status(job_id, status, message=""):
    url = f"{BASE_URL}/update_job_status"
    payload = {
        "job_id": job_id,
        "status": status,
        "message": message
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    return response.json()


update_job_status(args.processId, "DONE", f"{args.ins} has finished this job successfully!")
logging.info(f"Job {args.processId} marked as DONE")
logging.info(f"return success to the parent!")