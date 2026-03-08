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
from openpyxl import load_workbook
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import xlrd
import xlsxwriter as xlsw

# Load the Excel file with only the needed columns
excel_file = 'sdn_.xlsx'
# Load all columns from the Excel file
df = pd.read_excel(excel_file)
print(df.head())
print(df)
print(df.iloc[0])
print(df.iloc[1])
print(df.iloc[2])
print(df.iloc[2,1])
result=str(df.iloc[2,1])
dataframe=pd.DataFrame({'numberofFlowsPerAgent':[0],'tablesize':[0],'rewardAgingFactor':[0],'spatialReward':[0],'sdn':[0],'trace':[0],'rate1':[0],'rate2':[0]})
dataframe=dataframe.iloc[1:,:]

for row in range(len(df)):

    parser = argparse.ArgumentParser()
    parser.add_argument('--numberofFlowsPerAgent', default=df.iloc[row,0], type=int)
    parser.add_argument('--tablesize', default=df.iloc[row,1], type=int)
    parser.add_argument('--LFUTimeInterval', default=df.iloc[row,2], type=int)
    parser.add_argument('--agingfactor', default=df.iloc[row,3], type=float)
    parser.add_argument('--speculativesdn', default=df.iloc[row,4], type=str)
    parser.add_argument('--seed', default=df.iloc[row,5], type=int)
    parser.add_argument('--epsilon', default=df.iloc[row,6], type=float)
    parser.add_argument('--gamma', default=df.iloc[row,7], type=float)
    parser.add_argument('--target_replace_iter', default=df.iloc[row,8], type=int)
    parser.add_argument('--memory_capacity', default=df.iloc[row,9], type=int)
    parser.add_argument('--LR', default=df.iloc[row,10], type=float)
    parser.add_argument('--rewardAgingFactor', default=df.iloc[row,11], type=float)
    parser.add_argument('--spatialReward', default=df.iloc[row,12], type=float)
    parser.add_argument('--sdn', default=df.iloc[row,13], type=str)
    parser.add_argument('--dataset', default=df.iloc[row,14], type=int)
    parser.add_argument('--trace', default=df.iloc[row,15], type=int)
    parser.add_argument('--NN', default=df.iloc[row,16], type=int)
    parser.add_argument('--LTI', default=df.iloc[row,17], type=float)
    parser.add_argument('--RTI', default=df.iloc[row,18], type=float)

    os.environ['WANDB_API_KEY'] = "d3b7a9f392f8535c450e67c32f2aa2fe9ecf21e9"
    wandb.login()
    wandb.init(project="sdn_results", entity="nwsl", group="sdn", job_type="speculativeSDN_")
#    wandb.config = {"numberofFlowsPerAgent":1 ,"tablesize":1 ,"LFUTimeInterval":1 ,"agingfactor":1 ,"speculativesdn":"1" ,"seed": 1, "epsilon":1, "gamma":1, "target_replace_iter":1, "memory_capacity":1, "LR": 1, "rewardAgingFactor": 1, "spatialReward": 1, "sdn": "1", "dataset": 1, "trace": 1, "NN": 1, "LTI": 1, "RTI": 1}
    args = parser.parse_args()
    wandb.config=args
    wandb.log({"numberofFlowsPerAgent":wandb.config.numberofFlowsPerAgent, "tablesize": wandb.config.tablesize, "LFUTimeInterval": wandb.config.LFUTimeInterval, "agingfactor": wandb.config.agingfactor, "speculativesdn": wandb.config.speculativesdn, "seed": wandb.config.seed, "epsilon": wandb.config.epsilon, "gamma": wandb.config.gamma, "target_replace_iter": wandb.config.target_replace_iter, "memory_capacity": wandb.config.memory_capacity, "LR": wandb.config.LR, "rewardAgingFactor": wandb.config.rewardAgingFactor, "spatialReward": wandb.config.spatialReward, "sdn": wandb.config.sdn, "dataset": wandb.config.dataset, "trace": wandb.config.trace, "NN": wandb.config.NN, "LTI": wandb.config.LTI, "RTI": wandb.config.RTI})
    
    second=df.iloc[row,22].split()
    print(df.iloc[row,23])
    if pd.isnull(df.iloc[row,23])!=True:
        speculatedflow=df.iloc[row,23].split()
        second_=df.iloc[row,24].split()
        speculatedflow = [int(row) if row.isdigit() else float(row) for row in speculatedflow]
        second_ = [int(row) if row.isdigit() else float(row) for row in second_]
    hitrate=df.iloc[row,25].split()
    trafficrate=df.iloc[row,26].split()

#    [float(row) for row in second]
#    list(map(float, second.split(' ')))
    second = [int(row) if row.isdigit() else float(row) for row in second]
    hitrate = [int(row) if row.isdigit() else float(row) for row in hitrate]
    trafficrate = [int(row) if row.isdigit() else float(row) for row in trafficrate]
 

    for a in range(len(trafficrate)):
        step_=float(second[a])
        wandb.log({'second': float(second[a])})#, step=second)
        wandb.log({'hitrate': float(hitrate[a])})#second[row])
        wandb.log({'trafficrate': trafficrate[a]})

    if df.iloc[row,4]=="speculativereactive":
        for a in range(len(second_)):
            wandb.log({'second_': second_[a]})#, step=second)
            wandb.log({'speculatedflow': speculatedflow[a]})
        rate1=0
        rate2=0
        for a in range(len(df)):
            if df.iloc[a,4]=="reactive":
                if df.iloc[a,1]==df.iloc[row,1] and df.iloc[a,15]==df.iloc[row,15]:
                    b_=0
                    for b in range(len(second_)):
                        if second[b]>49:
                            b_=b
                            break
                    hitrate_=df.iloc[a,25].split()
                    hitrate_= [int(row_) if row_.isdigit() else float(row_) for row_ in hitrate_]

                    r1=0
                    r2=0
                    rate=0
                    for b in range(len(second_)):
                        if b>b_:
                            r1+=hitrate[b]
                            r2+=hitrate_[b]
                    if r2==0:
                        rate=0
                    else:
                        rate=((r1-r2)/r2)*100
                    rate1=rate

                    speculatedflow_ = [b / df.iloc[row,1] for b in speculatedflow]

                    b_=0
                    for b in range(len(second)):
                        if second[b]>49:
                            b_=b
                            break
                    r3=0
                    counter=0
                    for b in range(len(second)):
                        if b>b_:
                            r3+=speculatedflow_[b]
                            counter+=1
                    r3=r3/counter
                    rate_=rate1/r3
                    rate2=rate_

                    dataframe=dataframe.append({'numberofFlowsPerAgent':df.iloc[row,0],'tablesize':df.iloc[row,1],'rewardAgingFactor':df.iloc[row,11],'spatialReward':df.iloc[row,12],'sdn':df.iloc[row,13],'trace':df.iloc[row,15],'rate1':rate1,'rate2':rate2}, ignore_index=True)
                    dataframe.to_excel('sdn___.xlsx',index=False)

                
    wandb.finish()
