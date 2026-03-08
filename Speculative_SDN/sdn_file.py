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


#numberofflows=[10]
#table_size=[10, 30, 50]
#lfu=[10]
#agingfactor=[50]
#speculativesdn=["reactive"]
#seed=[1]
#epsilon=[1]
#gamma=[9]
#target_replace_iter=[100]
#memory_capacity=[10]
#LR=[9]
#rewardAgingFactor=[3]
#spatialReward=[3]
#sdn=["trace"]
#dataset=[1]
#trace=[1, 2, 3]
#NN=[1]
#LTI=[100]
#RTI=[10]

#numberofflows=[10]
#table_size=[100, 50, 200]
#lfu=[10]
#agingfactor=[50]
#speculativesdn=["reactive"]
#seed=[1]
#epsilon=[1]
#gamma=[9]
#target_replace_iter=[100]
#memory_capacity=[10]
#LR=[9]
#rewardAgingFactor=[3]
#spatialReward=[3]
#sdn=["trace"]
#dataset=[2]
#trace=[1, 2, 3]
#NN=[1]
#LTI=[1]
#RTI=[.5]

numberofflows=[5, 10]
table_size=[10, 30, 50]
lfu=[10]
agingfactor=[50]
speculativesdn=["speculativereactive"]
seed=[1]
epsilon=[1]
gamma=[9]
target_replace_iter=[100]
memory_capacity=[10]
LR=[9]
rewardAgingFactor=[3, 5, 9]
spatialReward=[3, 5, 9]
sdn=["trace", "source", "destination"]
dataset=[1]
trace=[1, 2, 3]
NN=[1]
LTI=[100]
RTI=[10]

#numberofflows=[10, 20, 30]
#table_size=[100, 50, 200]
#lfu=[10]
#agingfactor=[50]
#speculativesdn=["speculativereactive"]
#seed=[3, 5, 7]
#epsilon=[1]
#gamma=[9]
#target_replace_iter=[100]
#memory_capacity=[10]
#LR=[9]
#rewardAgingFactor=[3, 5, 9]
#spatialReward=[3, 5, 9]
#sdn=["trace", "source", "destination"]
#dataset=[2]
#trace=[1, 2, 3]
#NN=[1]
#LTI=[1]
#RTI=[.5]


str_=("python3 SpeculativeSDN_.py --numberofFlowsPerAgent={0} --tablesize={1} --LFUTimeInterval={2} --agingfactor={3} --speculativesdn={4} --seed={5} --epsilon={6} --gamma={7} --target_replace_iter={8} --memory_capacity={9} --LR={10} --rewardAgingFactor={11} --spatialReward={12} --sdn={13} --dataset={14} --trace={15} --NN={16} --LTI={17} --RTI={18} >> results/{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}_{12}_{13}_{14}_{15}_{16}_{17}_{18}.txt \n")
n=0
for a1 in range(len(numberofflows)):
 for a2 in range(len(table_size)):
  for a3 in range(len(lfu)):
   for a4 in range(len(agingfactor)):
    for a5 in range(len(speculativesdn)):
     for a6 in range(len(seed)):
      for a7 in range(len(epsilon)):
       for a8 in range(len(gamma)):
        for a9 in range(len(target_replace_iter)):
         for a10 in range(len(memory_capacity)):
          for a11 in range(len(LR)):
           for a12 in range(len(rewardAgingFactor)):
            for a13 in range(len(spatialReward)):
             for a14 in range(len(sdn)):
              for a15 in range(len(dataset)):
               for a16 in range(len(trace)):
                for a17 in range(len(NN)):
                 for a18 in range(len(LTI)):
                  for a19 in range(len(RTI)):
                          str=str_.format(numberofflows[a1], table_size[a2], lfu[a3], agingfactor[a4], speculativesdn[a5], seed[a6], epsilon[a7], gamma[a8], target_replace_iter[a9], memory_capacity[a10], LR[a11], rewardAgingFactor[a12], spatialReward[a13], sdn[a14], dataset[a15], trace[a16], NN[a17], LTI[a18], RTI[a19])
#                          with open("output_sdn.txt", "a") as text_file:
#                                text_file.write(str)
                          if n<70:
                            with open("run1.txt", "a") as text_file:
                                text_file.write(str)
                            print(100)
                          if n>=70 and n<140:
                            with open("run2.txt", "a") as text_file:
                                text_file.write(str)
                            print(200)
                          if n>=140 and n<210:
                            with open("run3.txt", "a") as text_file:
                                text_file.write(str)
                            print(300)
                          if n>=210 and n<280:
                            with open("run4.txt", "a") as text_file:
                                text_file.write(str)
                            print(400)
                          if n>=280 and n<350:
                            with open("run5.txt", "a") as text_file:
                                text_file.write(str)
                            print(500)
                          if n>=350 and n<420:
                            with open("run6.txt", "a") as text_file:
                                text_file.write(str)
                            print(600)
                          if n>=420 and n<490:
                            with open("run7.txt", "a") as text_file:
                                text_file.write(str)
                            print(700)
#                          if n>700 and n<=800:
#                            with open("output_8.txt", "a") as text_file:
#                                text_file.write(str)
#                            print(800)
                          n+=1
                          print(n)
#files= os.listdir("sdn_/")
#file=[]
#print(files)
#for k in range(len(files)):
#    str=files[k]
#    str=str[:-4]
#    file.append(str.split('_'))
#print(file)
#for k in range(len(files)):
#    with open(files[k]) as sdn:
##        lines= [line.rstrip() for line in file]
#        for line in sdn:
#                lines=line.rstrip()
#                print(lines)
##                print(line)
##    print(line[0])
##    print(line[1])
##                for k in range(len(file)):
##        print(sdn)
#                file[k].append(lines)
##    file.append(line[1])
#                print(file)
#wb=load_workbook("sdn_.xlsx")
#ws=wb.worksheets[0]
#for row in file:
#    ws.append(row)
#wb.save("sdn_.xlsx")
