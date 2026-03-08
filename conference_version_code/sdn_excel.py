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
#table_size=[30]
#lfu=[3000]
#agingfactor=[99]
#speculativesdn=["speculative"]
#seed=[3]
#epsilon=[1]
#gamma=[9]
#target_replace_iter=[100]
#memory_capacity=[10]
#LR=[9]
#rewardAgingFactor=[9]
#spatialReward=[9]
#sdn=[9]
#dataset=[9]
#trace=[9]
#NN=[9]
#LTI=[9]
#RTI=[9]
#str=("python3 SpeculativeSDN_trace-based1.py --numberofFlowsPerAgent={0} --tablesize={1} --LFUTimeInterval={2} --agingfactor={3} --speculativesdn={4} --seed={5} --epsilon={6} --gamma={7} --target_replace_iter={8} --memory_capacity={9} --LR={10} --rewardAgingFactor={11} --spatialReward={12} --sdn={13} --dataset={14} --trace={15} --NN={16} --LTI={17} --RTI={18} >> results/{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}_{12}_{13}_{14}_{15}_{16}_{17}_{18}.txt \n")
#for a in range(len(numberofflows)):
# for c in range(len(table_size)):
#  for g in range(len(lfu)):
#   for k in range(len(agingfactor)):
#    for l in range(len(speculativesdn)):
#     for b in range(len(seed)):
#      for u in range(len(epsilon)):
#       for e in range(len(gamma)):
#        for r in range(len(target_replace_iter)):
#         for q in range(len(memory_capacity)):
#          for f in range(len(LR)):
#           for p in range(len(rewardAgingFactor)):
#            for w in range(len(spatialReward)):
#             for y in range(len(sdn)):
#              for u in range(len(dataset)):
#               for z in range(len(trace)):
#                for r in range(len(NN)):
#                 for q in range(len(LTI)):
#                  for f in range(len(RTI)):
#                          str=str.format(numberofflows[a], table_size[c], lfu[g], agingfactor[k], speculativesdn[l], seed[b], epsilon[u], gamma[e], target_replace_iter[r], memory_capacity[q], LR[f], rewardAgingFactor[k], spatialReward[l], sdn[b], dataset[u], trace[], NN[], LTI[], RTI[])
#                          with open("output.txt", "a") as text_file:
#                            text_file.write(str)
          


files= os.listdir("sdn/")
file=[]
print(files)
for k in range(len(files)):
    str=files[k]
    str=str[:-4]
    file.append(str.split('_'))
print(file)
for k in range(len(files)):
    with open(files[k]) as sdn:
#        lines= [line.rstrip() for line in file]
        for line in sdn:
                lines=line.rstrip()
                print(lines)
#                print(line)
#    print(line[0])
#    print(line[1])
#                for k in range(len(file)):
#        print(sdn)
                file[k].append(lines)
#    file.append(line[1])
                print(file)
wb=load_workbook("sdn_.xlsx")
ws=wb.worksheets[0]
for row in file:
    ws.append(row)
wb.save("sdn_.xlsx")
