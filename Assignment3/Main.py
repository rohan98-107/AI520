# Assignment 3 - Search & Destroy
# Rohan Rele, Alex Eng, Aakash Raman, Adarsh Patel

import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

# We should eventually organize this into an Agent class, Landscape class etc.
# But for now I'm just going to dump everything here

TARGET = 1
EMPTY = 0

FLAT = (EMPTY,0.1)
HILLY = (EMPTY,0.3)
FOREST = (EMPTY,0.7)
MAZE = (EMPTY,0.9)

def setCell():
    x = random.randint(1,100)
    if x <= 20:
        return FLAT
    elif 20 < x <= 50:
        return HILLY
    elif 50 < x <= 80:
        return FOREST
    else:
        return MAZE

def generateLandscape(dim):
    landscape = [[setCell() for i in range(dim)] for j in range(dim)]
    return landscape

def flattenStatus(tup):
    return tup[1]
def flattenFNrate(tup):
    return tup[0]

def printLandscape(ls):
    size = len(ls)
    temp = [[flattenStatus(j) for j in i] for i in ls]
    plt.figure(figsize=(size/20, size/20), dpi=500)
    sns.heatmap(temp, vmin=0, vmax=1, linewidth=0.01, linecolor='black',
                    annot=False, annot_kws={"size": 15/size},
                    square=True, cbar=False,
                    xticklabels=False,
                    yticklabels=False,
                    cmap='Greens')
    plt.show()

def searchCell(tup):
    if tup[0] == EMPTY:
        return False
    else:
        p = 1 - tup[1]
        if random.uniform(0, 1) < p:
            return True
        else:
            return False
