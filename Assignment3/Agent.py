# Assignment 3 - Search & Destroy
# Rohan Rele, Alex Eng, Aakash Raman, Adarsh Patel

import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import math
sns.set(font_scale=1)
from copy import deepcopy

from Landscape import *

class agent:
    # track total number of cell searches (and movements for movement-restricted agent)
    num_actions = 0

    def __init__(self, landscape, rule, start_i = -1, start_j = -1):
        self.ls = landscape
        d = self.ls.dim

        if rule == 1 or rule == 2 or rule == 3 or rule == 4:
            self.rule = rule
        else:
            print("Invalid rule, set to 1 by default")
            self.rule = 1

        self.knowledge = [[agentCell() for j in range(d)] for i in range(d)]
        for i in range(d):
            for j in range(d):
                self.knowledge[i][j].setBelief(1/(d**2))
        self.i = random.randint(0, self.ls.dim-1) if start_i == -1 else start_i
        self.j = random.randint(0, self.ls.dim-1) if start_j == -1 else start_j

    def printBelief(self):
        size = len(self.knowledge)
        plt.figure(figsize=(size / 5, size / 5), dpi=500)
        belief = [[self.knowledge[i][j].getBelief() for j in range(size)] for i in range(size)]
        sns.heatmap(belief, vmin=0, vmax=1, linewidth=0.01, linecolor='black',
                    annot=True, annot_kws={"fontsize": 1},
                    square=True, cbar=False,
                    xticklabels=False,
                    yticklabels=False,
                    cmap='Blues')
        plt.title('Agent Belief', fontsize=size / 20, ha='center')
        plt.show()

    def saveBelief(self, filename):
        size = len(self.knowledge)
        plt.figure(figsize=(size / 5, size / 5), dpi=500)
        belief = [[self.knowledge[i][j].getBelief() for j in range(size)] for i in range(size)]
        sns.heatmap(belief, vmin=0, vmax=1, linewidth=0.01, linecolor='black',
                    annot=True, annot_kws={"fontsize": size / 20}, fmt='0.2g',
                    square=True, cbar=False,
                    xticklabels=False,
                    yticklabels=False,
                    cmap='Blues')
        plt.title('Agent Belief', fontsize=size / 5, ha='center')
        plt.savefig('./imgs/belief_{}.png'.format(filename), dpi=500)

    def searchCell(self,cell): #search a landCell
        if not cell.isTarget():
            return False
        else:
            p = 1 - cell.getTerrain()
            if random.uniform(0, 1) < p:
                return True
            else:
                return False

    def getVisited(self):
        n = self.ls.dim
        coords = []
        for x in range(n):
            for y in range(n):
                if self.knowledge[x][y].getStatus():
                    coords.append((x,y))
        return coords

    def probNotFound(self):
        n = self.ls.dim
        res = 0
        coords = self.getVisited()
        res = (n**2 - len(coords))/(n**2)
        for coord in coords:
            res += (self.ls.landscape[coord].getTerrain())*(self.knowledge[coord].getBelief())
        return res

    def probFound(self,x,y):
        return (1-self.ls.landscape[x][y].getTerrain())*self.knowledge[x][y].getBelief()

    def updateBelief(self,x,y):

        #P(H|E) = P(E|H)P(H)/P(E)
        #H: Target in cell
        #E: Target not found
        curr_belief = self.knowledge[x][y].getBelief()
        num = self.ls.landscape[x][y].getTerrain()*curr_belief
        denom = self.probNotFound()
        remainder = abs(curr_belief - (num/denom))
        self.knowledge[x][y].setBelief(num/denom)
        for i in range(self.ls.dim):
            for j in range(self.ls.dim):
                if i == x and j == y:
                    continue
                else:
                    temp = self.knowledge[i][j].getBelief()
                    self.knowledge[i][j].setBelief(temp + (temp*remainder)/(1-remainder))
        return self.knowledge

    def getMaxLikCell(self,i,j):
        if self.rule == 1:
            #get max i for P(Target in Cell i)
            belief = np.array([[self.knowledge[i][j].getBelief() for j in range(self.ls.dim)] for i in range(self.ls.dim)])
            return np.unravel_index(belief.argmax(),belief.shape)
        elif self.rule == 2:
            #get max i for P(Target FOUND in Cell i)
            belief = [[self.knowledge[i][j].getBelief()*(1-self.ls.landscape[i][j].getTerrain()) for j in range(self.ls.dim)] for i in range(self.ls.dim)]
            belief = np.array(belief)
            return np.unravel_index(belief.argmax(),belief.shape)
        elif self.rule == 3:
            belief = [[self.knowledge[i][j].getBelief()*(1-self.ls.landscape[i][j].getTerrain()) for j in range(self.ls.dim)] for i in range(self.ls.dim)]
            belief = np.array(belief)
            return np.unravel_index(belief.argmax(),belief.shape)
        else:
            pass

    def findTarget(self):
        i = self.i; j = self.j
        self.num_actions += 1
        while not self.searchCell(self.ls.landscape[i][j]):
            self.knowledge = self.updateBelief(i,j)
            next_i,next_j = self.getMaxLikCell(i,j)
            self.num_actions += 1
            if self.rule == 3 or self.rule == 4:
                self.num_actions += math.abs(i - next_i) + math.abs(j - next_j)
            i,j = next_i,next_j

        return (i,j)
