# Assignment 3 - Search & Destroy
# Rohan Rele, Alex Eng, Aakash Raman, Adarsh Patel

import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

from Landscape import *

class agent:
    # track total number of cell searches (and movements for movement-restricted agent)
    num_actions = 0

    def __init__(self, landscape):
        self.ls = landscape
        d = self.ls.dim
        self.knowledge = [[agentCell() for j in range(d)] for i in range(d)]
        for i in range(d):
            for j in range(d):
                self.knowledge[i][j].setBelief(1/(d**2))

    def printBelief(self):
        size = len(self.knowledge)
        plt.figure(figsize=(size/10, size/10), dpi=500)
        belief = [[self.knowledge[i][j].getBelief() for j in range(size)] for i in range(size)]
        sns.heatmap(belief, vmin=0, vmax=1, linewidth=0.01, linecolor='black',
                        annot=True, annot_kws={"size": 15/size},
                        square=True, cbar=False,
                        xticklabels=False,
                        yticklabels=False,
                        cmap='Blues')
        plt.title('Agent Belief', fontsize=size / 15, ha='center')
        plt.show()

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
        if self.searchCell(self.ls.landscape[x][y]):
            for i in range(self.ls.dim):
                for j in range(self.ls.dim):
                    self.knowledge[i][j].setBelief(0)
            self.knowledge[x][y].setBelief(1)
        else:
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

test_ls = landscape(5)
test_ls.printLandscape()
test_agent = agent(test_ls)
test_agent.printBelief()
test_agent.updateBelief(2,3)
test_agent.printBelief()
