# Assignment 3 - Search & Destroy
# Rohan Rele, Alex Eng, Aakash Raman, Adarsh Patel

import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

dirs = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (-1, 1), (1, -1)]

TARGET = 1
EMPTY = 0

FLAT = 0.1
HILLY = 0.3
FOREST = 0.7
MAZE = 0.9

class landscape:

    dim = 0
    landscape = [[]]
    target = []

    def __init__(self, dim):
        self.dim = dim
        self.landscape = [[self.setCell() for _ in range(self.dim)] for _ in range(self.dim)]

        target_x = random.randint(0, dim - 1)
        target_y = random.randint(0, dim - 1)
        self.landscape[target_x][target_y] = [TARGET, self.flattenFNrate(self.landscape[target_x][target_y])]
        self.target = [target_x, target_y]


    def setCell(self):
        x = random.randint(1, 100)
        if x <= 20:
            return (EMPTY, FLAT)
        elif 20 < x <= 50:
            return (EMPTY, HILLY)
        elif 50 < x <= 80:
            return (EMPTY, FOREST)
        else:
            return (EMPTY, MAZE)

    # moves target to one of its neighbors (at random) and returns a terrain type the new target is NOT (at random)
    def moveTarget(self):
        x = self.target[0]; y = self.target[1]
        self.landscape[x][y] = (EMPTY, self.flattenFNrate(self.landscape[x][y]))

        nbrs = []
        for dx, dy in dirs:
            if 0 <= x + dx < self.dim and 0 <= y + dy < self.dim:
                nbrs.append((x + dx, y + dy))

        self.target = random.choice(nbrs); new_x = self.target[0]; new_y = self.target[1]
        self.landscape[new_x][new_y] = (TARGET, self.flattenFNrate(self.landscape[new_x][new_y]))

        terrains = {FLAT, HILLY, FOREST, MAZE}
        new_terrain = {(self.landscape[new_x][new_y])[1]}
        return random.choice(list(terrains.difference(new_terrain)))


    def flattenStatus(self, tup):
        return tup[0]

    def flattenFNrate(self, tup):
        return tup[1]

    def searchCell(self, tup):
        if tup[0] == EMPTY:
            return False
        else:
            p = 1 - tup[1]
            if random.uniform(0, 1) < p:
                return True
            else:
                return False

    def printLandscape(self):
        size = len(self.landscape)
        temp = [[self.flattenFNrate(j) for j in i] for i in self.landscape]
        plt.figure(figsize=(size/10, size/10), dpi=500)
        sns.heatmap(temp, mask=(temp==-1), vmin=0, vmax=1, linewidth=0.01, linecolor='black',
                        annot=False, annot_kws={"size": 15/size},
                        square=True, cbar=False,
                        xticklabels=False,
                        yticklabels=False,
                        cmap='Greens')
        plt.title('Landscape: Target at ({}, {})'.format(self.target[0], self.target[1]), fontsize=size/15, ha='center')
        plt.show()

test = landscape(50)
test.printLandscape()

