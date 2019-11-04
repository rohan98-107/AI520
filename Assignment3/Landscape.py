# Assignment 3 - Search & Destroy
# Rohan Rele, Alex Eng, Aakash Raman, Adarsh Patel

import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

dirs = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (-1, 1), (1, -1)]

from Cell import *

class landscape:

    dim = 0
    landscape = [[]]
    target_x = 0
    target_y = 0

    def __init__(self, dim):
        self.dim = dim
        self.landscape = [[landCell() for _ in range(self.dim)] for _ in range(self.dim)]

        target_x = random.randint(0, dim - 1)
        target_y = random.randint(0, dim - 1)
        self.landscape[target_x][target_y].target = PRESENT
        self.target_x = target_x
        self.target_y = target_y

    # moves target to one of its neighbors (at random) and returns a terrain type the new target is NOT (at random)
    def moveTarget(self):
        x = self.target_x; y = self.target_y
        self.landscape[x][y].target = ABSENT

        nbrs = []
        for dx, dy in dirs:
            if 0 <= x + dx < self.dim and 0 <= y + dy < self.dim:
                nbrs.append((x + dx, y + dy))

        new_nbr = random.choice(nbrs)
        self.target_x = new_nbr[0]; self.target_y = new_nbr[1]
        self.landscape[self.target_x][self.target_y].target = PRESENT

        terrains = {FLAT, HILLY, FOREST, MAZE}
        new_terrain = {(self.landscape[new_x][new_y])[1]}
        return random.choice(list(terrains.difference(new_terrain)))

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
