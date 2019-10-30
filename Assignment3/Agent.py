# Assignment 3 - Search & Destroy
# Rohan Rele, Alex Eng, Aakash Raman, Adarsh Patel

import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

from Landscape import *

class agent:

    knowledge = [[]]

    # track total number of cell searches (and movements for movement-restricted agent)
    num_actions = 0

    def __init__(self, landscape):
        self.knowledge = [[(1/(landscape.dim**2)) for _ in range(landscape.dim)] for _ in range(landscape.dim)]

    def printKnowledge(self):
        size = len(self.knowledge)
        plt.figure(figsize=(size/10, size/10), dpi=500)
        sns.heatmap(self.knowledge, vmin=0, vmax=1, linewidth=0.01, linecolor='black',
                        annot=False, annot_kws={"size": 15/size},
                        square=True, cbar=False,
                        xticklabels=False,
                        yticklabels=False,
                        cmap='Greens')
        plt.title('Agent Knowledge', fontsize=size / 15, ha='center')
        plt.show()

ls = landscape(50)
ls.printLandscape()

ag = agent(ls)
ag.printKnowledge()