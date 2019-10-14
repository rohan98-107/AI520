import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dirs = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (-1, 1), (1, -1)]

MINE = -6
DETONATED = -8
HIDDEN = -1
SAFE = 0

class MineSweeper:
    dim = 0
    num_mines = 0
    board = [[]]
    gameOver = False
    success = False

    def __init__(self, dim, num_mines):
        self.dim = dim

        if num_mines < 0 or num_mines > dim**2:
            print("Invalid number of mines specified --> set to 0")
            num_mines = 0
        self.num_mines = num_mines

        board = [[0 for _ in range(dim)] for _ in range(dim)]

        for n in range(num_mines):
            x = random.randint(0, dim-1)
            y = random.randint(0, dim-1)
            while board[x][y] == MINE:
                x = random.randint(0, dim-1)
                y = random.randint(0, dim-1)
            board[x][y] = MINE

        temp = np.array(board)

        coords = zip(*np.where(temp == MINE))
        for x, y in coords:
            for i, j in dirs:
                if 0 <= x + i < dim and 0 <= y + j < dim and temp[x+i][y+j] != MINE:
                    temp[x + i][y + j] += 1
                else:
                    continue

        self.board = temp.tolist()


    def isMine(self,x,y):
        if self.board[x][y] == MINE:
            print("Boom!")
            self.gameOver = True

    def keepScore():
        '''
        fill this in
        '''
        return

    def printBoard(self):
        # Color Legend:
        # dark blue (-8): mine
        # white (0): no mines
        # light red (1) --> dark red (8): 1-8 mines

        imgsize = int(self.dim / 10)
        fontsize = 12.5 / (self.dim)
        dpi = 500

        plt.figure(figsize=(imgsize, imgsize), dpi=dpi)
        sns.heatmap(self.board, vmin=-8, vmax=8, linewidth=0.01, linecolor='lightgray',
                    annot=True, annot_kws={"size": fontsize},
                    square=True, cbar=False,
                    xticklabels=False,
                    yticklabels=False,
                    cmap='seismic')
        plt.show()

    def saveBoard(self, filename):
        imgsize = int(self.dim / 10)
        fontsize = 75 / (self.dim)
        dpi = 1000

        plt.figure(figsize=(imgsize, imgsize), dpi=dpi)
        sns.heatmap(self.board, vmin=-8, vmax=8, linewidth=0.01, linecolor='lightgray',
                    annot=True, annot_kws={"size": fontsize},
                    square=True, cbar=False,
                    xticklabels=False,
                    yticklabels=False,
                    cmap='seismic')
        plt.savefig('{}.png'.format(filename), dpi=dpi)


# test board visualizations
'''
dim = 30
density = 0.15
num_mines = int(density*(dim**2))
game = MineSweeper(dim, num_mines)
game.printBoard()
#game.saveBoard('test')
'''
