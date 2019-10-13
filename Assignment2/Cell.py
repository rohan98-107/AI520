from copy import deepcopy
from MineSweeper import *

class Cell:
    def __init__(self):
        self.value = HIDDEN
        self.expectedval = 0.0
        self.probability = 1.0

class ProbAgent:
    def __init__(self,gameObj):

        self.game = gameObj

        #playerKnowledge
        self.playerKnowledge = nodes = [[Cell() for j in range(gameObj.dim)] for i in range(gameObj.dim)]

        # keep track of these throughout game to yield final mineFlagRate
        self.numFlaggedMines = 0
        self.numDetonatedMines = 0

        # these are the two metrics we can judge performance with
        self.mineFlagRate = 0
        self.totalSolveTime = 0

    def revealSquare(self,x,y):
        if not 0 <= x < self.game.dim or not 0 <= y < self.game.dim:
            print("Invalid")
            return
        if self.game.isMine():
            print("Detonated mine")
            self.numDetonatedMines += 1
            self.playerKnowledge[i][j].value = DETONATED

        self.playerKnowledge[x][y].value = self.game.board[x][y]

    def makeFirstMove(self):
        dim = len(self.game.board)-1
        corners = [(0,0),(0,dim),(dim,0),(dim,dim)]
        i,j = random.choice(corners)
        self.revealSquare(i,j)
        return i,j

    def guess(self):
        x = 0
        y = 0
        while self.playerKnowledge[x][y].value != HIDDEN:
            x = random.randint(0, self.game.dim-1)
            y = random.randint(0, self.game.dim-1)

        self.revealSquare(i,j)

    def solveGame(self):

        x,y = self.makeFirstMove()
        density = self.game.num_mines/(self.game.dim**2)
        while self.playerKnowledge[x][y].value != HIDDEN:

            for i,j in dirs:
                exp = 1
                if 0 <= x + i < dim and 0 <= y + j < dim and temp[x+i][y+j] != MINE:
                    exp += 1

                self.playerKnowledge[x][y].probability = (1-density)**exp

                '''
                I scrapped the conditional probability tables...
                ...they only worked for the first few moves and the rest
                ...ended up just being guessing.
                '''

        return
