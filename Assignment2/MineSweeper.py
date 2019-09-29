# Assignment 2 - Minesweeper
# Rohan Rele, Alex Eng & Aakash Raman
# Generation/Setup code

import random
import numpy as np

dirs = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (-1, 1), (1, -1)]

class MineSweeper:

    dim = 0
    board = [[]]
    playerMoves = [[]]
    gameOver = False
    success = False

    def __init__(self, dim, num_mines):
        self.dim = dim
        self.playerMoves = [[False for x in range(dim)] for y in range(dim)]

        board = [[0 for x in range(dim)] for y in range(dim)]

        for n in range(num_mines):
            x = random.randint(0, dim-1)
            y = random.randint(0, dim-1)

            if board[x][y] != -1:
                board[x][y] = -1
            else:
                n -= 1

        temp = np.array(board)

        coords = zip(*np.where(temp == -1))
        for x, y in coords:
            for i, j in dirs:
                if 0 <= x + i < dim and 0 <= y + j < dim and temp[x+i][y+j] != -1:
                    temp[x + i][y + j] += 1
                else:
                    continue

        self.board = temp.tolist()

    def revealSquare(self,x,y):
        if self.board[x][y] == -1:
            print("Boom!")
            self.gameOver = True

        self.playerMoves[x][y] = True
        for i, j in dirs:
            if self.playerMoves[x+i][y+j] == False and self.board[x+i][y+j] == 0:
                self.playerMoves[x+i][y+j] = True
