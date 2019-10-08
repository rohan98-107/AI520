# Assignment 2 - Minesweeper
# Rohan Rele, Alex Eng, Aakash Raman, and Adarsh Patel
# Baseline agent code

from MineSweeper import *
import queue

def solveBaseline(game):
    q = queue.Queue(maxsize = -1)

    # randomly add a hidden cell to Queue
    q.put(addRandomCell(game))

    while not q.empty():

        (x, y) = q.get()

        oldQueueSize = q.qsize()

        if game.board[x][y] == MINE:
            game.playerKnowledge[x][y] = DETONATED
            print('\nBOOM! Mine detonated at {}, {}\n\n'.format(x, y))
            continue

        game.playerKnowledge[x][y] = SAFE
        clue = game.board[x][y]
        print('\nCell {}, {} safely revealed with clue: {}'.format(x, y, clue))
        numSafeNbrs, numMineNbrs, numHiddenNbrs, numTotalNbrs = getCellNeighborData(game, x, y)
        print('\t# safe neighbors: {}\n\t# mine neighbors: {}\n\t# hidden neighbors: {}\n\t# total neighbors: {}\n\n'\
              .format(numSafeNbrs, numMineNbrs, numHiddenNbrs, numTotalNbrs))

        if numHiddenNbrs == 0:
            pass
        elif (clue - numMineNbrs) == numHiddenNbrs:
            print('All neighbors of {}, {} must be mines.\n'.format(x, y))
            for i, j in dirs:
                if 0 <= x + i < game.dim and 0 <= y + j < game.dim:
                    game.playerKnowledge[x+i][y+j] = MINE
                    print('Neighbor {}, {} flagged as a mine.\n'.format(x+i, y+j))
        elif (8 - clue - numSafeNbrs) == numHiddenNbrs:
            print('All neighbors of {}, {} must be safe.\n'.format(x, y))
            for i, j in dirs:
                if 0 <= x + i < game.dim and 0 <= y + j < game.dim:
                    game.playerKnowledge[x+i][y+j] = SAFE
                    q.put((x+i, y+j))
                    print('Neighbor {}, {} flagged as safe and enqueued for next visitation.\n'.format(x + i, y + j))

        if q.qsize() == oldQueueSize:
            if numHiddenCells(game) != 0:
                q.put(addRandomCell(game))
                print('Revealing cell {}, {} led to no conclusive next move. Will randomly reveal a cell next.\n'.format(x, y))

    return game


# utility function to grab relevant metadata given game, (x,y) coordinates
def getCellNeighborData(game, x, y):
    numSafeNbrs = 0;    numMineNbrs = 0;    numHiddenNbrs = 0;  numTotalNbrs = 0
    for i, j in dirs:
        if 0 <= x + i < game.dim and 0 <= y + j < game.dim:
            numTotalNbrs += 1
            if (game.playerKnowledge[x+i][y+j] == MINE) or (game.playerKnowledge[x+i][y+j] == DETONATED):
                numMineNbrs += 1
            elif game.playerKnowledge[x+i][y+j] == SAFE:
                numSafeNbrs += 1
            elif game.playerKnowledge[x+i][y+j] == HIDDEN:
                numHiddenNbrs += 1
    return numSafeNbrs, numMineNbrs, numHiddenNbrs, numTotalNbrs

# utility function to return a random x,y from the game's currently hidden cells
def addRandomCell(game):
    (x, y) = np.where(game.playerKnowledge == HIDDEN)
    i = np.random.randint(len(x))
    return (x[i], y[i])

def numHiddenCells(game):
    return (game.playerKnowledge == HIDDEN).sum()

# test game
dim = 20
density = 0.10
num_mines = int(density*(dim**2))

game = MineSweeper(dim, num_mines)
game.saveBoard('test')

solvedGame = solveBaseline(game)

num_undetonated_mines = (solvedGame.playerKnowledge == MINE).sum()

print('\n\nGAME OVER\n\nSuccessful undetonated mine detection rate:{}'.format(num_undetonated_mines / num_mines))
