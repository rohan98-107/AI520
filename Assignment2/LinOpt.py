from MineSweeper import *
from Agent import *
from LinAlg import *
from BruteForceAgent import *
from collections import deque
from copy import deepcopy
import sys
import time
import numpy as np
import pandas as pd

#Simplex Functions

def enteringVar(tableau):
	obj = tableau[-1,:-1]
	col = obj.argmin()
	return col

def departingVar(tableau,col):
	rhs = tableau[:-1,-1]
	c = tableau[:-1,col]
	temp = np.array([0] * len(rhs))
	for i in range(0,len(rhs)):
		if c[i] != 0 and rhs[i]/c[i] >= 0:
			temp[i] = rhs[i]/c[i]
		else:
			temp[i] = 100000
	row = temp.argmin()
	return row

def checkOptimal(tableau):
	obj = tableau[-1,:-1]
	if min(obj) >= 0:
		return True
	else:
		return False

def checkNoSolution(tableau):
    obj = tableau[-1, :-1]
    for e in range(len(obj)):
        if e < 0:
            for i in range(len(tableau)-1):
                if tableau[i][e] > 0:
                    return False
    return True

def rowReduce(tableau,row,col):

    m = len(tableau)-1
    for i in range(0,m+1):
        if i != row:
            c = -1*(tableau[i][col]/tableau[row][col])
            tableau[i] += c*tableau[row]

    tableau[row] *= (1/tableau[row][col])
    return tableau

def simplex(tableau):
    while (not checkOptimal(tableau)):
        e = enteringVar(tableau)
        d = departingVar(tableau,e)
        tableau = rowReduce(tableau,d,e)
        if checkNoSolution(tableau):
            return None
    return tableau, tableau[-1,-1]

# cautious matrix: has inequalities [row] <= clue-numMineNbrs
def simplexCautious(matrix):
    # concatenate I_n to right (without last col) for slacking vars to resolve <= into =
    tableau = np.concatenate((matrix[:, :-1], np.identity(matrix.shape[0])), axis = 1)
    # concatenate back the last col
    tableau = np.column_stack((tableau, matrix[:, -1]))
    # add row for objective function: all -1s except for slacking vars
    obj_row = np.concatenate(([-1]*(matrix.shape[1] - 1), [0]*(matrix.shape[0] + 1)))
    tableau = np.row_stack((tableau, obj_row))

    # simplex as usual
    tableau = simplex(tableau)

    return tableau

# optimistic matrix: has inequalities [row] >= clue-numMineNbrs
def simplexOptimistic(matrix):
    # Phase I: solve for artificial variables
    # concatenate -I_n to right (without last col) for slacking vars to resolve >= into =
    tableau = np.concatenate((matrix[:, :-1], -1*np.identity(matrix.shape[0])), axis=1)
    # concatenate I_n to right (without last col) for artificial vars (to get init basic fsbl soln)
    tableau = np.column_stack((tableau, np.identity(matrix.shape[0])))
    # concatenate back the last col
    tableau = np.column_stack((tableau, matrix[:, -1]))
    # add obj row: max -(all artificial vars)
    obj_row = np.concatenate(([0]*(matrix.shape[1]-1+matrix.shape[0]), [1]*matrix.shape[0], [0]))
    tableau = np.row_stack((tableau, obj_row))
    # solve Phase I with simplex
    tableau = simplex(tableau)

    # Phase I fail: if z != 0, then no optimal solution exists
    if tableau is None:
        return None

    # Phase II: update and simplex
    tableau = tableau[0]
    # delete artificial vars' columns
    tableau = np.column_stack((tableau[:, :(matrix.shape[1]-1 + matrix.shape[0])], tableau[:, -1]))
    # update objective function: all -1s except for slacking vars
    obj_row = np.concatenate(([-1]*(matrix.shape[1] - 1), [0]*(matrix.shape[0] + 1)))
    tableau[-1] = obj_row
    # simplex as usual
    tableau = simplex(tableau)

    return tableau

# inherit from brute force agent (and lin alg agent)
# override enqueueBestPossibleCells to use lin opt instead of simple gaussian elim
# question: what to do when no optimal sol'n is found?
class lin_opt_agent(brute_force_agent):
    def __init__(self, game, useMineCount, order, uncertaintyType):
        brute_force_agent.__init__(self, game, useMineCount, order)
        self.uncertaintyType = uncertaintyType

    # override LinAlg.py method to use linear optimization to solve system of constraints
    # with objective function: Max z = sum(x_i) where x_i is all possible mines detected
    # note: max detected mines <=> min detonated mines
    # return a random x,y from the game's currently hidden cells iff no new safe cells can be recovered
    def enqueueBestPossibleCells(self, recrunch, q):
        safesFound = False;
        dim = self.game.dim
        if recrunch:
            information_cells = []
            hidden_cells = []
            for x in range(dim):
                for y in range(dim):
                    if self.playerKnowledge[x][y] == HIDDEN:
                        hidden_cells.append((x,y))
                    # if it's not safe, then there's no clue we can use
                    if self.playerKnowledge[x][y] == SAFE:
                        clue = self.game.board[x][y]
                        numSafeNbrs, numMineNbrs, numHiddenNbrs, numTotalNbrs = self.getCellNeighborData(x, y)

                        # if we have hidden neighbors, then add to list of cells to use
                        # in system of equations
                        if numHiddenNbrs != 0:
                            information_cells.append((x,y,clue,numMineNbrs, numHiddenNbrs))

            #if we have information, solve the system
            if information_cells:
                # create matrix of zeroes to start
                # each row will represent information provided by one clue
                # we take each hidden neighbor (nx,ny) of the cell with the clue
                # and then set row[dim*nx + ny] = 1, and the last value in row
                # equal to clue - already discovered mines. This encodes a
                # boolean of whether a spot contributes to a clue, and when we solve
                # the equation for the dim*dim values, constraining each to 1 or 0
                # we can know if (x,y) is a mine iff the variable corresponding to
                # the (dim * x + y)th column is 1
                matrix = np.zeros((len(information_cells) + (1 if self.useMineCount else 0), dim*dim + 1), dtype = 'float64')
                for i in range(len(information_cells)):
                    x, y, clue, mineNeighbors, numHiddenNbrs = information_cells[i]
                    hidden_nbrs = self.get_hidden_neighbors(x,y)
                    for nx, ny in hidden_nbrs:
                        matrix[i][dim * nx + ny] = 1
                    matrix[i][dim * dim] = clue - mineNeighbors
                if self.useMineCount:
                    for x,y in hidden_cells:
                        matrix[len(information_cells), dim * x + y] = 1
                    matrix[len(information_cells),dim*dim] = self.game.num_mines-self.numFlaggedMines-self.numDetonatedMines


                if self.uncertaintyType == 'cautious':
                    matrix = simplexCautious(matrix)
                elif self.uncertaintyType == 'optimistic':
                    matrix = simplexOptimistic(matrix)
                else:
                    rref(matrix)

                if matrix is not None:
                    #keeping track of flags here is purely for debugging purposes
                    flags = []
                    for row in matrix:
                        #
                        positives = []
                        negatives = []
                        #scale the row for code reuseability
                        if row[-1] < 0 :
                            row *= -1
                        #for each variable
                        for i in range(dim*dim):
                            #first get the cell it corresponds to
                            x = i // dim
                            y = i % dim
                            # if the coeffecient in this row is positive put it in
                            # positives. if negative then negatives, if 0 then we don't care
                            if row[i] >0:
                                positives.append((x,y, row[i]))
                            elif row[i] < 0:
                                negatives.append((x,y, row[i]))

                        #if the row has no information move on
                        if len(positives) == len(negatives) == 0:
                            continue

                        #if the row sums to 0, we can potentially find definitively safe cells
                        if row[-1] == 0:
                            # if we have both positive and negative coeffecients, the variables could all be 0
                            # or there could be some mines, the sum of the coeffecients of the mines is 0
                            # which can happen since we have positive and negative coeffecients
                            # so we don't know anything definitive (multiple combinations satisfy the equation)
                            if len(positives) >0 and len(negatives) > 0:
                                continue
                            else:
                                # however if only one list is populated then everything
                                # in the list must have value 0 for equation to hold (so everything is safe)
                                # we can do positives + negatives since exactly one has values
                                for x,y,_ in positives + negatives:
                                    # if self.game.board[x][y] == MINE:
                                    assert self.game.board[x][y] != MINE
                                    self.playerKnowledge[x][y] = SAFE
                                    q.append((x,y)) #add the safe cell to the queue to visit next
                                    safesFound = True
                        else:
                            # if the row sums to a positive then since each variable is only
                            # 0 or 1 (ie. can not be negative), if the sum of the
                            # positive coeffecients exactly matches the last value of the row
                            # (the other side of the equation), then all the variables with
                            # positive coeffecients must be 1, and so we flag them. The sum can never be less since
                            # each variable is at most 1, and if it is greater, then multiple combinations
                            # of the variables in the positive and negative lists can be mines
                            if sum([coeffecient for x,y,coeffecient in positives]) == row[-1]:
                                for x,y,_ in positives:
                                    if self.playerKnowledge[x,y] == MINE:
                                        continue
                                    assert self.game.board[x][y] == MINE
                                    self.playerKnowledge[x][y] = MINE
                                    flags.append((x,y))
                                    self.numFlaggedMines += 1



        # if lin alg didn't give us anything then pick a random cell
        if (recrunch is False) or (safesFound is False):
            if recrunch is True and self.logging:
                print('\tRe-processing did not find new safe cells; proceeding to randomly select hidden cell.\n')
            tuple = self.probability_method()
            if tuple:
                q.append(tuple)

# utility function to run game, save initial & solved boards, and print play-by-play to log txt file
# game metrics: mine safe detection rate and solve time calculated here; outputted to log
def linOptGameDriver(dim, density, logFileName, uncertaintyType):
    #sys.stdout = open('{}_log.txt'.format(logFileName), 'w')

    num_mines = int(density*(dim**2))

    print('\n\n***** GAME STARTING *****\n\n{} by {} board with {} mines\n\nSolving with LINEAR OPTIMIZATION strategy\n\nUncertainty: {}'\
          .format(dim, dim, num_mines, uncertaintyType))
    order = random.shuffle([i for i in range(dim**2)])
    game = MineSweeper(dim, num_mines)
    #game.saveBoard('{}_init_board'.format(logFileName))

    agent = lin_opt_agent(game, True, order, uncertaintyType)
    #agent.enableLogging()
    agent.solve()

    print('\n\n***** GAME OVER *****\n\nGame ended in {} seconds\n\nSafely detected (without detonating) {}% of mines'\
          .format(agent.totalSolveTime, agent.mineFlagRate*100))

    #agent.savePlayerKnowledge('{}_solved_board'.format(logFileName))




'''
t = [[1,0,0,0,0,0,0],
     [1,1,1,0,1,0,5],
     [1,0,1,0,0,0,2]]

t = np.array(t,dtype=float)

# change to simplexOptimistic(t) or simplexCautious(t)
print(simplexCautious(t))
'''



'''
linOptGameDriver(20, 0.2, 'test', 'cautious')
'''

