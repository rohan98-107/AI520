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
    for i in range(len(obj)):
        if obj[i] < 0:
            return i
    return None

def departingVar(tableau,col):
	rhs = tableau[:-1,-1]
	c = tableau[:-1,col]
	temp = np.array([0] * len(rhs))
	for i in range(0,len(rhs)):
		if c[i] != 0 and (rhs[i]/c[i]).all() >= 0:
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

def hasNoSolution(tableau):
    obj = tableau[-1, :-1]
    if np.sum(obj < 0) == 0:
        return False
    for e in range(len(obj)):
        if obj[e] < 0:
            for i in range(len(tableau)-1):
                if tableau[i][e] > 0:
                    return False
    return True

def rowReduce(tableau,row,col):

    if tableau[row][col] == 0:
        return None

    m = len(tableau)-1
    for i in range(0,m+1):
        if i != row:
            c = -1*(tableau[i][col]/tableau[row][col])
            tableau[i] += c*tableau[row]

    tableau[row] *= (1/tableau[row][col])
    return tableau

def makeBasic(tableau):
    for c in range(tableau.shape[1]):
        col = tableau[:, c]
        # if a std basis need to row reduce on the one
        if (np.sum(col[:-1] == 1) == 1) and (np.sum(col[:-1] == 0) == len(col[:-1])-1):
            pivotRow = np.where(col[:-1] == 1)[0][0]
            tableau = rowReduce(tableau, pivotRow, c)
            if tableau is None:
                return None
    return tableau

def simplex(tableau):
    i = 0

    while (not checkOptimal(tableau)):
        e = enteringVar(tableau)
        d = departingVar(tableau,e)
        tableau = rowReduce(tableau,d,e)
        if tableau is None or hasNoSolution(tableau):
            return None

        i+=1

        # hardcode stop if get stuck in degeneracy / bland's rule cycling issues
        if i > 20000:
            return None
    return tableau

# cautious matrix: has inequalities [row] <= clue-numMineNbrs
def simplexCautious(matrix):
    # concatenate I_n to right (without last col) for slacking vars to resolve <= into =
    tableau = np.column_stack((matrix[:, :-1], np.identity(matrix.shape[0])))
    # concatenate back the last col
    tableau = np.column_stack((tableau, matrix[:, -1]))
    # add row for objective function: all -1s except for slacking vars
    obj_row = np.concatenate(([-1]*(matrix.shape[1] - 1), [0]*(matrix.shape[0]+1)))
    tableau = np.row_stack((tableau, obj_row))

    # simplex as usual
    tableau = simplex(tableau)

    # if sol'n, cut out cols for slackvars and obj row (only returning row vals as in rref)
    if tableau is not None:
        tableau = np.column_stack((tableau[:, :matrix.shape[0]], tableau[:, -1]))[:-1, :]

    return tableau

# optimistic matrix: has inequalities [row] >= clue-numMineNbrs
def simplexOptimistic(matrix):
    # Phase I: solve for artificial variables
    # concatenate -I_n to right (without last col) for slacking vars to resolve >= into =
    tableau = np.column_stack((matrix[:, :-1], -1*np.identity(matrix.shape[0], dtype=int)))
    # concatenate I_n to right (without last col) for artificial vars (to get init basic fsbl soln)
    tableau = np.column_stack((tableau, np.identity(matrix.shape[0])))
    # concatenate back the last col
    tableau = np.column_stack((tableau, matrix[:, -1]))
    # add obj row: max -(all artificial vars)
    obj_row = np.concatenate(([0]*(matrix.shape[1]-1+matrix.shape[0]), [1]*matrix.shape[0], [0]))
    tableau = np.row_stack((tableau, obj_row))

    # row reduce obj row (art vars basic)
    tableau = makeBasic(tableau)
    # solve Phase I with simplex
    tableau = simplex(tableau)
    # Phase I fail: if z != 0, then no optimal solution exists
    if tableau is None or tableau[-1, -1] != 0:
        return None

    # Phase II: update and simplex
    # delete artificial vars' columns
    tableau = np.column_stack((tableau[:, :(matrix.shape[1]-1 + matrix.shape[0])], tableau[:, -1]))
    # update objective function: all -1s except for slacking vars
    obj_row = np.concatenate(([-1]*(matrix.shape[1] - 1), [0]*(matrix.shape[0] + 1)))
    tableau[-1] = obj_row
    # row reduce to get basic vars std col vectors
    tableau = makeBasic(tableau)


    # simplex as usual
    tableau = simplex(tableau)

    # if sol'n, cut out cols for slackvars and obj row (only returning row vals as in rref)
    if tableau is not None:
        tableau = np.column_stack((tableau[:, :matrix.shape[0]], tableau[:, -1]))[:-1, :]

    return tableau

# inherit from brute force agent (and lin alg agent)
# override enqueueBestPossibleCells to use lin opt instead of simple gaussian elim
# when no optimal soln found then fall to normal strategy (bruteforce)
class lin_opt_agent(brute_force_agent):
    def __init__(self, game, useMineCount, order, uncertaintyType):
        brute_force_agent.__init__(self, game, useMineCount, order)
        self.uncertaintyType = uncertaintyType
        self.logging = False

        # override agent.py method to use linear algebra to solve system of equations
        # return a random x,y from the game's currently hidden cells iff no new safe cells can be recovered

    def enqueueBestPossibleCells(self, recrunch, q):
        safesFound = False;
        dim = self.game.dim
        if recrunch:
            if self.logging:
                print("game:")
                for row in self.game.board:
                    print(row)
                print()
                print("knowledge:")
                print(self.playerKnowledge)
                print()
            information_cells = []
            hidden_cells = []
            for x in range(dim):
                for y in range(dim):
                    if self.playerKnowledge[x][y] == HIDDEN:
                        hidden_cells.append((x, y))
                    # if it's not safe, then there's no clue we can use
                    if self.playerKnowledge[x][y] == SAFE:
                        clue = self.game.board[x][y]
                        numSafeNbrs, numMineNbrs, numHiddenNbrs, numTotalNbrs = self.getCellNeighborData(x, y)

                        # if we have hidden neighbors, then add to list of cells to use
                        # in system of equations
                        if numHiddenNbrs != 0:
                            information_cells.append((x, y, clue, numMineNbrs, numHiddenNbrs))

            # if we have information, solve the system
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
                matrix = np.zeros((len(information_cells) + (1 if self.useMineCount else 0), dim * dim + 1),
                                  dtype='float64')
                for i in range(len(information_cells)):
                    x, y, clue, mineNeighbors, numHiddenNbrs = information_cells[i]
                    hidden_nbrs = self.get_hidden_neighbors(x, y)
                    for nx, ny in hidden_nbrs:
                        matrix[i][dim * nx + ny] = 1
                    matrix[i][dim * dim] = clue - mineNeighbors
                    if self.logging:
                        print("generated following row using ({},{}): ".format(x, y))
                        print(matrix[i])
                        print()

                if self.useMineCount:
                    for x, y in hidden_cells:
                        matrix[len(information_cells), dim * x + y] = 1
                    matrix[len(
                        information_cells), dim * dim] = self.game.num_mines - self.numFlaggedMines - self.numDetonatedMines
                    print("generated following row using total mine count: ")
                    print(matrix[len(information_cells)])
                    print()
                # row reduce our matrix to solve the system
                if self.logging:
                    print("information matrix:")
                    print(matrix)
                    print()

                backup = deepcopy(matrix)
                if self.uncertaintyType == 'cautious':
                    matrix = simplexCautious(matrix)
                elif self.uncertaintyType == 'optimistic':
                    matrix = simplexOptimistic(matrix)
                else:
                    rref(matrix)

                if matrix is None:
                    #if self.logging:
                    print("Linear optimization for {} uncertainty FAILED. Proceeding with regular Lin Alg.".format(self.uncertaintyType))
                    matrix = backup
                    rref(matrix)
                else:
                    #if self.logging:
                    print("Linear optimization for {} uncertainty SUCCEEDED.".format(self.uncertainty))

                if self.logging:
                    print("solved matrix:")
                    print(matrix)
                    print()

                # keeping track of flags here is purely for debugging purposes
                flags = []
                for row in matrix:
                    #
                    if self.logging:
                        print("using row: ")
                        print(row)
                    positives = []
                    negatives = []
                    # scale the row for code reuseability
                    if row[-1] < 0:
                        row *= -1
                    # for each variable
                    for i in range(dim * dim):
                        # first get the cell it corresponds to
                        x = i // dim
                        y = i % dim
                        # if the coeffecient in this row is positive put it in
                        # positives. if negative then negatives, if 0 then we don't care
                        if row[i] > 0:
                            positives.append((x, y, row[i]))
                        elif row[i] < 0:
                            negatives.append((x, y, row[i]))

                    # if the row has no information move on
                    if len(positives) == len(negatives) == 0:
                        if self.logging:
                            print()
                        continue

                    # if the row sums to 0, we can potentially find definitively safe cells
                    if row[-1] == 0:
                        # if we have both positive and negative coeffecients, the variables could all be 0
                        # or there could be some mines, the sum of the coeffecients of the mines is 0
                        # which can happen since we have positive and negative coeffecients
                        # so we don't know anything definitive (multiple combinations satisfy the equation)
                        if len(positives) > 0 and len(negatives) > 0:
                            if self.logging:
                                print()
                            continue
                        else:
                            # however if only one list is populated then everything
                            # in the list must have value 0 for equation to hold (so everything is safe)
                            # we can do positives + negatives since exactly one has values
                            for x, y, _ in positives + negatives:
                                # if self.game.board[x][y] == MINE:
                                if self.uncertaintyType in ['none', 'randomReveal']:
                                    assert self.game.board[x][y] != MINE
                                if self.logging:
                                    print("deduced ({},{}) to be safe via lin alg".format(x, y))
                                self.playerKnowledge[x][y] = SAFE
                                q.append((x, y))  # add the safe cell to the queue to visit next
                                safesFound = True
                    else:
                        # if the row sums to a positive then since each variable is only
                        # 0 or 1 (ie. can not be negative), if the sum of the
                        # positive coeffecients exactly matches the last value of the row
                        # (the other side of the equation), then all the variables with
                        # positive coeffecients must be 1, and so we flag them. The sum can never be less since
                        # each variable is at most 1, and if it is greater, then multiple combinations
                        # of the variables in the positive and negative lists can be mines
                        if sum([coeffecient for x, y, coeffecient in positives]) == row[-1]:
                            for x, y, _ in positives:
                                if self.playerKnowledge[x, y] == MINE:
                                    if self.logging:
                                        print()
                                    continue
                                if self.logging:
                                    print("deduced ({},{}) to be a mine via lin alg".format(x, y))
                                if self.uncertaintyType in ['none', 'randomReveal']:
                                    assert self.game.board[x][y] == MINE
                                self.playerKnowledge[x][y] = MINE
                                flags.append((x, y))
                                self.numFlaggedMines += 1
                    if self.logging:
                        print()
            else:
                if self.logging:
                    print("no information for lin alg")
                    print()

        # if lin alg didn't give us anything then pick a random cell
        if (recrunch is False) or (safesFound is False):
            if recrunch is True and self.logging:
                print(
                    '\tRe-processing via lin alg not find new safe cells; proceeding to randomly select hidden cell.\n')
            tuple = self.probability_method()
            if tuple:
                q.append(tuple)

# utility function to run game, save initial & solved boards, and print play-by-play to log txt file
# game metrics: mine safe detection rate and solve time calculated here; outputted to log
def linOptGameDriver(dim, density, logFileName, uncertaintyType):
    sys.stdout = open('{}_log.txt'.format(logFileName), 'w')

    num_mines = int(density*(dim**2))

    print('\n\n***** GAME STARTING *****\n\n{} by {} board with {} mines\n\nSolving with LINEAR OPTIMIZATION strategy\n\nUncertainty: {}'\
          .format(dim, dim, num_mines, uncertaintyType))
    order = [i for i in range(dim**2)]
    random.shuffle(order)
    game = MineSweeper(dim, num_mines)
    #game.saveBoard('{}_init_board'.format(logFileName))

    agent = lin_opt_agent(game, True, order, uncertaintyType)
    agent.enableLogging()
    agent.solve()

    print('\n\n***** GAME OVER *****\n\nGame ended in {} seconds\n\nSafely detected (without detonating) {}% of mines'\
          .format(agent.totalSolveTime, agent.mineFlagRate*100))

    #agent.savePlayerKnowledge('{}_solved_board'.format(logFileName))

'''
t = [[1, -1, 1,  2],
     [1, 1, -1,  4],
     [-1, 1, 1, 6]]
t = np.array(t,dtype=float)

# change to simplexOptimistic(t) or simplexCautious(t)
print(simplexOptimistic(t))
'''






linOptGameDriver(10, 0.2, 'cautious', 'cautious')


