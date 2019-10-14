from MineSweeper import *
from collections import deque
from copy import deepcopy
import sys
import time
from Agent import *

#converts matrix to row reduce echelon form
def rref(matrix):
    m,n = matrix.shape
    #pivot column, starts at -1, so when
    col = 0
    for row in range(m):
        #if pivot column is out of bounds we are done
        if col >= n:
            return
        #keep looping through until we find a column to use for pivot
        #(needs a nonzero value in row r >= current row)
        while matrix[row][col] == 0:
            #if current row has 0, its fine, just need to find a lower row to swap with
            swap = -1
            for r in range(row+1, m):
                # if we find a row with non zero value in this column save it
                if matrix[r][col] != 0:
                    swap = r
                    break
            if swap == -1:
                #if we found nothing to swap with then this column has no non-zero
                #value in this row or lower, so move on to next colum, if it is in bounds
                col += 1
                if col >= n:
                    return
                continue
            #swap
            matrix[row], matrix[swap]= matrix[swap].copy(), matrix[row].copy()
            break
        #scale the current row to have a leading 1
        matrix[row] /= matrix[row][col]

        # zero this column for every other row by subtracting the correct
        # multiple of this row
        for r in [x for x in range(m) if x != row]:
            if matrix[r][col] != 0:
                matrix[r] -= matrix[row] * (matrix[r][col] / matrix[row][col])
        col += 1

class lin_alg_agent(agent):
    def __init__(self, game):
        agent.__init__(self,game)

        #gets all neighbors surrounding (x,y) that we haven't tested yet
    def get_hidden_neighbors(self, x, y):
        neighbors = []
        for i, j in dirs:
            if 0 <= x + i < self.game.dim and 0 <= y + j < self.game.dim and self.playerKnowledge[x + i][y + j] == HIDDEN:
                neighbors.append((x+i,y+j))
        return neighbors

    # override agent.py method to use linear algebra to solve system of equations
    # return a random x,y from the game's currently hidden cells iff no new safe cells can be recovered
    def enqueueBestPossibleCells(self, recrunch, q):
        safesFound = False;
        dim = self.game.dim
        if recrunch:
            information_cells = []
            for x in range(self.game.dim):
                for y in range(self.game.dim):
                    # if it's not safe, then there's no clue we can use
                    if self.playerKnowledge[x][y] != SAFE:
                        pass
                    # if we have a clue check if we can possibly do inference using neighbors
                    else:
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
                matrix = np.zeros((len(information_cells), dim*dim + 1), dtype = 'float64')
                for i in range(len(information_cells)):
                    x, y, clue, mineNeighbors, numHiddenNbrs = information_cells[i]
                    hidden_nbrs = self.get_hidden_neighbors(x,y)
                    for nx, ny in hidden_nbrs:
                        matrix[i][dim * nx + ny] = 1
                    matrix[i][dim * dim] = clue - mineNeighbors

                #row reduce our matrix to solve the system
                rref(matrix)
                #keeping track of flags here is purely for debugging purposes
                flags = []
                # for row in self.game.board:
                #     print(row)
                # print()
                # print(self.playerKnowledge)
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
                                assert self.game.board[x][y] == MINE
                                self.playerKnowledge[x][y] = MINE
                                flags.append((x,y))
                                self.numFlaggedMines += 1
                #
                # print("found safe " + str(q))
                # print("flagged " + str(flags))

        # if lin alg didn't give us anything then pick a random cell
        if (recrunch is False) or (safesFound is False):
            if recrunch is True and self.logging:
                print('\tRe-processing did not find new safe cells; proceeding to randomly select hidden cell.\n')
            (x_tuple, y_tuple) = np.where(self.playerKnowledge == HIDDEN)
            if len(x_tuple) == 0:
                return
            i = np.random.randint(len(x_tuple))
            x = x_tuple[i]; y = y_tuple[i]
            q.append((x, y))

# utility function to run game, save initial & solved boards, and print play-by-play to log txt file
# game metrics: mine safe detection rate and solve time calculated here; outputted to log
def linearAlgebraGameDriver(dim, density, logFileName):
    #sys.stdout = open('{}_log.txt'.format(logFileName), 'w')

    num_mines = int(density*(dim**2))

    print('\n\n***** GAME STARTING *****\n\n{} by {} board with {} mines\n\nSolving with LINEAR ALGEBRA strategy\n\n'\
          .format(dim, dim, num_mines))

    game = MineSweeper(dim, num_mines)
    game.saveBoard('{}_init_board'.format(logFileName))

    agent = lin_alg_agent(game)
    # agent.enableLogging()
    agent.solveBaseline()

    print('\n\n***** GAME OVER *****\n\nGame ended in {} seconds\n\nSafely detected (without detonating) {}% of mines'\
          .format(agent.totalSolveTime, agent.mineFlagRate*100))

    agent.savePlayerKnowledge('{}_solved_board'.format(logFileName))

# compares baseline and lin alg system of equations using same boards and outputs trial by trial time and mine
# detection percent as well as average over all boards
def comparisonGameDriver(dim, density, trials):
    num_mines = int(density*(dim**2))
    baseline_cumulative_time = 0
    baseline_cumulative_rate = 0
    lin_alg_cumulative_time = 0
    lin_alg_cumulative_rate = 0
    for i in range(trials):
        game = MineSweeper(dim, num_mines)
        la_agent = lin_alg_agent(game)
        baselineAgent = agent(game)
        la_agent.solveBaseline()
        baselineAgent.solveBaseline()
        print('Trial {}:\n\tBaseline finished in {} seconds detecting {}% of mines\n\tLin alg finished in {} seconds detecting {}% of mines'\
              .format(i+1, baselineAgent.totalSolveTime, baselineAgent.mineFlagRate*100, la_agent.totalSolveTime, la_agent.mineFlagRate*100))
        baseline_cumulative_time+=baselineAgent.totalSolveTime
        baseline_cumulative_rate+=baselineAgent.mineFlagRate*100
        lin_alg_cumulative_time+=la_agent.totalSolveTime
        lin_alg_cumulative_rate+=la_agent.mineFlagRate*100
    print('\n\n\n\n\nFinished {} trials:\n\tBaseline average was {} seconds detecting {}% of mines\n\tLin alg finished in {} seconds detecting {}% of mines'\
          .format(i+1, baseline_cumulative_time/trials, baseline_cumulative_rate/trials, lin_alg_cumulative_time/trials, lin_alg_cumulative_rate/trials))


# # test game driver
dim = 50
density = 0.25
trialFileName = 'lin_alg'

# linearAlgebraGameDriver(dim, density, trialFileName)
comparisonGameDriver(dim,density,50)
