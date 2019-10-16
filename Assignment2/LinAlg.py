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
    def __init__(self, game, useMineCount, order):
        agent.__init__(self,game, order)
        self.useMineCount = useMineCount

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
                    if self.logging:
                        print("generated following row using ({},{}): ".format(x,y))
                        print(matrix[i])
                        print()

                if self.useMineCount:
                    for x,y in hidden_cells:
                        matrix[len(information_cells), dim * x + y] = 1
                    matrix[len(information_cells),dim*dim] = self.game.num_mines-self.numFlaggedMines-self.numDetonatedMines
                    print("generated following row using total mine count: ")
                    print(matrix[len(information_cells)])
                    print()
                #row reduce our matrix to solve the system
                if self.logging:
                    print("information matrix:")
                    print(matrix)
                    print()
                rref(matrix)
                if self.logging:
                    print("rref'd matrix:")
                    print(matrix)
                    print()
                #keeping track of flags here is purely for debugging purposes
                flags = []
                for row in matrix:
                    #
                    if self.logging:
                        print("using row: ")
                        print(row)
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
                        if self.logging:
                            print()
                        continue

                    #if the row sums to 0, we can potentially find definitively safe cells
                    if row[-1] == 0:
                        # if we have both positive and negative coeffecients, the variables could all be 0
                        # or there could be some mines, the sum of the coeffecients of the mines is 0
                        # which can happen since we have positive and negative coeffecients
                        # so we don't know anything definitive (multiple combinations satisfy the equation)
                        if len(positives) >0 and len(negatives) > 0:
                            if self.logging:
                                print()
                            continue
                        else:
                            # however if only one list is populated then everything
                            # in the list must have value 0 for equation to hold (so everything is safe)
                            # we can do positives + negatives since exactly one has values
                            for x,y,_ in positives + negatives:
                                # if self.game.board[x][y] == MINE:
                                assert self.game.board[x][y] != MINE
                                if self.logging:
                                    print("deduced ({},{}) to be safe via lin alg".format(x,y))
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
                                    if self.logging:
                                        print()
                                    continue
                                if self.logging:
                                    print("deduced ({},{}) to be a mine via lin alg".format(x,y))
                                assert self.game.board[x][y] == MINE
                                self.playerKnowledge[x][y] = MINE
                                flags.append((x,y))
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
                print('\tRe-processing via lin alg not find new safe cells; proceeding to randomly select hidden cell.\n')
            tuple = self.probability_method()
            if tuple:
                q.append(tuple)

# utility function to run game, save initial & solved boards, and print play-by-play to log txt file
# game metrics: mine safe detection rate and solve time calculated here; outputted to log
def linearAlgebraGameDriver(dim, density, logFileName):
    sys.stdout = open('{}_log.txt'.format(logFileName), 'w')

    num_mines = int(density*(dim**2))

    print('\n\n***** GAME STARTING *****\n\n{} by {} board with {} mines\n\nSolving with LINEAR ALGEBRA strategy\n\n'\
          .format(dim, dim, num_mines))
    order = [i for i in range(dim**2)]
    random.shuffle(order)
    game = MineSweeper(dim, num_mines)
    # game.saveBoard('{}_init_board'.format(logFileName))

    agent = lin_alg_agent(game, True, order)
    agent.enableLogging()
    agent.solve()

    print('\n\n***** GAME OVER *****\n\nGame ended in {} seconds\n\nSafely detected (without detonating) {}% of mines'\
          .format(agent.totalSolveTime, agent.mineFlagRate*100))

    # agent.savePlayerKnowledge('{}_solved_board'.format(logFileName))

# compares baseline and lin alg system of equations using same boards and outputs trial by trial time and mine
# detection percent as well as average over all boards
def baselineVsLinAlgComparisonGameDriver(dim, density, trials, useMineCount = False):
    print("baseline vs brute with lin alg, dim {}, density {}, trials {}, useMineCount={}".format(dim,density,trials,useMineCount))
    num_mines = int(density*(dim**2))
    baseline_cumulative_time = 0
    baseline_cumulative_rate = 0
    lin_alg_cumulative_time = 0
    lin_alg_cumulative_rate = 0
    for i in range(trials):
        game = MineSweeper(dim, num_mines)
        order = [i for i in range(dim**2)]
        random.shuffle(order)
        la_agent = lin_alg_agent(game, useMineCount, order)
        baselineAgent = agent(game, order)
        la_agent.solve()
        baselineAgent.solve()
        baseline_cumulative_time+=baselineAgent.totalSolveTime
        baseline_cumulative_rate+=baselineAgent.mineFlagRate*100
        lin_alg_cumulative_time+=la_agent.totalSolveTime
        lin_alg_cumulative_rate+=la_agent.mineFlagRate*100
        if i % 50 == 49:
            print('\n\n\n\n\nFinished {} trials:\n\tBaseline average was {} seconds detecting {}% of mines\n\tLin alg finished in {} seconds detecting {}% of mines'\
              .format(i+1, baseline_cumulative_time/(i+1), baseline_cumulative_rate/(i+1), lin_alg_cumulative_time/(i+1), lin_alg_cumulative_rate/(i+1)))
