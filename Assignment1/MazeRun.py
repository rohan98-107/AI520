# Assignment 1 - Maze Runner
# Rohan Rele, Alex Eng & Aakash Raman
# Problem 1 / Main Script

import numpy as np
import random
from colorama import init, Fore, Back, Style
import matplotlib.pyplot as plt
import seaborn as sns
import math
import heapq
import collections
import time
from copy import deepcopy

init(autoreset=True)

# key for maze
EMPTY = 0
BLOCKED = -4
VISITED = 3
FAILED = -3
TARGET_VISITED = 2
TARGET_FAILED = -2
FIRE = -8

dirs = [(0, 1), (1, 0), (-1, 0), (0, -1)] # Right, Down, Up, Left


def generateMaze(dim, p):
    # if probability is invalid
    if not 0 <= p <= 1:
        print("error, invalid probability")
        return None

    # with probability p, mark a cell as being blocked 

    maze = np.zeros((dim, dim))
    for i in range(0, dim):
        for j in range(0, dim):
            if random.uniform(0, 1) < p:
                maze[i][j] = BLOCKED
            else:
                pass

    # set start and goal to empty cells
    maze[0][0] = EMPTY
    maze[dim - 1][dim - 1] = EMPTY
    return maze

def resetMaze(maze):
    res = maze
    res[res != BLOCKED] = EMPTY # set all indices where maze is blocked
    return res

def findNodesExpanded(maze, dim):
    r = np.array(maze)
    return dim * dim - ((r == EMPTY).sum() + (r == BLOCKED).sum())
    #count all empty and blocked cells, subtract from total

def printMaze(result): # method to visualize maze in console output
    dim = len(result)
    if not 0 < dim:
        print("error")
        return
    for i in range(0, dim):
        for j in range(0, dim):
            if result[i][j] == VISITED:
                print(Fore.GREEN + Back.GREEN + Style.BRIGHT + '+' + str(result[i][j]), end='')
            elif result[i][j] == TARGET_VISITED:
                print(Fore.CYAN + Back.CYAN + Style.BRIGHT + '+' + str(result[i][j]), end='')
            elif result[i][j] == FAILED:
                print(Fore.YELLOW + Back.YELLOW + Style.DIM + str(result[i][j]), end='')
            elif result[i][j] == TARGET_FAILED:
                print(Fore.WHITE + Back.WHITE + Style.DIM + str(result[i][j]), end='')
            elif result[i][j] == BLOCKED:
                print(Fore.RED + Back.RED + Style.BRIGHT + str(result[i][j]), end='')
            elif result[i][j] == FIRE:
                print(Fore.MAGENTA + Back.MAGENTA + Style.BRIGHT + str(result[i][j]), end='')
            else:
                print(Fore.BLACK + Back.BLACK + '+' + str(result[i][j]), end='')
        print()
    return


def printMazeHM(result): # method to print high dimensional mazes in jupyter notebook
    if len(result) < 50:
        size = 3
    elif len(result) < 100:
        size = 5
    elif len(result) < 150:
        size = 7
    else:
        size = 10

    plt.figure(figsize=(size, size), dpi=1000)
    sns.heatmap(result, vmin=-4, vmax=4, square=True, cbar=False, xticklabels=False, yticklabels=False, cmap='PiYG')
    plt.show()


def printMazeHM_orig(result): # fast method to visualize a wide range of dimensional mazes
    plt.figure(dpi=100)
    sns.heatmap(result, vmin=-4, vmax=4, linewidth=0.5, square=True, cbar=False, xticklabels=False, yticklabels=False,
                cmap='PiYG')
    plt.show()


def numVisited(maze): # count all cells that are not empty or blocked
    res = 0
    for i in range(len(maze)):
        for j in range(len(maze)):
            if maze[i][j] not in [EMPTY, BLOCKED]:
                res += 1
    return res


def isValid(maze, cell_x, cell_y):
    dim = len(maze)
    if not (0 <= cell_x < dim and 0 <= cell_y < dim): # boundaries of maze
        return False
    # if cell is blocked, a previously "failed" cell or on fire, return false
    if maze[cell_x][cell_y] < 0:
        return False
    else:
        return True

def DFS(maze):

    # initialize stack and list of parent cells
    stack = collections.deque([(0, 0)])
    max_stack_length = 0
    n = len(maze) - 1
    parents = [[(-1, -1)] * (n+1) for t in range(n+1)]

    # standard DFS algorithm
    while stack and not maze[n][n] == VISITED:
        x, y = stack.pop()
        if maze[x][y] == EMPTY:
            maze[x][y] = VISITED
            for dx, dy in dirs[::-1]:
                i = x + dx
                j = y + dy

                if isValid(maze, i, j) and maze[i][j] != VISITED:
                    stack.append((i, j))
                    parents[i][j] = (x,y)

            max_stack_length = max(max_stack_length, len(stack))
            # compare current length of stack to max length, used later for part 3

    if not maze[n][n] == VISITED: # if while loop terminates before goal visited, stack is empty, no solution
        for i in range(n + 1):
            for j in range(n + 1):
                if (maze[i][j] == VISITED):
                    maze[i][j] = FAILED # mark all visited nodes as failed, no node led to goal
        return maze, None, max_stack_length

    # 1. trace back the successful path to the start using parents list
    path = []
    parent_i = n
    parent_j = n
    while (parent_i >= 0) & (parent_j >= 0):
        path.append((parent_i, parent_j))
        parent = parents[parent_i][parent_j]
        parent_i = parent[0]
        parent_j = parent[1]
    path.reverse()
    # 2. mark all coordinates visited but not in final path as failures
    for i in range(n + 1):
        for j in range(n + 1):
            if (maze[i][j] == VISITED) & ((i, j) not in path):
                maze[i][j] = FAILED

    return maze, path, max_stack_length

# modified DFS with different direction preferences (left first, then up) -
# - used for comparison/analysis purposes
# exact same algorithm/ comments as above except for one change
def DFS_leftUp(maze):
    stack = collections.deque([(0, 0)])
    max_stack_length = 0
    n = len(maze) - 1
    parents = [[(-1, -1)] * (n+1) for t in range(n+1)]

    while stack and not maze[n][n] == VISITED:

        x, y = stack.pop()
        if maze[x][y] == EMPTY:
            maze[x][y] = VISITED
            for dx, dy in dirs: # directions are re-ordered to prioritize left/up before right/down
                i = x + dx
                j = y + dy

                if isValid(maze, i, j) and maze[i][j] != VISITED:
                    stack.append((i, j))
                    parents[i][j] = (x,y)

            max_stack_length = max(max_stack_length, len(stack))

    if not maze[n][n] == VISITED:
        for i in range(n + 1):
            for j in range(n + 1):
                if (maze[i][j] == VISITED):
                    maze[i][j] = FAILED
        return maze, None, max_stack_length

    path = []
    parent_i = n
    parent_j = n
    while (parent_i >= 0) & (parent_j >= 0):
        path.append((parent_i, parent_j))
        parent = parents[parent_i][parent_j]
        parent_i = parent[0]
        parent_j = parent[1]
    path.reverse()

    for i in range(n + 1):
        for j in range(n + 1):
            if (maze[i][j] == VISITED) & ((i, j) not in path):
                maze[i][j] = FAILED

    return maze, path, max_stack_length

def BFS(maze, root_x=0, root_y=0, ):
    # enqueue starting point and mark it as visited
    q = collections.deque()
    q.append((root_x, root_y))
    maze[root_x][root_y] = VISITED
    dim = len(maze) - 1

    parents = [[(-1, -1)] * (dim + 1) for t in range(dim + 1)]

    # terminating condition is when the bottom right corner (goal) has been visited
    while maze[dim][dim] != VISITED:

        # if queue is empty, there were no possible next moves, thus, no-solution
        if not q:
            # mark all visited paths as failures
            for i in range(dim + 1):
                for j in range(dim + 1):
                    if maze[i][j] == VISITED:
                        maze[i][j] = FAILED
            # print("No-Solution")
            return maze, None

        # dequeue
        x, y = q.popleft()
        # check all available next moves in BFS fashion (looping through all of them)
        # for every such move, if there is a valid next move and that cell has not been visited, visit it...
        # ...and then enqueue it
        for dx, dy in dirs:
            i = x + dx
            j = y + dy
            if isValid(maze, i, j) and maze[i][j] != VISITED:
                maze[i][j] = VISITED
                q.append((i, j))
                parents[i][j] = (x, y)

    # if the loop is terminated via the terminating condition, there was a solution
    # print("Found-Solution")

    # need to annotate maze successes & failures
    # 1. retrace final path via parents matrix
    path = []
    parent_i = dim;
    parent_j = dim
    while (parent_i >= 0) & (parent_j >= 0):
        path.append((parent_i, parent_j))
        parent = parents[parent_i][parent_j]
        parent_i = parent[0]
        parent_j = parent[1]
    path.reverse()

    # 2. mark all coordinates visited but not in final path as failures
    for i in range(dim + 1):
        for j in range(dim + 1):
            if (maze[i][j] == VISITED) & ((i, j) not in path):
                maze[i][j] = FAILED
    return maze, path


def dist_euclid(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def dist_manhattan(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)


def A_star(maze, heuristic):
    dim = len(maze)

    # define a 2D list of the 'parent' cell for all maze cells to backtrack the correct path
    parents = [[(-1, -1)] * dim for t in range(dim)]

    # fringe (priority heap) will have tuples (priority, (i,j))
    # closed set aka visited list will have tuples (i,j)
    fringe_hp = []
    visited = []

    # enqueue (0,0) with distance 0 from S
    heapq.heappush(fringe_hp, (0, (0, 0)))
    max_hp_size = len(fringe_hp)

    # loop until either the goal is found or the fringe is empty
    while len(fringe_hp) > 0:
        # pop min elt of fringe & mark as visited
        min_fringe_elt = heapq.heappop(fringe_hp)
        i = min_fringe_elt[1][0];
        j = min_fringe_elt[1][1]
        if not maze[i][j] == VISITED:
            maze[i][j] = VISITED
            visited.append(min_fringe_elt[1])

            # termination case: solution was found at goal
            if (i == dim - 1) & (j == dim - 1):
                # print('Found-Solution')

                # annotate maze with solution information:
                # 1. retrace solved path
                path = []
                s = i
                t = j
                while (s >= 0) & (t >= 0):
                    maze[s][t] = VISITED
                    path.append((s, t))
                    parent = parents[s][t]
                    s = parent[0]
                    t = parent[1]
                path.reverse()

                # 2. mark bad path blocks (visited but not in final path)
                bad_paths = list(set(visited).difference(set(path)))
                for bad_block in bad_paths:
                    s = bad_block[0];
                    t = bad_block[1]
                    maze[s][t] = FAILED

                return maze, path, max_hp_size

            for dx, dy in dirs:
                child = (i + dx, j + dy)  # if not solution, then check all children (L,R,U,D)
                if (isValid(maze, child[0], child[1])) and (maze[child[0]][child[1]] != VISITED):
                    parents[child[0]][child[1]] = (i, j)

                    # calculate path length from S for heuristic purposes - inefficient, I know
                    path_length = 0
                    parent_i = child[0]
                    parent_j = child[1]
                    while (parent_i >= 0) & (parent_j >= 0):
                        parent = parents[parent_i][parent_j]
                        parent_i = parent[0];
                        parent_j = parent[1]
                        path_length += 1

                    # calculate distance heuristic h based on param (checked already)
                    h = heuristic(child[0], child[1], dim - 1, dim - 1)

                    pushNeeded = True

                    # if child already in fringe & no update needed, then skip
                    # else delete child so you can push it in again
                    for q in range(len(fringe_hp)):
                        elt_wt = (fringe_hp[q])[0]
                        elt_i = ((fringe_hp[q])[1])[0]
                        elt_j = ((fringe_hp[q])[1])[1]

                        if (elt_i == child[0]) & (elt_j == child[1]):
                            if ((path_length + h) > elt_wt):
                                pushNeeded = False
                            else:
                                fringe_hp[q] = fringe_hp[0]
                                garbage = heapq.heappop(fringe_hp)
                                heapq.heapify(fringe_hp)
                            break

                    # insert new child or update child in fringe
                    if pushNeeded:
                        heapq.heappush(fringe_hp, (path_length + h, (child[0], child[1])))

            if len(fringe_hp) > max_hp_size:
                max_hp_size = len(fringe_hp)

    # if fringe is empty without reaching goal, this was a failure
    # print("No-Solution")

    # annotate bad path blocks (visited but not in final path)
    for bad_block in visited:
        s = bad_block[0]
        t = bad_block[1]
        maze[s][t] = FAILED

    return maze, None, max_hp_size


def A_star_euclid(maze):
    return A_star(maze, dist_euclid)


def A_star_manhattan(maze):
    return A_star(maze, dist_manhattan)


def bdBFS(maze):
    dim = len(maze)
    parents = [[(-1, -1)] * dim for t in range(dim)]
    #enqueue start into source queue
    s_q = collections.deque([(0, 0)])
    #enqueue goal into target queue
    t_q = collections.deque([(dim - 1, dim - 1)])
    s_path_terminal = None
    t_path_terminal = None
    while s_q and t_q and not s_path_terminal:
        x1, y1 = s_q.popleft()
        x2, y2 = t_q.popleft()
        if not maze[x1][y1] == VISITED:
            # source bfs
            maze[x1][y1] = VISITED
            for dx, dy in dirs:
                newX1 = x1 + dx
                newY1 = y1 + dy
                if isValid(maze, newX1, newY1) and not maze[newX1][newY1] == VISITED:
                    # if target has been to new coordinates we are and coming from source we have full path
                    # so mark these locations as terminal for the two BFS's
                    if maze[newX1][newY1] == TARGET_VISITED:
                        s_path_terminal = (x1, y1)
                        t_path_terminal = (newX1, newY1)
                        break
                    # else add to source fringe to continue search
                    else:
                        parents[newX1][newY1] = (x1, y1)
                        s_q.append((newX1, newY1))

        if not s_path_terminal and not maze[x2][y2] == TARGET_VISITED:
            # target bfs
            maze[x2][y2] = TARGET_VISITED
            for dx, dy in dirs:
                newX2 = x2 + dx
                newY2 = y2 + dy
                if isValid(maze, newX2, newY2) and not maze[newX2][newY2] == TARGET_VISITED:
                    # if source has been to new coordinates we are and coming from target we have full path
                    # so mark these locations as terminal for the two BFS's
                    if maze[newX2][newY2] == VISITED:
                        t_path_terminal = (x2, y2)
                        s_path_terminal = (newX2, newY2)
                        break
                    # else add to target fringe to continue search
                    else:
                        parents[newX2][newY2] = (x2, y2)
                        t_q.append((newX2, newY2))

    #check if we found a path before either queue empty
    if not s_path_terminal:
        # mark all visited paths as failures
        for i in range(dim):
            for j in range(dim):
                if maze[i][j] > 0:
                    maze[i][j] = FAILED
        # print("No-Solution")
        return maze, None
    # print("Found-Solution")

    # need to annotate maze successes & failures
    # 1. retrace final path via parents matrix
    s_path = []
    parent_i = s_path_terminal[0];
    parent_j = s_path_terminal[1]
    while (parent_i >= 0) & (parent_j >= 0):
        s_path.append((parent_i, parent_j))
        parent = parents[parent_i][parent_j]
        parent_i = parent[0]
        parent_j = parent[1]
    #do same thing again for target parents
    t_path = []
    parent_i = t_path_terminal[0];
    parent_j = t_path_terminal[1]
    while (parent_i >= 0) & (parent_j >= 0):
        t_path.append((parent_i, parent_j))
        parent = parents[parent_i][parent_j]
        parent_i = parent[0]
        parent_j = parent[1]
    path = s_path[::-1] + t_path

    # 2. mark all coordinates visited but not in final path as failures
    #have different target and source failed notations for visualizations
    for i in range(dim):
        for j in range(dim):
            if (i, j) not in path:
                if maze[i][j] == VISITED:
                    maze[i][j] = FAILED
                elif maze[i][j] == TARGET_VISITED:
                    maze[i][j] = TARGET_FAILED
    return maze, path


# simple utility driver to run multiple trials of one algo at a time

def algoTrialDriver(dim, wall_probability, algo, num_trials):
    num_successes = 0
    total_time = 0
    success_time = 0

    for i in range(1, num_trials + 1):

        maze = generateMaze(dim, wall_probability)

        if (algo == 'A_STAR_EUCLID'):
            start = time.time()
            result, path, maxheap = A_star(maze, dist_euclid)
        elif (algo == 'A_STAR_MANHATTAN'):
            start = time.time()
            result, path, maxheap = A_star(maze, dist_manhattan)
        elif (algo == 'DFS'):
            start = time.time()
            r = DFS(maze)
            result, path = (r[0],r[1])
        elif (algo == 'BFS'):
            start = time.time()
            result, path = BFS(maze)
        elif (algo == 'BD_BFS'):
            start = time.time()
            result, path = bdBFS(maze)
        else:
            result = None;
            path = None

        end = time.time()

        if path is not None:
            num_successes += 1

        total_time += (end - start)

    # return success rate (solvability rate) and total time for n trials
    return (num_successes / num_trials), (total_time / num_trials)


def aStarTrialDriver(dim, wall_probability, success_sample_size):
    num_successes = 0

    total_visited_AEuc = 0
    total_time_AEuc = 0
    total_visited_AMan = 0
    total_time_AMan = 0

    while (num_successes < success_sample_size):
        mz = generateMaze(dim, wall_probability)
        mz_x = deepcopy(mz)
        mz_y = deepcopy(mz)

        euc_start = time.time()
        # add param max_fringe_size in returned if needed -- took out here to save runtime
        euc_mz, euc_path = A_star(mz_x, dist_euclid)
        euc_time = time.time() - euc_start

        man_start = time.time()
        # see above comment
        man_mz, man_path, man_fringesize = A_star(mz_y, dist_manhattan)
        man_time = time.time() - man_start

        if (euc_path is not None) & (man_path is not None):
            num_successes += 1
            total_visited_AEuc += numVisited(euc_mz)
            total_time_AEuc += euc_time
            total_visited_AMan += numVisited(man_mz)
            total_time_AMan += man_time

    return ((total_visited_AEuc / success_sample_size) / (dim ** 2)), (total_time_AEuc / success_sample_size), \
           ((total_visited_AMan / success_sample_size) / (dim ** 2)), (total_time_AMan / success_sample_size)


def dfsTrialDriver(dim, wall_probability, num_trials): # function to compare standard DFS to DFS_leftUp
    total_time_RD = 0
    total_time_LU = 0

    total_visited_RD = 0
    total_visited_LU = 0

    for i in range(num_trials):
        mz = generateMaze(dim, wall_probability)
        mz_x = deepcopy(mz)
        mz_y = deepcopy(mz)

        RD_start = time.time()
        rd_mz, rd_path, rd_maxsize = DFS(mz_x)
        total_time_RD += time.time() - RD_start
        total_visited_RD += numVisited(rd_mz)

        LU_start = time.time()
        lu_mz, lu_path = DFS_leftUp(mz_y)
        total_time_LU += time.time() - LU_start
        total_visited_LU += numVisited(lu_mz)

    # returns success rate of both and total visited nodes for both
    return (total_time_RD / num_trials), (total_time_LU / num_trials), \
           ((total_visited_RD / num_trials) / (dim ** 2)), ((total_visited_LU / num_trials) / (dim ** 2))
