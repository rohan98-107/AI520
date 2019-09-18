# Assignment 1 - Maze Runner
# Rohan Rele, Alex Eng & Aakash Raman
import numpy as np
import random
from colorama import init, Fore, Back, Style
import math
import heapq
from copy import copy, deepcopy
import collections
import time

init(autoreset=True)

# key for maze
EMPTY = 0
BLOCKED = -1
VISITED = 1
TARGET_VISITED = 2
FAILED = -2
TARGET_FAILED = -3
dirs = [(0, 1), (1, 0), (-1, 0), (0, -1)]


def generateMaze(dim, p):
    # if probability is invalid
    if not 0 <= p <= 1:
        print("error, invalid probability")
        return None

    # a probability of 0.53 corresponds to picking a number from 1-53 out of the numbers from 1-100
    # a probability of 0.1 corresponds to picking the number 1 out of the numbers from 1-10
    # a probability of p corresponds to picking a number from 1-p*10^h out of the numbers...
    # ...from 1-10^h where h is the number of nonzero decimal places in p

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


def printMaze(result):
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
            else:
                print(Fore.BLACK + Back.BLACK + '+' + str(result[i][j]), end='')
        print()
    return


def isValid(maze, cell_x, cell_y):
    dim = len(maze)
    if not (0 <= cell_x < dim and 0 <= cell_y < dim):
        return False
    # if cell is blocked or is a previously "failed" cell
    if maze[cell_x][cell_y] < 0:
        return False
    else:
        return True


def DFS(maze, root_x=0, root_y=0, stack=None, path=None):
    # initializing stack
    if stack is None:
        stack = []
    # initializing path
    if path is None:
        path = []

    # if there are no moves from start, there is no solution
    if maze[0][0] == FAILED:
        print("No-Solution")
        return maze, path, len(path)

    # if recursion reaches bottom right corner of matrix (maze), we are done
    if root_x == len(maze) - 1 and root_y == len(maze) - 1:
        maze[root_x][root_y] = VISITED
        path.append((root_x, root_y))
        print("Found-Solution")
        return maze, path, len(path)

    # at every recursive call, append the cell's coordinates to the stack and mark it as visited
    stack.append((root_x, root_y))
    maze[root_x][root_y] = VISITED
    path.append((root_x, root_y))

    # checking to see if the recursion can move up, down, left, right, if there is an available move...
    # ...perform DFS on that move
    for dx, dy in dirs:
        i = root_x + dx
        j = root_y + dy
        if isValid(maze, i, j) and (i, j) not in stack:
            return DFS(maze, i, j, stack, path)
    # if there are no available moves, pop the coordinate on the current recusive step, mark it as failed...
    # ... and then pop the coordinate underneath it and continue DFS from that point

    stack.remove((root_x, root_y))
    maze[root_x][root_y] = FAILED
    path.remove((root_x, root_y))
    if not stack:
        prev = (root_x, root_y)
    else:
        prev = stack.pop()
        path.remove((prev[0], prev[1]))
    return DFS(maze, prev[0], prev[1], stack, path)


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
            print("No-Solution")
            return maze

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
    print("Found-Solution")

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
    return maze


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
    fringe_hp = [];
    visited = []

    # enqueue (0,0) with distance 0 from S
    heapq.heappush(fringe_hp, (0, (0, 0)))

    # loop until either the goal is found or the fringe is empty
    while len(fringe_hp) > 0:
        # pop min elt of fringe & mark as visited
        min_fringe_elt = heapq.heappop(fringe_hp)
        i = min_fringe_elt[1][0];
        j = min_fringe_elt[1][1]
        maze[i][j] = VISITED
        visited.append(min_fringe_elt[1])

        # termination case: solution was found at goal
        if (i == dim - 1) & (j == dim - 1):
            print('Found-Solution')

            # annotate maze with solution information:
            # 1. retrace solved path
            path = []
            s = i;
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

            return maze

        for dx, dy in dirs:
            child = (i + dx, j + dy)  # if not solution, then check all children (L,R,U,D)
            if isValid(maze, child[0], child[1]) and child not in visited:
                visited.append(child)
                parents[child[0]][child[1]] = (i, j)

                # calculate path length from S for heuristic purposes - inefficient, I know
                path_length = 0
                parent_i = child[0];
                parent_j = child[1]
                while (parent_i >= 0) & (parent_j >= 0):
                    parent = parents[parent_i][parent_j]
                    parent_i = parent[0];
                    parent_j = parent[1]
                    path_length += 1

                # calculate distance heuristic h based on param (checked already)
                h = heuristic(child[0], child[1], dim - 1, dim - 1)

                # if child already in fringe & no update needed, then skip
                for q in fringe_hp:
                    if (q[1] == (child[0], child[1])) & (path_length + h > q[0]):
                        continue

                # insert new child or update child in fringe
                heapq.heappush(fringe_hp, (path_length + h, (child[0], child[1])))

    # if fringe is empty without reaching goal, this was a failure
    print("No-Solution")

    # annotate bad path blocks (visited but not in final path)
    for bad_block in visited:
        s = bad_block[0]
        t = bad_block[1]
        maze[s][t] = FAILED

    return maze


def bdBFS(maze):
    dim = len(maze)
    parents = [[(-1, -1)] * dim for t in range(dim)]
    maze[0][0] = VISITED
    maze[dim - 1][dim - 1] = TARGET_VISITED
    s_q = collections.deque([(0, 0)])
    t_q = collections.deque([(dim - 1, dim - 1)])
    s_path_terminal = None
    t_path_terminal = None
    while s_q and t_q and not s_path_terminal:
        x1, y1 = s_q.popleft()
        x2, y2 = t_q.popleft()
        for dx, dy in dirs:
            newX1 = x1 + dx
            newX2 = x2 + dx
            newY1 = y1 + dy
            newY2 = y2 + dy

            # source bfs
            if isValid(maze, newX1, newY1) and not maze[newX1][newY1] == VISITED:
                # if target has been there and coming from source we have full path
                if maze[newX1][newY1] == TARGET_VISITED:
                    s_path_terminal = (x1, y1)
                    t_path_terminal = (newX1, newY1)
                    break
                # else add to source fringe to continue search
                else:
                    maze[newX1][newY1] = VISITED
                    parents[newX1][newY1] = (x1, y1)
                    s_q.append((newX1, newY1))
            # target bfs
            if isValid(maze, newX2, newY2) and not maze[newX2][newY2] == TARGET_VISITED:
                # if source has been there and coming from target we have full path
                if maze[newX2][newY2] == VISITED:
                    t_path_terminal = (x2, y2)
                    s_path_terminal = (newX2, newY2)
                    break
                # else add to target fringe to continue search
                else:
                    maze[newX2][newY2] = TARGET_VISITED
                    parents[newX2][newY2] = (x2, y2)
                    t_q.append((newX2, newY2))
    if not s_path_terminal:
        # mark all visited paths as failures
        for i in range(dim):
            for j in range(dim):
                if maze[i][j] > 0:
                    maze[i][j] = FAILED
        print("No-Solution")
        return maze
    print("Found-Solution")

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
    for i in range(dim):
        for j in range(dim):
            if (i, j) not in path:
                if maze[i][j] == VISITED:
                    maze[i][j] = FAILED
                elif maze[i][j] == TARGET_VISITED:
                    maze[i][j] = TARGET_FAILED
    return maze


# simple utility driver to run multiple trials of one algo at a time

def algoTrialDriver(dim, wall_probability, algo, num_trials):
    for i in range(1, num_trials + 1):
        print('\n', '-' * 20)
        print('TRIAL ', i, ' OF ', num_trials)
        print()

        maze = generateMaze(dim, wall_probability)

        print('Running...', algo, '\n')

        if (algo == 'A_STAR_EUCLID'):
            result = A_star(maze, dist_euclid)
        elif (algo == 'A_STAR_MANHATTAN'):
            result = A_star(maze, dist_manhattan)
        elif (algo == 'DFS'):
            result = DFS(maze, stack=None)[0]
        elif (algo == 'BFS'):
            result = BFS(maze)
        elif (algo == 'BD_BFS'):
            result = bdBFS(maze)

        printMaze(result)

        print('\n', '-' * 20)


# algos A_STAR_EUCLID, A_STAR_MANHATTAN, BFS, DFS, BD_BFS'
dim = 15
wall_probability = 0.3
algo = 'BFS'
num_trials = 5

algoTrialDriver(dim, wall_probability, algo, num_trials)

'''Uncomment this to test BFS vs DFS head-to-head on the same maze'''
'''
dim = 15
wall_probability = 0.3

maze1 = generateMaze(dim, wall_probability)
maze2 = deepcopy(maze1)

print('DFS')
result1 = DFS(maze1, stack = None)
printMaze(result1)


print('\n\n')

print('BFS')
result2 = BFS(maze2)
printMaze(result2)

maze = generateMaze(15,0.3)
result = DFS(maze)
print(result)
printMaze(result[0])
print((result[0] == 1).sum())
'''
