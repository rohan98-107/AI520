# Assignment 1 - Maze Runner
# Rohan Rele, Alex Eng & Aakash Raman
# Problems 3 & 4

# fitness function f(param1,param2,param3) = f(p1,p2,p3)
# param1 := functional runtime
# param2 := dim^2 - (len(path)+failed) = no. of empty cells
# param3 := max size of fringe (maximum backtracks)

# we are already given the fitness function / algorithm pairs
# Pair 1: DFS w/ maximal fringe (stack) size
# Pair 2: A* Manhattan with Maximal Nodes Expanded

from test import *


def findNodesExpanded(maze, dim):
    r = np.array(maze)
    return dim * dim - (r == EMPTY or r == BLOCKED).sum()


def resetMaze(maze):
    res = maze
    res[res == VISITED] = EMPTY
    res[res == FAILED] = EMPTY
    res[res == TARGET_FAILED] = EMPTY
    res[res == TARGET_VISITED] = EMPTY

    return res


def makeHarder(init_solved_maze, init_path, metric_choice, algo_choice, init_maxsize=0, fixed_dim=110, fixed_p=0.31):
    result_maze = None
    result_path = None

    if metric_choice == 'maxsize': # must use DFS
        F = init_maxsize
    elif metric_choice == 'nodes_expanded': # must use A_star
        F = findNodesExpanded(init_solved_maze, fixed_dim)

    tpath = init_path[1:-1]  # remove Start and Goal
    tmaze = init_solved_maze

    while True:

        for px, py in tpath:

            tmaze[px][py] = BLOCKED
            tmaze = resetMaze(tmaze)
            tmaze_p, tpath_p, Fp = algo_choice(tmaze)

            if not tpath_p:
                tmaze[px][py] = VISITED
                continue

            if Fp > F:
                F = Fp
                tmaze = deepcopy(tmaze_p)
                tpath = tpath_p[1:-1]

        result_maze = tmaze
        result_path = tpath
        print("Found local optima")
        break

    return result_maze, result_path


start = generateMaze(100, 0.3)
printMazeHM(start)
start_solved, path, maxsize = DFS(start)
printMazeHM(start_solved)
res = makeHarder(start_solved, path, 'maxsize', DFS, init_maxsize=maxsize)
printMazeHM(res[0])
print(res[1])
