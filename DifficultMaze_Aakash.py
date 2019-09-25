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
from MazeRun import *

f = lambda time, p: p * (math.exp(-1 * time))


def makeHarder(init_solved_maze, init_path, metric_choice, algo_choice, init_maxsize=0, fixed_dim=110, fixed_p=0.31):
    result_maze = None
    result_path = None

    if metric_choice == 'maxsize':  # must use DFS
        F = init_maxsize
    elif metric_choice == 'nodes_expanded':  # must use A_star with Manhattan
        F = findNodesExpanded(init_solved_maze, fixed_dim)

    tpath = init_path[1:-1]  # remove Start and Goal
    tmaze = init_solved_maze
    count = 1

    while count < 100000000000:

        for px, py in tpath:

            count *= 1.25
            tmaze[px][py] = BLOCKED
            tmaze = resetMaze(tmaze)
            tmaze_p, tpath_p, maxsize = algo_choice(tmaze)

            if metric_choice == 'maxsize':
                Fp = maxsize
            elif metric_choice == 'nodes_expanded':
                Fp = findNodesExpanded(tmaze_p, fixed_dim)

            if not tpath_p:
                tmaze[px][py] = VISITED
                if random.random() < f(count, 0.1):
                    for dx, dy in dirs:
                        if not isValid(tmaze, px + dx,
                                       py + dy) and 0 <= px + dx < fixed_dim and 0 <= py + dy < fixed_dim:
                            tmaze[px + dx][py + dy] = EMPTY
                continue

            if Fp > F:
                F = Fp
                tmaze = deepcopy(tmaze_p)
                tpath = tpath_p[1:-1]
            else:
                if random.random() < f(count, 0.1):
                    F = Fp
                    tmaze = deepcopy(tmaze_p)
                    tpath = tpath_p[1:-1]

        result_maze = resetMaze(tmaze)
        result, result_path, result_maxsize = algo_choice(result_maze)
        print("Found local optima")
        break

    return result, result_path, result_maxsize


def SimulatedAnnealingDriver(metric_choice, algo_choice, dim = 110, p= 0.305, numTrials = 100, prints = False):
    best_maze = None
    best_metric = 0
    for i in range(numTrials):
        while True:
            start = generateMaze(dim, p)
            #printMazeHM_orig(start)
            #start_solved, path, maxsize = A_star_manhattan(start)
            start_solved, path, maxsize = algo_choice(start)
            #printMazeHM_orig(start_solved)
            if path:
                break
            # print("No-Solution")

        res = makeHarder(start_solved, path, metric_choice, algo_choice, init_maxsize=maxsize)
        # print("test")
        #res = makeHarder(start_solved, path, 'nodes_expanded', A_star_manhattan, init_maxsize=maxsize)
        if(res[2] > best_metric):

            print("original maxsize: " + str(maxsize))
            #printMazeHM_orig(res[0])
            #print(res[1])
            print("new maxsize: " + str(res[2]))
            #printMazeHM_orig(DFS_again(res[0])[0])
            resetMaze(res[0])
            best_maze = res[0]
            best_metric = res[2]
    return best_maze, best_metric


maze, metric = SimulatedAnnealingDriver('maxsize', DFS_again, dim = 175, numTrials = 100 )
print("best maxsize: " + str(metric))
printMazeHM_orig(maze)
printMazeHM_orig(DFS_again(maze)[0])
