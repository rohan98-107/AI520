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

    while count < 1000000000:

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
                if random.random() < f(count, 0.23):
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
                if random.random() < f(count, 0.23):
                    F = Fp
                    tmaze = deepcopy(tmaze_p)
                    tpath = tpath_p[1:-1]

        result_maze = resetMaze(tmaze)
        result, result_path, result_maxsize = algo_choice(result_maze)
        print("Found local optima")
        break

    return result, result_path, result_maxsize


while True:
    start = generateMaze(110, 0.3)
    printMazeHM_orig(start)
    #start_solved, path, maxsize = A_star_manhattan(start)
    start_solved, path, maxsize = DFS_revised(start)
    printMazeHM_orig(start_solved)
    if path:
        break
    print("No-Solution")

print("original maxsize: " + str(maxsize))
res = makeHarder(start_solved, path, 'maxsize', DFS_revised, init_maxsize=maxsize)
#res = makeHarder(start_solved, path, 'nodes_expanded', A_star_manhattan, init_maxsize=maxsize)
printMazeHM_orig(res[0])
print(res[1])
print("new maxsize: " + str(res[2]))
