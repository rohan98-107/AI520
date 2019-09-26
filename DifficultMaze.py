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

from MazeRun import *


def findNodesExpanded(maze, dim):
    r = np.array(maze)
    return dim * dim - ((r == EMPTY).sum() + (r == BLOCKED).sum())


def resetMaze(maze):
    res = maze
    res[res == VISITED] = EMPTY
    res[res == FAILED] = EMPTY
    res[res == TARGET_FAILED] = EMPTY
    res[res == TARGET_VISITED] = EMPTY

    return res


f = lambda time, p: p * (math.exp(-1 * time))


def makeHarder(init_solved_maze, init_path, metric_choice, algo_choice, fixed_dim, init_maxsize=0):
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

            count *= 1.75
            tmaze[px][py] = BLOCKED
            tmaze = resetMaze(tmaze)
            tmaze_p, tpath_p, maxsize = algo_choice(tmaze)

            if metric_choice == 'maxsize':
                Fp = maxsize
            elif metric_choice == 'nodes_expanded':
                Fp = findNodesExpanded(tmaze_p, fixed_dim)

            if not tpath_p:
                tmaze[px][py] = VISITED
                if random.random() < f(count, 0.25):
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
                if random.random() < f(count, 0.25):
                    F = Fp
                    tmaze = deepcopy(tmaze_p)
                    tpath = tpath_p[1:-1]
                else:
                    tmaze[px][py] = VISITED

        result_maze = resetMaze(tmaze)
        result, result_path, result_maxsize = algo_choice(result_maze)

    # print("Found local optima")
    return result, result_path, result_maxsize


def SimulatedAnnealingDriver(metric_choice, algo_choice, dim=110, p=0.305, numTrials=100):
    best_maze = None
    best_metric = 0
    for i in range(numTrials):
        while True:
            start = generateMaze(dim, p)
            start_solved, path, maxsize = algo_choice(start)
            if path:
                break
        '''
        while True:
            start = generateMaze(dim, p)
            start_solved, path, maxsize = algo_choice(start)
            s = findNodesExpanded(start_solved, dim)
            if s >= 20000 and path:
                break
        '''
        
        # s = findNodesExpanded(start_solved, dim)
        res = makeHarder(start_solved, path, metric_choice, algo_choice, fixed_dim=dim, init_maxsize=maxsize)
        n = findNodesExpanded(res[0], dim)
        # if res[2] > best metric:
        if n > best_metric:
            best_metric = n
            resetMaze(res[0])
            best_maze = res[0]
            # best_metric = res[2]

        # print("Trial " + str(i) + ": " + str(maxsize) + "->" + str(res[2]))
        print("Trial " + str(i) + ": " + str(s) + "->" + str(n))

    return best_maze, best_metric


# maze, metric = SimulatedAnnealingDriver('maxsize', DFS, dim=175, numTrials=1)
maze, metric = SimulatedAnnealingDriver('nodes_expanded', A_star_manhattan, dim=175, numTrials=1)
print("best metric: " + str(metric))
printMazeHM_orig(maze)
# printMazeHM_orig(DFS(maze)[0])
printMazeHM_orig(A_star_manhattan(maze)[0])
