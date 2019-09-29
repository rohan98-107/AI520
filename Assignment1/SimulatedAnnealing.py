# Assignment 1 - Maze Runner
# Rohan Rele, Alex Eng & Aakash Raman
# Problem 3

# fitness function f(param1,param2,param3) = f(p1,p2,p3)
# param1 := functional runtime
# param2 := dim^2 - (len(path)+failed) = no. of empty cells
# param3 := max size of fringe (maximum backtracks)

# we are already given the fitness function / algorithm pairs
# Pair 1: DFS w/ maximal fringe (stack) size
# Pair 2: A* Manhattan with Maximal Nodes Expanded

from MazeRun import *


def findNodesExpanded(maze, dim): # subtract sum of empty and blocked cells from total number of cells
    r = np.array(maze)
    return dim * dim - ((r == EMPTY).sum() + (r == BLOCKED).sum())


def resetMaze(maze):
    res = maze
    res[res == VISITED] = EMPTY
    res[res == FAILED] = EMPTY
    res[res == TARGET_FAILED] = EMPTY
    res[res == TARGET_VISITED] = EMPTY

    return res


f = lambda time, p: p * (math.exp(-1 * time)) # decay function for simulated annealing


def makeHarder(init_solved_maze, init_path, metric_choice, algo_choice, fixed_dim, init_maxsize=0):
    result_maze = None
    result_path = None

    if metric_choice == 'maxsize':  # must use DFS
        F = init_maxsize
    elif metric_choice == 'nodes_expanded':  # must use A_star with Manhattan
        F = findNodesExpanded(init_solved_maze, fixed_dim)

    tpath = init_path[1:-1]  # remove Start and Goal
    tmaze = init_solved_maze # initialize a temporary maze
    count = 1 # count is used for time in decay function

    while count < 1000000000: # set arbitrary limit for count

        for px, py in tpath: # iterate through solved path

            count *= 1.75 # increase count exponentially as later trials will take longer
            tmaze[px][py] = BLOCKED # block point on path
            # try solving with blocked cell
            tmaze = resetMaze(tmaze)
            tmaze_p, tpath_p, maxsize = algo_choice(tmaze)

            # set F prime accordingly
            if metric_choice == 'maxsize':
                Fp = maxsize
            elif metric_choice == 'nodes_expanded':
                Fp = findNodesExpanded(tmaze_p, fixed_dim)

            # if introduced blockage makes maze unsolvable, undo the blockage
            if not tpath_p:
                tmaze[px][py] = VISITED
                if random.random() < f(count, 0.25): # with some probability, as a function of above decay func, ...
                # ... break down a random wall around a node where path is 'stuck' - avoid local optima
                    for dx, dy in dirs:
                        if not isValid(tmaze, px + dx,
                                       py + dy) and 0 <= px + dx < fixed_dim and 0 <= py + dy < fixed_dim:
                            tmaze[px + dx][py + dy] = EMPTY
                continue

            if Fp > F: # if F prime > F , the maze is harder than the previous best maze, replace the temporary maze and path
                F = Fp
                tmaze = deepcopy(tmaze_p)
                tpath = tpath_p[1:-1]
            else: # if it is not better
                if random.random() < f(count, 0.25): # with some probability (decay function), "allow" the mistake (simulated annealing)
                    F = Fp
                    tmaze = deepcopy(tmaze_p)
                    tpath = tpath_p[1:-1]
                else: # if probability condition not satisfied, undo the blockage and restart loop
                    tmaze[px][py] = VISITED

        result_maze = resetMaze(tmaze) # after loop completes iterating through path, set result_maze to last best maze
        result, result_path, result_maxsize = algo_choice(result_maze) # solve last best mazed and return its info

    # print("Found local optima")
    return result, result_path, result_maxsize


def SimulatedAnnealingDriver(metric_choice, algo_choice, dim=110, p=0.305, numTrials=100):
    best_maze = None
    best_metric = 0
    for i in range(numTrials):

        # keep generating Mazes until one is solvable
        while True:
            start = generateMaze(dim, p)
            start_solved, path, maxsize = algo_choice(start)
            if path:
                break

        # - uncomment this code to visualize one trial
        # print("Starting Maze - solved")
        # printMazeHM_orig(start_solved)

        if metric_choice == 'nodes_expanded':
            s = findNodesExpanded(start_solved, dim) # metric for original maze
            res = makeHarder(start_solved, path, metric_choice, algo_choice, fixed_dim=dim, init_maxsize=maxsize)
            n = findNodesExpanded(res[0], dim) # metric for harder maze

            # compare metric to best metric
            if n > best_metric: # if maze is harder than best_maze, replace best_metric and best_maze
                best_metric = n
                resetMaze(res[0])
                best_maze = res[0]

            print("Trial " + str(i) + ": " + str(s) + "->" + str(n)) # print trial progress

        # same as above except metric is calculated differently
        if metric_choice == 'maxsize':
            res = makeHarder(start_solved, path, metric_choice, algo_choice, fixed_dim=dim, init_maxsize=maxsize)
            if res[2] > best_metric:
                best_metric = res[2]
                resetMaze(res[0])
                best_maze = res[0]

            print("Trial " + str(i) + ": " + str(maxsize) + "->" + str(res[2]))


    return best_maze, best_metric


# THE COMMENTED CODE HERE IS IMPORTANT
# Uncomment A_star trial + last line together or...
# Uncomment DFS trial + 2nd to last line together

maze, metric = SimulatedAnnealingDriver('maxsize', DFS, dim=75, numTrials=5)
# maze, metric = SimulatedAnnealingDriver('nodes_expanded', A_star_manhattan, dim=50, numTrials=1)
print("best metric: " + str(metric))
print()
print("Hardest Maze - unsolved")
printMazeHM_orig(maze)
print("Hardest Maze - solved")
printMazeHM_orig(DFS(maze)[0])
# printMazeHM_orig(A_star_manhattan(maze)[0])
