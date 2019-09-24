

from MazeRun import *

def beamSearch(mazes = None, k = 10, dim = 50, p = 0.2, algo_choice = DFS_revised, metric_choice = "maxsize"):
    fringe = []
    if mazes == None:
        while len(fringe) < k:
            new_maze = generateMaze(dim,p)
            tmaze_p, tpath_p, maxsize = algo_choice(new_maze)
            if not tpath_p:
                continue
            if metric_choice == 'maxsize':
                metric = maxsize
            elif metric_choice == 'nodes_expanded':
                metric = findNodesExpanded(tmaze_p, dim)
            resetMaze(tmaze_p)
            fringe.append((tmaze_p, metric, tpath_p))
    else:
        for maze in mazes:
            tmaze_p, tpath_p, maxsize = algo_choice(maze)
            if not tpath_p:
                continue
            if metric_choice == 'maxsize':
                metric = maxsize
            elif metric_choice == 'nodes_expanded':
                metric = findNodesExpanded(tmaze_p, dim)
            resetMaze(tmaze_p)
            fringe.append((tmaze_p,metric,tpath_p))

    final = fringe[0][0]
    best_metric = fringe[0][1]
    for maze, metric, path in fringe[1:]:
        if metric > best_metric:
            best_metric = metric
            final = maze
    iters = 0
    print("original random fringe: ")
    print([x[1] for x in fringe])
    print()
    while True:
        iters += 1
        print("iter: " + str(iters))
        local_optima = 0
        new_fringe = []
        for maze, metric, path in fringe:
            is_local_optima = True

            for x,y in random.sample(path[1:-1],20):
            # for x,y in path[1:-1]:
                new_maze = deepcopy(maze)
                new_maze[x][y] = BLOCKED
                if metric_choice == 'nodes_expanded':
                    for dx, dy in dirs:
                        i = x + dx
                        j = y + dy
                        if isValid(maze, i, j):
                            new_maze[i][j] = BLOCKED
                tmaze_p, tpath_p, maxsize = algo_choice(new_maze)
                if tpath_p:
                    if metric_choice == 'maxsize':
                        t_metric = maxsize
                    elif metric_choice == 'nodes_expanded':
                        t_metric = findNodesExpanded(tmaze_p, dim)
                    if t_metric >= metric:
                        is_local_optima = False
                        new_fringe.append((tmaze_p,t_metric, tpath_p))
                    resetMaze(tmaze_p)

            if is_local_optima:
                if metric > best_metric:
                    best_metric = metric
                    final = maze
                local_optima += 1

        print("found " + str(local_optima) + " local_optima")
        if local_optima == min(k,len(fringe)):
            break
        fringe = heapq.nlargest(k, new_fringe, key = lambda x: x[1])
        print([x[1] for x in fringe])
        print()
    return final, best_metric

algo = 'a'
algo_choice = DFS_again if algo == 'DFS' else A_star_manhattan
metric_choice = 'maxsize' if algo == 'DFS' else 'nodes_expanded'
p = .2 if algo == 'DFS' else .2
hard_maze, difficulty = beamSearch(k = 20, dim = 175, algo_choice = algo_choice, metric_choice = metric_choice, p = p)
printMazeHM_orig(hard_maze)
print("best metric: " + str(difficulty))
a,b,c = algo_choice(hard_maze)
printMazeHM_orig(a)
print(b)
print(c)
