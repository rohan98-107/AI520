

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
    while True:
        iters += 1
        print("iter: " + str(iters))
        local_optima = 0
        for i in range(len(fringe)):
            maze, metric, path = fringe[i]
            is_local_optima = True
            for x,y in path[1:-1]:
                new_maze = deepcopy(maze)
                new_maze[x][y] = BLOCKED
                tmaze_p, tpath_p, maxsize = algo_choice(new_maze)
                if tpath_p:
                    if metric_choice == 'maxsize':
                        t_metric = maxsize
                    elif metric_choice == 'nodes_expanded':
                        t_metric = findNodesExpanded(tmaze_p, dim)
                    if t_metric > metric:
                        is_local_optima = False
                    resetMaze(tmaze_p)
                    fringe.append((tmaze_p,t_metric, tpath_p))

            if is_local_optima:
                if metric > best_metric:
                    best_metric = metric
                    final = maze
                local_optima += 1

        print("found " + str(local_optima) + " local_optima")
        if local_optima == k:
            break
        fringe = heapq.nlargest(k, fringe, key = lambda x: x[1])
        print([x[1] for x in fringe])
    return final, best_metric



hard_maze, difficulty = beamSearch(k = 10, dim=110, algo_choice = DFS_again, metric_choice = 'maxsize')
printMazeHM(hard_maze)
print("best metric: " + str(difficulty))
a,b,c = DFS_again(hard_maze)
printMazeHM(a)
print(b)
print(c)
