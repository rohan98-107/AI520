

from MazeRun import *

def beamSearch(k = 10, dim = 50, p = 0.2, algo_choice = DFS_revised, metric_choice = "maxsize", sample = 20, remove = True, cap = 0):
    fringe = []
    while len(fringe) < k * 3:
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
    fringe = heapq.nlargest(k, fringe, key = lambda x: x[1])

    final = fringe[0][0]
    best_metric = fringe[0][1]
    iters = 0

    sample_string = " sample: " + str(sample) if sample > 0 else " not sampling"
    remove_string = " removing: " + str(remove)
    print("running with metric: " + metric_choice + " dim= " + str(dim) + " p= " + str(p) + " climbers(k) = " + str(k) + sample_string + remove_string)
    print("original random fringe metric: ")
    print([x[1] for x in fringe])
    print()
    while True:
        iters += 1
        print("iter: " + str(iters))
        local_optima = 0
        new_fringe = []
        for maze, metric, path in fringe:
            is_local_optima = True
            children = set()
            nodes_to_consider = random.sample(path[1:-1],sample) if sample > 0 else path[1:-1]
            if remove:
                for x,y in nodes_to_consider:
                    for dx,dy in dirs:
                        new_x = x + dx
                        new_y = y + dy
                        if 0 <= new_x < dim and 0 <= new_y < dim and maze[new_x][new_y] == BLOCKED:
                            children.add((new_x,new_y))
            else:
                children = nodes_to_consider

            for x,y in children:
                # print(child)
                new_maze = deepcopy(maze)
                new_maze[x][y] = EMPTY if remove else BLOCKED
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
            # print("largest in current iter: ")
            # print([x[1] for x in heapq.nlargest(k, new_fringe, key = lambda x: x[1])])

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
        if cap > 0 and iters >= cap:
            if fringe[0][1] > best_metric:
                final = fringe[0][0]
                best_metric = fringe[0][1]
            break
    return final, best_metric

algo = 'a'
algo_choice = DFS_again if algo == 'DFS' else A_star_manhattan
metric_choice = 'maxsize' if algo == 'DFS' else 'nodes_expanded'
p = .2 if algo == 'DFS' else .2
k = 20 if algo == 'DFS' else 10
sample = 20 if algo == 'DFS' else 5
hard_maze, difficulty = beamSearch(k = k, dim = 175, algo_choice = algo_choice, metric_choice = metric_choice, p = p, sample = sample, remove = True, cap = 30)
printMazeHM_orig(hard_maze)
print("best metric: " + str(difficulty))
a,b,c = algo_choice(hard_maze)
printMazeHM_orig(a)
print(b)
print(c)


# if metric_choice == 'nodes_expanded':
#     children = []
#     for _ in range(5):
#         children.append(random.sample(path[1:-1],sample))
# else:
#     to_block = random.sample(path[1:-1],sample) if sample > 0 else path[1:-1]
#     children = [[x] for x in to_block]
# for child in children:
#     # print(child)
#     new_maze = deepcopy(maze)
#     for x,y in child:
#         new_maze[x][y] = BLOCKED
