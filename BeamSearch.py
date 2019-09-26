

from MazeRun import *

def beamSearch(k = 10, dim = 50, p = 0.2, algo_choice = DFS_revised, metric_choice = "maxsize", sample = 20, remove = True, cap = 0):
    fringe = []

    #come up with initial spots for k climbers, k*3 is optional can just use
    #k , but k*3 allows for a little randomness at beginning to help

    while len(fringe) < k * 3:
        new_maze = generateMaze(dim,p)
        tmaze_p, tpath_p, maxsize = algo_choice(new_maze)
        if not tpath_p: #ignore unsolvable mazes
            continue
        if metric_choice == 'maxsize':
            metric = maxsize
        elif metric_choice == 'nodes_expanded':
            metric = findNodesExpanded(tmaze_p, dim)
        resetMaze(tmaze_p)
        #when adding tp fringe store metric so we don't have to recompute to see
        #if child is better than parent also store path because we will be using
        #the concept of removing or adding blockages along path to identify children
        fringe.append((tmaze_p, metric, tpath_p))
    fringe = heapq.nlargest(k, fringe, key = lambda x: x[1])

    #after heap.nlargest front of list is best metric
    final = fringe[0][0]
    best_metric = fringe[0][1]
    iters = 0

    # sample_string = " sample: " + str(sample) if sample > 0 else " not sampling"
    # remove_string = " removing: " + str(remove)
    # print("running with metric: " + metric_choice + " dim= " + str(dim) + " p= " + str(p) + " climbers(k) = " + str(k) + sample_string + remove_string)
    # print("original random fringe metric: ")
    # print([x[1] for x in fringe])
    # print()
    while True:
        iters += 1

        local_optima = 0
        new_fringe = []
        for maze, metric, path in fringe:
            is_local_optima = True #initially assume local optima
            children = set()

            #these are the nodes we will use for blocking, or we will unblock their
            #surrounding nodes if sample > 0 then we only use a random sample along path
            #to save run time, otherwise use whole path
            nodes_to_consider = random.sample(path[1:-1],sample) if sample > 0 else path[1:-1]

            #we can only remove or add blocks at a time, as we allow for children to be considered
            #for next iteration if child metric >= parent. If it was strictly greater we could
            #allow removing and adding blocks in same beam search, but the >= allows for a child to
            #eventually find combinations of adding one block that doesn't help but then a second
            #that relies on first addition that does help
            if remove:
                for x,y in nodes_to_consider:
                    #for every node in sample or path, we make one child for each
                    #surrounding block that is unblocked as we are removing
                    for dx,dy in dirs:
                        new_x = x + dx
                        new_y = y + dy
                        if 0 <= new_x < dim and 0 <= new_y < dim and maze[new_x][new_y] == BLOCKED:
                            children.add((new_x,new_y))
            else:
                #just use the sample or path if we are blocking
                children = nodes_to_consider

            for x,y in children:
                # print(child)
                new_maze = deepcopy(maze) # make a copy
                new_maze[x][y] = EMPTY if remove else BLOCKED
                tmaze_p, tpath_p, maxsize = algo_choice(new_maze)

                #confirm that the maze is solvable before using the metric and child
                if tpath_p:
                    if metric_choice == 'maxsize':
                        t_metric = maxsize
                    elif metric_choice == 'nodes_expanded':
                        t_metric = findNodesExpanded(tmaze_p, dim)
                    if t_metric >= metric:
                        is_local_optima = False #no longer a local optima the second one worthwhile child is found
                        new_fringe.append((tmaze_p,t_metric, tpath_p)) #adding to new fringe because we don't care to revisit parent in next iter
                    resetMaze(tmaze_p)


            #if local optima update best if needed
            if is_local_optima:
                if metric > best_metric:
                    best_metric = metric
                    final = maze
                local_optima += 1

        #end if every climber is at a local optima, we can have less that k climbers
        #if in previous iteration <k children better than parent, if after such a
        #scenario though climbers can come back if enough children found in later
        #iteration
        if local_optima == min(k,len(fringe)):
            break
        #get best k children for next iteration
        fringe = heapq.nlargest(k, new_fringe, key = lambda x: x[1])

        #these prints are to monitor progress due to slow run time
        # print([x[1] for x in fringe])
        # print()

        #can optionally cap iterations to reduce run time
        if cap > 0 and iters >= cap:
            #need to check if fringe contains better than previous best as best
            #only updated at local optima
            if fringe[0][1] > best_metric:
                final = fringe[0][0]
                best_metric = fringe[0][1]
            break
    return final, best_metric



def beamDriver(algo = 'A_star'):
    #we only need to solve for two specific algorithm and metric combinations
    #so just set them
    algo_choice = DFS_again if algo == 'DFS' else A_star_manhattan
    metric_choice = 'maxsize' if algo == 'DFS' else 'nodes_expanded'

    #this initial p can help find better solutions, our implementation of A_star
    #explores most nodes when completely empty map DFS_needs some blockages
    #to manipulate algorithm into poor path finding.
    p = .2 if algo == 'DFS' else .15

    #less default climbers for A_star due to longer runtime
    k = 20 if algo == 'DFS' else 10

    #sample of nodes along parent path to use for children, if set to 0 we look
    #at all children. Default is to use a sample because checking all children
    #for dim 175 (min path length 348 and min children 346) will take too long
    sample = 20 if algo == 'DFS' else 5

    #capping iterations at 30 will allow for it to terminate within a few hours
    #without cap and if sample ==0 then it will take days to run
    cap = 30

    #try both removing or adding blocks, different algorithms will be more suitable for one or the other
    remove_hard_maze, remove_difficulty = beamSearch(k = k, dim = 175, algo_choice = algo_choice, metric_choice = metric_choice, p = p, sample = sample, remove = True, cap = cap)
    add_hard_maze, add_difficulty = beamSearch(k = k, dim = 175, algo_choice = algo_choice, metric_choice = metric_choice, p = p, sample = sample, remove = False, cap = cap)

    #pick better of adding and removing
    hard_maze = remove_hard_maze if remove_difficulty > add_difficulty else add_hard_maze
    difficulty = max(remove_difficulty, add_difficulty)

    printMazeHM_orig(hard_maze)
    print("best metric: " + str(difficulty))
    a,b,c = algo_choice(hard_maze)
    printMazeHM_orig(a)

beamDriver(algo = 'DFS')
