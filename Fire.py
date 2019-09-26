from MazeRun import *

dirs2 = [(1, 0), (0, 1), (-1, 0), (0, -1)]

def generateValidFireMaze(dim, p):
    while True:
        maze = generateMaze(dim,p)
        tmaze_p, tpath_p, maxsize = A_star_manhattan(maze)
        if not tpath_p:
            continue
        resetMaze(tmaze_p)
        tmaze_p, tpath_p, maxsize = A_star_manhattan(np.rot90(tmaze_p))
        if tpath_p:
            return resetMaze(tmaze_p)

def trySurvivingBaseLine(maze, path, q):
    dim = len(maze)
    onFire = set()
    onFire.add((0,dim-1))
    maze[0][0] = VISITED
    maze[0][dim-1] = FIRE
    for x,y in path[1:]:
        maze[x][y] = VISITED
        for fire_x, fire_y in list(onFire):
            for dx, dy in dirs:
                spread_x = fire_x + dx
                spread_y = fire_y + dy
                if isValid(maze,spread_x,spread_y) and random.uniform(0, 1) < q:
                    onFire.add((spread_x, spread_y))
                    if not maze[spread_x][spread_y] == VISITED:
                        maze[spread_x][spread_y] = FIRE
        if (x, y) in onFire:
            return False, maze, (x,y), onFire
    return True, maze, None, onFire

def baseLineFireDriver(qs = [0.1 * x for x in range(1,11)], dim = 40, p = 0.305, algo_choice = A_star_manhattan, trials = 100):
    print("baseline fire.py dim=" + str(dim)+" trials=" + str(trials))
    data = {round(x,2):0 for x in qs}
    for i in range(trials):
        maze = generateValidFireMaze(dim,p)
        maze, path, maxsize = algo_choice(maze)
        maze = resetMaze(maze)
        for q in data.keys():
            success, maze, burn_location, fire_at_termination =  trySurvivingBaseLine(maze, path, q)
            if success:
                data[q] +=1
            maze = resetMaze(maze)
        if (i+1) %10 == 0:
            print(data)
    return data

def min_manhattan_distance_to_nearest_fire(fires, x , y):
    return min([dist_manhattan(x,y,fx,fy) for fx, fy in fires])

def min_euclid_distance_to_nearest_fire(fires, x , y):
    return min([dist_euclid(x,y,fx,fy) for fx, fy in fires])

def A_star_fire(maze, fires, x, y, weight = 0.5):
    n = len(maze)
    parents = [[(-1, -1)] * (n) for t in range(n)]
    parents[x][y] = (-2,-2)
    heap = [(0,0,(x,y))]
    while heap and parents[n-1][n-1] == (-1,-1):
        prio, length,coordinates = heapq.heappop(heap)
        x, y = coordinates
        for dx, dy in dirs2:
            new_x = x + dx
            new_y = y + dy
            if isValid(maze, new_x, new_y) and parents[new_x][new_y] == (-1,-1):
                parents[new_x][new_y] = (x,y)
                manhattan_distance_to_goal = (n -1 - new_x + n - 1 - new_y)
                priority =  length + 1 + manhattan_distance_to_goal - weight * min_manhattan_distance_to_nearest_fire(fires, new_x, new_y)
                heapq.heappush(heap, (priority,length+1,(new_x,new_y)))
    if parents[n-1][n-1] == (-1,-1):
        return -1, -1, None
    path = []
    parent_i = n -1
    parent_j = n -1
    while parents[parent_i][parent_j] != (-2,-2):
        path.append((parent_i, parent_j))
        parent_i, parent_j = parents[parent_i][parent_j]
    path.reverse()
    return path[0][0], path[0][1], path




def smarterFireSurvivor(maze, q, weight = 0.5):
    n = len(maze)
    maze[0][n-1] = FIRE
    maze[0][0] = VISITED
    fires = set()
    fires.add((0,n-1))
    curr_x = 0
    curr_y = 0
    smart_path = [(0,0)]
    while curr_x != n-1 or curr_y != n-1: #not at end
        x, y, path = A_star_fire(maze, fires, curr_x, curr_y, weight = weight)
        smart_path.append((x,y))
        # print( str(x) + "," + str(y))
        # print(path)
        if x == -1 and y == -1:
            return False, maze, (x,y), fires, smart_path
        for fire_x, fire_y in list(fires):
            for dx, dy in dirs:
                spread_x = fire_x + dx
                spread_y = fire_y + dy
                if isValid(maze,spread_x,spread_y) and random.uniform(0, 1) < q:
                    fires.add((spread_x, spread_y))
                    if not maze[spread_x][spread_y] == VISITED:
                        maze[spread_x][spread_y] = FIRE
        if maze[x][y] == FIRE:
            return False, maze, (x,y), fires, smart_path
        maze[x][y] = VISITED
        curr_x = x
        curr_y = y
        # printMaze(maze)
        # time.sleep(.5)
    return True, maze, None, fires, smart_path

# def path_heuristic(path, fires):
#     return len(path) - math.sqrt(sum([manhattan_distance_to_nearest_fire(fires,x,y) for x,y in path[1:]])/len(path))
#
# def all_paths(maze, x, y, fires):
#     queue = [[(x,y)]]
#     finished = []
#     n = len(maze)
#     cap = 5000
#     while queue:
#         next_queue = []
#         # print(len(queue))
#         for path in queue:
#             x, y = path[-1]
#             for dx, dy in dirs:
#                 new_x = x + dx
#                 new_y = y + dy
#                 coordinates = (new_x,new_y)
#                 if isValid(maze, new_x, new_y) and coordinates not in path:
#                     dup = path.copy()
#                     dup.append(coordinates)
#                     if coordinates == (n-1, n-1):
#                         finished.append(dup)
#                     else:
#                         next_queue.append(dup)
#         queue = heapq.nsmallest(cap, next_queue, key = lambda x: path_heuristic(x,fires) * \
#                             (dist_manhattan(x[-1][0],x[-1][1],n-1,n-1) + len(x))/len(x) )
#         print(len(queue))
#     return finished
#
# def smartestFireSurvivor(maze, q , weight = 0.5):
#     n = len(maze)
#     maze[0][n-1] = FIRE
#     maze[0][0] = VISITED
#     fires = set()
#     fires.add((0,n-1))
#     curr_x = 0
#     curr_y = 0
#     moves = 0
#     smart_path = [(0,0)]
#     while curr_x != n-1 or curr_y != n-1: #not at end
#
#         paths = all_paths(maze,curr_x,curr_y, fires)
#         if len(paths) == 0:
#             print("failed")
#             printMaze(maze)
#             return False, maze, (curr_x,curr_y), fires, smart_path
#         best_path = paths[0]
#         best_metric = path_heuristic(paths[0], fires)
#         for path in paths[1:]:
#             metric = path_heuristic(path, fires)
#             if metric < best_metric:
#                 best_path = path
#                 best_metric = metric
#         x,y = best_path[1]
#         smart_path.append((x,y))
#         for fire_x, fire_y in list(fires):
#             for dx, dy in dirs:
#                 spread_x = fire_x + dx
#                 spread_y = fire_y + dy
#                 if isValid(maze,spread_x,spread_y) and random.uniform(0, 1) < q:
#                     fires.add((spread_x, spread_y))
#                     if not maze[spread_x][spread_y] == VISITED:
#                         maze[spread_x][spread_y] = FIRE
#         if maze[x][y] == FIRE:
#             return False, maze, (x,y), fires, smart_path
#         maze[x][y] = VISITED
#         curr_x = x
#         curr_y = y
#         # printMaze(maze)
#         moves += 1
#     return True, maze, None, fires, smart_path


def smarterFireDriver(qs = [0.1 * x for x in range(1,11)], dim = 40, p = 0.305, trials = 100, smarterFireFunction = smarterFireSurvivor, weights = [.5], repeats_per_maze = 10):
    print("smarter fire.py dim=" + str(dim)+" trials=" + str(trials))
    data = {round(x,2):{round(weight,2):0 for weight in weights} for x in qs}
    for i in range(trials):
        maze = generateValidFireMaze(dim,p)

        for q in data.keys():
            for weight in data[q].keys():
                for k in range(repeats_per_maze):
                    success, maze, burn_location, fire_at_termination, path =  smarterFireFunction(maze, q, weight = weight)
                    if success:
                        data[q][weight] +=1
                    # print()
                    # print("q=" + str(q))
                    # printMaze(maze)
                    maze = resetMaze(maze)
        if (i+1) %10 == 0:
            for q in data.keys():
                print("q=" + str(q) + "  : " + str(data[q]))
            for _ in range(3):
                print()
    return data

def combinedFireDriver(qs = [0.1 * x for x in range(1,11)], dim = 40, p = 0.305, trials = 100, smarterFireFunction = smarterFireSurvivor, weight = 0.0, trials_per_maze = 10):
    print("combined fire.py dim=" + str(dim)+" trials=" + str(trials) + " weight=" + str(weight))
    data = {round(x,2):[0,0] for x in qs}
    for i in range(trials):
        maze = generateValidFireMaze(dim,p)
        maze, path, maxsize = A_star_manhattan(maze)
        base_maze = resetMaze(maze)
        smart_maze = deepcopy(maze)
        for q in data.keys():
            for _ in range(trials_per_maze):
                success, smart_maze, burn_location, fire_at_termination, smart_path =  smarterFireFunction(smart_maze, q, weight = weight)
                base_success, base_maze, base_burn_location, base_fire_at_termination =  trySurvivingBaseLine(base_maze, path, q)
                if success:
                    data[q][1] +=1
                if base_success:
                    data[q][0] +=1
                # if base_success and not success:
                #     print("base")
                #     printMaze(base_maze)
                #     print("smart")
                #     printMaze(smart_maze)
                #     print(smart_path)
                # print()
                # print("q=" + str(q))
                # printMaze(maze)
                base_maze = resetMaze(base_maze)
                smart_maze = resetMaze(smart_maze)
        if (i+1) %10 == 0:
            print(data)
    return data

# smarterFireDriver(dim = 20, trials = 200, smarterFireFunction = smarterFireSurvivor, weights = [x*.2 for x in range(6)])
# baseLineFireDriver(dim=80, trials = 200)
combinedFireDriver(dim=40, trials = 500, smarterFireFunction = smarterFireSurvivor, weight = 1)
