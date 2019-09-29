from MazeRun import *

def simulateFire(maze, fires, q):
    #take the current set of fires
    for fire_x, fire_y in list(fires):
        #for each surrounding node
        for dx, dy in dirs:
            spread_x = fire_x + dx
            spread_y = fire_y + dy
            #if in bounds and empty and with probability q
            if isValid(maze,spread_x,spread_y) and random.uniform(0, 1) < q:
                #add fire to set of fires and mark maze
                fires.add((spread_x, spread_y))
                maze[spread_x][spread_y] = FIRE
    return maze, fires

def generateValidFireMaze(dim, p):
    while True:
        maze = generateMaze(dim,p)
        tmaze_p, tpath_p = bdBFS(maze)
        # if generated maze doesn't have a path from top left to bottom right try again
        if not tpath_p:
            continue
        resetMaze(tmaze_p)
        tmaze_p, tpath_p = bdBFS(np.rot90(tmaze_p))
        # if it has a path from top right to bottom left return otherwise keep looping
        if tpath_p:
            return resetMaze(tmaze_p)

#base line solution just takes a path (shortest path) and tries it
def trySurvivingBaseLine(maze, path, q):
    dim = len(maze)
    onFire = set()
    #add initial fire location to set
    onFire.add((0,dim-1))
    #marking as visited just for visualization purposes
    maze[0][0] = VISITED
    maze[0][dim-1] = FIRE
    for x,y in path[1:]:
        #move on step at a time
        maze[x][y] = VISITED
        #and after moving simulate the fires spreading
        maze, onfire = simulateFire(maze, onFire, q)
        #if we got set on fire we lose, return failure, maze, where we died, and current set of fires
        if (x, y) in onFire:
            return False, maze, (x,y), onFire
    #made it to end without getting set on fire so success
    return True, maze, None, onFire

#driver for baseline, unneccesary for analysis, use comparisonFireDriver for comparing new algorithm
#to baseline on same set of mazes
def baseLineFireDriver(qs = [0.1 * x for x in range(1,11)], dim = 40, p = 0.305, algo_choice = bdBFS, trials = 100):
    print("baseline fire.py dim=" + str(dim)+" trials=" + str(trials))
    data = {round(x,2):0 for x in qs}
    for i in range(trials):
        #for each trial gnerate a valid maze
        maze = generateValidFireMaze(dim,p)
        #get shortest path using bdBFS
        maze, path= algo_choice(maze)
        #reset maze solely for visualization purposes
        maze = resetMaze(maze)
        for q in data.keys():
            success, maze, burn_location, fire_at_termination =  trySurvivingBaseLine(maze, path, q)
            #if success log it
            if success:
                data[q] +=1
            #reset Maze to remove fires
            maze = resetMaze(maze)
        #printing every so often for responsiveness
        if (i+1) %10 == 0:
            print(data)
    return data

def min_manhattan_distance_to_nearest_fire(fires, x , y):
    return min([dist_manhattan(x,y,fx,fy) for fx, fy in fires])

def min_euclid_distance_to_nearest_fire(fires, x , y):
    return min([dist_euclid(x,y,fx,fy) for fx, fy in fires])

#A star function for smarter maze runner we run this at every step to find a path to goal
#it return the next step we should take and the rest of the found path (unused)
def A_star_fire(maze, fires, x, y, weight = 1, heuristic_modifier = min_manhattan_distance_to_nearest_fire):
    n = len(maze)
    parents = [[(-1, -1)] * (n) for t in range(n)]
    #mark the start locations parent uniquely
    parents[x][y] = (-2,-2)
    #initializes priority queue
    heap = [(0,0,(x,y))]
    while heap and parents[n-1][n-1] == (-1,-1):
        #get highest priority element from fringe
        prio, length, coordinates = heapq.heappop(heap)
        x, y = coordinates
        #for each neighbor
        for dx, dy in dirs:
            new_x = x + dx
            new_y = y + dy
            #if in bounds of maze, and empty (not blocked or on fire) and unvisited
            if isValid(maze, new_x, new_y) and parents[new_x][new_y] == (-1,-1):
                parents[new_x][new_y] = (x,y) # mark visited
                manhattan_distance_to_goal = (n -1 - new_x + n - 1 - new_y)
                # modify simple manhattan with modifier (manhattan or euclidean distance to min_euclid_distance_to_nearest_fire)
                modifier = weight * heuristic_modifier(fires, new_x, new_y)
                # priority is basic manhattan adjusted so higher priority (higher actual priority means lesser number
                # for the below number since min heap) given to nodes farther from fire
                priority =  length + 1 + manhattan_distance_to_goal - modifier
                #add to fringe
                heapq.heappush(heap, (priority, length + 1, (new_x,new_y)))
    #failure if we couldn't find a path (since maze was originally solvable this means fire blocks us)
    if parents[n-1][n-1] == (-1,-1):
        return -1, -1, None
    #reconstruct path using parents (note we never add the starting point)
    path = []
    parent_i = n -1
    parent_j = n -1
    while parents[parent_i][parent_j] != (-2,-2):
        path.append((parent_i, parent_j))
        parent_i, parent_j = parents[parent_i][parent_j]
    path.reverse() #reverse cuz we want the last coordinates added, but we could easily just take from end of path
    #return the first step along path and the path itself
    return path[0][0], path[0][1], path

def smarterFireSurvivor(maze, q, weight = 1, heuristic_modifier = min_manhattan_distance_to_nearest_fire):
    #initialize maze and fire like in baseline
    n = len(maze)
    maze[0][n-1] = FIRE
    maze[0][0] = VISITED
    fires = set()
    fires.add((0,n-1))
    curr_x = 0
    curr_y = 0
    smart_path = [(0,0)] #keeps track of current path (unused)
    while curr_x != n-1 or curr_y != n-1: #not at end
        #at each step compute an 'optimal' path, here x,y are the first step along the path
        x, y, path = A_star_fire(maze, fires, curr_x, curr_y, weight = weight, heuristic_modifier = heuristic_modifier)

        smart_path.append((x,y))

        #if A_star_fire failed then fire has blocked us, so we have failed
        if x == -1 and y == -1:
            return False, maze, (x,y), fires, smart_path

        #simulate the fire
        maze, fires = simulateFire(maze, fires, q)

        #if our new location got set on fire then failure
        if maze[x][y] == FIRE:
            return False, maze, (x,y), fires, smart_path

        #take a step
        maze[x][y] = VISITED
        curr_x = x
        curr_y = y

    return True, maze, None, fires, smart_path

#driver to test different weights for same function
def smarterFireDriver(qs = [0.1 * x for x in range(1,11)], dim = 40, p = 0.305, trials = 100, smarterFireFunction = smarterFireSurvivor, weights = [1], trials_per_maze = 10):
    print("smarter fire.py dim=" + str(dim)+" trials=" + str(trials))
    #initialize a map from q to a map from weight to successes
    data = {round(x,2):{round(weight,2):0 for weight in weights} for x in qs}
    for i in range(trials):
        #new maze for each trial
        maze = generateValidFireMaze(dim,p)
        #for each q for each weight repeat trials_per_maze times to reduce noise
        for q in data.keys():
            for weight in data[q].keys():
                for k in range(trials_per_maze):
                    success, maze, burn_location, fire_at_termination, path =  smarterFireFunction(maze, q, weight = weight)
                    # mark success if we made it to end, and then reset maze
                    if success:
                        data[q][weight] +=1
                    maze = resetMaze(maze)
        #print out data every so often because this takes some time to run
        if (i+1) %10 == 0:
            for q in data.keys():
                print("q=" + str(q) + "  : " + str(data[q]))
            for _ in range(3):
                print()
    return data

#compares our smarter fire function to baseline using same mazes
#number of mazes = trials, repeats per maze = trials_per_maze
#so total number of trials per method per q is trials * trials_per_maze
#10 trials per maze to reduce noise
def comparisonFireDriver(qs = [0.1 * x for x in range(1,11)], dim = 40, p = 0.305, trials = 100, smarterFireFunction = smarterFireSurvivor, weight = 0.0, trials_per_maze = 10):
    print("comparison fire.py dim=" + str(dim)+" trials=" + str(trials) + " weight=" + str(weight))
    #data maps each q to a list
    #first item in list is number of baseline successes
    #second is number of manhattan modifier successes
    #third is number of euclid modifier successes
    data = {round(x,2):[0,0,0] for x in qs}
    for i in range(trials):
        #generate a new maze per trial
        maze = generateValidFireMaze(dim,p)
        #find a path to use for baseline, bdBFS for computing shortest path quickly
        maze, path = bdBFS(maze)
        #reset maze and then make copies to use same maze with both smarter A* versions
        base_maze = resetMaze(maze)
        manhattan_maze = deepcopy(maze)
        euclid_maze = deepcopy(maze)
        #repeat each maze multiple times per q to reduce noise
        for q in data.keys():
            for _ in range(trials_per_maze):
                #run with a_star modified by both manhattan distance and euclidean distance to fire
                #for comparison to baseline
                manhattan_success, manhattan_maze, _, _, _ =  smarterFireFunction(manhattan_maze, q, weight = weight, heuristic_modifier = min_manhattan_distance_to_nearest_fire)
                euclid_success, euclid_maze, _, _, _ =  smarterFireFunction(euclid_maze, q, weight = weight, heuristic_modifier = min_euclid_distance_to_nearest_fire)
                base_success, base_maze, base_burn_location, base_fire_at_termination =  trySurvivingBaseLine(base_maze, path, q)
                #mark successes
                if base_success:
                    data[q][0] +=1
                if manhattan_success:
                    data[q][1] +=1
                if euclid_success:
                    data[q][2] +=1

                #reset so we can reuse the maze
                base_maze = resetMaze(base_maze)
                manhattan_maze = resetMaze(manhattan_maze)
                euclid_maze = resetMaze(euclid_maze)
        #print every so often
        if (i+1) %10 == 0:
            print(data)
    return data

#analysis done using dim 20 since dim 40 takes a while to run

# smarterFireDriver(dim = 20, trials = 200, smarterFireFunction = smarterFireSurvivor, weights = [x*.1 for x in range(11)], trials_per_maze = 5)
# baseLineFireDriver(dim=20, trials = 200)
comparisonFireDriver(dim=20, trials = 500, smarterFireFunction = smarterFireSurvivor, weight = .2)
