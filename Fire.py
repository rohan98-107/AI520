from MazeRun import *

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

def trySurviving(dim = 110, p = 0.305, algo_choice = A_star_manhattan, q = .5):
    maze = generateValidFireMaze(dim,p)
    maze, path, maxsize = algo_choice(maze)
    resetMaze(maze)
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
            return False, maze, path, (x,y), onFire
    return True, maze, path, None, onFire

data = []
for q in [0.05 * x for x in range(1,21)]:
    successes = 0
    for i in range(100): #probably want this to be > 1000 instead of 100 as there's a good amount of variance with 100
        success, maze, path, burn_location, fire_at_termination =  trySurviving(dim = 20, q=q)
        if success:
            # print(success)
            successes += 1
            # print(q)
            # printMaze(maze)
            # print(path)
            # print()
            # print(burn_location)
            # print()
            # print(fire_at_termination)
    data.append((q,successes))
print(data)
