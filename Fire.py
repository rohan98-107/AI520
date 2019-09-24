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

def trySurviving(maze, path, q):
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

def fireDriver(qs = [0.1 * x for x in range(1,11)], dim = 110, p = 0.305, algo_choice = A_star_manhattan, mazes_per_q = 100, trials_per_maze = 5):
    data = {round(x,2):0 for x in qs}
    for i in range(mazes_per_q):
        maze = generateValidFireMaze(dim,p)
        maze, path, maxsize = algo_choice(maze)
        maze = resetMaze(maze)
        for q in data.keys():
            for _ in range(trials_per_maze):
                success, maze, burn_location, fire_at_termination =  trySurviving(maze, path, q)
                if success:
                    data[q] +=1
                maze = resetMaze(maze)
        if (i+1) %10 == 0:
            print(data)
    return data

fireDriver()
