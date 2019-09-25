from MazeRun import *

def hard_dfs_maze(dim):
    maze = np.zeros((dim, dim))
    maze[:,1] = BLOCKED
    maze[dim-2,:] = BLOCKED
    maze[0][1] = EMPTY
    maze[dim-2][0] = EMPTY
    maze[dim-1][1] = EMPTY
    return maze

maze, path, maxsize = DFS_again(hard_dfs_maze(175))
printMazeHM_orig(maze)
print(path)
print(maxsize)
