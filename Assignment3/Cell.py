PRESENT = 1
ABSENT = 0

FLAT = 0.1
HILLY = 0.3
FOREST = 0.7
MAZE = 0.9

VISTED = True
UNVISITED = False

class cell:

    def __init__(self):
        x = random.randint(1, 100)
        self.target = ABSENT
        self.status = UNVISITED
        if x <= 20:
            self.terrain = FLAT
        elif 20 < x <= 50:
            self.terrain = HILLY
        elif 50 < x <= 80:
            self.terrain = FOREST
        else:
            self.terrain = MAZE

    def getTerrain(self):
        return self.terrain

    def getStatus(self):
        return self.visited
