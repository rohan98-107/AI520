import random

PRESENT = 1
ABSENT = 0

FLAT = 0.1
HILLY = 0.3
FOREST = 0.7
MAZE = 0.9

VISTED = True
UNVISITED = False

class landCell:

    def __init__(self):
        x = random.randint(1, 100)
        self.target = ABSENT
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


class agentCell:

    def __init__(self,landCell):

        self.belief = 0
        self.terrain = landCell.terrain
        self.status = UNVISITED

    def getBelief(self):
        return self.belief

    def getTerrain(self):
        return self.terrain

    def getStatus(self):
        return self.status
