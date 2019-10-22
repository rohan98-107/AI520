import sys
sys.path.append('..')
from MineSweeper import *
from collections import deque
from copy import deepcopy
import sys
import time
from Agent import *
from LinAlg import *
from BruteForceAgent import *
from BruteForceAgentWithoutLinAlg import *


def totalComparisonGameDriver(dim, density, trials, useMineCount = False):
    print("total comparison, dim {}, density {}, trials {}, useMineCount={}".format(dim,density,trials,useMineCount))
    num_mines = int(density*(dim**2))
    baseline_cumulative_time = 0
    baseline_cumulative_rate = 0
    brute_cumulative_time = 0
    brute_cumulative_rate = 0
    la_cumulative_time = 0
    la_cumulative_rate = 0
    la_brute_cumulative_time = 0
    la_brute_cumulative_rate = 0
    for i in range(trials):
        game = MineSweeper(dim, num_mines)
        order = [i for i in range(dim**2)]
        random.shuffle(order)
        brute_agent = brute_force_no_lin_alg_agent(game, useMineCount, order)
        la_brute_agent = brute_force_agent(game,useMineCount,order)
        baselineAgent = agent(game, order)
        la_agent = lin_alg_agent(game,useMineCount,order)
        brute_agent.solve()
        la_brute_agent.solve()
        baselineAgent.solve()
        la_agent.solve()
        baseline_cumulative_time+=baselineAgent.totalSolveTime
        baseline_cumulative_rate+=baselineAgent.mineFlagRate*100
        brute_cumulative_time+=brute_agent.totalSolveTime
        brute_cumulative_rate+=brute_agent.mineFlagRate*100
        la_cumulative_time+=la_agent.totalSolveTime
        la_cumulative_rate+=la_agent.mineFlagRate*100
        la_brute_cumulative_time+=la_brute_agent.totalSolveTime
        la_brute_cumulative_rate+=la_brute_agent.mineFlagRate*100
        if i % 10 == 9:
            print('\n\n\n\n\nFinished {} trials:'.format(i+1))
            print('\tBaseline: {} seconds \tBrute {} seconds \tLin alg {} seconds \tCombined {} seconds'.format(\
                round(baseline_cumulative_time/(i+1),2),round(brute_cumulative_time/(i+1),2),round(la_cumulative_time/(i+1),2),round(la_brute_cumulative_time/(i+1),2)))
            print('\tBaseline: {}% \tBrute {}% \t\tLin alg {}% \t\tCombined {}%'.format(\
                round(baseline_cumulative_rate/(i+1),2),round(brute_cumulative_rate/(i+1),2),round(la_cumulative_rate/(i+1),2),round(la_brute_cumulative_rate/(i+1),2)))

def mineCountComparisonGameDriver(dim, density, trials, agent_choice):
    #print("total comparison, dim {}, density {}, trials {}, agent_choice: {}".format(dim,density,trials,agent_choice.__name__))
    num_mines = int(density*(dim**2))
    baseline_cumulative_time = 0
    baseline_cumulative_rate = 0
    mine_cnt_cumulative_time = 0
    mine_cnt_cumulative_rate = 0
    for i in range(trials):
        game = MineSweeper(dim, num_mines)
        order = [i for i in range(dim**2)]
        random.shuffle(order)
        baselineAgent = agent_choice(game,False,order)
        mine_cnt_agent = agent_choice(game,True,order)
        mine_cnt_agent.solve()
        baselineAgent.solve()
        baseline_cumulative_time+=baselineAgent.totalSolveTime
        baseline_cumulative_rate+=baselineAgent.mineFlagRate*100
        mine_cnt_cumulative_time+=mine_cnt_agent.totalSolveTime
        mine_cnt_cumulative_rate+=mine_cnt_agent.mineFlagRate*100
        '''
        if i % 10 == 9:
            print('\n\n\n\n\nFinished {} trials:'.format(i+1))
            print('\t{}: {} seconds \tUsing mine cnt: {} seconds'.format(\
                agent_choice.__name__,round(baseline_cumulative_time/(i+1),2),round(mine_cnt_cumulative_time/(i+1),2)))
            print('\t{}: {}% \tUsing mine cnt: {}%'.format(\
                agent_choice.__name__,round(baseline_cumulative_rate/(i+1),2),round(mine_cnt_cumulative_rate/(i+1),2)))
        '''

    return (baseline_cumulative_rate/trials), (mine_cnt_cumulative_rate/trials), (baseline_cumulative_time/trials), (mine_cnt_cumulative_time/trials)


# totalComparisonGameDriver(10,.3,1000)
# linearAlgebraWithBruteGameDriver(10,.2,"lin_alg_with_brute")
#mineCountComparisonGameDriver(10, .2, 1000, brute_force_agent)


