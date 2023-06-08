from PlanningPbm import getPlanningPbm
from utils import PriorityQueue
import time
from ranking import getRanking, getScheduleFromRanking
import numpy as np

PLAN = getPlanningPbm("mockel")
SD = PLAN.getFirstSchedule(randomFlag=False) # schedule deter

class TreeNode:
    def __init__(self, obj, bi, constraint, nextconstraint, parent=None, warmstart=None):
        self.bsup = obj
        self.binf = bi
        self.c = constraint
        self.nc = nextconstraint
        self.parent = parent
        self.warmstart = warmstart

    def copy(self):
        return TreeNode(self.bsup, self.binf, self.c, self.nc, self.parent)
    
def get_constraint(completionTimes, schedule, typeConst, prevConst):
    # Return the list [a,b,c]
    # where a is the machine and b and c are the two operations
    # in function of the constraint strategy

    # Inputs:
    # completionTimes: list of completion times for each operation from the ranking method
    # schedule: schedule that is the index basis
    # typeConst: type of constraint strategy
    # prevConst: list of constraints already added to the model


    ijList = [(prevConst[i], prevConst[j], prevConst[k]) for i, j, k in zip(range(0, len(prevConst), 4), range(1, len(prevConst), 4), range(2, len(prevConst), 4))]

    if typeConst == "CM-FO":
        for _, i in enumerate(PLAN.criticMach):            
            sorted_indices = np.argsort(completionTimes[i])
            if PLAN.nbOpPerMach[i] == 1:
                continue
            for j in range(PLAN.nbOpPerMach[i]):
                if sorted_indices[j] == 0:
                    continue
                if completionTimes[i][sorted_indices[j]] < completionTimes[i][sorted_indices[j-1]] + schedule[i][sorted_indices[j]][3]:
                    if (i, sorted_indices[j], sorted_indices[j-1]) not in ijList:
                        return [i,sorted_indices[j], sorted_indices[j-1]]
        return [None,None,None]
    
    elif typeConst == "CM-LV":
        maxx = 0
        maxi = -1
        maxj = -1
        maxj2 = -1
        for _, i in enumerate(PLAN.criticMach):
            sorted_indices = np.argsort(completionTimes[i])
            if PLAN.nbOpPerMach[i] == 1:
                continue
            for j in range(PLAN.nbOpPerMach[i]):
                if sorted_indices[j] == 0:
                    continue
                diff = completionTimes[i][sorted_indices[j-1]] + schedule[i][sorted_indices[j]][3] - completionTimes[i][sorted_indices[j]]
                if (diff > maxx) and ((i, sorted_indices[j], sorted_indices[j-1]) not in ijList):
                    maxx = diff
                    maxi = i
                    maxj = sorted_indices[j]
                    maxj2 = sorted_indices[j-1]
        if maxi != -1:
            return [maxi, maxj, maxj2]
            
        else:
            return [None,None,None]


    elif typeConst == "CM-LV-M":
        nbConst = 2
        maxTab = np.zeros((1,nbConst))
        maxiTab = np.ones((1,nbConst)) * -1
        maxjTab = np.ones((1,nbConst)) * -1
        maxj2Tab = np.ones((1,nbConst)) * -1
        for _, i in enumerate(PLAN.criticMach):
            sorted_indices = np.argsort(completionTimes[i])
            if PLAN.nbOpPerMach[i] == 1:
                continue
            for j in range(PLAN.nbOpPerMach[i]):
                if sorted_indices[j] == 0:
                    continue
                diff = completionTimes[i][sorted_indices[j-1]] + schedule[i][sorted_indices[j]][3] - completionTimes[i][sorted_indices[j]]
                min = np.min(maxTab)
                if (diff > min) and ((i, sorted_indices[j], sorted_indices[j-1]) not in ijList):
                    minIdx = np.argmin(maxTab)
                    maxTab[0][minIdx] = diff
                    maxiTab[0][minIdx] = i
                    maxjTab[0][minIdx] = sorted_indices[j]
                    maxj2Tab[0][minIdx] = sorted_indices[j-1]
        if (maxiTab==-1).all():
            return [None,None,None]
        else:
            idx = np.where(maxiTab != -1)[1]
            c = []
            for i in range(len(idx)):
                c = c + [int(maxiTab[0][idx[i]]), int(maxjTab[0][idx[i]]), int(maxj2Tab[0][idx[i]])]
            return c

    elif typeConst == "CM-5LV":
        maxTab = np.zeros((1,5))
        maxiTab = np.ones((1, 5)) * -1
        maxjTab = np.ones((1, 5)) * -1
        maxj2Tab = np.ones((1, 5)) * -1
        for _, i in enumerate(PLAN.criticMach):
            sorted_indices = np.argsort(completionTimes[i])
            if PLAN.nbOpPerMach[i] == 1:
                continue
            for j in range(PLAN.nbOpPerMach[i]):
                if sorted_indices[j] == 0:
                    continue
                diff = completionTimes[i][sorted_indices[j-1]] + schedule[i][sorted_indices[j]][3] - completionTimes[i][sorted_indices[j]]
                min = np.min(maxTab)
                if (diff > min) and ((i, sorted_indices[j], sorted_indices[j-1]) not in ijList):
                    minIdx = np.argmin(maxTab)
                    maxTab[0][minIdx] = diff
                    maxiTab[0][minIdx] = i
                    maxjTab[0][minIdx] = sorted_indices[j]
                    maxj2Tab[0][minIdx] = sorted_indices[j-1]
        if (maxiTab==-1).all():
            return [None,None,None]
        else:
            idx = np.where(maxiTab != -1)[0]
            r = np.random.randint(0,len(idx))
            return [int(maxiTab[0][idx[r]]), int(maxjTab[0][idx[r]]), int(maxj2Tab[0][idx[r]])]
            
    
    elif typeConst == "LJ":
        l = np.argsort(PLAN.nbOpPerJob)[::-1]
        idxOp = PLAN.getOpIndexInSched(schedule)
        for i, val in enumerate(l):
            for j in range(0,PLAN.nbOpPerJob[val]):
                m = idxOp[val][j][0]
                o = idxOp[val][j][1]
                sorted_indices = np.argsort(completionTimes[m])
                idx_o = np.where(sorted_indices == o)[0][0]
                if idx_o == 0:
                    continue                
                o2 = sorted_indices[idx_o-1]                
                if completionTimes[m][o] < completionTimes[m][o2] + schedule[m][o][3]:
                    if (m,o,o2) not in ijList:                         
                        return [m,o,o2]
        return [None,None,None]
    
    else:
        print("typeConst not implemented")
        return [None,None,None]
    
def branch_and_bound(typeConst="criticMach", maxIter=2**PLAN.nbOp, prioType = "borneInf", rankingConstraint = [], LS = False, timeRun = 60):
    # Implementation of the branch and bound algorithm

    # Inputs:
    #  - typeConst: type of constraint to use for the branching (constraint strategy)
    # - maxIter: maximum number of iterations
    # - prioType: type of priority to use for the search (expansion strategy 
    # - rankingConstraint: list of constraints to use for the ranking (set S)
    # - LS: boolean, if True, use the lukewarm start

    # Outputs:  bestNode, bestConst, bestIter, nbBreakBorne, maxLen, nbNoNextConst, optimizationTime, modelTime, minLB
    #  - bestNode: best node found
    #  - bestConst: constraints add to the intial model to get the best node
    #  - bestIter: iteration at which the best node was found
    #  - nbBreakBorne: number of times the branching was stopped because the lower bound was higher than the upper bound (pruning)
    #  - maxLen: maximum number of constraints added during the search
    #  - nbNoNextConst: number of times the branching was stopped because there was no more constraints to add (leaf reached)
    #  - optimizationTime: time spent in the optimization
    #  - modelTime: time spent in the model creation
    #  - minLB: minimum lower bound found at the end of the search

    # Initialization: output variables
    nbBreakBorne = 0
    nbNoNextConst = 0
    maxLen = 0
    optimizationTime = 0
    modelTime = 0

    # Initialization: get initial schedule from the ranking method
    ranking, end1,_, optiT, modelT = getRanking(PLAN, SD,setS=rankingConstraint)
    optimizationTime += optiT
    modelTime += modelT
    schedule, _ = getScheduleFromRanking(PLAN, SD, ranking)

    complRank, endRank,_, optiT, modelT = getRanking(PLAN, schedule,setS=rankingConstraint)
    optimizationTime += optiT
    modelTime += modelT
    borneInf = PLAN.getObjective(endRank)

    SCHED, compl = getScheduleFromRanking(PLAN, schedule, complRank)

    _, endReal = PLAN.getStartEnd(SCHED)
    borneSup = PLAN.getObjective(endReal)

    # Initialization: root node
    nextConst = get_constraint(compl, SCHED, typeConst, [])
    if LS:
        LS = complRank
    else:
        LS = []
    root = TreeNode(borneSup, borneInf, None, nextConst, parent=None, warmstart=LS)

    bestNode = root
    bestIter = 0
    bestConst = root.nc

    fringe = PriorityQueue()
    fringe.push(root, 0)

    iter = 0
    startTime = time.time()
    endTime = time.time()
    while len(fringe.heap) > 0 and (iter < maxIter or endTime-startTime < timeRun):
        priority, item = fringe.pop()

        if item.bsup < bestNode.bsup: # Save best node
            bestConst = get_constraint_from_node(item)
            bestIter = iter
            bestNode = item            

        if item.nc is not None: # If there is possible children
            if item.c is not None: # If this is not the root node, get the constraints from the node to the root
                constList = get_constraint_from_node(item)
            else: # If this is the root node
                constList = item.nc

            for i in range(2): # For each child
                ranking, endRank,_, optiT, modelT = getRanking(PLAN, SCHED, constraintsBB=constList+[i],setS=rankingConstraint, complTimeInit=item.warmstart)
                optimizationTime += optiT
                modelTime += modelT

                if endRank[0] == None: # If the child is infeasible, continue (not added to the fringe)
                    continue

                borneInf = PLAN.getObjective(endRank)

                if borneInf < bestNode.bsup: # If the child is not pruned
                    schedule, _ = getScheduleFromRanking(PLAN, SCHED, ranking)
                    _, endReal = PLAN.getStartEnd(schedule)
                    borneSup = PLAN.getObjective(endReal)
                    nextConst = get_constraint(ranking, SCHED, typeConst,constList+[i] )
                    
                    if nextConst[0] != None: # If there is a constraint to add
                        if LS:
                            LS_child = ranking
                        else:
                            LS_child = []
                        n = TreeNode(borneSup, borneInf, i, nextConst, parent=item, warmstart = LS_child)

                        if prioType == "GS-low": # Priotity depends on the expansion strategy
                            prio =borneInf
                        elif prioType == "DFS":
                            prio = - (-priority+1)
                        elif prioType == "GS-up":
                            prio = borneSup
                        elif prioType == "BFS":
                            prio = priority+1
                        else:
                            print("error type priority")

                        fringe.push(n,prio) # Add the child to the fringe
                    else:
                        nbNoNextConst += 1
                else:
                    nbBreakBorne += 1 
        iter+=1
        endTime = time.time()

    # Get the smallest lower bound of the nodes not expanded
    minLB = np.inf
    for i in range(len(fringe.heap)):
        if fringe.heap[i][2].binf < minLB:
            minLB = fringe.heap[i][2].binf

    return bestNode, bestConst, bestIter, nbBreakBorne, maxLen, nbNoNextConst, optimizationTime, modelTime, minLB

def get_constraint_from_node(node):
    # return the list of constraints from the node to the root
    constList = []
    tmpNode = node.copy()
    while tmpNode.parent is not None:
        constList.append(tmpNode.parent.nc[0])
        constList.append(tmpNode.parent.nc[1])
        constList.append(tmpNode.parent.nc[2])
        constList.append(tmpNode.c)
        tmpNode = tmpNode.parent
    constList.append(node.nc[0])
    constList.append(node.nc[1])
    constList.append(node.nc[2])
    return constList
  
if __name__ == "__main__":

    typeConstList = ["CM-FO", "CM-LV", "LJ", "CM5-LV"]
    typePrioList = ["DFS", "BFS", "GS-up", "GS-low"]
    setToTest = [[0,0,0,0,0,0,0], [0,0,1,0,0,0,0]] # set (1),(17)


    # TEST: utility of warm start
    # ---------------------------   

    f = open("bnb_result_rukewarm_start.txt", "w")  
    
    count = 0
    for c in setToTest:
        for i, tc in enumerate(typeConstList):
            for j, tp in enumerate(typePrioList):
                timeList = []
                optiTimeList = []
                modelTimeList = []
                for warmStartBool in [True, False]:
                    startTime = time.time()
                    _,_,_,_,_,_, optiTime, modelTime,_ = branch_and_bound(typeConst=tc, maxIter=10, prioType = tp, rankingConstraint = c, LS = warmStartBool)
                    endTime = time.time()
                    timeList.append(endTime-startTime)
                    optiTimeList.append(optiTime)
                    modelTimeList.append(modelTime)
                r = tc + " & " + tp + " & " + str(c) + " & " + str(round(timeList[0], 4)) + " & " + str(round(timeList[1], 4)) + " & " + str(round(optiTimeList[0], 4)) + " & " + str(round(optiTimeList[1], 4)) + " & " + str(round(modelTimeList[0], 4)) + " & " + str(round(modelTimeList[1], 4)) + "\\\\"
                if optiTimeList[0] < optiTimeList[1]:
                    count += 1
                print(r)
                f.write(r + "\n")
    f.close()
    print(count)