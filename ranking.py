from PlanningPbm import getPlanningPbm, PlanningPbm
import gurobipy
from gurobipy import *
from gurobipy import quicksum, GRB
import numpy as np
import itertools
import time
import matplotlib.pyplot as plt

def getRanking(plan: PlanningPbm, schedule, constraintsBB=[None], complTimeInit = [], setS = []):
    # Compute the ranking of the operations in the schedule wuth the relaxation of the problem
    # Input:
    #   - plan: planning problem
    #   - schedule: schedule of the planning problem
    #   - constraintsBB: list of constraints to apply in the branch and bound
    #   - complTimeInit: initial completion time for the schedule
    #   - setS: list of constraints to apply in the relaxation (set S)
    # Output:
    #   - compl: ranking of the operations in the schedule
    #   - end: ranking of the jobs in the schedule
    #   - nbConst: number of constraints added in the relaxation
    #   - model.Runtime: time to solve the relaxation (optimization time)
    #   - round(endTime-startTime, 4): total time to solve the relaxation

    startTime = time.time()

    # Create a new model
    # ------------------
    model = gurobipy.Model("schedule")
    model.Params.LogToConsole = 0 # disable console output

    # Create variables
    # ------------------
    Completion = {}
    End= {}
    for m in range(plan.nbMach):
        for o in range(plan.nbOpPerMach[m]):
            Completion[m,o]=model.addVar(vtype=GRB.CONTINUOUS,lb=0,name="c_%s_%s"%(m,o))
            if complTimeInit != []:
                Completion[m,o].start = complTimeInit[m][o]
    for j in range(plan.nbJobs):
        End[j]=model.addVar(vtype=GRB.CONTINUOUS,lb=0,name="e_%s"%(j))

    # Set objective
    # -------------
    model.setObjective(quicksum([(End[j]) for j in range(plan.nbJobs)]), GRB.MINIMIZE)     

    # Add initial constraints
    # -----------------------

    # Constraints on completion time individually for each operation
    for m in range(plan.nbMach):
        for o in range(0,plan.nbOpPerMach[m]):
            model.addConstr(Completion[m,o] >= schedule[m][o][2] + schedule[m][o][3])

    idxOp = plan.getOpIndexInSched(schedule)
    for j in range(plan.nbJobs):

        # deal with jobs with no operation (i.e. op only on machine with infinite capacity)
        if plan.nbOpPerJob[j] == 0:
            continue

        # Get end 
        oIdx = max(plan.nbOpPerJob[j]-1, 0)
        m = idxOp[j][oIdx][0]
        mo = idxOp[j][oIdx][1]
        model.addConstr(End[j] == Completion[m,mo]+schedule[m][mo][4])

        # Precedence constraints
        for o in range(1,plan.nbOpPerJob[j]):            
            m1, mo1 = idxOp[j][o][0], idxOp[j][o][1] # operation o of job j
            m0, mo0 = idxOp[j][o-1][0], idxOp[j][o-1][1]  # operation o-1 of job j
            model.addConstr(Completion[m1,mo1] >= Completion[m0,mo0] + schedule[m0][mo0][4] + schedule[m1][mo1][2] + schedule[m1][mo1][3])


    # Add constraints depending on the set S
    # --------------------------------------

    nbConst = 0
    if len(setS)>0:
        
        if setS[0] == 1: # Contraints on each pair of operation on the same machine            
            for m in range(plan.nbMach):
                for o1 in range(0,plan.nbOpPerMach[m]):
                    for o2 in range(o1+1,plan.nbOpPerMach[m]):
                        lhs = Completion[m,o1] * schedule[m][o1][3] + Completion[m,o2] * schedule[m][o2][3]
                        rhs1 = 1/2 * ((schedule[m][o1][3] + schedule[m][o2][3])**2)
                        rhs2 = 1/2 * (schedule[m][o1][3]**2 + schedule[m][o2][3]**2)
                        model.addConstr(lhs >= rhs1 + rhs2)
                        nbConst += 1

        if setS[1] == 1 or setS[2] == 1: 
            for m in range(plan.nbMach):
                lhs = 0
                rhs1 = 0
                rhs2 = 0
                for o in range(plan.nbOpPerMach[m]):
                    lhs += Completion[m,o] * schedule[m][o][3]
                    rhs1 += schedule[m][o][3]
                    rhs2 += schedule[m][o][3]**2
                if setS[1] == 1: # constraints on all operations
                    model.addConstr(lhs >= 1/2 * rhs1**2 +  1/2 * rhs2)
                    nbConst += 1
                
                if setS[2] == 1: # constraint on all operations expect one                    
                    for o in range(plan.nbOpPerMach[m]):
                        tmpLhs = lhs - Completion[m,o] * schedule[m][o][3]
                        tmpRhs1 = rhs1 - schedule[m][o][3]
                        tmpRhs2 = rhs2 - schedule[m][o][3]**2
                        model.addConstr(tmpLhs >= 1/2 * tmpRhs1**2 +  1/2 * tmpRhs2)
                        nbConst += 1
        
        sizeSet = []
        startNb = 2 
        for i in range(3,len(setS)):
            if setS[i] ==1:
                sizeSet.append(startNb)
            startNb += 1
        
        if len(sizeSet)>0: # constraints on successive set of operations on the same machine

            for m in range(plan.nbMach):
                for size in sizeSet:
                    for o1 in range(0,plan.nbOpPerMach[m]-size+1):
                        lhs = 0
                        rhs1 = 0
                        rhs2 = 0
                        for o in range(size):
                            lhs += Completion[m,o1+o] * schedule[m][o1+o][3]
                            rhs1 += schedule[m][o1+o][3]
                            rhs2 += schedule[m][o1+o][3]**2
                        model.addConstr(lhs >= 1/2 * rhs1**2 +  1/2 * rhs2)
                        nbConst += 1


    # Add constraints from B&B
    # ------------------------
    if constraintsBB[0] is not None:  # Constraints add from B&B
        nbConst = int(len(constraintsBB)/4)
        for i in range(nbConst):
            
            m = constraintsBB[4*i]
            o1 = constraintsBB[4*i+1]
            o2 = constraintsBB[4*i+2]
            op = constraintsBB[4*i+3]
            #print("Ranking ", m, o1, o2, op)
            if op == 0:
                model.addConstr(Completion[m,o1] >= Completion[m,o2] + schedule[m][o1][3])        
                nbConst += 1      
            else:
                model.addConstr(Completion[m,o2] >= Completion[m,o1] + schedule[m][o2][3])
                nbConst += 1
    
    # Optimize model
    # ---------------
    model.optimize()

    # Get the solution
    # ---------------
    if model.status == 2:
        cmplSol = model.getAttr('x', Completion)
        compl = []
        for _, value in enumerate(plan.nbOpPerMach):
            compl.append(np.zeros(value))
        for m in range(plan.nbMach):
            for o in range(plan.nbOpPerMach[m]):
                compl[m][o] = cmplSol[m,o]

        end = []
        endSol = model.getAttr('x', End)
        for j in range(plan.nbJobs):
            end.append(endSol[j])

        endTime = time.time()
        
        return compl, end, nbConst, model.Runtime, round(endTime-startTime, 4)
        
    else:
        endTime = time.time()
        return None, [None], nbConst, model.Runtime, round(endTime-startTime, 4)

def getScheduleFromRanking(plan: PlanningPbm, schedule, ranking):
    # sort operations according to ranking
    # Output: new schedule and ranking ordered
    newSched = [[] for i in range(plan.nbMach)]
    compl = [[] for i in range(plan.nbMach)]
    for i in range(plan.nbMach):
        sorted_indices = np.argsort(ranking[i])
        for j in range(plan.nbOpPerMach[i]):
            newSched[i].append(schedule[i][sorted_indices[j]])
            compl[i].append(ranking[i][sorted_indices[j]])
    return newSched, compl


if __name__ == "__main__":

    # Get planning problem
    # --------------------
    plan = getPlanningPbm("mecanelec")
    scheduleDeter = plan.getFirstSchedule(randomFlag=False)
    

    # TEST: set S, subset of operations
    # --------------------------------
    combinations = list(itertools.product([0,1], repeat=7)) # 7 constraints or repeat = 4 for 4 constraints

    f = open("RankingResults.txt", "w") #open text file

    lbList = np.zeros(len(combinations))
    ubList = np.zeros(len(combinations))
    nbConstList = np.zeros(len(combinations))

    for i in range(len(combinations)):

        c = list(combinations[i]) # set S
        startTime = time.time()
        complG, endJobG, nbConst,_,_ = getRanking(plan, scheduleDeter,setS=c)
        endTime = time.time()

        borneInf = plan.getObjective(endJobG)

        newSched, _ = getScheduleFromRanking(plan, scheduleDeter, complG)        
        _, endJobRanking = plan.getStartEnd(newSched)
        objCurr = plan.getObjective(endJobRanking)

        r = str(i+1) + " & " + str(c) + " & " + str(int(borneInf)) + " & " + str(int(objCurr)) + " & " + str(round(endTime-startTime, 4)) + " & " + str(nbConst) + "\\ \n"
        f.write(r)
        lbList[i] = borneInf
        ubList[i] = objCurr
        nbConstList[i] = nbConst

    f.close()

    plt.figure()
    plt.plot(lbList, 'o')
    plt.plot(nbConstList, 'o')
    plt.legend(["Lower bound", "Number of constraints"])
    plt.xlabel("Identification number of the set S")
    plt.show()

    plt.figure()
    plt.plot(ubList, 'o')
    plt.plot(nbConstList, 'o')
    plt.legend(["Upper bound", "Number of constraints"])
    plt.xlabel("Identification number of the set S")
    plt.show()