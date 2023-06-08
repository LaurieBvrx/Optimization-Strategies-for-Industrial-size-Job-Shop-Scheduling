import numpy as np
import time
import math
import random
import copy
from PlanningPbm import PlanningPbm, getPlanningPbm
import colorama
import matplotlib.pyplot as plt
from ranking import getRanking, getScheduleFromRanking
import itertools
import os
import ast

def simulated_annealing(p: PlanningPbm, schedule, start, endJob, initial_temperature=1000, cooling_rate=0.9999, stopping_temperature=0.01, max_iterations=40000, neighbor_size=20, neighType = 1):
    # Function to solve the planning problem using simulated annealing
    # p: PlanningPbm object
    # schedule: initial schedule
    # start: initial completion times of the operations
    # endJob: initial completion times of the jobs

    currentSchedule = copy.deepcopy(schedule)
    currentEnd = endJob.copy()
    currentObjective = p.getObjective(currentEnd)

    bestSchedule = copy.deepcopy(currentSchedule)
    bestStart = copy.deepcopy(start)
    bestEnd = currentEnd.copy()
    bestObjective = currentObjective

    temperature = initial_temperature
    iteration = 0

    while temperature > stopping_temperature and iteration < max_iterations:
        # if iteration % 1000 == 0:
        #     print("iteration: " + str(iteration))
        #     print("curr obj: " + str(currentObjective) + " best obj: " + str(bestObjective) + " temp: " + str(temperature))
        if neighType == 1:
            newSchedule, newStart, newEnd = get_new_solution1(p, currentSchedule, neighbor_size)
        elif neighType == 2:
            newSchedule, newStart, newEnd = get_new_solution2(p, currentSchedule, neighbor_size)
        elif neighType == 3:
            newSchedule, newStart, newEnd = get_new_solution3(p, currentSchedule, neighbor_size)
        else:
            print("Error: neighType should be 1, 2 or 3")
            return
        
        newObjective = p.getObjective(newEnd)

        if newObjective < currentObjective : # keep the new solution if it is better

            currentSchedule = copy.deepcopy(newSchedule)
            currentEnd = newEnd.copy()
            currentObjective = newObjective

            if newObjective < bestObjective and p.isScheduleFeasible(newSchedule):
                bestSchedule = copy.deepcopy(newSchedule)
                bestStart = copy.deepcopy(newStart)
                bestEnd = newEnd.copy()
                bestObjective = newObjective
        else: # keep the new solution with a probability that depends on the temperature
            r = random.random()
            proba = math.exp((currentObjective - newObjective) / temperature)
            if r < proba:
                currentSchedule = copy.deepcopy(newSchedule)
                currentEnd = newEnd.copy()
                currentObjective = newObjective

        temperature *= cooling_rate
        iteration += 1
    
    return bestSchedule, bestStart, bestEnd, bestObjective

def get_new_solution1(p: PlanningPbm, schedule, neighbors):
    # Function to generate a new solution by making a small random change to the current solution
    # Here, we swap two random operations on a random machine
    # => Implementation of the first neighborhood

    new_schedule = copy.deepcopy(schedule) 

    idx = np.where(p.nbOpPerMach > 1)[0] # get the indices of the machines with more than one operation
    idx = idx.tolist() # convert to list
    machine = random.choice(idx) # choose a random machine
    i = random.sample(range(len(schedule[machine])),1)[0] # choose a random operation on the machine
    j = random.sample(range(max(0,i-neighbors), min(i+neighbors, len(schedule[machine])-1)),1)[0]
    new_schedule[machine][i], new_schedule[machine][j] = new_schedule[machine][j], new_schedule[machine][i]
    new_start, new_end = p.getStartEnd(new_schedule)

    return new_schedule, new_start, new_end

def get_new_solution2(p: PlanningPbm, schedule, neighbors):
    # Function to generate a new solution by making a small random change to the current solution
    # Here, we swap three random operations on a random machine
    # => Implementation of the second neighborhood

    new_schedule = copy.deepcopy(schedule)

    idx = np.where(p.nbOpPerMach > 1)[0] # get the indices of the machines with more than one operation
    idx = idx.tolist() # convert to list
    machine = random.choice(idx) # choose a random machine

    if p.nbOpPerMach[machine] == 2: # do the swap with 2 operations if there are only 2 operations on the machine
        i = 0
        j = 1
        new_schedule[machine][i], new_schedule[machine][j] = new_schedule[machine][j], new_schedule[machine][i]
        new_start, new_end = p.getStartEnd(new_schedule)
    else:
        i = random.sample(range(len(schedule[machine])),1)[0] # choose a random operation on the machine
        j = random.sample(range(max(0,i-neighbors), min(i+neighbors, len(schedule[machine])-1)),1)[0]
        k = random.sample(range(max(0,i-neighbors), min(i+neighbors, len(schedule[machine])-1)),1)[0]
        new_schedule[machine][i], new_schedule[machine][j], new_schedule[machine][k] = new_schedule[machine][j], new_schedule[machine][k], new_schedule[machine][i]
        new_start, new_end = p.getStartEnd(new_schedule)

    return new_schedule, new_start, new_end

def get_new_solution3(p: PlanningPbm, schedule, neighbors):
    # Function to generate a new solution by making a small random change to the current solution
    # Here, we swap four random operations.
    # We test the 4! = 24 possible combinations and keep the best one
    # => Implementation of the third neighborhood

    new_schedule = copy.deepcopy(schedule)

    idx = np.where(p.nbOpPerMach > 1)[0] # get the indices of the machines with more than one operation
    idx = idx.tolist() # convert to list
    machine = random.choice(idx) # choose a random machine
    i = random.sample(range(len(schedule[machine])),1)[0] # choose a random operation on the machine
    j = random.sample(range(max(0,i-neighbors), min(i+neighbors, len(schedule[machine])-1)),1)[0]
    k = random.sample(range(max(0,i-neighbors), min(i+neighbors, len(schedule[machine])-1)),1)[0]
    l = random.sample(range(max(0,i-neighbors), min(i+neighbors, len(schedule[machine])-1)),1)[0]

    opi = new_schedule[machine][i]
    opj = new_schedule[machine][j]
    opk = new_schedule[machine][k]
    opl = new_schedule[machine][l]
    op = [opi, opj, opk, opl]

    # all permutations of the list [0,1,2,3]
    lst = [0, 1, 2, 3]
    permutations = list(itertools.permutations(lst))

    best_obj = np.inf
    best_schedule = None
    best_start = None
    best_end = None

    for perm in permutations:
        new_schedule[machine][i], new_schedule[machine][j], new_schedule[machine][k], new_schedule[machine][l] = op[perm[0]], op[perm[1]], op[perm[2]], op[perm[3]]
        new_start, new_end = p.getStartEnd(new_schedule)
        obj = p.getObjective(new_end)
        if obj < best_obj:
            best_obj = obj
            best_schedule = copy.deepcopy(new_schedule)
            best_start = new_start.copy()
            best_end = new_end.copy()
    
    return best_schedule, best_start, best_end
    


def plotNeighborSize(ns, objList, objectiveRef,neighType=1):

    plt.xlabel("window size w", fontsize=12)
    plt.ylabel("Objective value", fontsize=12)
    plt.axhline(y=objectiveRef, color='black', linestyle='--')
    plt.scatter(ns, np.mean(objList, axis=1), marker='o',zorder=3,c='blue')

    # Add value labels to each point
    i = 0
    for x, y in zip(ns, np.mean(objList, axis=1)):
        plt.text(x, y + np.std(objList[i]) +10000, str(int(y)), ha='center', va='bottom',rotation=90)
        i += 1
    
    # Error bar
    plt.errorbar(ns, np.mean(objList, axis=1), yerr=np.std(objList, axis=1), fmt='none', ecolor='blue', capsize=3, zorder=2)
    plt.xticks(ns)
    plt.legend(["Reference"],fontsize=10,loc='upper right')
    plt.show()
    
    # Save figure
    if neighType==1:
        x="1"
    elif neighType==2:
        x="2"
    elif neighType==3:
        x="3"
    else:
        x="unknown"
    plt.savefig("SAresults/neighborSize" + x + ".png")

if __name__ == "__main__":

    # --------------------------------------------
    # |       INFORMATION ABOUT THE PROBLEM      |
    # --------------------------------------------

    plan = getPlanningPbm("mockel2")
    scheduleDeter = plan.getFirstSchedule(randomFlag=False)
    startDeter, endJobDeter = plan.getStartEnd(scheduleDeter)
    deterObjective = plan.getObjective(endJobDeter)
    print("Initial objective value: " + str(deterObjective))


    # --------------------------------------------
    # |         SIMULATED ANNEALING              |
    # --------------------------------------------

    TYPE = 1
    ns = [1,2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    nbtests = 10
    objList = np.zeros((len(ns), nbtests))
    for i in range(len(ns)):
        print("Neighbor size: " + str(ns[i]))
        for j in range(nbtests):            
            # Simulated annealing
            startTimeAnnealing = time.time()
            if TYPE == 1 or TYPE ==2 :
                best_schedule, best_start, best_end, best_obj = simulated_annealing(plan,scheduleDeter, startDeter, endJobDeter, initial_temperature=1000, cooling_rate=0.9999, neighbor_size=ns[i], neighType=TYPE)
            else:
                best_schedule, best_start, best_end, best_obj = simulated_annealing(plan,scheduleDeter, startDeter, endJobDeter, initial_temperature=100, cooling_rate=0.9965, neighbor_size=ns[i], neighType=TYPE)
            endTimeAnnealing = time.time()
            objList[i,j] += best_obj
            print("Time : " + str(endTimeAnnealing - startTimeAnnealing) + " , Best objective value: " + str(best_obj))
    
    plotNeighborSize(ns, objList, deterObjective, TYPE)
    

    # --------------------------------------------
    # |            RANKING + SA                  |
    # --------------------------------------------

    # Ranking method
    S = [1,1,0,0,1,1,1] # 24
    complG, endJobG, _,_,_ = getRanking(plan, scheduleDeter, setS=S)
    lb = plan.getObjective(endJobG)
    newSched, _ = getScheduleFromRanking(plan, scheduleDeter, complG)
    compl, endJobRanking = plan.getStartEnd(newSched)
    ub = plan.getObjective(endJobRanking)
    print("Ranking: Borne inf = " + str(lb) + ", Objective = " + str(ub))

    #SA
    best_schedule, best_start, best_end, best_obj = simulated_annealing(plan,newSched, compl, endJobRanking, neighType=1)
    print("SA: Best objective = " + str(best_obj))
    plan.saveSchedule(best_schedule, "schedule_rank_SA.txt")

    # Checking
    scheduleVerif = plan.loadSchedule("schedule_rank_SA.txt")
    complVerif, endVerif = plan.getStartEnd(scheduleVerif)
    print("Checking: Best objective = ", plan.getObjective(endVerif))


    # --------------------------------------------
    # |              DIVE + SA                   |
    # --------------------------------------------

    # DIVE (read schedule found by DIVE)
    currPath = os.getcwd()
    # Read the contents of the text file
    with open(currPath + "/DIVEresults/dives_2h_sched.txt", 'r') as file:
        contents = file.read()
    # Convert the contents into a list of lists of tuples
    schedule = ast.literal_eval(contents)
    compl, end = plan.getStartEnd(schedule)
    print("Dive: Objective = ", plan.getObjective(end))

    # SA
    best_schedule, best_start, best_end, best_obj = simulated_annealing(plan,schedule, compl, end, neighType=1)
    print("SA: Best objective = : " + str(best_obj))
    plan.saveSchedule(best_schedule, "schedule_dive_SA.txt")

    # Checking
    scheduleVerif = plan.loadSchedule("schedule_dive_SA.txt")
    complVerif, endVerif = plan.getStartEnd(scheduleVerif)
    print("Checking: Best objective = ", plan.getObjective(endVerif))