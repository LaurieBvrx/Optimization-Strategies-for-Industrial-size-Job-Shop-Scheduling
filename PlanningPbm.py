import numpy as np
import os
import random
from utils import *
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import gurobipy as gp
from gurobipy import quicksum, GRB
import time


class PlanningPbm:

    # Class to store the job shop problem from the files

    def __init__(self, pathJob, pathOp, pathMach):
        self.pathJob = pathJob
        self.pathOp = pathOp
        self.pathMach = pathMach

        self.nbJobs = self.get_njobs() # number of jobs
        self.totNbOp = self.getNbLines(pathOp) - 1 # total number of operations
        self.totNbMach = self.getNbLines(pathMach) - 1 # total number of machines

        self.machDict, self.machCapInfDict = self.getMachinesDict() # dictionary with the machines names as keys and the index of the machine as value
        self.nbMach = len(self.machDict) # number of machines with finite capacity
        self.nbOpPerJob, self.totNbOpPerJob = self.getNbOpPerJob()
        self.nbOpPerMach = self.getNbOpPerMach() # number of operations per machine

        self.listOp = self.getListOp() # list of operations for each job

        self.nbOp = 0
        for i in range(self.nbJobs):
            self.nbOp += len(self.listOp[i])
        self.machOrder = self.getMeanOpPerMach() # first index = largest mean

        self.criticMach = self.getCriticMachine() # first index = largest duration

    # INFORMATION ABOUT THE PROBLEM
    def getListOfJobs(self, path):
        # return the list of job names in the file
        listJob = []
        with open(path, 'r') as f:
            line = f.readline()
            while line:
                line = f.readline()
                if line != "":
                    listJob.append(line.split(";")[0])
        return listJob

    def get_njobs(self):
        njobs0 = self.getNbLines(self.pathJob) - 1 # -1 because of header
        listjob1 = self.getListOfJobs(self.pathOp)
        njobs1 = len(set(listjob1))
        if njobs0 != njobs1:
            listjob0 = self.getListOfJobs(self.pathJob)
            njobs0 = len(set(listjob0))
            if njobs0 < njobs1:
                diff = set(listjob1) - set(listjob0)
                #print("There are " + str(len(diff)) + " jobs in the operation file that are not in the OF file")
                #print("The jobs are: " + str(diff))
                return njobs0
            else:
                diff = set(listjob0) - set(listjob1)
                #print("There are " + str(len(diff)) + " jobs in the OF file that are not in the operation file")
                #print("The jobs are: " + str(diff))
                return njobs1
        else:
            return njobs0       
        
    def getNbLines(self, path):
        # return the number of lines in a file
        with open(path, 'r') as f:
            lines = f.readlines()
            return len(lines)
    
    def getMachinesDict(self):
        # return a dictionary with the machines names as keys and the index of the machine as value
        # return the list of machines with infinite capacity
        usedMachines = []
        with open(self.pathOp, 'r') as f:
            line = f.readline()
            line = f.readline()
            while len(line) != 0:
                machineName = line.split(";")[2]
                if line != "" and machineName not in usedMachines:
                    usedMachines.append(machineName)
                line = f.readline()

        machinesDict = {}
        machinesCapInfDict = {} # Machines with infinite capacity
        with open(self.pathMach, 'r') as f:
            line = f.readline() # Skip header
            line = f.readline()
            i = 0
            j = 0
            while len(line) != 0:
                lineSplit = line.split(";")            
                if lineSplit[0] in usedMachines:
                    if int(lineSplit[3][0]) == 3: # Infinite capacity
                        machinesCapInfDict[lineSplit[0][:10]] = i
                        i+=1
                    else:
                        machinesDict[lineSplit[0][:10]] = j
                        j+=1
                line = f.readline()
        return machinesDict, machinesCapInfDict
    
    def getNbOpPerJob(self):
        # return the list of jobs in the file
        noperations = np.zeros(self.nbJobs, dtype=int)
        nTotOp = np.zeros(self.nbJobs, dtype=int)
        with open(self.pathOp, 'r') as f:
            f.readline()
            line = f.readline().split(";")
            job = line[0]
            machine = line[2][:10]
            if machine in self.machDict:
                noperations[0] = 1
            nTotOp[0] = 1
            i=0
            line = f.readline().split(";")
            while len(line) > 1:
                jobCurrent = line[0]
                machCurr = line[2][:10]
                
                if jobCurrent == job:
                    if machCurr in self.machDict:
                        noperations[i] += 1
                    nTotOp[i] +=1
                else:
                    i += 1
                    job = jobCurrent
                    if machCurr in self.machDict:
                        noperations[i] += 1
                    nTotOp[i] +=1
                line = f.readline().split(";")
        return noperations, nTotOp

    def getNbOpPerMach(self):
        noperations = np.zeros(len(self.machDict), dtype=int)
        with open(self.pathOp, 'r') as f:
            f.readline() # Skip header
            for i in range(self.nbJobs):
                for j in range(self.totNbOpPerJob[i]):
                    lineSplit = f.readline().split(";")
                    if lineSplit[2][:10] in self.machDict:
                        noperations[self.machDict[lineSplit[2][:10]]] += 1
        return noperations
    
    def getListOp(self):
        operations = [[] for _ in range(self.nbJobs)]

        with open(self.pathOp, 'r') as f:
            f.readline()
            line = f.readline().split(";")

            for j in range(self.nbJobs):                
                count = 0 
                for _ in range(self.nbOpPerJob[j]):                    

                    if len(line) <= 1:
                        break                    
                    machine = line[2][:10]
                    
                    durBefore = 0
                    duration = 0
                    durAfter = 0

                    while machine not in self.machDict: #find inf capacity machine before
                        dur = (float(line[3]) + float(line[4])) #* 60
                        if dur == 0:
                            dur = 5*24
                        durBefore += dur
                        line = f.readline().split(";")
                        machine = line[2][:10]
                        count += 1

                    duration = (float(line[3]) + float(line[4])) #* 60
                    if duration == 0:
                        duration = 5*24
                    count += 1

                    line = f.readline().split(";")
                    if len(line) > 1:
                        machineAfter = line[2][:10]
                        while machineAfter not in self.machDict and count < self.totNbOpPerJob[j] : #find inf capacity machine after
                            dur = (float(line[3]) + float(line[4])) #* 60
                            if dur == 0:
                                dur = 5*24
                            durAfter += dur
                            line = f.readline().split(";")
                            if len(line) <= 1:
                                break
                            machineAfter = line[2][:10]
                            count += 1
                    #print([machine, durBefore, duration, durAfter])
                    operations[j].append([self.machDict[machine], durBefore, duration, durAfter])
        return operations                

    def getMeanOpPerMach(self):
        meanOpPerMach = np.zeros(len(self.machDict))
        for i in range(len(self.listOp)):
            for j in range(len(self.listOp[i])):
                meanOpPerMach[self.listOp[i][j][0]] += j
        meanOpPerMach = meanOpPerMach / self.nbOpPerMach
        return np.argsort(meanOpPerMach)[::-1]
    
    def getCriticMachine(self):
        criticMach = np.zeros(len(self.machDict))
        for i in range(len(self.listOp)):
            for j in range(len(self.listOp[i])):
                criticMach[self.listOp[i][j][0]] += self.listOp[i][j][2]
        return np.argsort(criticMach)[::-1]
    
    # ABOUT THE SCHEDULE
    def getObjective(self, endJob):
        # return the objective function value
        obj=0
        if endJob[0] == None:
            return np.inf
        for i in range(self.nbJobs):
            obj += endJob[i] 

        return obj

    def getFirstSchedule(self, randomFlag = True):
        # schedule : List of list of operations for each machine

        # Deterministic schedule
        schedule = [[] for i in range(self.nbMach)]
        for i in range(self.nbJobs):
            for j in range(len(self.listOp[i])):
                schedule[self.listOp[i][j][0]].append((i,j, self.listOp[i][j][1], self.listOp[i][j][2], self.listOp[i][j][3]))

        # random schedule: shuffle the operations of each machine
        if randomFlag:
            for i in range(len(schedule)):
                random.shuffle(schedule[i])

        return schedule

    def saveSchedule(self,schedule, filename):
        with open(filename, 'a') as f:
            for i in range(len(schedule)):
                line = ""
                for j in range(len(schedule[i])):
                    line += "(" + str(schedule[i][j][0]) + "," + str(schedule[i][j][1]) + "," + str(schedule[i][j][2]) + "," + str(schedule[i][j][3]) + "," + str(schedule[i][j][4]) + ");"
                line += "\n"
                f.write(line)

    def loadSchedule(self, filename):
        schedule = [[] for i in range(self.nbMach)]
        with open(filename, 'r') as f:
            for i in range(self.nbMach):
                line = f.readline().split(";")
                for j in range(len(line)-1):
                    #remove the parenthesis
                    line[j] = line[j][1:-1]

                    op = line[j].split(",")
                    schedule[i].append((int(op[0]), int(op[1]), float(op[2]), float(op[3]), float(op[4])))
        return schedule
    
    def isScheduleFeasible(self, schedule):
        
        counterOpMach = np.zeros(self.nbMach, dtype=int)
        counterOpJob = [[0] for _ in range(self.nbJobs)]      

        change = True
        while (counterOpMach != self.nbOpPerMach).any():
            if not change:
                return False

            change = False
            for i in range(self.nbMach):
                if counterOpMach[i] == self.nbOpPerMach[i]: # if all the operations of the machine are scheduled
                    continue
                
                op = schedule[i][counterOpMach[i]]
                if counterOpJob[op[0]][0] == op[1]:
                    change = True
                    
                    counterOpMach[i] += 1
                    counterOpJob[op[0]][0] += 1
                    counterOpJob[op[0]] = reduceList(counterOpJob[op[0]])
        return True

    def getStartEnd(self, schedule, penality = 5000):
        start = []
        end = []
        for _, value in enumerate(self.nbOpPerMach):
            start.append(np.zeros(value))
            end.append(np.zeros(value))
        
        counterOpMach = np.zeros(self.nbMach, dtype=int)
        counterOpJob = [[0] for _ in range(self.nbJobs)]
        endMach = np.zeros(self.nbMach, dtype=float)
        endJob = np.zeros(self.nbJobs, dtype=float)

        change = True
        counter =  np.zeros((1,2))
        while (counterOpMach != self.nbOpPerMach).any():
            #print("")
            #print("counter OP", counterOpJob)
            if not change:
                # free an operation on the machine with the lower mean of numero of operations
                # get machine number
                l = 1
                m = self.machOrder[self.nbMach - l]
                while counterOpMach[m] == self.nbOpPerMach[m]:
                    l += 1
                    m = self.machOrder[self.nbMach - l]

                op = schedule[m][counterOpMach[m]]

                s = max(endMach[m], endJob[op[0]] + op[2])
                start[m][counterOpMach[m]] = s
                end[m][counterOpMach[m]] = s + op[3]

                endMach[m] = s + op[3] + penality
                endJob[op[0]] = s + op[3] + op[4] + penality
                
                counterOpMach[m] += 1
                counterOpJob[op[0]].append(op[1]) # add the operation to the list of operations of the job
                counterOpJob[op[0]].sort() # sort the list of operations of the job
                counterOpJob[op[0]] = reduceList(counterOpJob[op[0]]) # reduce the list of operations of the job
                counter[0][0] += 1

            change = False
            for i in range(self.nbMach):
                if counterOpMach[i] == self.nbOpPerMach[i]: # if all the operations of the machine are scheduled
                    continue
                
                op = schedule[i][counterOpMach[i]]

                if counterOpJob[op[0]][0] == op[1]:
                    change = True
                    s = max(endMach[i], endJob[op[0]] + op[2])
                    start[i][counterOpMach[i]] = s
                    end[i][counterOpMach[i]] = s + op[3]

                    endMach[i] = s + op[3]
                    endJob[op[0]] = s + op[3] + op[4]

                    counterOpMach[i] += 1
                    counterOpJob[op[0]][0] += 1
                    counterOpJob[op[0]] = reduceList(counterOpJob[op[0]])
                    counter[0][1] += 1
                    
        #print("counter", counter)
        return end, endJob 
    
    def getStartEndGurobi(self, schedule): 

        # Create a new model
        model = gp.Model("schedule")
        model.Params.LogToConsole = 0 # disable console output

        # Create variables
        Completion = {}
        Start = {}
        End= {}
        for m in range(self.nbMach):
            for o in range(self.nbOpPerMach[m]):
                Completion[m,o]=model.addVar(vtype=GRB.CONTINUOUS,lb=0,name="c_%s_%s"%(m,o))
                Start[m,o]=model.addVar(vtype=GRB.CONTINUOUS,lb=0,name="s_%s_%s"%(m,o))
        for j in range(self.nbJobs):
            End[j]=model.addVar(vtype=GRB.CONTINUOUS,lb=0,name="e_%s"%(j))

        # Set objective
        model.setObjective(quicksum([(End[j]) for j in range(self.nbJobs)]), GRB.MINIMIZE)

        # Add constraints
        # Constraints on the operation on the machine
        for m in range(self.nbMach):
            for o in range(0,self.nbOpPerMach[m]):
                if o == 0:
                    model.addConstr(Completion[m,0] >= schedule[m][0][3])
                else:
                    model.addConstr(Completion[m,o] >= Completion[m,o-1] +  schedule[m][o][3])
                model.addConstr(Start[m,o] + schedule[m][o][3] == Completion[m,o])
                model.addConstr(Start[m,o] >= schedule[m][o][2])

        idxOp = self.getOpIndexInSched(schedule)
        for j in range(self.nbJobs):

            # deal with jobs with no operation (i.e. op only on machine with infinite capacity)
            if self.nbOpPerJob[j] == 0:
                continue

            # Get end 
            oIdx = max(self.nbOpPerJob[j]-1, 0)
            m = idxOp[j][oIdx][0]
            mo = idxOp[j][oIdx][1]
            model.addConstr(End[j] == Completion[m,mo]+schedule[m][mo][4])

            # Precedence constraints
            for o in range(1,self.nbOpPerJob[j]):            
                m1, mo1 = idxOp[j][o][0], idxOp[j][o][1] # operation o of job j
                m0, mo0 = idxOp[j][o-1][0], idxOp[j][o-1][1]  # operation o-1 of job j
                model.addConstr(Completion[m1,mo1] >= Completion[m0,mo0] + schedule[m0][mo0][4] + schedule[m1][mo1][2] + schedule[m1][mo1][3])


        # Optimize model
        model.optimize()

        # Print solution
        if model.status == 2:

            # get start into list of list
            startSol = model.getAttr('x', Start)
            start = []
            for _, value in enumerate(self.nbOpPerMach):
                start.append(np.zeros(value))
            for m in range(self.nbMach):
                for o in range(self.nbOpPerMach[m]):
                    start[m][o] = startSol[m,o]
            
            # get end into a list
            endSol = model.getAttr('x', End)
            endReturn = []
            for j in range(self.nbJobs):
                endReturn.append(endSol[j])
            
            return startSol,endReturn
        else:
            return None,[None]
        
    def getOpIndexInSched(self, schedule):
        opIndex = [[] for _ in range(self.nbJobs)]
        for j in range(self.nbJobs):
            for o in range(self.nbOpPerJob[j]):
                m = self.listOp[j][o][0]
                for i in range(len(schedule[m])):
                    if schedule[m][i][0] == j and schedule[m][i][1] == o:
                        opIndex[j].append((m,i))
                        break
        return opIndex
    
    def plotSchedule(self, schedule, completionTime, endJob):
        cmap = plt.get_cmap('viridis', self.nbJobs)
        # generate a vector of distinct colors
        colorsJob = [cmap(i) for i in range(self.nbJobs)]
        #colorsJob = ['tab:blue', 'tab:purple', 'tab:green','tab:orange']
        fig, ax = plt.subplots()
        ax.set_xlim(0, int(max(endJob)/5) + 1)

        for i in range(len(schedule)):
            for j in range(len(schedule[i])):
                jobNb = schedule[i][j][0]
                cmplPt = completionTime[i][j]
                startPt = cmplPt - schedule[i][j][3]
                aftPt = cmplPt + schedule[i][j][4]           
                ax.broken_barh([(startPt, cmplPt-startPt)], (i-0.5, 1), facecolors=colorsJob[jobNb])
                ax.broken_barh([(cmplPt, aftPt-cmplPt)], (i-0.5, 1), facecolors=colorsJob[jobNb], alpha = 0.05)

                #ax.annotate(str(jobNb) + ", " + str(schedule[i][j][1]), (startPt, i + 0.5), color='black', ha='center', va='center')

        # Define the function to update the plot based on the slider value
        def update(val):
            i = int(val)
            ax.set_xlim(i, i+max(endJob)+1)
            fig.canvas.draw_idle()

        # Create the slider widget
        axcolor = 'lightgoldenrodyellow'
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.05], facecolor=axcolor)
        slider = Slider(ax_slider, 'Index', 0, max(endJob)/5, valinit=0, valstep=1)

        # Connect the slider widget to the update function
        slider.on_changed(update)

        # Add labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Machine')
        ax.set_title('Job Shop Scheduling')

        # Display the plot and the slider widget
        plt.show()


    # TEST OF THE TWO METHODS TO GET COMPLETION TIMES
    def getScheduleList(self):
        schedules = []
        for k in range(self.nbJobs):

            schedule = [[] for i in range(self.nbMach)]
            for l in range(self.nbJobs):
                i = l + k if l + k < self.nbJobs else l + k - self.nbJobs

                for j in range(len(self.listOp[i])):
                    schedule[self.listOp[i][j][0]].append((i,j, self.listOp[i][j][1], self.listOp[i][j][2], self.listOp[i][j][3]))
            schedules.append(schedule)

        return schedules

def getPlanningPbm(test: str):
    currentPath = os.getcwd()
    mockelPath = currentPath + "/data/" + test
    jobPath = mockelPath + "\DataExportOF.csv"
    operationsPath = mockelPath + "\DataExportOperation.csv"
    machinePath = mockelPath + "\MoyFab.csv"

    return PlanningPbm(jobPath, operationsPath, machinePath)


if __name__ == "__main__":

    # --------------------------------------------
    # |       INFORMATION ABOUT THE PROBLEM      |
    # --------------------------------------------

    plan = getPlanningPbm("mockel")

    schedules = plan.getScheduleList()
    timem1 = 0
    timem2 = 0

    for i in range(len(schedules)):
        startTime = time.time()
        startDeter, endJobDeter = plan.getStartEnd(schedules[i])
        endTime = time.time()
        timem1 += (endTime - startTime)

        startTime = time.time()
        startDeterGu, endJobDeterGu = plan.getStartEndGurobi(schedules[i])
        endTime = time.time()
        timem2 += (endTime - startTime)
    
    print("nb schedules: " + str(len(schedules)))
    print("Time with my function: " + str(timem1))
    print("Time with gurobi: " + str(timem2))

    print("average time with my function: " + str(timem1/len(schedules)))
    print("average time with gurobi: " + str(timem2/len(schedules)))