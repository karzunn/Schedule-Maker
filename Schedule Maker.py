import random
import numpy as np

def repair(Selection,Tasks,FreeTime): #Function that converts invalid solutions to valid ones.
    T=np.transpose(Selection)
    for i in range(len(Selection[0])):
        if (np.sum(T[i])>1):
            idx=random.choice(np.where(T[i] == 1)[0])
            T[i]=T[i]*0
            T[i][idx]=1
    Selection=np.transpose(T)

    for i in range(len(FreeTime)):
        while sum(Tasks*Selection[i])>FreeTime[i]:
            Selection[i][random.choice(np.where(Selection[i] == 1)[0])]=0
    return Selection

def fitness(Selection,Tasks,Freetime): #function to evaluate the fitness score of a solution
    return np.sum(Freetime-np.sum(Selection*Tasks,axis=1))

def crossover(A,B): #function to perform a genetic crossover between two solutions, A and B.
    crossoverpt=round(random.uniform(0,len(A)-1))
    Child1=np.concatenate((A[:,0:crossoverpt],B[:,crossoverpt:]),axis=1)
    Child2=np.concatenate((B[:,0:crossoverpt],A[:,crossoverpt:]),axis=1)
    return Child1,Child2

def mutate(Selection): #function to perform a genetic mutation to a solution.
    Selection[round(random.uniform(0,len(Selection)-1))][round(random.uniform(0,len(Selection[0])-1))]=abs(Selection[round(random.uniform(0,len(Selection)-1))][round(random.uniform(0,len(Selection[0])-1))]-1)
    return Selection

def breed(Selection,Population,Tasks,FreeTime,mutatepercent,crossoverpercent,prunepercent): #Function that intakes a series of random solutions, evaluates fitness scores, uses roulette wheel selection, and breeds the selected solutions.
    Fitness=[]
    for y in range(Population): #Generate fitness scores of the input population
        Selection[y]=repair(Selection[y],Tasks,FreeTime)
        Fitness.append(1/(fitness(Selection[y],Tasks,FreeTime)+1))
    
    rel_fit=np.array(Fitness)/np.sum(Fitness) #Compute relative fitness scores by dividing by the sum of all scores
    
    cp=crossoverpercent
    
    pp=1/(1-prunepercent)
    
    parents=[] #Here, roulette wheel selection is used to select potential parents, based on the desired crossover percent
    while len(parents)<=round(Population/(4/cp))*2: #A form of roulette wheel selection, where the lowest scores can be pruned by adjusting the "prunepercent"
        r = random.uniform(min(rel_fit)+(max(rel_fit)-min(rel_fit))/pp,max(rel_fit))
        for i in range(Population):
            if r<=rel_fit[i]:
                parents.append(i)
                break
    nextgen=[] #Here, roullete wheel selection is used to select potential solutions that will transfer to the next generation without crossover.
    while len(nextgen)<round(Population-len(parents)*2): #
        r = random.uniform(min(rel_fit)+(max(rel_fit)-min(rel_fit))/pp,max(rel_fit))
        for i in range(Population):
            if r<=rel_fit[i]:
                nextgen.append(i)
                break
    
    gen2=[Selection[nextgen[i]] for i in range(len(nextgen))]

    pickedparents=[]
    children=[]
    for i in range(int(len(parents))): #Crossover process
        randnum1=round(random.uniform(0,(Population/(2/cp))-1))
        randnum2=round(random.uniform(0,(Population/(2/cp))-1))
        while randnum1==randnum2 and [randnum1,randnum2] in pickedparents:
            randnum2=round(random.uniform(0,(Population/(2/cp))-1))
        pickedparents.append([randnum1,randnum2])
        c1,c2=crossover(Selection[parents[randnum1]],Selection[parents[randnum2]])
        c1=repair(c1,Tasks,FreeTime)
        c2=repair(c2,Tasks,FreeTime)
        children.append(c1)
        children.append(c2)
        
    Finalists=children+gen2
    
    mutations=[] #Mutation process based on mutation percent
    for x in range(round(Population*mutatepercent)):
        mutation=round(random.uniform(0,Population-1))
        while mutation in mutations:
            mutation=round(random.uniform(0,Population-1))
        mutations.append(mutation)
        Finalists[mutation]=mutate(Finalists[mutation])
        Finalists[mutation]=repair(Finalists[mutation],Tasks,FreeTime)

    return Finalists








TimeSlot=[]
FreeTime=[]
Tasks=[]
TaskNames=[]

while True:
    try:
        a=input("Enter times, separated by a dash. Separate time slots by a comma. Example: 7:30-9:30,12:30-1:30 \n")
        b=a.split(",")
        for i in range(len(b)):
            TimeSlot.append(b[i])
            time1,time2 = b[i].split("-")
            hours, minutes = time1.split(":")
            sumtime1=int(hours)*60+int(minutes)
            hours, minutes = time2.split(":")
            sumtime2=int(hours)*60+int(minutes)
            if sumtime1>sumtime2:
                sumtime2=sumtime2+60*12
            FreeTime.append(sumtime2-sumtime1) 
    except ValueError:
        print("Sorry, I didn't understand that.")
        continue
    else:
        break

while True:
    try:
        a=input("Give the name of the task connected to the length of the task in minutes by a dash. Separate tasks with a comma.\n Example: Dishes-30,Vacuum-50,Coding-60 \n")
        b=a.split(",")
        for i in range(len(b)):
            TN,TT=b[i].split("-")
            TaskNames.append(TN)
            Tasks.append(int(TT))
    except ValueError:
        print("Sorry, I didn't understand that.")
        continue
    else:
        break        
    
FreeTime=np.array(FreeTime)
Tasks=np.array(Tasks)





print("\n Generating Schedule...\n")

Population=50 #population size can be adjusted here

Selection=[[[random.randint(0, 1) for x in range(len(Tasks))] for y in range(len(FreeTime))] for z in range(Population)]
for i in range(len(Selection)):
    Selection[i]=repair(Selection[i],Tasks,FreeTime)

currentmax=0
improved=True
miniter=1
while improved==True:
    Children=breed(Selection,Population,Tasks,FreeTime,0.15,1,0.25) #breeding parameters can be adjusted here
    Selection=Children.copy()
    FitnessFinal=[None]*Population
    std=[]
    for y in range(Population):
        FitnessFinal[y]=1/(fitness(Selection[y],Tasks,FreeTime)+1)
        std.append(np.std(FreeTime-np.sum(Selection[y]*Tasks,axis=1)))
    if currentmax==max(FitnessFinal) and miniter>50:
        improved=False
    miniter=miniter+1
    currentmax=max(FitnessFinal)

print("\n\n----------SCHEDULE----------\n")

sortedstd=list(set(std))
for d in range(len(sortedstd)):
    for i in range(len(std)):
        if FitnessFinal[i]==max(FitnessFinal) and std[i]==sortedstd[d]:
            Choice=Selection[i]
            break

idx=[]
T=np.transpose(Choice)
for i in range(len(Tasks)):
    if (np.sum(T[i])==0):
        idx.append(i)
        
unused=""
for i in range(len(idx)):
    unused=unused+" "+TaskNames[idx[i]]

for d in range(len(TimeSlot)):
    tasklist=" "
    for i in range(len(Tasks)):
        if Choice[d][i]==1:
            tasklist=TaskNames[i]+", "+tasklist
            
    print("\nFrom {}, you should do the following tasks: {}          |          Unused Time: {}".format(TimeSlot[d],tasklist,(FreeTime[d]-np.sum(Choice[d]*Tasks))))
print("\n\nYou have {} minutes of extra time total, and no room for the following tasks: {}".format(fitness(Choice,Tasks,FreeTime),unused))
