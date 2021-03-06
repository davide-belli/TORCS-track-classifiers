import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from random import shuffle
from os import listdir
from os.path import isfile, join
import os
import matplotlib.pyplot as plt

TIMESPAN = 25

def read_data(inp, target, X, y):
    with open('./' + inp, 'r') as f:
        this_X = []
        first = True
        for line in f:
            data = line.strip().split(",")
            if first:
                first = False
            else:
                data = list(map(float, data))
                v1 = np.average(np.array(data[-33:-31]))
                v2 = np.average(np.array(data[-10:-6]))
                this_X.append([v1]+[v2])
        for i in range(TIMESPAN,len(this_X)):
            #print([]+data[-34:-30]+data[-10:-6])
            #input()
            v1 = this_X[i][0]
            v1old = this_X[i-TIMESPAN][0]
            v2 = this_X[i][1]
            v2old = this_X[i-TIMESPAN][1]
            X.append([v1-v1old]+[v2-v2old])
            y.append(target)
    return X, y

def read_test(inp, target):
    result_X = []
    result_y = []
    this_X = []
    with open('./' + inp, 'r') as f:
        first = True
        n = 0
        for line in f:
            data = line.strip().split(",")
            if first:
                first = False
            else:
                data = list(map(float, data))
                v1 = np.average(np.array(data[-33:-31]))
                v2 = np.average(np.array(data[-10:-6]))
                this_X.append([v1] + [v2])
        for i in range(TIMESPAN, len(this_X)):
            # print([]+data[-34:-30]+data[-10:-6])
            # input()
            v1 = this_X[i][0]
            v1old = this_X[i - TIMESPAN][0]
            v2 = this_X[i][1]
            v2old = this_X[i - TIMESPAN][1]
            result_X.append([v1 - v1old] + [v2 - v2old])
            result_y.append(target)
    return result_X, result_y
    
    
    
road_files = [f for f in listdir("./road/") if isfile(join("./road/", f))]
dirt_files = [f for f in listdir("./dirt/") if isfile(join("./dirt/", f))]
test_files = [f for f in listdir("./output/") if isfile(join("./output/", f))]
test_path = "./output/"
road_path = "./road/"
dirt_path = "./dirt/"

print("road ", road_files)
print("dirt ", dirt_files)
print("test", test_files)




input("Start parsing test...")

Xt_l = []
yt_l = []

for n, inp in enumerate(test_files):
    if inp.startswith("manual"):
        print(inp, "Hit manual dirt track")
        this_Xt, this_yt = read_test(test_path + inp, 1)
    else:
        this_Xt, this_yt = read_test(test_path + inp, 0)
    Xt_l.append(this_Xt)
    yt_l.append(this_yt)
    print("Read file ", n, "/", len(test_files))

input("lens")
print(len(Xt_l))
print(len(Xt_l[0]))
print(len(Xt_l[0][0]))

input("   ssss   ")
Xt = []
yt = []
for i in range(len(Xt_l)):
    
    Xt.append(np.array(Xt_l[i]))
    yt.append(np.array(yt_l[i]))
    
print("Test data shapes, should be [n_files] x [n_lines] x [n_features]: ")
print(Xt[0].shape)
print(yt[0].shape)
print("len", len(Xt))
print("len[0]", len(Xt[0]))
print("len[0][0]", len(Xt[0][0]))




input("Start parsing train...")

for i in range(20):
    print("Combination ", i)
    
    path = "./experiment/complete_delta/"+str(i)+"/"
    if not os.path.exists(path):
        os.makedirs(path)
    
    shuffle(road_files)
    shuffle(dirt_files)
    this_road = road_files[:4]
    this_dirt = dirt_files[:4]
    
    combD = ",".join(this_dirt)
    combR = ",".join(this_road)

    X = []
    y = []
    
    for n, inp in enumerate(this_dirt):
        X, y = read_data(dirt_path+inp, 1, X, y)
        print("Read dirt file ", n, "/", len(this_dirt))
    for n, inp in enumerate(this_road):
        X, y = read_data(road_path+inp, 0, X, y)
        print("Read road file ", n, "/", len(this_road))
    
    #print(data1[0])
    #print(data1[0][2])
    
    
    X=np.array(X)
    y=np.array(y)
    print(X.shape)
    print(y.shape)
    
    print("Starting training ", i)
    
    clf = svm.SVC()
    clf.fit(X, y)
    
    print("Finished training ", i)

    with open(path + "a_results.txt", 'a') as f:
        f.write("Dirt: " + combD + "\n")
        f.write("Road: " + combR + "\n")

    total_correct = 0
    total_wrong = 0
    race_pred = 0
        
    for n in range(len(Xt)):
        
        name = test_files[n]
        correct_pred = 0
        wrong_pred = 0
        plotX = []
        plotY = []
        plotT = []
        
        for i in range(len(yt[n])):
            plotX.append(i)
            plotT.append(yt[n][i])
            pred = clf.predict(Xt[n][i].reshape(-1, len(Xt[n][0])))
            plotY.append(pred)
            if pred == yt[n][i]:
                correct_pred += 1
            else:
                wrong_pred += 1
        
        precision = correct_pred / (correct_pred + wrong_pred)
        print("The final precision of SVM is: ", precision)
        total_correct += correct_pred
        total_wrong += wrong_pred
        if correct_pred > wrong_pred:
            race_pred += 1
        

        print("Finished testing ", i)

        with open(path + "a_results.txt", 'a') as f:
            f.write("\n\nTest on race: " + str(name))
            f.write("\nCorrect Pred: " + str(correct_pred))
            f.write("\t|\tWrong Pred: " + str(wrong_pred))
            f.write("\nPrecision: " + str(precision))

        fig = plt.figure()
        plt.scatter(plotX, plotY, c="red", s=3)
        plt.scatter(plotX, plotT, c="blue", s=1)
        plt.savefig(path + str(name) + '.png', format='png')
        plt.close()
        

        print("Finished plotting ", i)
    
    total_precision = total_correct / (total_correct + total_wrong)
    with open(path + "a_results.txt", 'a') as f:
        f.write("\nFINAL SCORE!\nCorrect Pred: " + str(total_correct))
        f.write("\t|\tWrong Pred: " + str(total_wrong))
        f.write("\nTotal Precision: " + str(total_precision))
        f.write("\n The number of correctly predicted circuits is: " + str(race_pred) + "/" + str(len(test_files)))
        







#plt.scatter(speed1, rpm1, c="red", label="forza", s=2)
#plt.scatter(speed2, rpm2, c="blue", label="alpine")
#plt.show()

#plt.close()
#plt.scatter(speed1, true_rpm1, c="red", label="forza")
#plt.scatter(speed2, true_rpm2, c="blue", label="alpine")
#plt.show()

#plt.close()
#plt.scatter(rpm1, true_rpm1, c="red", label="forza")
#plt.scatter(rpm2, true_rpm2, c="blue", label="alpine")
#plt.show()