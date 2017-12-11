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
        this_y = []
        first = True
        n = 0
        for line in f:
            data = line.strip().split(",")
            if first:
                first = False
            elif n < TIMESPAN:
                n += 1
                # print(n)
                this_X.append([] + data[-34:-30] + data[-10:-6])
                this_y.append(target)
            else:
                data = list(map(float, data))
                #print([]+data[-34:-30]+data[-10:-6])
                #input()
                # print([]+data[-34:-30]+data[-10:-6]+this_X[-TIMESPAN][:8])
                this_X.append([]+data[-34:-30]+data[-10:-6]+this_X[-TIMESPAN][:8])
                this_y.append(target)
    return X+this_X[TIMESPAN:], y+this_y[TIMESPAN:]

def read_test(inp, target):
    this_X = []
    this_y = []
    with open('./' + inp, 'r') as f:
        first = True
        n = 0
        for line in f:
            data = line.strip().split(",")
            if first:
                first = False
            elif n < TIMESPAN:
                n += 1
                # print(n)
                this_X.append([] + data[-34:-30] + data[-10:-6])
                this_y.append(target)
            else:
                data = list(map(float, data))
                #print([]+data[-34:-30]+data[-10:-6])
                #input()
                # print(len(this_X))
                # print([]+data[-34:-30]+data[-10:-6]+this_X[-TIMESPAN][:8])
                # input("ok")
                this_X.append([]+data[-34:-30]+data[-10:-6]+this_X[-TIMESPAN][:8])
                this_y.append(target)
    return this_X[TIMESPAN:], this_y[TIMESPAN:]
    
    
    
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
    
    path = "./experiment/complete_union/"+str(i)+"/"
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