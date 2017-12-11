import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import random
import pickle
from os import listdir
from os.path import isfile, join
import os
import matplotlib.pyplot as plt


def read_data(inp, target, X, y):
    with open('./' + inp, 'r') as f:
        count = 0
        for line in f:
            data = line.strip().split(",")
            if count <52:
                count += 1
            elif count < 5000:
                data = list(map(float, data))
                gears = gears_zero[:]
                current_gear = int(data[0])
                gears[current_gear] += 1
                X.append([]+gears+data[1:9])
                y.append(target)
                count += 1
    return X, y

def read_test(inp, target):
    this_X = []
    this_y = []
    with open('./' + inp, 'r') as f:
        count = 0
        for line in f:
            data = line.strip().split(",")
            if count < 52:
                count += 1
            elif count < 5000:
                data = list(map(float, data))
                gears = gears_zero[:]
                current_gear = int(data[0])
                gears[current_gear] = 1
                this_X.append([]+gears+data[1:9])
                # print(this_X[-1])
                this_y.append(target)
                count += 1
    return this_X, this_y
    
    
    
road_files = [f for f in listdir("./road/") if isfile(join("./road/", f))]
dirt_files = [f for f in listdir("./dirt/") if isfile(join("./dirt/", f))]
test_files = [f for f in listdir("./output/") if isfile(join("./output/", f))]
test_path = "./output/"
road_path = "./road/"
dirt_path = "./dirt/"

print("road files: ", len(road_files))
print("dirt files: ", len(dirt_files))
print("test files: ", len(test_files))
gears_zero = [0,0,0,0,0,0,0,0]


input("Start parsing test...")

Xt_l = []
yt_l = []

for n, inp in enumerate(test_files):
    if inp.startswith("dirt"):
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


random.seed(1234)
best_model = []
best_pred = 0
best_acc = []
top_acc = 0
top_acc_model = -1

clf = svm.SVC()
source_path = "./experiment/svm_updated/"

input("Start parsing train...")

for i in range(50):
    print("Combination ", i)
    number = i
    
    path = source_path + str(i) + "/"
    if not os.path.exists(path):
        os.makedirs(path)
    
    random.shuffle(road_files)
    random.shuffle(dirt_files)
    this_road = road_files[:2]
    this_dirt = dirt_files[:2]
    
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
            
            print(str(name)+" "+str(pred)+" "+str(yt[n][i]))
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
        
    if race_pred == best_pred:
        best_model.append([i])
        best_acc.append([precision])
    elif race_pred > best_pred:
        best_pred = race_pred
        best_model = [i]
        best_acc = [precision]
        
    if precision > top_acc:
        top_acc = precision
        top_acc_model = i
    
    total_precision = total_correct / (total_correct + total_wrong)
    with open(path + "a_results.txt", 'a') as f:
        f.write("\nFINAL SCORE!\nCorrect Pred: " + str(total_correct))
        f.write("\t|\tWrong Pred: " + str(total_wrong))
        f.write("\nTotal Precision: " + str(total_precision))
        f.write("\nThe number of correctly predicted circuits is: " + str(race_pred) + "/" + str(len(test_files)))

    with open(source_path + "results.txt", 'a') as f:
        f.write("\n\nModel number " + str(number) + "\nCorrect Pred: " + str(total_correct))
        f.write("\t|\tWrong Pred: " + str(total_wrong))
        f.write("\nTotal Precision: " + str(total_precision))
        f.write("\nThe number of correctly predicted circuits is: " + str(race_pred) + "/" + str(len(test_files)))

    pickle.dump(clf, open(path+"model.pickle.dat", "wb"))
    
    print("\nEnd of "+ str(number) + "Total Precision: " + str(total_precision))
    print("\n The number of correctly predicted circuits is: " + str(race_pred) + "/" + str(len(test_files)))
    
    
with open(source_path + "results.txt", 'a') as f:
    f.write("\n\n\n FINAL RESULTS")
    f.write("\n Most correct predictions: " + str(best_pred))
    f.write("\n Achiveved in models: " + "".join(best_model))
    f.write("\n With precision: "+ "".join(best_acc))
    f.write("\n\n While best PRECISION is: " + str(top_acc) + " achieved in model " + str(top_acc_model))

print("\n\n\n FINAL RESULTS")
print("\n Most correct predictions: " + str(best_pred))
print("\n Achiveved in models: " + "".join(best_model))
print("\n With precision: " + "".join(best_acc))
print("\n\n While best PRECISION is: " + str(top_acc) + " achieved in model " + str(top_acc_model))
    
    
    
    
    
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