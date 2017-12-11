import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from random import shuffle
from os import listdir
from os.path import isfile, join


def read_data(inp, target):
    with open('./' + inp, 'r') as f:
        first = True
        for line in f:
            data = line.strip().split(",")
            if first:
                first = False
            else:
                data = list(map(float, data))
                #print([]+data[-34:-30]+data[-10:-6])
                #input()
                X.append([]+data[-34:-30]+data[-10:-6])
                y.append(target)

def read_test(inp, target):
    with open('./' + inp, 'r') as f:
        first = True
        for line in f:
            data = line.strip().split(",")
            if first:
                first = False
            else:
                data = list(map(float, data))
                Xt.append([]+data[-34:-30]+data[-10:-6])
                yt.append(target)

train_files = [f for f in listdir("./train/") if isfile(join("./train/", f))]
test_files = [f for f in listdir("./output/") if isfile(join("./output/", f))]
test_path = "./output/"
train_path = "./train/"

print("train ", train_files)
print("test", test_files)

input("Start parsing...")

X = []
y = []
Xt = []
yt = []

for n, inp in enumerate(train_files):
    if inp.startswith("manual"):
        print(inp, "Hit manual dirt track")
        read_data(train_path+inp, 1)
    else:
        read_data(train_path+inp, 0)
    print("Read file ", n, "/", len(train_files))

for n, inp in enumerate(test_files):
    if inp.startswith("manual"):
        print(inp, "Hit manual dirt track")
        read_test(test_path+inp, 1)
    else:
        read_test(test_path+inp, 0)
    print("Read file ", n, "/", len(test_files))

#print(data1[0])
#print(data1[0][2])


X=np.array(X)
y=np.array(y)
Xt=np.array(Xt)
yt=np.array(yt)
# y=y.reshape(-1,1)
print(X.shape)
print(y.shape)
print(Xt.shape)
print(yt.shape)
print("len", len(Xt[0]))

input("Continue with training, parsing finised")

clf = svm.SVC()
clf.fit(X, y)

input("Finished training ")

correct_pred = 0
wrong_pred = 0

for i in range(len(yt)):
    print(Xt[i])
    pred = clf.predict(Xt[i].reshape(-1,len(Xt[0])))
    print(pred)
    if pred == yt[i]:
        correct_pred += 1
    else:
        wrong_pred += 1

precision = correct_pred / (correct_pred + wrong_pred)
print("The final precision of SVM is: ", precision)







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