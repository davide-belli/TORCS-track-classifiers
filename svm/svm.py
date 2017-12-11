import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from random import shuffle

inp1 = "./output/forza_1"
inp2 = "./output/manual_2017-11-29_17-48-14"

X = []
y = []
X_t = []
y_t = []
data1 = []
data2 = []

with open('./' + inp1, 'r') as f:
    first = True
    for line in f:
        data = line.strip().split(",")
        if first:
            first = False
        else:
            data = list(map(float, data))
            data1.append(data)

with open('./' + inp2, 'r') as f:
    first = True
    for line in f:
        data = line.strip().split(",")
        if first:
            first = False
        else:
            data = list(map(float, data))
            data2.append(data)


shuffle(data1)
shuffle(data2)
limit1 = len(data1)//10*8
limit2 = len(data2)//10*8

print("split ",len(data1), limit1)
#print(data1[0])
#print(data1[0][2])
input("Continue")
for n in range(len(data1)):
    if n < limit1:
        #print(n)
        X.append(data1[n])
        y.append(0)
    else:
        #print("appending t")
        X_t.append(data1[n])
        y_t.append(0)

for n, point in enumerate(data2):
    if n < limit2:
        X.append(point)
        y.append(1)
    else:
        X_t.append(point)
        y_t.append(1)

X=np.array(X)
y=np.array(y)
X_t = np.array(X_t)
y_t = np.array(y_t)
# y=y.reshape(-1,1)
print(X.shape)
print(y.shape)
print("lenyt", len(y))
print(X_t.shape)
print(y_t.shape)
print("lenyt", len(y_t))

input("Start training")

clf = svm.SVC()
clf.fit(X, y)

input("Finished training ")

correct_pred = 0
wrong_pred = 0

for i in range(len(y_t)):
    print(X_t[i])
    pred = clf.predict(X_t[i].reshape(-1,85))
    print(pred)
    if pred == y_t[i]:
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