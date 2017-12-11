import numpy as np
import matplotlib.pyplot as plt

inp1 = "./output/forza_1"
inp2 = "./output/manual_2017-11-29_17-48-14"

c=0
with open('./' + inp1, 'r') as f:
    header = f.readline()
    header = header.split(",")
    id1 = header.index("wheelSpinVel_0")
    id2 = header.index("speedX_0")
    id3 = header.index("rpm_0")
    xs1 = []
    xr1 = []
    xrpm1 = []
    first = True
    for line in f:
        data = line.split(",")
        if not first:
            first = False
        elif c<50:
            xs1.append(data[id2])
            speeds = [float(data[id1]),float(data[id1+1]),float(data[id1+2]),float(data[id1+3])]
            #print(speeds)
            #input("wait")
            xr1.append(sum(speeds)/4)
            xrpm1.append(data[id3])
            c+=1

c=0
with open('./' + inp2, 'r') as f:
    header = f.readline()
    header = header.split(",")
    id1 = header.index("wheelSpinVel_0")
    id2 = header.index("speedX_0")
    id3 = header.index("rpm_0")
    xs2 = []
    xr2 = []
    xrpm2 = []
    first = True
    for line in f:
        data = line.split(",")
        if  not first:
            first = False
        elif c<50:
            xs2.append(data[id2])
            speeds = [float(data[id1]),float(data[id1+1]),float(data[id1+2]),float(data[id1+3])]
            xr2.append(sum(speeds)/4)
            xrpm2.append(data[id3])
            c+=1

speed1 = np.array(xs1)
speed2 = np.array(xs2)
rpm1 = np.array(xr1)
rpm2 = np.array(xr2)
true_rpm1 = np.array(xrpm1)
true_rpm2 = np.array(xrpm2)
            
#print(speed1, speed2)
#print(rpm1, rpm2)

plt.scatter(speed1, rpm1, c="red", label="forza", s=2)
plt.scatter(speed2, rpm2, c="blue", label="alpine")
plt.show()

plt.close()
plt.scatter(speed1, true_rpm1, c="red", label="forza")
plt.scatter(speed2, true_rpm2, c="blue", label="alpine")
plt.show()

plt.close()
plt.scatter(rpm1, true_rpm1, c="red", label="forza")
plt.scatter(rpm2, true_rpm2, c="blue", label="alpine")
plt.show()