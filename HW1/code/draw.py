import matplotlib.pyplot as plt
import numpy as np
import pickle

x = np.linspace(-3, 3, 30)
y = 2*x + 1
z = x**2

x = [0, 10, 20]
y = [0, 0.85, 0.85]

with open("acces.txt", "rb") as fp:
    acces = pickle.load(fp)

plt.figure(num=3, figsize=(5,5))
plt.title("Accuracy")
plt.plot(acces, label="Training")
plt.plot(x, y, label="Testing")
plt.xlabel("epochs")
plt.legend(loc="upper left")
plt.show()

