import pickle
import matplotlib.pyplot as plt

with open('losses.txt', 'rb') as f:
    losses = pickle.load(f)
with open('acces.txt', 'rb') as f:
    acces = pickle.load(f)
# with open('sec.txt', 'rb') as f:
#     sec = pickle.load(f)


plt.figure(figsize=(5,5))
plt.title("loss")
plt.plot(losses, label="Training")
plt.xlabel("epochs")
plt.legend(loc="upper right")

plt.figure(num=2, figsize=(5,5))
plt.title("Accuracy")
plt.plot(acces, label="Training")
# plt.plot(sec, label="Testing")
plt.xlabel("epochs")
plt.legend(loc="upper left")
plt.show()