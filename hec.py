import matplotlib.pyplot as plt
import numpy as np
import math
import random

k = np.linspace(-math.pi,math.pi,15)
x = np.cos(k)
y = np.sin(k)
# x = np.linspace(-1,1,50)
# c = random.uniform(-1,1)
# y = np.array([i+c for i in x])
train_data = np.array([[x[i], y[i]] for i in range(len(x))])

test_data = train_data

theta = [0,0]
l1 = 1
l2 = 1
lattice = [1,10,10]
epochs = 150
input_dim = len(train_data[0])
output_dim = len(theta)
sigma0 = 5
sigmaf = 0.1
etaw0 = 0.9
etawf = 0.1
etat0 = 0.8
etatf = 0.5
etaA0 = etat0
etaAf = etatf

def get_x(theta):
    return (l1*math.cos(theta[0]) + l2*math.cos(theta[0] + theta[1]))

def get_x1(theta):
    return [0, l1*math.cos(theta[0]), l1*math.cos(theta[0]) + l2*math.cos(theta[0] + theta[1])]

def get_y(theta):
    return (l1*math.sin(theta[0]) + l2*math.sin(theta[0] + theta[1]))

def get_y1(theta):
    return [0, l1*math.sin(theta[0]), l1*math.sin(theta[0]) + l2*math.sin(theta[0] + theta[1])]


class Neuron():
    def __init__(self, weights, index, fitness = 0):
        self.weights = weights
        self.index = index
        self.yg = np.random.uniform(low=-0.1, high=0.1, size=(output_dim,))
        self.Ag = np.random.uniform(low=-0.1, high=0.1, size=(output_dim, input_dim))

def norm(a):
    return sum(a**2)**0.5

def dist(n1, n2):
    return norm(n1.index-n2.index)

def h(n, winner, sigma):
    return math.e**((-1*dist(n, winner))/(2*sigma**2))

def fitness(neuron, x):
    return norm(neuron.weights-x)

def coarse(x, h_list, h_sum):
    return np.sum([h_list[j]*(neurons[j].yg + np.matmul(neurons[j].Ag, x - neurons[j].weights)) for j in range(len(neurons))], axis=0)/h_sum

def fine(x, v0, h_list, h_sum):
    return np.sum([h_list[j]*(np.matmul(neurons[j].Ag, x - v0)) for j in range(len(neurons))], axis=0)/h_sum

def get_delta_theta(del_v, h_list, h_sum):
    return np.sum([h_list[j]*(np.matmul(neurons[j].Ag, del_v)) for j in range(len(neurons))], axis=0)/h_sum

def predict(x):
    sigma = 0.1
    fitnesses = []
    for neuron in neurons:
        fitnesses.append(fitness(neuron, x))

    winner = neurons[fitnesses.index(min(fitnesses))]
    h_list = [h(neuron, winner, sigma) for neuron in neurons]
    h_sum = sum(h_list)
    y = np.sum([h_list[j]*(neurons[j].yg + np.matmul(neurons[j].Ag, x - neurons[j].weights)) for j in range(len(neurons))], axis=0)/h_sum
    return y

neurons=np.array([[[[Neuron(np.array([random.uniform(-0.1,0.1) for x in range(input_dim)]),np.array([k,i,j]))] for j in range(lattice[2])] for i in range(lattice[1])] for k in range(lattice[0])])
neurons = neurons.flatten()

stat = []
t=0
for e in range(epochs):
    print(e+1)
    tmax = float(epochs*len(train_data))
    for data in train_data:
        etaw = etaw0 * ((etawf/etaw0)**(t/tmax))
        etat = etat0 * ((etatf/etat0)**(t/tmax))
        etaA = etaA0 * ((etaAf/etaA0)**(t/tmax))
        sigma = sigma0 * (sigmaf/sigma0)**(t/tmax)
        fitnesses = []
        for neuron in neurons:
            fitnesses.append(fitness(neuron, data))
        winner = neurons[fitnesses.index(min(fitnesses))]

        h_list = [h(neuron, winner, sigma) for neuron in neurons]
        h_sum = sum(h_list)

        theta0 = coarse(data, h_list, h_sum)
        v0 = np.array([get_x(theta0), get_y(theta0)])
        theta1 = theta0 + fine(data, v0, h_list, h_sum)
        v1 = np.array([get_x(theta1), get_y(theta1)])
        delta_v = v1-v0
        delta_theta = theta1-theta0
        theta0_new = coarse(v0, h_list, h_sum)
        theta1_new = coarse(v1, h_list, h_sum)
        delta_theta_new = get_delta_theta(delta_v, h_list, h_sum)

        for i in range(len(neurons)):
            neuron = neurons[i]
            neuron.weights += etaw * h_list[i] * (data - neuron.weights)
            neuron.yg += etat * (h_list[i]/h_sum)*(theta0 - theta0_new)
            neuron.Ag += etaA * np.matmul((delta_theta - delta_theta_new).reshape(2,1), delta_v.reshape(1,2)) * (h_list[i]/(h_sum*(norm(delta_v)**2)))
        #print(v1)
        stat.append(norm(v0-data))
        t+=1

error = []
for data in test_data:
    theta = predict(data)
    x = get_x1(theta)
    y = get_y1(theta)
    error.append(norm(data-np.array([x[2],y[2]])))
    plt.plot(x,y)
    plt.scatter(x[2],y[2],color='red')
print(np.average(error))
for data in train_data:
    plt.scatter(data[0], data[1])
for neuron in neurons:
    plt.scatter(neuron.weights[0], neuron.weights[1], color="green")
plt.axis([-2,2,-2,2])
plt.show()
plt.plot(stat)
plt.show()
