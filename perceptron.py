#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 15:35:38 2022

@author: diegorodriguez
"""
from csv import reader
from random import seed
from random import randrange

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup
        



# Split a dataset into a train and test set
def train_test_split(dataset, split=0.60):
	train = list()
	train_size = split * len(dataset)
	dataset_copy = list(dataset)
	while len(train) < train_size:
		index = randrange(len(dataset_copy))
		train.append(dataset_copy.pop(index))
	return train, dataset_copy

#make predictions 
def predict_pct(row, weights):
    activation = weights[0]
    for i in range(len(row) - 1):
        activation+= weights[i + 1] * row[i]
    return 1.0 if activation >=0.0 else 0.0

#estimate coefficients using gradient descent
def train_coef(train, l_rate, n_epoch):
    weights = [0.0 for i in range(len(train[0]))]
    #iterating over n_epochs
    for epoch in range(n_epoch):
        sum_error = 0.0
        #iterating over rows
        for row in train:
            yhat = predict_pct(row, weights)
            error = row[-1] - yhat
            sum_error += error**2
            weights[0] = weights[0] + l_rate * error
            #iterating over weights
            for i in range(len(row) - 1):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
                print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return weights 

# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
	weights = [0.0 for i in range(len(train[0]))]
	for _ in range(n_epoch):
		for row in train:
			prediction = predict_pct(row, weights)
			error = row[-1] - prediction
			weights[0] = weights[0] + l_rate * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
	return weights

seed(1)
# load and prepare data
filename = 'sonar.all-data.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert string class to integers
str_column_to_int(dataset, len(dataset[0])-1)
train, test = train_test_split(dataset)
# evaluate algorithm
l_rate = 0.01
n_epoch = 5
weights = train_coef(test, l_rate, n_epoch)
print(weights)

for row in test:
  prediction = predict_pct(row, weights)
  print("Expected=%d, Predicted=%d" % (row[-1], prediction))
  
  
  
