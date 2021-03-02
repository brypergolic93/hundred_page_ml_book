# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 09:32:52 2021

@author: Bryan
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from csv import reader

marketing_file_open = open('advertising.csv')
marketing_data_read = reader(marketing_file_open)
marketing_data_list = list(marketing_data_read)
marketing_data = np.array(marketing_data_list)

# segment each column into an array and convert the data from string to float
sales = marketing_data[1:-1,3].astype(float)
tv_ad = marketing_data[1:-1,0].astype(float)
radio_ad = marketing_data[1:-1,1].astype(float)
newspaper_ad = marketing_data[1:-1,2].astype(float)

############## Gradient descent functions

# Update parameters w and b during one epoch. x_i = spendings, y_i = sales
def update_w_and_b(spendings, sales, w, b, alpha):
    
    dl_dw = 0.0
    dl_db = 0.0
    
    N = len(spendings)
    
    # Partial derivatives wrt to w and b:
    for i in range(N):
        dl_dw += -2*spendings[i]*(sales[i] - (w*spendings[i] + b))
        dl_db += -2*(sales[i] - (w*spendings[i] + b))
        
    # Update w and b. Divide by N here as well
    w = w - (1/float(N))*dl_dw*alpha
    b = b - (1/float(N))*dl_db*alpha

    return w, b


# Loop over multiple epochs
def train(spendings, sales, w, b, alpha, epochs):
    
    for e in range(epochs):
        w, b = update_w_and_b(spendings, sales, w, b, alpha)
        
        # log the progress
        if e % 1 == 0:
            print("Epoch: ", e, "Loss: ", avg_loss(spendings, sales, w, b))
            
        '''    
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(tv_ad,sales, 'k.')
        ax.set_xlabel('TV Marketing Costs ($M)')
        ax.set_ylabel('Sales (Units)')
        ax.set_title("Epoch: %1.0f" % e)
        plt.plot(spendings, w*spendings + b, '-')
        plt.show()
        '''
        
    return w, b


# Average loss function that computes mean squared     
def avg_loss(spendings, sales, w, b):
    
    N = len(spendings)
    total_error = 0.0
    
    for i in range(N):
        # Mean squared error equation
        total_error += (sales[i] - (w*spendings[i] + b))**2
        
    return total_error/float(N)

w_init = 0.0
b_init = 7.0
alpha_init = 0.00001
epochs_init = 10

w, b = train(tv_ad, sales, w_init, b_init, alpha_init, epochs_init)

fig = plt.figure()
ax = plt.axes()
ax.plot(tv_ad,sales, 'k.')
ax.set_xlabel('TV Marketing Costs ($M)')
ax.set_ylabel('Sales (Units)')
plt.plot(tv_ad, w*tv_ad + b, '-')
plt.show()

# Create predictive function
def predict(x, w, b):
    
    return w*x + b

x_new = 175.0
y_new = predict(x_new, w, b)
print("\nPredicted sales spending $%3.2f million in TV advertising: $%2.2f million" % (x_new,y_new))