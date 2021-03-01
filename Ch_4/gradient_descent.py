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

fig = plt.figure()
ax = plt.axes()
ax.plot(tv_ad,sales, 'k.')
ax.set_xlabel('TV Marketing Costs ($ M)')
ax.set_ylabel('Sales ($ M)')
plt.show()


