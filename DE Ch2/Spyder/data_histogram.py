# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 13:41:34 2019

@author: admin
"""

import matplotlib.pyplot as plt
import pandas as pd

# name of the file to read from
r_filenameCSV = 'realEstate_trans_full.csv'

# name of the output file
w_filenameCSV = 'realEstate_corellations.csv'

# read the data and select only sales of flats 
# with up to 4 beds
csv_read = pd.read_csv(r_filenameCSV)
csv_read = csv_read.query('beds < 5')

# generate histogram by number of beds
csv_read.hist(
    column='price', 
    by='beds', 
    xlabelsize=8, 
    ylabelsize=8, 
    sharex=False, 
    sharey=False,
    figsize=(9,7)
)

# save to file
plt.savefig(
    'priceByBeds_histogram.pdf'
)

# show the figures
plt.show()