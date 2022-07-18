# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 10:30:36 2022

@author: slaven
"""

import matplotlib.pyplot as plt

def plotPortfolioValueChange(portfolioValue, startRange, endRange):
    plt.figure()
    plt.title('Change in portfolio value over time ({} to {})'.format(startRange.strftime('%Y-%m-%d'), endRange.strftime('%Y-%m-%d')))
    plt.plot(portfolioValue)
    plt.xlabel('time t in steps of {}'.format('15m'))
    plt.ylabel('in USD')