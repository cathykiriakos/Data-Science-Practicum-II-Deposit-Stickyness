#!/usr/bin/env python
# coding: utf-8

# # Forecasting deposit stickyness in custodian banking institutions  
# 
# ## Project Description: 
# I currently work for a large custody bank that as most other organizations are dealing with uncertainty driven by the Covid -19 epidemic.  Unlike many other retail-oriented services we experience deposit growth during periods of stress.  The question that arises in all of our executive team meetings is if the deposits will stay on, and if not how long will the duration of elevated deposits be. The importance to my field of work is determining how will those coming deposit actions of our clients will influence the liquidity position of the bank, and how we need to manage preemptively for those impending actions. This project leveraging historical balance sheet information available from custodian banking institutions, will forecast ramp up and ramp down inflection points for deposit growth and subsequent decline using various macroeconomic indicators as independent variables in an effort to get a sense of what’s to come for our bank's deposit behavior. 
# 
# ## Data Science Task: 
# To determine and apply the most appropriate supervised learning algorithm to forecast deposit inflection points during economic stress, in an effort to forecast current deposit trends at my banking institution
# 
# ## Data:  
# This project will take historical balance sheet information, specifically foreign office, and domestic deposits from some of the large custody banks including Northern Trust, Bank of New York, and State Street Bank & Trust using publicly available balance sheet information from SNL or via the Edgar API tool, and then obtain historical macroeconomic indicators from FRED, including interest rates, stock indices like Dow Jones S&P, GDP Growth, CPI, VIX unemployment and treasury yield in an effort to forecast the inflection point in deposit growth and subsequent declines resulting from periods of economic stress. 
# 
# For this effort I will be looking at quarterly banking deposit data as my dependent variable, with a goal of going back as far as possible with historical data I was unable to capture the 1987 black Monday, but did get the dot com bust of the 90's. This data was obtained via S&P’s SNL.  My independent variables were obtained via FRED pretty heavily to obtain the most up to date historical macroeconomic indicator information, and yahoo finance to capture S&P as well as  [3]. 
#   
# 
# ## Data Analysis:
# Random fores regression was applied to handle non-normal distribution of deposits across all banks
# 
# 
# ## First steps - Independent Data Step: 
# We will obtain all the necesary data for our independent variables, whose sources are largely FRED and Yahoo Finance for the Dow Jones and S&P 500 data [6][7]

# In[389]:


#import packages
import pandas as pd
from datetime import datetime
import datetime
import numpy as np
import csv
import os 
import time
import re


# In[390]:


#Vix 
vix = pd.read_csv('VIXCLS.csv',
                  sep=r',',
                  skipinitialspace = True, 
                 converters = {'DATE':pd.to_datetime,
                              'VIXCLS': np.float()})
vix


# In[391]:


#Fed Funds Rate
ff = pd.read_csv('FEDFUNDS (1).csv',
                sep =r',', 
                skipinitialspace = True, 
                converters = {'DATE': pd.to_datetime,
                             'FEDFUNDS': np.float()})
ff.dtypes


# In[392]:


#Start Data Frame with all independent variables 
import numpy as np
indep = pd.merge(ff,vix, 
                how = 'outer',
                on = 'DATE')


# In[393]:


#CPI
cpi = pd.read_csv('CPIAUCSL.csv',
                sep =r',', 
                skipinitialspace = True, 
                converters = {'DATE': pd.to_datetime,
                             'CPIAUCSL': np.float()})
#Merge with independent variables
indep = pd.merge(indep,cpi, 
                how = 'outer',
                on = 'DATE')


# In[394]:


#Dow Jones Industrial Average
djii = pd.read_csv('HistoricalPrices.csv',
                sep =r',', 
                skipinitialspace = True, 
                converters = {'DATE': pd.to_datetime,
                             'djii_close': np.float()})
#Merge with independent variables
indep = pd.merge(indep,djii, 
                how = 'outer',
                on = 'DATE')


# In[395]:


#Household debt
hd = pd.read_csv('HistoricalPrices.csv',
                sep =r',', 
                skipinitialspace = True, 
                converters = {'DATE': pd.to_datetime,
                             'FODSP': np.float()})
#Merge with independent variables
indep = pd.merge(indep,hd, 
                how = 'outer',
                on = 'DATE')


# In[396]:


#Unemployment Rate 
un = pd.read_csv('UNRATE.csv',
                sep =r',', 
                skipinitialspace = True, 
                converters = {'DATE': pd.to_datetime,
                             'UNRATE': np.float()})
#Merge with independent variables
indep = pd.merge(indep,un, 
                how = 'outer',
                on = 'DATE')


# In[397]:


#10-Year Treasury Yeild
ty = pd.read_csv('DGS10.csv',
                sep =r',', 
                skipinitialspace = True, 
                converters = {'DATE': pd.to_datetime,
                             '10yrTreasYeild': np.float()})
#Merge with independent variables
indep = pd.merge(indep,ty, 
                how = 'outer',
                on = 'DATE')


# In[398]:


#Ted Spread
ts = pd.read_csv('TEDRATE.csv',
                sep =r',', 
                skipinitialspace = True, 
                converters = {'DATE': pd.to_datetime,
                             'TEDRATE': np.float()})
#Merge with independent variables
indep = pd.merge(indep,ts, 
                how = 'outer',
                on = 'DATE')


# In[399]:


#Case Schiller Home Index
csi = pd.read_csv('CSUSHPINSA.csv',
                sep =r',', 
                skipinitialspace = True, 
                converters = {'DATE': pd.to_datetime,
                             'CS_Index': np.float()})
#Merge with independent variables
indep = pd.merge(indep,csi, 
                how = 'outer',
                on = 'DATE')


# In[400]:


#S&P 500 Close
snp = pd.read_csv('GSPC.csv',
                sep =r',', 
                skipinitialspace = True, 
                converters = {'DATE': pd.to_datetime,
                             'S&P500': np.float()})
#Merge with independent variables
indep = pd.merge(indep,snp, 
                how = 'outer',
                on = 'DATE')


# In[401]:


#GDP
gdp = pd.read_csv('GDP.csv',
                sep =r',', 
                skipinitialspace = True, 
                converters = {'DATE': pd.to_datetime,
                             'GDP': np.float()})
#Merge with independent variables
indep = pd.merge(indep,gdp, 
                how = 'outer',
                on = 'DATE')


# In[402]:


indep


# In[403]:


indep.to_excel("output.xlsx")


# In[404]:


#Reload Cleaned Dataframe 
indep = pd.read_excel ("Final Data.xlsx")
print (indep)


# ## Next Step: Dependent Variables 
# 
# After a come to Jesus moment, realizing that the FDIC Bankfind API did not contain any of the needed balance sheet information that I required; and determining that a good deal of people that work with the Edgar API to parse balance sheet information spend months and or all of their free time to get a single data point for their needs - I decided that this route of data capture while cool and a little flashy was way too involved of a project than I could bite off in the next coming weeks. 
# 
# I pivoted my approach, while my SNL add on at the office was not functioning appropriately I did remember that I have an S&P Market Intelligence login for peer comparisons.  I used their MI Report builder, while clunky got the job done.  I was able to pull down excel balance sheet information for Northern Trust, State Street Bank & Trust, and Bank of New York Mellon information in 40 quarter lots from 1Q 1990 to 3Q 2020. 
# 
# Since it was all in excel, I tacked this into the output file from my cleaned independent variable data from above, providing a clean dataset with all needed data points.  I will be limited to June 2020 which is the most recent period available on S&P unfortunately. However with March through June data we will be able to see the deposit ramp up that many custodian banking institutions witnessed as a result of the market uncertainty and other macro factors that we're not quite sure are at this point yet; as we're speculating that supranational funds are ramping up funds for Covid 19 initiaitves - however no hard correlations of this concept have been fleshed out at this point in time.
# 
# Enough of the talk, below is an upload of the complete data set "Final Data."

# In[405]:


#Load Complete Data Set 
data = pd.read_excel ("Final Data.xlsx")
print (data)


# In[406]:


from scipy import stats 
import seaborn
import scipy


# We will start by calling info to understand the data within the dataframe: 

# In[407]:


data.info()


# In[408]:


#convert CS_Index to float
data['CS_Index'] = pd.to_numeric(data['CS_Index'], errors='coerce')
data.info()


# In[409]:


#drop quarter column
data = data.drop(['Quarter'], axis = 1)
data.head()


# In[410]:


#clean whitespace in columns  
data.columns = data.columns.str.replace(' ', '')


# In[507]:


data.describe()


# ## Exploratory Data Analysis 

# In[411]:


#Subset for each bank
BONYDom=data[['DATE','BONYDom','VIXCLS',
'djii_close_x','10yrTreasYeild','TEDRATE','AdjClose','GDP',
'CS_Index','FEDFUNDS','CPIAUCSL','UNRATE']]

BONYFo=data[['DATE','BONYFo','VIXCLS',
'djii_close_x','10yrTreasYeild','TEDRATE','AdjClose','GDP',
'CS_Index','FEDFUNDS','CPIAUCSL','UNRATE']]

BONYFFP=data[['DATE','BONYFFP_Repo','VIXCLS',
'djii_close_x','10yrTreasYeild','TEDRATE','AdjClose','GDP',
'CS_Index','FEDFUNDS','CPIAUCSL','UNRATE']]

SSBTDom=data[['DATE','SSBTDom','VIXCLS',
'djii_close_x','10yrTreasYeild','TEDRATE','AdjClose','GDP',
'CS_Index','FEDFUNDS','CPIAUCSL','UNRATE']]

SSBTFo=data[['DATE','SSBTFo','VIXCLS',
'djii_close_x','10yrTreasYeild','TEDRATE','AdjClose','GDP',
'CS_Index','FEDFUNDS','CPIAUCSL','UNRATE']]

SSBTFFP=data[['DATE','SSBTFFP_Repo','VIXCLS',
'djii_close_x','10yrTreasYeild','TEDRATE','AdjClose','GDP',
'CS_Index','FEDFUNDS','CPIAUCSL','UNRATE']]

NTRSDom=data[['DATE','NTRSDom','VIXCLS',
'djii_close_x','10yrTreasYeild','TEDRATE','AdjClose','GDP',
'CS_Index','FEDFUNDS','CPIAUCSL','UNRATE']]

NTRSFo=data[['DATE','NTRSFo','VIXCLS',
'djii_close_x','10yrTreasYeild','TEDRATE','AdjClose','GDP',
'CS_Index','FEDFUNDS','CPIAUCSL','UNRATE']]

NTRSFFP=data[['DATE','NTRSFFP_Repo','VIXCLS',
'djii_close_x','10yrTreasYeild','TEDRATE','AdjClose','GDP',
'CS_Index','FEDFUNDS','CPIAUCSL','UNRATE']]


# In[412]:


from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[413]:


#BONYDom Correlation Matrix
BONYDomCorr = BONYDom.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(BONYDomCorr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(BONYDomCorr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# Looking at our BONY domestic deposits heatmap we can see that our more saturated correlations include: 
# Dow Jones Close (djii_close_x)
# S&P 500 Close (AdjClose)
# GDP
# Case schiller home index (CS_Index) 
# Consumer Price Index (CPIAUCSL) 
# 
# So we can investigate the BONY deposits further using those independent variables. 

# In[414]:


#Histogram of dependent variable to understand distribution: 
sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
sns.distplot(
    BONYDom['BONYDom'], norm_hist=False, kde=False, bins=20, hist_kws={"alpha": 1}
).set(xlabel='Domestic Deposits', ylabel='Count');


# In[415]:


sns.set_style("darkgrid")
sns.lineplot(data = BONYDom, x='DATE', y='BONYDom')


# In[416]:


#Show univariate distribution of each variable
g = sns.PairGrid(BONYDom)
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)


# In[417]:


#BONYFo Correlation Matrix
BONYFoCorr = BONYFo.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(BONYFoCorr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(BONYFoCorr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# Looking at our BONY foreign office deposits heatmap we can see that our more saturated correlations, which are mostly negative include: 
# VIX (VIXCLS)
# Dow Jones Close (djii_close_x)
# 10-year Treasury Yeild (10yrTreasYeild)
# S&P 500 Close (AdjClose)
# GDP
# Case schiller home index (CS_Index) 
# Consumer Price Index (CPIAUCSL) 
# 
# So we can investigate the BONY deposits further using those independent variables. 

# In[418]:


#Histogram of dependent variable to understand distribution: 
sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
sns.distplot(
    BONYFo['BONYFo'], norm_hist=False, kde=False, bins=20, hist_kws={"alpha": 1}
).set(xlabel='Foreign Office Deposits', ylabel='Count');


# In[419]:


sns.set_style("darkgrid")
sns.lineplot(data = BONYFo, x='DATE', y='BONYFo')


# In[420]:


#Show univariate distribution of each variable
g = sns.PairGrid(BONYFo)
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)


# In[421]:


#BONYFFP Correlation Matrix
BONYFFPCorr = BONYFFP.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(BONYFFPCorr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(BONYFFPCorr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# Looking at our BONY Fed Fund deposits heatmap we can see that our more saturated correlations, which are mostly negative include: 
# 
# Dow Jones Close (djii_close_x)
# 10-year Treasury Yeild (10yrTreasYeild)
# S&P 500 Close (AdjClose)
# GDP
# Case schiller home index (CS_Index) 
# Fed Funds Rate (FEDFUNDS)
# Consumer Price Index (CPIAUCSL) 
# 
# So we can investigate the BONY deposits further using those independent variables

# In[422]:


#Histogram of dependent variable to understand distribution: 
sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
sns.distplot(
    BONYFFP['BONYFFP_Repo'], norm_hist=False, kde=False, bins=20, hist_kws={"alpha": 1}
).set(xlabel='FFP', ylabel='Count');


# In[423]:


sns.set_style("darkgrid")
sns.lineplot(data = BONYFFP, x='DATE', y='BONYFFP_Repo')


# In[424]:


#Show univariate distribution of each variable
g = sns.PairGrid(BONYFFP)
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)


# In[425]:


#SSBTDom Correlation Matrix
SSBTDomCorr = SSBTDom.corr()

# Generate a mask Domr the upper triangle
mask = np.triu(np.ones_like(SSBTDomCorr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(SSBTDomCorr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# Surprisingly SSB&T domestic deposits are not highly correlated to any of the macro factors that I investigated.  Due to this I will skip out analysis on what moves SSB&T domestic deposits.  

# In[426]:


#Histogram of dependent variable to understand distribution: 
sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
sns.distplot(
    SSBTDom['SSBTDom'], norm_hist=False, kde=False, bins=20, hist_kws={"alpha": 1}
).set(xlabel='Domestic Deposits', ylabel='Count');


# In[427]:


sns.set_style("darkgrid")
sns.lineplot(data = SSBTDom, x='DATE', y='SSBTDom')


# In[428]:


#Show univariate distribution of each variable
g = sns.PairGrid(SSBTDom)
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)


# In[429]:


#SSBTFo Correlation Matrix
SSBTFoCorr = SSBTFo.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(SSBTFoCorr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(SSBTFoCorr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# For SSB&T foreign office deposits do have some higher level corrrelations that we can investigate, including: 
# 10-Year Treasury Yeild
# GDP
# CS_Inex
# Fed Funds Rate 
# CPI

# In[430]:


#Histogram of dependent variable to understand distribution: 
sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
sns.distplot(
    SSBTFo['SSBTFo'], norm_hist=False, kde=False, bins=20, hist_kws={"alpha": 1}
).set(xlabel='Foreign Office Deposits', ylabel='Count');


# In[431]:


sns.set_style("darkgrid")
sns.lineplot(data = SSBTFo, x='DATE', y='SSBTFo')


# In[432]:


#Show univariate distribution of each variable
g = sns.PairGrid(SSBTFo)
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)


# In[433]:


#SSBTFFP Correlation Matrix
SSBTFFPCorr = SSBTFFP.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(SSBTFFPCorr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(SSBTFFPCorr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# For SSB&T Fed Funds deposits have stronger corrrelations that we can investigate, including: 
# djii close 
# adj_close_x - S&P
# 10-Year Treasury Yeild
# GDP
# CS_Index
# Fed Funds Rate 
# CPI

# In[434]:


#Histogram of dependent variable to understand distribution: 
sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
sns.distplot(
    SSBTFFP['SSBTFFP_Repo'], norm_hist=False, kde=False, bins=20, hist_kws={"alpha": 1}
).set(xlabel='FFP', ylabel='Count');


# In[435]:


sns.set_style("darkgrid")
sns.lineplot(data = SSBTFFP, x='DATE', y='SSBTFFP_Repo')


# In[436]:


#Show univariate distribution of each variable
g = sns.PairGrid(SSBTFFP)
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)


# In[437]:


#NTRSDom Correlation Matrix
NTRSDomCorr = NTRSDom.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(NTRSDomCorr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(NTRSDomCorr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# For Northern Trust we have some stronger correlations on Domestic deposits and can investigate: 
# DJI Close 
# S&P Close 
# GDP
# Case Schiller Index 
# Fed Funds Rate 
# CPI 

# In[438]:


#Histogram of dependent variable to understand distribution: 
sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
sns.distplot(
    NTRSDom['NTRSDom'], norm_hist=False, kde=False, bins=20, hist_kws={"alpha": 1}
).set(xlabel='Domestic Deposits', ylabel='Count');


# In[439]:


sns.set_style("darkgrid")
sns.lineplot(data = NTRSDom, x='DATE', y='NTRSDom')


# In[440]:


#Show univariate distribution of each variable
g = sns.PairGrid(NTRSDom)
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)


# In[441]:


#NTRSFo Correlation Matrix
NTRSFoCorr = NTRSFo.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(NTRSFoCorr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(NTRSFoCorr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# Similar to BONY, Northern Trust has a greater proportion of negatively correlated variables, but in this scenario since I know that these account for the buik of deposits we will investigate the following variables: 
# Djii Close 
# 10-year treasury yeild 
# S&P500 close 
# GDP 
# Case Schiller 
# Fed Funds 
# CPI 

# In[442]:


#Histogram of dependent variable to understand distribution: 
sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
sns.distplot(
    NTRSFo['NTRSFo'], norm_hist=False, kde=False, bins=20, hist_kws={"alpha": 1}
).set(xlabel='Foreign Office Deposits', ylabel='Count');


# In[443]:


sns.set_style("darkgrid")
sns.lineplot(data = NTRSFo, x='DATE', y='NTRSFo')


# In[444]:


#Show univariate distribution of each variable
g = sns.PairGrid(NTRSFo)
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)


# In[445]:


#NTRSFFP Correlation Matrix
NTRSFFPCorr = NTRSFFP.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(NTRSFFPCorr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(NTRSFFPCorr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[446]:


#Histogram of dependent variable to understand distribution: 
sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
sns.distplot(
    NTRSFFP['NTRSFFP_Repo'], norm_hist=False, kde=False, bins=20, hist_kws={"alpha": 1}
).set(xlabel='FFP', ylabel='Count');


# In[447]:


sns.set_style("darkgrid")
sns.lineplot(data = NTRSFFP, x='DATE', y='NTRSFFP_Repo')


# In[448]:


#Show univariate distribution of each variable
g = sns.PairGrid(NTRSFFP)
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)


# These time series plots might be interesting plotted against each other by institution. Lets give that a try: 
# 
# ### Domestic Deposits by Bank: 

# In[449]:


sns.set_style("darkgrid")
sns.lineplot(data = data, x='DATE', y='NTRSDom', color = 'g').set_title('NTRS vs. BONY Domestic')
ax2 = plt.twinx()
sns.lineplot(data = data, x = 'DATE', y = "BONYDom", color = 'b')
ax


# In[450]:


sns.set_style("darkgrid")
sns.lineplot(data = data, x='DATE', y='NTRSDom', color = 'g').set_title('NTRS vs. SSBT Domestic')
ax2 = plt.twinx()
sns.lineplot(data = data, x = 'DATE', y = "SSBTDom", color = 'b')
ax


# ### Foreign Office Deposits by Bank:

# In[451]:


sns.set_style("darkgrid")
sns.lineplot(data = data, x='DATE', y='NTRSFo', color = 'g').set_title('NTRS vs. BONY Foreign Office Deposits')
ax2 = plt.twinx()
sns.lineplot(data = data, x = 'DATE', y = "BONYFo", color = 'b')
ax


# In[452]:


sns.set_style("darkgrid")
sns.lineplot(data = data, x='DATE', y='NTRSFo', color = 'g').set_title('NTRS vs. SSBT Foreign Office Deposits')
ax2 = plt.twinx()
sns.lineplot(data = data, x = 'DATE', y = "SSBTFo", color = 'b')
ax


# ### FFP & Repo by Bank:

# In[453]:


sns.set_style("darkgrid")
sns.lineplot(data = data, x='DATE', y='NTRSFFP_Repo', color = 'g').set_title('NTRS vs. BONY FFP & Repo')
ax2 = plt.twinx()
sns.lineplot(data = data, x = 'DATE', y = "BONYFFP_Repo", color = 'b')
ax


# In[454]:


sns.set_style("darkgrid")
sns.lineplot(data = data, x='DATE', y='NTRSFFP_Repo', color = 'g').set_title('NTRS vs. SSBT FFP & Repo')
ax2 = plt.twinx()
sns.lineplot(data = data, x = 'DATE', y = "SSBTFFP_Repo", color = 'b')
ax


# ### What if we transform data into quarterly averages??? 
# We can resample based on the date and transform into quarterly data, across the board there are some independent variables that have no bearing on the correlations within the data.  Now we will transform, and strip out some of the data that does not seem to have much impact on our dependent variables we will transform our dataframe and re-run some of this analysis to see if there are any material changes in the correlations. 
# 
# In addition knowing that our FFP and Repo deposits are likely structural and not a strong indication of client behavior based on macroeconomic factors we can remove the FFP analysis from our views for this 2nd round of descriptive analysis with quarterly data. 

# In[455]:


data.set_index('DATE', inplace=True)
data.head()


# In[456]:


data2 = data.resample('QS').sum()


# In[457]:


data2.info()


# In[458]:


#Remove Ted Spread and Unemployment from data series along with first column unamed 
data2 = data2.drop(['UNRATE','djii_close_y', 'TEDRATE'], axis = 1) 


# In[459]:


data2.drop(data2.columns[data2.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)


# In[646]:


data2.describe()


# In[460]:


data2.head()


# In[508]:


data2.describe()


# In[509]:


#delete 0's 
data2 = data2[data2!=0].dropna()


# In[510]:


#Check to ensure 0 values for deposits are deleted from data frame, showing no min 0 values in our summary stats: 
data2.describe()


# In[511]:


#Subset for each bank
BONYDom2=data2[['BONYDom','VIXCLS',
'djii_close_x','10yrTreasYeild','AdjClose','GDP',
'CS_Index','FEDFUNDS','CPIAUCSL']]

BONYFo2=data2[['BONYFo','VIXCLS',
'djii_close_x','10yrTreasYeild','AdjClose','GDP',
'CS_Index','FEDFUNDS','CPIAUCSL']]

SSBTDom2=data2[['SSBTDom','VIXCLS',
'djii_close_x','10yrTreasYeild','AdjClose','GDP',
'CS_Index','FEDFUNDS','CPIAUCSL']]

SSBTFo2=data2[['SSBTFo','VIXCLS',
'djii_close_x','10yrTreasYeild','AdjClose','GDP',
'CS_Index','FEDFUNDS','CPIAUCSL']]

NTRSDom2=data2[['NTRSDom','VIXCLS',
'djii_close_x','10yrTreasYeild','AdjClose','GDP',
'CS_Index','FEDFUNDS','CPIAUCSL']]

NTRSFo2=data2[['NTRSFo','VIXCLS',
'djii_close_x','10yrTreasYeild','AdjClose','GDP',
'CS_Index','FEDFUNDS','CPIAUCSL']]


# In[512]:


#BONYDom Correlation Matrix
BONYDomCorr2 = BONYDom2.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(BONYDomCorr2, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(BONYDomCorr2, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# Pretty strong correlations across the board for BONY domestic deposits, with the exception of VIX. 

# In[513]:


#Histogram of dependent variable to understand distribution: 
sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
sns.distplot(BONYDom2['BONYDom'], norm_hist=False, kde=False, bins=20, hist_kws={"alpha": 1}
).set(xlabel='Domestic Deposits', ylabel='Count');


# In[515]:


#Show univariate distribution of each variable
g = sns.PairGrid(BONYDom2)
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)


# In[516]:


#BONYFo Correlation Matrix
BONYFoCorr2 = BONYFo2.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(BONYFoCorr2, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(BONYFoCorr2, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# Similar to last set of results, negative correlation across the boad with our BONY Foreign Office deposits, we can keep with the same analysis using this quarterly data. 

# In[517]:


#Histogram of dependent variable to understand distribution: 
sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
sns.distplot(BONYFo2['BONYFo'], norm_hist=False, kde=False, bins=20, hist_kws={"alpha": 1}
).set(xlabel='Foreign Office Deposits', ylabel='Count');


# In[518]:


#Show univariate distribution of each variable
g = sns.PairGrid(BONYFo2)
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)


# In[519]:


#SSBTDom Correlation Matrix
SSBTDomCorr2 = SSBTDom2.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(SSBTDomCorr2, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(SSBTDomCorr2, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# Showing much weaker correlations with the quarterly adjusted data for SSB&T domestic - with only our stock indices close change being the strongest correlations.  

# In[520]:


#Histogram of dependent variable to understand distribution: 
sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
sns.distplot(SSBTDom2['SSBTDom'], norm_hist=False, kde=False, bins=20, hist_kws={"alpha": 1}
).set(xlabel='Domestic Deposits', ylabel='Count');


# In[521]:


#Show univariate distribution of each variable
g = sns.PairGrid(SSBTDom2)
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)


# In[522]:


#SSBTFo Correlation Matrix
SSBTFoCorr2 = SSBTFo2.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(SSBTFoCorr2, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(SSBTFoCorr2, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[523]:


#Histogram of dependent variable to understand distribution: 
sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
sns.distplot(SSBTFo2['SSBTFo'], norm_hist=False, kde=False, bins=20, hist_kws={"alpha": 1}
).set(xlabel='Foreign Office Deposits', ylabel='Count');


# In[524]:


#Show univariate distribution of each variable
g = sns.PairGrid(SSBTFo2)
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)


# In[525]:


#NTRSDom Correlation Matrix
NTRSDomCorr2 = NTRSDom2.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(NTRSDomCorr2, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(NTRSDomCorr2, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# Stronger correlations across the board for Domestic deposits with quarterly adjusted data, VIX still not a strong indicator.

# In[526]:


#Histogram of dependent variable to understand distribution: 
sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
sns.distplot(NTRSDom2['NTRSDom'], norm_hist=False, kde=False, bins=20, hist_kws={"alpha": 1}
).set(xlabel='Domestic Deposits', ylabel='Count');


# In[527]:


#Show univariate distribution of each variable
g = sns.PairGrid(NTRSDom2)
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)


# In[528]:


#NTRSFo Correlation Matrix
NTRSFoCorr2 = NTRSFo2.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(NTRSFoCorr2, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(NTRSFoCorr2, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# Again stronger correlations with the quarterly data set, we will utilize all of our independent variables in our analysis on foreign office deposits. 

# In[529]:


#Histogram of dependent variable to understand distribution: 
sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
sns.distplot(NTRSFo2['NTRSFo'], norm_hist=False, kde=False, bins=20, hist_kws={"alpha": 1}
).set(xlabel='Foreign Office Deposits', ylabel='Count');


# Certainly a weirder distribution on this view, we will have to give this some more consideration in our choice of algorithm for forecasting. 

# In[530]:


#Show univariate distribution of each variable
g = sns.PairGrid(NTRSFo2)
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)


# ## A second view of quarterly deposits by bank: 

# In[531]:


sns.set_style("darkgrid")
sns.lineplot(data = data2, x='DATE', y='NTRSFo', color = 'g').set_title('NTRS vs. BONY Foreign Office Deposits')
ax2 = plt.twinx()
sns.lineplot(data = data2, x = 'DATE', y = "BONYFo", color = 'b')
ax


# In[532]:


sns.set_style("darkgrid")
sns.lineplot(data = data2, x='DATE', y='NTRSFo', color = 'g').set_title('NTRS vs. SSBT Foreign Office Deposits')
ax2 = plt.twinx()
sns.lineplot(data = data2, x = 'DATE', y = "SSBTFo", color = 'b')
ax


# In[533]:


sns.set_style("darkgrid")
sns.lineplot(data = data2, x='DATE', y='NTRSDom', color = 'g').set_title('NTRS vs. BONY Domestic Deposits')
ax2 = plt.twinx()
sns.lineplot(data = data2, x = 'DATE', y = "BONYDom", color = 'b')
ax


# In[534]:


sns.set_style("darkgrid")
sns.lineplot(data = data2, x='DATE', y='NTRSDom', color = 'g').set_title('NTRS vs. SSBT Domestic Deposits')
ax2 = plt.twinx()
sns.lineplot(data = data2, x = 'DATE', y = "SSBTDom", color = 'b')
ax


# ## Thoughts dependent variable distributions and appropriate algorithm usage: 
# A good deal of the dependent variables in our analysis would likely fall into the category where 75% of the data would fall within two standard deviations; or 89% falling within 3 standard deviations with the exception of the SSB&T FFP deposits.  Since these deposits are not necessarily and indicator of client behavior and are more structural in nature, it may make sense to strip the FFP & Repo from our analysis at this point.  
# 
# The non-linear nature of our data  application of the Chebyshev bound may be a good approach to the remainde of the analysis due to the factor that our data would not necessarily fall into a normal distribution[8], we could also consider leveraging the Chernoff Bound/Hoeffding Distribution for this analysis. 
# 
# Some other appraoches could be stick with using non-linear algorithms, including support vector regression, k-nearest neighbors and extra tree.  In addition to the non-linear we could also investigate performance on ensemble algorithms like Gradient Boosting Machines and Random Forest.
# 
# With some reading I feel like selecting a non-Linear regression tree using Scikit-learn would be a good appraoach for this time series forecasting exercise 
# 
# Subset names for each bank
# BONYDom2
# BONYFo2
# SSBTDom2
# SSBTFo2
# NTRSDom2
# NTRSFo2

# In[754]:


import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt


# ### BONY Domestic Depoists Analysis: 
# 
# The first line of code creates an object of the target variable called 'target_column_BD'. The second line gives us the list of all the features, excluding the target variable 'BONYDom'.
# 
# We have seen above that the units of the variables differ significantly and may influence the modeling process. To prevent this, we will do normalization via scaling of the predictors between 0 and 1. The third line performs this task.
# 
# The fourth line displays the summary of the normalized data. We can see that all the independent variables have now been scaled between 0 and 1. The target variable remains unchanged. [8]

# In[755]:


#Create arrays for features and response variables BONYDom2
target_col_BD = ['BONYDom']
predictorsBD = list(set(list(BONYDom2.columns))-set(target_col_BD))
BONYDom2[predictorsBD] = BONYDom2[predictorsBD]/BONYDom2[predictorsBD].max()
BONYDom2.describe()


# We will build our model on the training set and evaluate its performance on the test set. The first couple of lines of code below create arrays of the independent (X) and dependent (y) variables, respectively. The third line splits the data into training and test dataset, with the 'test_size' argument specifying the percentage of data to be kept in the test data. The fourth line prints the shape of the training set[11]

# In[756]:


#Create Testng and Training Sets 
X_Bd = BONYDom2[predictorsBD].values
y_Bd = BONYDom2[target_col_BD].values

XBd_train, XBd_test, yBd_train, yBd_test = train_test_split(X, y, test_size=0.30, random_state=40)
print(XBd_train.shape); print(XBd_test.shape)


# CART Feature Importance Analysis: 
# We will first investigate feature importance for the CART models that will be leveraged for this analysis.   This analysis introduced by Jason Brownlee will be leveraged [12]

# In[757]:


# decision tree for feature importance on a regression problem
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot
# define the model
model = DecisionTreeRegressor()
# fit the model
model.fit(X_Bd, y_Bd)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# We can conduct our analysis based on the feature importance, we can first run the model as is and make adjustments as we see fit.  We will start our model selection by viewing the performance of decision trees. I'm having a hard time accepting that Vix is that important to this model. 

# In[758]:


dtreeBd = DecisionTreeRegressor(max_depth=8, min_samples_leaf=0.13, random_state=3)
dtreeBd.fit(XBd_train, yBd_train)


# Once the model is built on the training set, we can make the predictions. The first line of code below predicts on the training set. The second and third lines of code prints the evaluation metrics - RMSE and R-squared - on the training set. The same steps are repeated on the test dataset in the fourth to sixth lines.[10]

# In[759]:


# Code lines 1 to 3
predBd_train_tree= dtreeBd.predict(XBd_train)
print(np.sqrt(mean_squared_error(yBd_train,predBd_train_tree)))
print(r2_score(yBd_train, predBd_train_tree))

# Code lines 4 to 6
predBd_test_tree= dtreeBd.predict(X_test)
print(np.sqrt(mean_squared_error(yBd_test,predBd_test_tree))) 
print(r2_score(yBd_test, predBd_test_tree))


# The above output shows that the RMSE is 352.38 for train data and 325.17 for test data. On the other hand, the R-squared value is 72 percent for train data and 80 percent for test data. More improvement can be done by parameter tuning. We will be changing the values of the parameter, 'max_depth', to see how that affects the model performance[11].
# 
# The first four lines of code below instantiates and fits the regression trees with 'max_depth' parameter of 2 and 5, respectively. The fifth and sixth lines of code generate predictions on the training data, whereas the seventh and eight lines of code gives predictions on the testing data [11].

# In[760]:


# Code Lines 1 to 4: Fit the regression tree 'dtree1' and 'dtree2' 
dtree1Bd = DecisionTreeRegressor(max_depth=2)
dtree2Bd = DecisionTreeRegressor(max_depth=5)
dtree1Bd.fit(XBd_train, yBd_train)
dtree2Bd.fit(XBd_train, yBd_train)

# Code Lines 5 to 6: Predict on training data
tr1Bd = dtree1Bd.predict(XBd_train)
tr2Bd = dtree2Bd.predict(XBd_train) 

#Code Lines 7 to 8: Predict on testing data
y1Bd = dtree1Bd.predict(X_test)
y2Bd = dtree2Bd.predict(X_test) 


# Generate the evaluation metrics - RMSE and R-squared - for the first regression tree, 'dtree1Bd'.

# In[761]:


# Print RMSE and R-squared value for regression tree 'dtree1' on training data
print(np.sqrt(mean_squared_error(yBd_train,tr1Bd))) 
print(r2_score(yBd_train, tr1Bd))

# Print RMSE and R-squared value for regression tree 'dtree1' on testing data
print(np.sqrt(mean_squared_error(yBd_test,y1Bd))) 
print(r2_score(yBd_test, y1Bd)) 


# Not the best results when adjusting the max depth of our decision tree to 5, we will change our approach and try Random forest on the BONY domestic deposits to see if we get a better outcome of our model. 
# 
# The first line of code below instantiates the Random Forest Regression model with the 'n_estimators' value of 500. 'n_estimators' indicates the number of trees in the forest. The second line fits the model to the training data.
# 
# The third line of code predicts, while the fourth and fifth lines print the evaluation metrics - RMSE and R-squared - on the training set. The same steps are repeated on the test dataset in the sixth to eight lines of code.

# In[762]:


#RF model
model_rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=100)
model_rf.fit(XBd_train, yBd_train) 
pred_train_rf= model_rf.predict(XBd_train)
print(np.sqrt(mean_squared_error(yBd_train,pred_train_rf)))
print(r2_score(yBd_train, pred_train_rf))

pred_test_rf = model_rf.predict(XBd_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_rf)))
print(r2_score(yBd_test, pred_test_rf))


# Overall our Random Forest Model yeilds better results, our RMSE and R-Squared values on the training data are 153, and 94.7% respectively.  For the test data our RMSE and R-Squared yeilds 252 and 88% respectively, much better results versus the decision trees used before - showing that it is a superior model as compared to decision trees for forecasting. 
# 
# Lets quickly see if by reducing the explanatory variables down to those with the highest feature scores yeilds even better results.  We will make a new subset of the BONY Domestic deposits, reducing it down to the features with the most importance from our earlier investigation of classification feature importnace and see how our peformance stacks up against our model.  

# In[765]:


#Selecting 0,4,7 or  VIX, Dow Jones Close, GDP, and CPI
BONYDom3 = data2[['BONYDom','VIXCLS','djii_close_x','GDP','CPIAUCSL']]


# In[768]:


#Create arrays for features and response variables BONYDom3
target_col_BD3 = ['BONYDom']
predictorsBD3 = list(set(list(BONYDom3.columns))-set(target_col_BD3))
BONYDom3[predictorsBD] = BONYDom2[predictorsBD]/BONYDom2[predictorsBD3].max()
BONYDom3.describe()


# In[769]:


#Create arrays for features and response variables BONYDom3
X_Bd3 = BONYDom3[predictorsBD].values
y_Bd3 = BONYDom3[target_col_BD].values

XBd3_train, XBd3_test, yBd3_train, yBd3_test = train_test_split(X, y, test_size=0.30, random_state=40)
print(XBd_train.shape); print(XBd_test.shape)


# In[770]:


#RF model for BD3
model_rf.fit(XBd3_train, yBd3_train) 


# In[771]:


#Create Testng and Training Sets 
X_Bd3 = BONYDom3[predictorsBD3].values
y_Bd3 = BONYDom3[target_col_BD3].values

XBd3_train, XBd3_test, yBd3_train, yBd3_test = train_test_split(X_Bd3,y_Bd3, test_size=0.30, random_state=40)
print(XBd3_train.shape); print(XBd3_test.shape)


# In[772]:


Bd_pred_train_rf= model_rf.predict(XBd3_train)
print(np.sqrt(mean_squared_error(yBd3_train,Bd_pred_train_rf)))
print(r2_score(yBd3_train, Bd_pred_train_rf))

pred_test_rf = model_rf.predict(XBd3_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_rf)))
print(r2_score(yBd3_test, pred_test_rf))


# Reducing our features did not yeild better results, for BONY Domestic Deposits.  So now we can focus on our first Random Forest model and view performance metrics. 

# In[773]:


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Jupyter Notebook, include the following so that plots will display:
get_ipython().run_line_magic('matplotlib', 'inline')


# In[774]:


#Running original RF Regressor model once more to put this at the top of the analysis 
model_rf.fit(XBd_train, yBd_train) 
pred_train_rf= model_rf.predict(XBd_train)
print(np.sqrt(mean_squared_error(yBd_train,pred_train_rf)))
print(r2_score(yBd_train, pred_train_rf))

pred_test_rf = model_rf.predict(XBd_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_rf)))
print(r2_score(yBd_test, pred_test_rf))


# Our RMSE on the training set was 153mm with a R-Squared value is 95%, the best possible score is 1.0 or 100% so this model is a good one for BONY Domestic Deposits.  Below lets get a view of our precitions

# In[775]:


#Prediction on the modle to pass xBd_Test to get output y_pred providing an array of real numbers corresponding to the input array
yBd_Pred = model_rf.predict(XBd_test)
yBd_Pred


# In[776]:


mse = mean_squared_error(yBd_test, yBd_Pred)
rmse = np.sqrt(mse)
rmse


# To visualize a single decision tree for the BONY Domestic deposit model we can select one tree and save the tree as an image using the script below: 

# In[777]:


# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = model_rf.estimators_[5]
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = model_rf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = predictorsBD, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')


# In[778]:


#Use Pillow to show tree
from PIL import Image
im = Image.open('tree.png')
im.show()


# In[779]:


#Show tree in notebook 
from IPython.display import Image 
Image(filename = 'tree.png', width = 1000, height = 600)


# ### BONY Foreign Office Deposit Analysis: 
# We will implement our random forest regression on the BONY Foreign Office deposits 

# In[780]:


BONYFo.describe()


# In[781]:


BONYFo.head()


# In[782]:


#Create arrays for features and response variables BONYFo
target_col_BF = ['BONYFo']
predictorsBF = list(set(list(BONYFo2.columns))-set(target_col_BF))
BONYFo2[predictorsBF] = BONYFo2[predictorsBF]/BONYFo2[predictorsBF].max()
BONYFo2.describe()


# In[783]:


#Create Testng and Training Sets BONY Foreign Office Deposits 
X_Bf = BONYFo2[predictorsBF].values
y_Bf = BONYFo2[target_col_BF].values

XBf_train, XBf_test, yBf_train, yBf_test = train_test_split(X_Bf, y_Bf, test_size=0.30, random_state=40)
print(XBf_train.shape); print(XBf_test.shape)


# In[784]:


# decision tree for feature importance on a regression problem
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot
# define the model
model = DecisionTreeRegressor()
# fit the model
model.fit(X_Bf, y_Bf)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# In our BONY Foreign Office deposits we can see that our features play a more important role across the board, with the exception of 4 which is our S&P 500 Adjusted Close.  We will move forward with this analysis.
# 
# Now we can run our random forest regression.

# In[785]:


#RF model
model_rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=100)
model_rf.fit(XBf_train, yBf_train) 
pred_train_rf= model_rf.predict(XBf_train)
print(np.sqrt(mean_squared_error(yBf_train,pred_train_rf)))
print(r2_score(yBf_train, pred_train_rf))

pred_test_rf = model_rf.predict(XBf_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_rf)))
print(r2_score(yBf_test, pred_test_rf))


# Overall the RMSE and R-squared error for the training data is 53mm and 96%, and on the testing data we have slightly lower lesser accuracy in thatour RMSE is 851, and R-squared is 68%, lower than our domestic analysis. 

# In[786]:


#Prediction on the modle to pass xBd_Test to get output y_pred providing an array of real numbers corresponding to the input array
yBf_Pred = model_rf.predict(XBf_test)
yBf_Pred


# Given the fact that my main concern is forecasting NTRS deposits, I'm satisfied with this model we can move forward, for additional accuracy we could strip out the S&P 500 close from our analysis and see if results improve but BONY is not the most important factor in this analysis. 
# 
# To visualize a single decision tree for the BONY Foreign Office deposit model we can select one tree and save the tree as an image using the script below:

# In[787]:


# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = model_rf.estimators_[5]
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = model_rf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = predictorsBD, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')


# In[788]:


#Show tree in notebook 
from IPython.display import Image 
Image(filename = 'tree.png', width = 1000, height = 600)


# ### SSBT Domestic Deposits Analysis 

# In[789]:


#Create arrays for features and response variables SSBTDom2
target_col_SD = ['SSBTDom']
predictorsSD = list(set(list(SSBTDom2.columns))-set(target_col_SD))
SSBTDom2[predictorsSD] = SSBTDom2[predictorsSD]/SSBTDom2[predictorsSD].max()
SSBTDom2.describe()


# In[790]:


#Create Testng and Training Sets SSBT Domestic 
X_SD = SSBTDom2[predictorsSD].values
y_SD = SSBTDom2[target_col_SD].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
print(X_train.shape); print(X_test.shape)


# In[791]:


# decision tree for feature importance on a regression problem
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot
# define the model
model = DecisionTreeRegressor()
# fit the model
model.fit(X_SD, y_SD)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# For SSBT Domestic deposits it appears that we only have a couple of features within our model that are important, this would include VIX, 10 year Treasury Yeild (kind of), GDP, and Case Schiller Index. 
# 
# So we will narrow down our dataset to incorporate just these features for our analysis. 

# In[792]:


#Selecting 0,4,7 or  VIX, 10 year Treasury Yeild (kind of), GDP, and Case Schiller Index
SSBTDom3 = data2[['SSBTDom','VIXCLS','10yrTreasYeild','GDP','CS_Index']]


# In[793]:


#Create arrays for features and response variables SSBTDom3
target_col_SD3 = ['SSBTDom']
predictorsSD3 = list(set(list(SSBTDom3.columns))-set(target_col_SD3))
SSBTDom3[predictorsSD3] = SSBTDom3[predictorsSD3]/SSBTDom2[predictorsSD3].max()
SSBTDom3.describe()


# In[794]:


#Create Testng and Training Sets SSSBT Domestic 
X_SD3 = SSBTDom3[predictorsSD3].values
y_SD3 = SSBTDom3[target_col_SD3].values

XSD3_train, XSD3_test, ySD3_train, ySD3_test = train_test_split(X_SD3, y_SD3, test_size=0.30, random_state=40)
print(XSD3_train.shape); print(XSD3_test.shape)


# In[795]:


#RF model
model_rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=100)
model_rf.fit(XSD3_train, ySD3_train) 
pred_train_rf= model_rf.predict(XSD3_train)
print(np.sqrt(mean_squared_error(yBf_train,pred_train_rf)))
print(r2_score(ySD3_train, pred_train_rf))

pred_test_rf = model_rf.predict(XSD3_test)
print(np.sqrt(mean_squared_error(ySD3_test,pred_test_rf)))
print(r2_score(ySD3_test, pred_test_rf))


# Our model yeilded more optimal results with 760mm MSE and 98% on our RMSE on our training data, testing a 247mm MSE and 85% R-Squared, telling us that our subsetted data did a good job of predicting SSBT Domestic Deposits.  
# 
# Now we can get a view of our predcitions below:

# In[796]:


#Prediction on the modle to pass xSD_Test to get output y_pred providing an array of real numbers corresponding to the input array
ySD3_Pred = model_rf.predict(XSD3_test)
ySD3_Pred


# Now we can get a view of our trees: 

# In[797]:


# Pull out one tree from the forest
tree = model_rf.estimators_[5]
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = model_rf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = predictorsSD3, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')


# In[798]:


#Show tree in notebook 
from IPython.display import Image 
Image(filename = 'tree.png', width = 1000, height = 600)


# ### SSBT Foreign Office Deposit Analysis: 

# In[799]:


#Create arrays for features and response variables SSBTFo
target_col_SF = ['SSBTFo']
predictorsSF = list(set(list(SSBTFo2.columns))-set(target_col_SF))
SSBTFo2[predictorsSF] = SSBTFo2[predictorsSF]/SSBTFo2[predictorsSF].max()
SSBTFo2.describe()


# In[800]:


#Create Testng and Training Sets SSBT Foreign Office Deposits 
X_Sf = SSBTFo2[predictorsSF].values
y_Sf = SSBTFo2[target_col_SF].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
print(X_train.shape); print(X_test.shape)


# In[801]:


# decision tree for feature importance on a regression problem
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot
# define the model
model = DecisionTreeRegressor()
# fit the model
model.fit(X_Sf, y_Sf)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# On our SSBT Foreign Office Deposits, features 3,4,5, and 7 have the most importance this is: TEDRATE, AdjClose, and GDP we can adjust our data frame to include just those predictors: AdjClose, GDP, CS_Index, CPIAUCSL

# In[802]:


SSBTFo3 = SSBTFo2[['SSBTFo','AdjClose','GDP','CS_Index','CPIAUCSL']] 


# In[803]:


#Create arrays for features and response variables SSBTFo
target_col_SF3 = ['SSBTFo']
predictorsSF3 = list(set(list(SSBTFo3.columns))-set(target_col_SF3))
SSBTFo3[predictorsSF3] = SSBTFo3[predictorsSF3]/SSBTFo3[predictorsSF3].max()
SSBTFo3.describe()


# In[804]:


#Create Testng and Training Sets 
X_Sf3 = SSBTFo3[predictorsSF3].values
y_Sf3 = SSBTFo3[target_col_SF3].values

X_trainSF3, X_testSF3, y_trainSF3, y_testSF3 = train_test_split(X_Sf3, y_Sf3, test_size=0.30, random_state=40)
print(X_trainSF3.shape); print(X_testSF3.shape)


# In[805]:


#RF model
model_rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=100)
model_rf.fit(X_trainSF3, y_trainSF3) 
pred_train_rf= model_rf.predict(X_trainSF3)
print(np.sqrt(mean_squared_error(y_trainSF3,pred_train_rf)))
print(r2_score(y_trainSF3, pred_train_rf))

pred_test_rf = model_rf.predict(X_testSF3)
print(np.sqrt(mean_squared_error(y_testSF3,pred_test_rf)))
print(r2_score(y_testSF3, pred_test_rf))


# Our RMSE & R-Squared for the testing and training set fall within a resonable level of tolerance. Now we can get a look at our predcitions. 

# In[806]:


#Prediction on the modle to pass xSD_Test to get output y_pred providing an array of real numbers corresponding to the input array
ySF3_Pred = model_rf.predict(X_testSF3)
ySF3_Pred


# ### Northern Trust Domestic Deposit Analysis
# 
# We will go a little further in depth with this analysis as we'd like to apply this at some level to forecast deposit movement. 

# In[807]:


#Create arrays for features and response variables NTRNDom2
target_col_ND = ['NTRSDom']
predictorsND = list(set(list(NTRSDom2.columns))-set(target_col_ND))
NTRSDom2[predictorsND] = NTRSDom2[predictorsND]/NTRSDom2[predictorsND].max()
NTRSDom2.describe()


# In[808]:


#Create Testng and Training Sets NTRS Domestic 
X_ND = NTRSDom2[predictorsND].values
y_ND = NTRSDom2[target_col_ND].values

X_ND_train, X_ND_test, y_ND_train, y_ND_test = train_test_split(X_ND, y_ND, test_size=0.30, random_state=40)
print(X_ND_train.shape); print(X_ND_test.shape)


# In[809]:


# decision tree for feature importance on a regression problem
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot
# define the model
model = DecisionTreeRegressor()
# fit the model
model.fit(X_ND, y_ND)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# It will always surprise me the correlation between VIX and deposits, obviously for NTRS Domestic deposits this plays an important factor.  In addition 10-year treasury yield, and CPI.  Since there are only three indicators we can run the model as a whole and once more with these features only and see if we yield better results. 

# In[810]:


#RF model
model_rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=100)
model_rf.fit(X_ND_train, y_ND_train) 
pred_train_rf= model_rf.predict(X_ND_train)
print(np.sqrt(mean_squared_error(y_ND_train,pred_train_rf)))
print(r2_score(y_ND_train, pred_train_rf))

pred_test_rf = model_rf.predict(X_ND_test)
print(np.sqrt(mean_squared_error(y_ND_test,pred_test_rf)))
print(r2_score(y_ND_test, pred_test_rf))


# Our RMSE & R-Squared for the testing and training set fall within a resonable level of tolerance, 0.009, and 98% and 267mm 0.78. Let's see if adjusting our features to align with our feature importance analysis will improve the peformance of the model.

# In[811]:


NTRSDom3 = NTRSDom2[['NTRSDom','VIXCLS','10yrTreasYeild','CPIAUCSL']] 


# In[812]:


#Create arrays for features and response variables NTRNDom2
target_col_ND3 = ['NTRSDom']
predictorsND3 = list(set(list(NTRSDom3.columns))-set(target_col_ND3))
NTRSDom3[predictorsND3] = NTRSDom3[predictorsND3]/NTRSDom3[predictorsND3].max()
NTRSDom3.describe()


# In[813]:


#Create Testng and Training Sets NTRS3 Domestic 
X_ND3 = NTRSDom3[predictorsND3].values
y_ND3 = NTRSDom3[target_col_ND3].values

X_ND3_train, X_ND3_test, y_ND3_train, y_ND3_test = train_test_split(X_ND3, y_ND3, test_size=0.30, random_state=40)
print(X_ND3_train.shape); print(X_ND3_test.shape)


# In[814]:


#RF model
model_rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=100)
model_rf.fit(X_ND3_train, y_ND3_train) 
pred_train_rf= model_rf.predict(X_ND3_train)
print(np.sqrt(mean_squared_error(y_ND3_train,pred_train_rf)))
print(r2_score(y_ND3_train, pred_train_rf))

pred_test_rf = model_rf.predict(X_ND3_test)
print(np.sqrt(mean_squared_error(y_ND3_test,pred_test_rf)))
print(r2_score(y_ND3_test, pred_test_rf))


# Ok on our testing and training we have 77mm RMSE and 0.98 on our R-Squared which is very good, our predictions we have 0.02mm RMSE amd 0.77 R Squared, less accurate than our initial version with NTRS2 data.  We will rely on our NTRS2 random forest model with better accuracy to forecast our domestic deposits. 

# In[815]:


#Bringind down the initial RF model for NTRS Domestic Deposits 
model_rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=100)
model_rf.fit(X_ND_train, y_ND_train) 
pred_train_rf= model_rf.predict(X_ND_train)
print(np.sqrt(mean_squared_error(y_ND_train,pred_train_rf)))
print(r2_score(y_ND_train, pred_train_rf))

pred_test_rf = model_rf.predict(X_ND_test)
print(np.sqrt(mean_squared_error(y_ND_test,pred_test_rf)))
print(r2_score(y_ND_test, pred_test_rf))


# In[816]:


#Prediction on the modle to pass xSD_Test to get output y_pred providing an array of real numbers corresponding to the input array
y_ND_Pred = model_rf.predict(X_ND_test)
y_ND_Pred


# In[817]:


#Getting a full array of predictions on our dataset 
y_ND_PredFull = model_rf.predict(X_ND)
y_ND_PredFull


# In[818]:


#Appending predictions to dataframe of NTRS Domestic Deposits 
NTRSDom2['Predictions'] = y_ND_PredFull
NTRSDom2.head()


# In[822]:


# View of Tree 
# Pull out one tree from the forest
tree = model_rf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = predictorsND, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')

#Show tree in notebook 
from IPython.display import Image 
Image(filename = 'tree.png', width = 1000, height = 600)


# In[821]:


#Plot Predictions agains actuals: 
sns.set_style("darkgrid")
sns.lineplot(data = NTRSDom2, x='DATE', y='NTRSDom', color = 'g').set_title('NTRS Actual vs. Predicted Domestic Deposits')
ax2 = plt.twinx()
sns.lineplot(data = NTRSDom2, x = 'DATE', y = "Predictions", color = 'b')
ax


# Getting a visual on our predictions in blue versus our actuals in green, the random forest model peforms quite effectively in predicting domestic deposit movements. 
# 
# ### NTRS Foreign Office Deposits:

# In[823]:


#Create arrays for features and response variables NTRNDom2
target_col_NF = ['NTRSFo']
predictorsNF = list(set(list(NTRSFo2.columns))-set(target_col_NF))
NTRSFo2[predictorsNF] = NTRSFo2[predictorsNF]/NTRSFo2[predictorsNF].max()
NTRSFo2.describe()


# In[824]:


#Create Testng and Training Sets NTRS Foreign Office Deposits 
X_NF = NTRSFo2[predictorsNF].values
y_NF = NTRSFo2[target_col_NF].values

X_NF_train, X_NF_test, y_NF_train, y_NF_test = train_test_split(X_NF, y_NF, test_size=0.30, random_state=40)
print(X_NF_train.shape); print(X_NF_test.shape)


# In[825]:


#RF model
model_rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=100)
model_rf.fit(X_NF_train, y_NF_train) 
pred_train_rf= model_rf.predict(X_NF_train)
print(np.sqrt(mean_squared_error(y_NF_train,pred_train_rf)))
print(r2_score(y_NF_train, pred_train_rf))

pred_test_rf = model_rf.predict(X_NF_test)
print(np.sqrt(mean_squared_error(y_NF_test,pred_test_rf)))
print(r2_score(y_NF_test, pred_test_rf))


# Our random forest model and foreign office depoists are good friends! RMSE is good and high r-squared across both testing and training sets.  

# In[826]:


#Getting a full array of predictions on our dataset 
y_NF_PredFull = model_rf.predict(X_NF)
y_NF_PredFull


# In[827]:


#Appending predictions to dataframe of NTRS Foreign Office Deposits 
NTRSFo2['Predictions'] = y_NF_PredFull
NTRSFo2.head()


# In[829]:


# View of Tree 
# Pull out one tree from the forest
tree = model_rf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = predictorsNF, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')

#Show tree in notebook 
from IPython.display import Image 
Image(filename = 'tree.png', width = 1000, height = 600)


# In[830]:


#Plot Predictions agains actuals: 
sns.set_style("darkgrid")
sns.lineplot(data = NTRSFo2, x='DATE', y='NTRSFo', color = 'g').set_title('NTRS Actual vs. Predicted Foreign Office Deposits')
ax2 = plt.twinx()
sns.lineplot(data = NTRSFo2, x = 'DATE', y = "Predictions", color = 'b')
ax


# The results look amazing, proving that random forest regressors are an adequate methodology for forecasting foreign office deposits. It will be pretty cool to see this methodology applied to more granular data to get daily predictions.  Overall I'm very statisfied with this approach and look forward to helping apply the methodology for additional forecasting. 
# 
# Thanks! 

# ### References
# [1] python-edgar. (2019, November 9). Retrieved from https://pypi.org/project/python-edgar/
# 
# [2] Schroeder, J. (2019, August 27). Tutorial: real-time live feed of SEC filings using Python & socket.io. Retrieved from https://medium.com/@jan_5421/crawling-new-filings-on-sec-edgar-using-python-and-socket-io-in-real-time-5cba8c6a3eb8
# 
# [3] Chen, K. (2020). Use Python to download TXT-format SEC filings on EDGAR (Part I) | Kai Chen. Retrieved October 29, 2020, from http://kaichen.work/?p=59
# 
# [4] A. (2020). amaline/fdic-banks-api-python-client. Retrieved from https://github.com/amaline/fdic-banks-api-python-client
# 
# [5] D. (2020b). dpguthrie/bankfind. Retrieved from https://github.com/dpguthrie/bankfind
# 
# [6] Federal Reserve Economic Data | FRED | St. Louis Fed. (2020). Retrieved from https://fred.stlouisfed.org/
# 
# [7]Yahoo is now a part of Verizon Media. (2020). Retrieved from https://finance.yahoo.com/quote/%5EGSPC/history/
# 
# [8]Sarkar, T. (2018, November 5). What if your data is NOT Normal? - Towards Data Science. Medium. https://towardsdatascience.com/what-if-your-data-is-not-normal-d7293f7b8f0
# 
# [9] Chaipitakporn, C. (2019, October 26). Illustration with Python: Chebyshev’s Inequality - Analytics Vidhya. Medium. https://medium.com/analytics-vidhya/illustration-with-python-chebyshevs-inequality-b34be151c547
# 
# [10] Brownlee, J. (2020, August 28). How to Develop a Framework to Spot-Check Machine Learning Algorithms in Python. Machine Learning Mastery. https://machinelearningmastery.com/spot-check-machine-learning-algorithms-in-python/
# 
# [11] Singh, D. (2019, May 21). Non-Linear Regression Trees with scikit-learn. Pluralsight. https://www.pluralsight.com/guides/non-linear-regression-trees-scikit-learn
# 
# [12]Brownlee, J. (2020a, August 20). How to Calculate Feature Importance With Python. Machine Learning Mastery. https://machinelearningmastery.com/calculate-feature-importance-with-python/
