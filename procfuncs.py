#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some processing functions

@author: vincent
"""

import os
import sys
import csv
import copy
import math
import numpy as np
import netCDF4 as nc
import scipy.stats as stats
import matplotlib.pyplot as plt
import shapefile
import statsmodels.api as sm
import datetime
from shapely.geometry import Point,Polygon,MultiPoint,LineString
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn
from scipy.interpolate import griddata

### Some operators ###
lg_a = np.logical_and
lg_o = np.logical_or
######################

def datestring_to_datedec(date_string):
    '''
    Converts date of format 'YYYY-MM-DD' to a decimal date
    '''
    dayspmonth = [31,28,31,30,31,30,31,31,30,31,30,31]
    yy = int(date_string[0:4])
    if(yy%4==0):
        dayspmonth[1] = 29
    mm = date_string[5:7]
    if(mm[0]=='0'):
        mm = mm[-1]
    mm = int(mm)-1
    dd = date_string[-2:]
    if(dd[0]=='0'):
        dd = dd[-1]
    dd = int(dd)-1
    ddfull  = np.sum(dayspmonth[0:mm])+dd
    decdate = yy+((ddfull+1/2)/np.sum(dayspmonth)) #mid-day
    return(decdate)

def yearlymean_frommonthly(tm_month,vals,axis_time=None):
    '''
    Yearly mean
    Assumes that input data is monthly
    '''
    ndim   = len(np.shape(vals))
    # Move time axis at index 0 #
    if(ndim>=2):
        if(axis_time is None):
            print('Error: provide axis_time in smooth_movingaverage_time for multi-dimensional arrays')
            sys.exit()
        mv_vals = np.moveaxis(vals,axis_time,0)
    else:
        mv_vals = np.copy(vals)
    ntot = np.shape(mv_vals)[0]
    nyr  = int(ntot/12)
    if(nyr<1):
        raise ValueError('time series is less than 1 year in yearlymean_frommonthly')

    # Mean #
    tm_yr   = np.array([np.mean(tm_month[ii*12:(ii+1)*12]) for ii in range (nyr)])
    if(ndim==1):
        outvals = np.array([np.mean(mv_vals[ii*12:(ii+1)*12,]) for ii in range (nyr)])
    else:
        outvals = np.concatenate([np.mean(mv_vals[ii*12:(ii+1)*12,],axis=0)[np.newaxis,] for ii in range (nyr)],axis=0)

    # Re-move to original time axis #
    if(ndim>=2):
        outvals = np.moveaxis(outvals,0,axis_time)

    return(tm_yr,outvals)

def smooth_movingaverage_time(timer,vals,window,axis_time=None,set_nans=True):
    '''
    if set_nans is True: nan values at time steps with insufficient values within window
    '''
    ntot    = len(timer) #number of time steps
    ndim    = len(np.shape(vals))
    nstpmin = int(np.floor(window/np.mean(np.diff(timer))))-1 #minimum number of values involved in each mean
    # Move time axis at index 0 #
    if(ndim>=2):
        if(axis_time is None):
            print('Error: provide axis_time in smooth_movingaverage_time for multi-dimensional arrays')
            sys.exit()
        mv_vals = np.moveaxis(vals,axis_time,0)
    else:
        mv_vals = np.copy(vals)
    
    # Smoothing #
    outvals = np.zeros_like(mv_vals)
    for tt in range(ntot):
        inds = abs(timer-timer[tt])<=(window/2)
        if(np.sum(inds)>=nstpmin or set_nans==False):
            outvals[tt] = np.mean(mv_vals[inds,],axis=0)
        elif(set_nans):
            outvals[tt] = np.nan
    
    # Re-move to original time axis #
    if(ndim>=2):
        outvals = np.moveaxis(outvals,0,axis_time)
    
    return(outvals)
  
def remove_linear_fit(xvals,yvals,axis=None,mask_in=None):
    '''
    Removes linear fit from ydata regressed on xdata
    Works for 1d, 2d, 3d arrays
    axis specifies axis along which centering_and_detrending is performed
    Returns detrended array of zero mean, and intercept and trend values
    '''

    ndim = len(np.shape(yvals))
    if(ndim==1):
        trend,intercept  = np.polyfit(xvals,yvals,deg=1)
        outarray         = yvals-intercept-trend*xvals
    elif(axis is not None):
        temp      = np.copy(np.moveaxis(yvals,axis,0))
        trend     = np.zeros(np.shape(temp)[1:])
        intercept = np.zeros(np.shape(temp)[1:])
        if(mask_in is None):
            wmask = np.ones(np.shape(temp)[1:]).astype(bool)
        else:
            wmask = np.copy(np.reshape(mask_in,np.shape(temp)[1:]))
        if(ndim==2):
            for ii in range(np.shape(temp)[1]):
                if(wmask[ii]):
                    trend[ii],intercept[ii] = np.polyfit(xvals,temp[:,ii],deg=1)
                    temp[:,ii]              = temp[:,ii]-intercept[ii]-trend[ii]*xvals
        elif(ndim==3):
            for ii in range(np.shape(temp)[1]):
                for jj in range(np.shape(temp)[2]):
                    if(wmask[ii,jj]):
                        trend[ii,jj],intercept[ii,jj] = np.polyfit(xvals,temp[:,ii,jj],deg=1)
                        temp[:,ii,jj]                 = temp[:,ii,jj]-intercept[ii,jj]-trend[ii,jj]*xvals
        else:
            print('Error: detrending_linear not implemented for arrays with more than 3 dimensions')
        outarray = np.moveaxis(temp,0,axis)
    else:
        print('Error: axis argument missing in detrending_linear')

    return(outarray,intercept,trend)

def get_linear_coeff(xvals,yvals,axis=None,mask_in=None):
    '''
    Gets the linear coefficient of ydata regressed on xdata
    Works for 1d, 2d, 3d arrays
    axis specifies axis along which regression is performed
    Returns coefficient values
    '''

    _,_,coeff = remove_linear_fit(xvals,yvals,axis=axis,mask_in=mask_in)
    return(coeff)
      

      
def simple_lin_world_plot(lats2d,lons2d,vals2d,maskval=-99,limsval=None):
    '''
    Simple world plot
    Linear value scale
    '''
    fieldplot = np.copy(vals2d)
    if(np.isnan(maskval)==False):
        inds = vals2d>maskval
    else:
        inds = np.isnan(maskval)==False
    if(limsval is None):
        levs = np.linspace(np.quantile(fieldplot[inds],0.01),np.quantile(fieldplot[inds],0.99),101)
    else:
        levs = np.linspace(limsval[0],limsval[1],101)
    if(np.any(inds==False)):
        fieldplot[inds==False] = np.nan
    fig = plt.figure(figsize=[12,6])
    ax  = plt.subplot(111)
    ax.set_position([0.005,0,0.88,1])
    mycmap = plt.cm.jet
    mymap = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180,resolution='c')
    mymap.drawcoastlines()
    mymap.fillcontinents(color='w')
    mymap.drawparallels(np.arange(-90.,91.,30.))
    mymap.drawmeridians(np.arange(-180.,181.,60.))
    mymap.drawmapboundary(fill_color='w')
    contourplot = plt.contourf(lons2d,lats2d,fieldplot,levels=levs,cmap=mycmap,extend='both')
    cax = fig.add_axes([0.89,0.2,0.01,0.6])
    cbar = plt.colorbar(contourplot,cax=cax)


