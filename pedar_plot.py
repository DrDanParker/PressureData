###############################################################################
###
### pedar_plot
### This file is part of BiomechPressure
### This file was created by Dr Daniel Parker 
### includes a set of fuctions to process walking data from pedar 
###   
### Copyright (C) 2018 University of Salford - All Rights Reserved
### You may use, distribute and modify this code under the terms of MIT Licence
### See <filename> or go to <url> for full licence details
###
###############################################################################


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



class PedarPlot:
    """
    A class to plot processed pressure data collected using Pedar System
    """
    
    def __init__(self, dfile):
        self.dfile = dfile
        
                
    def set_frame(dframe):
        if dframe.shape[0] != 99:
            if dframe.shape[1] == 99:
                dframe = dframe.T
            else:
                print('Error: Number of Sensors Not Correct')
        return dframe



    


    def format_frame_map(self,fmap):
        for k,v in fmap.items():
            new_grid = self.format_frame(fmap[k])
            fmap[k] = new_grid
        return(fmap)
    
    
    def set_grid(dframe,fname):
        """ Sets up image """
        gridplot = plt.imshow(dframe, origin='lower', 
                        cmap=plt.get_cmap('jet'), aspect='equal',vmin=0,vmax=400)
        plt.colorbar(gridplot)
    
            
    def compare_grids(self,grids):
        plt.figure() 
        ngrids = len(grids)
        a = 1
        b = round(ngrids/5)
        for k,v in grids.items():
            plt.subplot(b,5,a).set_title(k .format(a))
            self.set_grid(grids[k],k)
            a = a+1
        plt.show()
    

myfile = 'C:\Temp\Max_Data_Simple.xlsx'
df_map = pd.read_excel(myfile, sheet_name=None)

fmap = format_frame_map(df_map)
keys = list(fmap.keys())

compare_grids(fmap[keys[1]])


'''
dat = df_map[sname[0]]
datT = dat.T

fmap = format_frame(datT)
df = fmap[list(fmap.keys())[0]]
df2 = fmap[list(fmap.keys())[1]]
grids = [df,df2]

compare_grids(fmap)
'''
    

    
 
