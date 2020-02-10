###############################################################################
###
### checksteps
### This file is part of CorePressure
### This file was created by Dr Daniel Parker 
### includes a set of fuctions to process walking data from pedar 
###   
### Copyright (C) 2018 University of Salford - All Rights Reserved
### You may use, distribute and modify this code under the terms of MIT Licence
### See <filename> or go to <url> for full licence details
###
###############################################################################


import os
import pandas as pd
import numpy as np

def bld_flist(fpath,ftype='.asc'):
        ''' builds list of all files in directory+subs with given file type '''
        flist = [os.path.join(r,file) for r,d,f in os.walk(fpath) for file in f 
            if file.endswith(ftype)]
        return(flist)

fpath = 'C:/Temp/SPM_LOW CYCLES/'
filelist = bld_flist(fpath,ftype='.xlsx')

step_count = []
low_steps = []

OutSet = pd.DataFrame(columns = ['EVA_Left','EVA_Right','P_Left','P_Right'])

for fname in filelist:
    df = pd.read_excel(fname, sheet_name=None)
    dvals = pd.DataFrame(index=[fname])
    for cond in df:
        
        step_count.append(len(df[cond]))
        cond_strip = cond.split('_')[-2:]
        cond_tag = cond_strip[0] + '_' + cond_strip[1]
        # print(cond_tag)
        dvals[cond_tag] = [len(df[cond])]
        if len(df[cond]) < 15:
            low_steps.append(cond)
    dvals = dvals.reindex(sorted(dvals.columns), axis=1)
    print(dvals)
    # dvals = pd.DataFrame(data=vals,columns=cols)
    OutSet = pd.concat([OutSet, dvals], axis=0)
    

print(OutSet)

OutSet.to_excel('C:/Temp/stepcount.xlsx')

print('Min: ' + str(min(step_count)))
print('Max: ' + str(max(step_count)))
print('Mean: ' + str(np.mean(step_count)))
print('No of low: ' + str(len(low_steps)))
print(low_steps)