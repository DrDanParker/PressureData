###############################################################################
# p1_data.py
###############################################################################

import os
import sys
import importlib
import itertools
import numpy as np
import pandas as pd

import local

from matplotlib import pyplot	 as plt
import pedar_walk; importlib.reload(pedar_walk)
from pedar_walk import PedarWalkData as pdat
import AProcessing as ap; importlib.reload(ap)


def closeall(): # this closes all figures in reverse order
        l = plt.get_fignums()
        l.reverse()
        for a in l:
            plt.close(a)
            
def join_l(l, sep):
    out_str = ''
    for el in l:
        out_str += str(el) + str(sep)
    return out_str[:-1]

def grp_by_prefix(filelist,seperator='_',level=0):
    pref = []
    used = set()
    for file in filelist:
        pref.append(file.split(seperator)[level])
    unique = [x for x in pref if x not in used and (used.add(x) or True)]
    groups = []    
    for i in range(0,len(unique)):
        grp = []
        for j in range(0,len(pref)):
            if pref[j] == unique[i]:
                grp.append(filelist[j])
        groups.append(grp)
    return(groups)

def clean_data(data,fname,num):
    # plt.figure()
    # plt.subplot(411)
    # plt.plot(data.max(axis=1),color='r')    
    # plt.plot(data.mean(axis=1),color='b')    
    # 
    for col in data.columns:
        # plt.subplot(412)
        # plt.plot(data[col])
        if data[col].max() < 10: # Sensors below low limit
            # plt.subplot(413)
            # plt.plot(data[col]) 
            data[col] = 0
        elif data[col].max() < data[col].mean()*1.8: # Sensors with constant load
            if data[col].max() < 100:
                # plt.subplot(413)
                # plt.plot(data[col])
                data[col] = 0
    # plt.subplot(414)
    # plt.plot(data.max(axis=1),color='r')    
    # plt.plot(data.mean(axis=1),color='b')    
                
    # plt.savefig('C:/Temp/CleanPlots/'+figname+'.png')
    return(data)

def find_blocks(dataset,figname):
        """
        takes 2d cyclic dataset and chunks file to blocks
        returns start and end locations of each block
        """

        Condit = max(dataset) *0.75
        PPeaks = ap.PeakDetect(dataset,span=30,condit=Condit)
        MCondit = 0.5*Condit
        PMins = ap.MinDetect(dataset,span=30,condit=MCondit)
        
        plt.figure()
        
        ax = plt.subplot(211)
        ax.plot(dataset)
        ax.plot(PPeaks[0],PPeaks[1],'*')
        ax.plot(PMins[0],PMins[1],'*')
        
        ### Zero between cycles:
        for Pmin in PMins[1]:
            dataset[dataset == Pmin] = 0
        
        ### Ave gap:
        gap = 0
        for j in range(1, len(PPeaks[0])):
            gap = gap + (PPeaks[0][j]-PPeaks[0][j-1])
        gap = gap/len(PPeaks[0])
        
        ### Zero start:
        dataset[:PPeaks[0][0]] = 0
        for j in range(1, len(PPeaks[0])):
            if PPeaks[0][j] - PPeaks[0][j-1] > (gap*1.3):
                if j < 5:
                    dataset[:PPeaks[0][j]] = 0
                else: 
                    break
            elif j > 5:
                if PPeaks[0][j] - PPeaks[0][j-1] < gap:
                    break
        
        ### Zero end:
        dataset[PPeaks[0][-1]:] = 0

        ### Zero between blocks:
        step = [0]
        for j in range(1, len(PPeaks[0])):
            if PPeaks[0][j] - PPeaks[0][j-1] > (gap*1.45): ### ADJUSTABLE LEVEL
                dataset[PPeaks[0][j-1]:PPeaks[0][j]] = 0
                step.append(PPeaks[0][j])
        step.append(len(dataset))
        
        ## Remove small blocks:
        for j in range(1,len(step)):
            nsteps = PPeaks[0][(PPeaks[0]>step[j-1]) & (PPeaks[0]<step[j])],
            if len(nsteps[0]) < 4:
                dataset[step[j-1]:step[j]] = 0
        ax.plot(step,[0]*len(step),'*',color='r')
    
        
        ax = plt.subplot(212)
        ax.plot(dataset)
        
        
        count= 0
        first= 0
        start = []
        
        for j in range(0,len(dataset)):
            count+=1
            if dataset[j] != 0:
                if count > 50:
                    start.append(j)
                    count=0
                else:
                    first+=1
                    count=0
                    if first == 1:
                        start.append(j)
                        
        level = [50]*len(start)
        ax.plot(start,level,'*',color='g')

        
        end = []
        
        for st in start:
            count = 0
            for j in range(st,len(dataset)):
                count+=1
                if dataset[j] == 0:
                    if count > 50:
                        end.append(j)
                        break
                else:
                    count=0
        if len(end) < len(start):
            end.append(len(dataset))
            
        level = [50]*len(start)

        ax.plot(end,level,'*',color='r')
        plt.savefig('C:/Temp/BlockPlots/'+figname+'.png')
        return (start,end)


def data_blocks(data,figname,good_walks):
        """
        Cleans and stacks blocks of data for each insole
        """

        # Setup Mean Cycles and find blocks
        cyc_finder = pd.DataFrame()
        for col in data:
            if  data[col].max() > 100:
                cyc_finder[col] = insole_df[col]
        cyc_filt = cyc_finder.mean(axis=1)
        start,end = find_blocks(cyc_filt,figname)
        
        plt.figure()
        
        ax = plt.subplot(211)
        ax.plot(data.mean(axis=1))
        plt.savefig('C:/Temp/BlockCutPlots/'+figname+'.png')
        walk_set = []
        
        if len(start) > 0:
            # Cut raw data to blocks
            # walks = []
            for j in range(1,len(start)):
                walk = data.iloc[start[j]:end[j],:]
                steps_data = walk.mean(axis=1)
                
                Condit = steps_data.max()*0.4
                stp = ap.PeakDetect(steps_data,span=30,condit=Condit)
                PMins = ap.MinDetect(steps_data,span=45,condit=15)
                
                for Pmin in PMins[1]:
                    steps_data[steps_data == Pmin] = 0
                
                
                if len(stp[0]) >= 3:
                    ax.plot(steps_data,color='y')
                    ax.plot(stp[0]+start[j],stp[1],'*',color='k')
                    ax.plot(PMins[0]+start[j],PMins[1],'*',color='b')
                    plt.savefig('C:/Temp/BlockCutPlots/'+figname+'.png')
                    part_a = steps_data.iloc[:stp[0][1]][::-1]
                    trip_a = np.argwhere(part_a==0)[0]
                    cut_a = int(stp[0][1] - trip_a + start[j])
                    ax.plot(cut_a,0,'*',color='r')
                    plt.savefig('C:/Temp/BlockCutPlots/'+figname+'.png')
                    part_b = steps_data.iloc[stp[0][-3]:]
                    trip_b = np.argwhere(part_b==0)[0]
                    cut_b = int(stp[0][-3]+trip_b+start[j])
                    ax.plot(cut_b,0,'*',color='r')
                    plt.savefig('C:/Temp/BlockCutPlots/'+figname+'.png')
                    trim = data.iloc[cut_a:cut_b,:]
                    walktrim = trim.mean(axis=1)
                    ax.plot(walktrim,color='r')
                    walk_set.append(trim)
                    plt.savefig('C:/Temp/BlockCutPlots/'+figname+'.png')
        
        if len(walk_set) != len(good_walks):
            print('Walks :' + str(len(walk_set)))
            print('GoodWalks :' + str(len(good_walks)))
        
        toproc = []
        for j in range(len(good_walks)):
            if good_walks[j] == 1:
                ax.plot(walk_set[j].mean(axis=1),color='g')
                toproc.append(walk_set[j])

        if len(toproc) > 1:
            ins = toproc[0].append(toproc[1:])
            ins = ins.reset_index(drop=True)
        else:
            print('NUMBER OF WALKS IS LOW')

        ax = plt.subplot(212)
        ax.plot(ins.mean(axis=1))
        
        
        
        plt.savefig('C:/Temp/BlockCutPlots/'+figname+'.png')
    
        return(ins)


def cut_to_steps(single_insole,figname):
        """
        Cuts clean/stacked data to individual cycles or steps for a given walk
        input is dataframe for a single insole only.
        
        returns:
            Array containing individual steps each as a dataframe
        
        """
    
        SMean = single_insole.mean(axis=1)
        
        Condit = max(SMean)*0.5
        CPeaks = ap.PeakDetect(SMean,span=20,condit=Condit)
        # MCondit = 10
        # CMins = ap.MinDetect(SMean,span=10,condit=MCondit)
        
        ### NEED MINIMUM SETUP
        CM = [np.argmin(SMean[:CPeaks[0][0]])]
        for i in range(1,len(CPeaks[0])):
            CM.append(np.argmin(SMean[CPeaks[0][i-1]:CPeaks[0][i]]))
        CM.append(np.argmin(SMean[CPeaks[0][-1]:]))
        
        Mins = np.zeros(len(CM))
        CMins = [CM,Mins]
        
        plt.figure()
        ax = plt.subplot(211)
        ax.plot(SMean)
        ax.plot(CPeaks[0],CPeaks[1],'*')
        ax.plot(CMins[0],CMins[1],'*')
                
        
        steps = []
        ax = plt.subplot(212)
        
        
        for i in range(len(CPeaks[0])):
            
            peak = CPeaks[0][i]
            min_b_peak = max([x for x in CMins[0] if x <= peak])
            min_a_peak = min([x for x in CMins[0] if x >= peak])
            
            Cycle = single_insole.iloc[min_b_peak:min_a_peak,:]
            Cycle = Cycle.reset_index(drop=True)
            plt.plot(Cycle)
            steps.append(Cycle)
            
        
        
        
        plt.savefig('C:/Temp/CycPlots/'+figname+'.png')
           
        return(steps)
    
def peak_pressure(steps):
        """
        Gives peak pressure variables from pedar insole
        
        input is array containing individual steps 

        returns: 
            Array of peak pressure for each sensor for each cycle
            Array of highest overall peak pressure 
            Array of average peak pressure across all steps
        """
        
        sensors = [str(x) for x in range(1,100)]
        peaks = pd.DataFrame(columns=sensors)
        # print(steps)
        
        # plt.figure()
        
        for i in range(len(steps)):
            tag = 'step_'+str(i+1)
            step_df = steps[i]
            # plt.plot(step_df.max())
            peaks.loc[tag] = step_df.max()
        
        # plt.savefig('C:/Temp/CycPlots/'+figname+'Peaks.png')
        
        return(peaks)        


def bld_str(f):
    a = ''
    for i in f:
        a = a+str(i)+'_'
    return a[:-1]

###############################################################################
# Test Script
###############################################################################

closeall()
mypath = 'C:/Temp/ToRun/'
filelist = [os.path.join(r,file) for r,d,f in os.walk(mypath) for file in f 
            if file.endswith('.asc')]

groups = grp_by_prefix(filelist,seperator='\\')

blocks = pd.read_excel('C:/Temp/Blocks.xlsx', sheet_name=None)
block_keys = list(blocks.keys())
for group in groups:
    grp_tags = os.path.splitext(os.path.basename(group[0]))[0].split('_')
    grp_tag = join_l(grp_tags[0:2],'_')
    grp_num = int(grp_tags[2])
    
    key = 'AM_P1 ('+str(grp_num)+')'
    ofile = 'C:/Temp/OutDir/' +key+'.xlsx'
    block = blocks[key]
    
    print('Processing - '+key)
    # fig = plt.figure(key+ '_StackBlock')
    num = 0
    grp_dat = {}
    
    for fname in group:
        num +=1
        ffname = bld_str(fname.split('.')[0].split('_')[-2:])
        g_w = block.loc[block['Condition'] == ffname,:]
        gws = list(g_w.values)[0][1:]
        
        good_walks = []
        for w in gws:
            try:
                good_walks.append(int(w))
            except:
                continue
        print(good_walks)
        data = pdat.openPD(fname)
        
        #Split to Left and Right Insole Raw Data:        
        heads = list(data.head(0))
        Left = data.loc[:,heads[1]:heads[99]]
        try:
            Right = data.loc[:,heads[100]:heads[198]]
       
        except IndexError:
            Right = data.loc[:,heads[100]:]
        pair = {'Left':Left , 'Right':Right}

        sensors = [str(x) for x in range(1,100)]
        
        for insole in pair:
            insole_df = pd.DataFrame(data=pair[insole])
            insole_df.columns = sensors
            figname = os.path.splitext(fname)[0].split('\\')[-1] + '_' + insole
            print(figname)
            
            cdat = clean_data(insole_df,figname,num)
            walks = data_blocks(cdat,figname,good_walks)
            
            steps = cut_to_steps(walks,figname)
            print(len(steps))
            peaks = peak_pressure(steps)
            grp_dat[figname] = peaks
            peaks.to_csv('C:/Temp/OutDir/'+figname+'.txt',header=False,index=False,sep='\t')
    
    writer = pd.ExcelWriter(ofile)
    for dat in grp_dat:
        grp_dat[dat].to_excel(writer,dat)
    writer.save()
      
    
    # plt.tight_layout()    
    # plt.savefig('C:/Temp/'+key+'.png')
    # plt.close()
