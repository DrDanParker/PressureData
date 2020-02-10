###############################################################################
###
### pedar_walk
### This file is part of CorePressure
### This file was created by Dr Daniel Parker 
### includes a set of fuctions to process walking data from pedar 
###   
### Copyright (C) 2018 University of Salford - All Rights Reserved
### You may use, distribute and modify this code under the terms of MIT Licence
### See <filename> or go to <url> for full licence details
###
###############################################################################

#Builtins:
import numpy as np
import pandas as pd
# import matplotlib as mpl
# mpl.rcParams['backend'] = "qt4agg"
# mpl.rcParams['backend.qt4'] = "PySide"

from matplotlib import pyplot	 as plt
from matplotlib.backends.backend_pdf import PdfPages
import importlib

#Independents:
import AProcessing as ap; importlib.reload(ap)


class PedarWalkData:
    """
    A class to process pressure data collected using Pedar System
    """
    
    def __init__(self, files):
        self.files = files


    def closeall(): # this closes all figures in reverse order
        l = plt.get_fignums()
        l.reverse()
        for a in l:
            plt.close(a)


    def pdf_plot(fname):
        pp = PdfPages('C:/Temp/'+fname+'.pdf')
        return(pp)

                        
    def openPD(fname):
        srows = [i for i in range(9)]
        fdat = pd.read_csv(fname,sep='\t',skiprows=srows)
        return(fdat)
        
    
    def find_blocks(dataset,figname,pp):
        """
        takes 2d cyclic dataset and chunks file to blocks
        returns start and end locations of each block
        """
        
        Condit = max(dataset)*0.75 
        PPeaks = ap.PeakDetect(dataset,span=30,condit=Condit)
        MCondit = 0.5*Condit
        PMins = ap.MinDetect(dataset,span=30,condit=MCondit)
        
        fig = plt.figure(figname + '_findBlock')
        plt.subplot(311)
        plt.plot(dataset)
        plt.plot(PPeaks[0],PPeaks[1],'*')
        plt.plot(PMins[0],PMins[1],'*')
        
        ### Zero between cycles:
        for Pmin in PMins[1]:
            dataset[dataset == Pmin] = 0
        
        ### Zero start:
        for j in range(1, len(PPeaks[0])):
            if PPeaks[0][j] - PPeaks[0][j-1] > 100:
                if j < 9:
                    dataset[:PPeaks[0][j]] = 0
                else: 
                    break
        
        ### Zero end:
        dataset[PPeaks[0][-1]:] = 0
        for j in range(len(PPeaks[0])-5, len(PPeaks[0])):
            if PPeaks[0][j] - PPeaks[0][j-1] > 100:
                dataset[PPeaks[0][j]:] = 0
            else: 
                break
                                            
        ### Zero between blocks:
        for j in range(1, len(PPeaks[0])):
            if PPeaks[0][j] - PPeaks[0][j-1] > 85:
                dataset[PPeaks[0][j-1]:PPeaks[0][j]] = 0

        plt.subplot(312)
        plt.plot(dataset)
        plt.plot(PPeaks[0],PPeaks[1],'*')
        
        ### chunk blocks:
        blocks = []
        bl = [PPeaks[0][0]]
        
        for i in range(1,len(PPeaks[0])):
            if i == len(PPeaks[0])-1:
                if len(bl) > 9:
                    blocks.append(bl)
            elif PPeaks[0][i] - PPeaks[0][i-1] < 90:
                bl.append(PPeaks[0][i])
            else:
                if len(bl) > 9:
                    blocks.append(bl)
                bl = []
        
        ### Setup outputs:
        start = []
        end = []
    
        for bl in blocks:
            bl = bl[1:-2]

            flip_data = np.array(dataset[:bl[0]])[::-1]
            slip_data = np.array(dataset[bl[-1]:])
            
            st = np.argmin(flip_data)+1
            en = np.argmin(slip_data)
            
            start.append(bl[0]-st)
            end.append(bl[-1]+en)
        
        ys = [75] * len(start)
        
        plt.subplot(313)
        plt.plot(dataset)
        plt.plot(start,ys,'*')
        plt.plot(end,ys,'*')
        
        pp.savefig(fig)
        plt.close()
        return (start,end)
    
    def one_insole_one_block(data,ffname,insole='Left'):
        heads = list(data.head(0))
        if insole == 'Left':
            pdat = data.loc[:,heads[1]:heads[99]]
            insole_label = 'Left'
        else:
            try:
                pdat = data.loc[:,heads[100]:heads[198]]
                insole_label = 'Right'
            except IndexError:
                pdat = data.loc[:,heads[100]:]
                insole_label = 'Right - Part'
       

        sensors = [str(x) for x in range(1,100)]
        insole_df = pd.DataFrame(data=pdat)
        insole_df.columns = sensors
        
        # Setup Mean Cycles and find blocks 
        insole_df['Mean'] = insole_df.mean(axis=1)
        steps = PedarWalkData.cut_to_steps(insole_df,insole_label)
        trim_steps = steps[20:120]
    
        peaks = PedarWalkData.peak_pressure(trim_steps,insole_label)

        
        return peaks


    
    
    def stack_blocks(data,ffname):
        """
        Cleans and stacks blocks of data for each insole
        """
        
        pp = PedarWalkData.pdf_plot(ffname)
        data_file = data
        heads = list(data_file.head(0))
            
        #Split to Left and Right Insole Raw Data:        
        Left = data_file.loc[:,heads[1]:heads[99]]
        try:
            Right = data_file.loc[:,heads[100]:heads[198]]
       
        except IndexError:
            Right = data_file.loc[:,heads[100]:]
       
        PDat = [Left , Right]
        insole_labels = ['Left','Right']
      
        data_labels = [] # Output Labels
        insole_pair = [] # Output Dataset
        
        for i in range(0, len(PDat)):    
            figname = ffname + '_' + insole_labels[i]
            data_labels.append(figname)

            sensors = [str(x) for x in range(1,100)]
            insole_df = pd.DataFrame(data=PDat[i])
            # insole_df(columns=sensors)
            insole_df.columns = sensors
            
            # Setup Mean Cycles and find blocks 
            insole_df['Mean'] = insole_df.mean(axis=1)
            start,end = PedarWalkData.find_blocks(insole_df['Mean'],figname,pp)
            
            if len(start) > 0:
                # Cut raw data to blocks
                walks = []
                walkNo = []
                for j in range(len(start)):
                    walks.append(insole_df[start[j]:end[j]])
                    walkNo.append(str(j))
                
                
                ### Combine walks where trial is acceptable:
                ##############
                # THIS NEEDS TO BE PREDEFINED
                ##############
                
                # good_walks = range(len(walks))
                
                ########################
                ########################
                
                if len(walks) > 1:
                    ins = walks[0].append(walks[1:])
                    ins = ins.reset_index(drop=True)
                    insole_pair.append(ins)
                elif len(walks) == 1:
                    ins = walks[0]
                    ins = ins.reset_index(drop=True)
                    insole_pair.append(ins)
                else:
                    insole_pair.append([])
                    print('    ' + figname + ': The number of walks is too low')
                    figname = 'ERROR'
                    data_labels.append(figname)
            else:
                insole_pair.append([])
                print('    ' + figname + ': The number of steps is too low')
                figname = 'ERROR'
                data_labels.append(figname)
        
        ''' Plots '''
        try:
            fig = plt.figure(ffname+ '_StackBlock')
            plt.subplot(221)
            plt.plot(Left.mean(axis=1))
            plt.subplot(222)
            plt.plot(Right.mean(axis=1))
            
            plt.subplot(223)
            plt.plot(insole_pair[0]['Mean'])
            plt.subplot(224)
            plt.plot(insole_pair[1]['Mean'])
        except TypeError:
            print('PlotError')
    
        
        pp.savefig(fig)
        pp.close()
        plt.close()
    
        return(insole_pair,data_labels)
        
        
    def cut_to_steps(single_insole,label):
        """
        Cuts clean/stacked data to individual cycles or steps for a given walk
        input is dataframe for a single insole only.
        
        returns:
            Array containing individual steps each as a dataframe
        
        """
        if label != 'ERROR':
            
            SMean = single_insole['Mean']
            Condit = max(SMean)*0.5
            CPeaks = ap.PeakDetect(SMean,span=20,condit=Condit)
            
            steps = []
                
            for peak in CPeaks[0]:
                
                flip_data = np.array(SMean[:peak])[::-1]
                slip_data = np.array(SMean[peak:])
                
                st = peak - np.argmin(flip_data)+1
                en = peak + np.argmin(slip_data)
                
                steps.append(single_insole[st:en])
                
            return(steps)
        
        else:
            print('Input Error: No Input Data')
            return([])    
            
        
    def peak_pressure(steps,figname):
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
        
        for i in range(len(steps)):
            tag = 'step_'+str(i+1)
            step_df = steps[i].loc[:]
            peaks.loc[tag] = step_df.max()
        
        
        return(peaks)
    
    def pedar_mask(dframe,sheet):
        ''' applys mask to dataframe and adds columns with mask names '''
        
        ### Pedar Mask Definitions - based on {}
        p_masks = {'1 Heel':[str(x) for x in range(1,10)],
                    '2 Lat Arch':[str(x) for x in range(10,20)],
                    '3 Med Arch':[str(x) for x in range(20,30)],
                    '4 1st MTH':[str(x) for x in range(30,40)],
                    '5 2-4th MTH':[str(x) for x in range(40,50)],
                    '6 5th MTH':[str(x) for x in range(50,60)],
                    '7 Hallux':[str(x) for x in range(60,70)],
                    '8 Lesser Toes':[str(x) for x in range(70,100)]}
        
        ### Create column for each mask with maximum pressure value from selected sensors
        for mask in sorted(p_masks):
            dframe[mask] = dframe[p_masks[mask]].max(axis=1)
        
        return(dframe)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        