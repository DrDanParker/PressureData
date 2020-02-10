###############################################################################
###
### p1_preproc
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
import numpy as np
import pandas as pd
import local
import math
import pedar_masks as pmask
from matplotlib import pyplot as plt
from collections import OrderedDict

def closeall(): # this closes all figures in reverse order
    l = plt.get_fignums()
    l.reverse()
    for a in l:
        plt.close(a)

def generate_fmap(df):
    ### Produces a dataframe for each condition in standard format
    grid_set = [5,7,7,7,7,7,7,7,7,7,7,7,7,6,4]
    fmap = {}
    for condition in df:
        pmap = df[condition].values
        ngrid = []
        n = 0
        for i in range(len(grid_set)):
            pdat = list(pmap[n:n+grid_set[i]])
            if grid_set[i] <6:
                pdat = [None] + pdat
            ngrid.append(pdat)
            n = n+grid_set[i]
        dgrid = pd.DataFrame(data=ngrid)
        fmap[condition] = dgrid
    return(fmap)

def set_grid(dframe,fname):
    """ Sets up image """
    ncheck = sum(n < 0 for n in dframe.values.flatten())
    if ncheck == 0:
        vmin = 10
        vmax = 400
        cmap = 'jet'
    else:
        vmin = -50
        vmax = 50
        cmap = 'seismic'
    gridplot = plt.imshow(dframe, origin='lower', 
                    cmap=plt.get_cmap(cmap), aspect=1.5,vmin=vmin,vmax=vmax)
    ax = plt.gca()
    rect = plt.Rectangle((-0.5,7.5),7,4,linewidth=2,edgecolor='k',facecolor='none')
    ax.add_patch(rect)
    plt.colorbar(gridplot)
    plt.title(fname)
    plt.xticks([])
    plt.yticks([])
    # plt.tight_layout()
    
def set_mask(dframe,fname,box):
    gridplot = plt.imshow(dframe, origin='lower', 
                    cmap=plt.get_cmap('jet'), aspect=1.5,vmin=10,vmax=400)
    ax = plt.gca()
    for b in box:
        rect = plt.Rectangle((b[0],b[1]),b[2],b[3],linewidth=1,edgecolor='k',facecolor='none')
        ax.add_patch(rect)
    plt.colorbar(gridplot)
    plt.title(fname)
    plt.xticks([])
    plt.yticks([])
    # plt.tight_layout()
    
# compare_fmaps(DAT[1],masks,figname)
#set_mask(dframe,fname,box)
def compare_fmaps(grids,masks,figname):
    plt.figure(figname) 
    ngrids = len(grids)
    n = 1
    b = math.ceil(ngrids/4)+1
    for grid in grids:
        plt.subplot(b,4,n).set_title(grid[1])
        set_grid(grids[grid],grid)
        n = n+1
    for mask in masks:
        plt.subplot(b,4,n).set_title(mask[1])
        set_mask(mask[0],mask[1],mask[2])
        n = n+1
    
    plt.show()
    
    

def whole_forefoot(dframe):
    wf_set = [x for x in range(55,83)]
    wf = dframe.iloc[wf_set].max(axis=0)
    return(wf)
    
def three_part_forefoot(dframe):
    p_masks = {'ThreePart_1':[55,56,62,63,69,70,76,77],
                'ThreePart_2':[57,58,64,65,71,72,78,79],
                'ThreePart_3':[59,60,61,66,67,68,73,74,75,80,81,82]}
    maskdata = {}
    for mask in p_masks:
        maskdata[mask] = dframe.iloc[p_masks[mask]].max(axis=0)
    return maskdata

def five_part_forefoot(dframe):
    p_masks = {'FivePart_1':[55,56,62,63,69,70,76,77],
                'FivePart_2':[57,58,64,65,71,72,78,79],
                'FivePart_3':[59,66,73,80],
                'FivePart_4':[60,67,74,81],
                'FivePart_5':[61,68,75,82]}
    maskdata = {}
    for mask in p_masks:
        maskdata[mask] = dframe.iloc[p_masks[mask]].max(axis=0)
    return maskdata



def apply_masks(grid,mask):
    masked = {}
    
    forefoot = grid.copy(deep=True)
    forefoot.loc[8:11,:] = mask['Forefoot']
    masked['Forefoot'] = forefoot
    three_part = grid.copy(deep=True)
    three_part.loc[8:11,0:2] = mask['ThreePart_1']
    three_part.loc[8:11,3:4] = mask['ThreePart_2']
    three_part.loc[8:11,5:6] = mask['ThreePart_3']
    masked['ThreePart'] = three_part
    five_part = grid.copy(deep=True)
    five_part.loc[8:11,0:2] = mask['FivePart_1']
    five_part.loc[8:11,3] = mask['FivePart_2']
    five_part.loc[8:11,4] = mask['FivePart_3']
    five_part.loc[8:11,5] = mask['FivePart_4']
    five_part.loc[8:11,6] = mask['FivePart_5']
    masked['FivePart'] = five_part
    
    return(masked)
    
def masksets(mask_type):
    
    if mask_type == 'Forefoot':
        grid_layout = [[-0.5,7.5,7,4]]
    elif mask_type == 'ThreePart':
        grid_layout = [[-0.5,7.5,3,4],[2.5,7.5,2,4],[4.5,7.5,2,4]]
    elif mask_type == 'FivePart':
        grid_layout = [[-0.5,7.5,3,4],[2.5,7.5,1,4],[3.5,7.5,1,4],[4.5,7.5,1,4],[5.5,7.5,1,4]]
    else:
        print('Bad Mask Type Given')
    return grid_layout
    
    
def bld_flist(fpath,ftype='.asc'):
        ''' builds list of all files in directory+subs with given file type '''
        flist = [os.path.join(r,file) for r,d,f in os.walk(fpath) for file in f 
            if file.endswith(ftype)]
        return(flist)

def out_data(ofile,sheet_map):
    writer = pd.ExcelWriter(ofile)
    for sheet in sheet_map:
        sheet_map[sheet].to_excel(writer,sheet)
    writer.save()

################################################################################
##
##  Run Code
##
################################################################################

closeall()

#spm_dat = pd.read_excel(spm_file)
fpath = 'C:/Users/hls376/Dropbox/UoS/Portals/[UoS] Price & Parker/Pressure_SPM/NewSPMOutput/'
opath = 'C:/Users/hls376/Dropbox/UoS/Portals/[UoS] Price & Parker/Pressure_SPM/OutData/'
spm_dir = 'C:/Users/hls376/Desktop/SPM_Output/'
filelist = bld_flist(fpath,ftype='.xlsx')
# spm_dir = filelist[-1]
filelist.pop()
#print(filelist)
#filelist = [filelist[0]]

for fname in filelist:
    
    df = pd.read_excel(fname, sheet_name=None)
    pdata = pd.DataFrame()
    masks = pd.DataFrame()
    
    for frame in df:
        
        
        
        mask = pd.Series()
        MaxVal = df[frame].mean(axis=0)         ## AVERAGE OF PEAKS
        pdata[frame] = MaxVal

        ### Calculate mask data
        wf_m = whole_forefoot(MaxVal)
        mask['Forefoot'] = wf_m
        tp_m = three_part_forefoot(MaxVal)
        for k, v in tp_m.items(): mask[k] = v
        fp_m = five_part_forefoot(MaxVal)
        for k, v in fp_m.items(): mask[k] = v        
        masks[frame] = mask
    
    
    
    ### Pull pSPM Data for participant into dataset
    n_parts = fname.split('/')[-1].split('(')
    pspm_names = []
    pspm_names.append(n_parts[0][:-1] +'_'+n_parts[1][:2] + '_Left.csv')
    pspm_names.append(n_parts[0][:-1] +'_'+n_parts[1][:2] + '_Right.csv')
    
    for pspm_name in pspm_names:
        try:
            tags = pspm_name.split('_')
            base = '_'.join(tags[:3])
            side = tags[-1][:-4]
            z = base + '_z_' + side
            thr = base + '_thrsh_' +side
            pspm_f = spm_dir + pspm_name
            pspm_dat = pd.read_csv(pspm_f,nrows=99,index_col=0)
            pspm_dat.index = pdata.index
            pdata[z] = pspm_dat['z']
            pdata[thr] = pspm_dat['thrsh']
        except:
            print('No SPM Data')
            pdata[z] = pd.DataFrame(data=['NaN']*99)
            pdata[thr] = pd.DataFrame(data=['NaN']*99)
    
    ### Group by side    
    Left = pdata.loc[:, pdata.columns.str.endswith('Left')]
    L_masks = masks.loc[:, masks.columns.str.endswith('Left')]
    Right = pdata.loc[:, pdata.columns.str.endswith('Right')]
    R_masks = masks.loc[:, masks.columns.str.endswith('Right')]
    
    ### Generate maps to plot
    L_map = generate_fmap(Left)
    R_map = generate_fmap(Right)

    PDAT = [[Left,L_map,L_masks],[Right,R_map,R_masks]]
    
    for DAT in PDAT:
        sheet_map = OrderedDict()
        fnams = list(DAT[0])
        side = fnams[0].split('_')[-1]
        oname = '_'.join(fnams[0].split('_')[:3]) + '_' + side + '.xlsx'
        
        for grid in DAT[1]:
            if 'EVA' in grid:
                eva = DAT[1][grid]
                eva_200 = DAT[1][grid].where(DAT[1][grid]>200,np.nan)
                eva_masked = apply_masks(DAT[1][grid],DAT[2][grid])
                 
            elif "3_P" in grid:
                flat = DAT[1][grid]
                flat_200 = DAT[1][grid].where(DAT[1][grid]>200,np.nan)
                flat_masked = apply_masks(DAT[1][grid],DAT[2][grid])
            elif "z" in grid:
                z = DAT[1][grid]
            elif "thrsh" in grid:
                thr = DAT[1][grid]
        
        sheet_map['EVA'] = eva
        sheet_map['EVA_200'] = eva_200
        for mask in eva_masked:
            sheet_map['EVA_Forefoot'] = eva_masked['Forefoot']
            sheet_map['EVA_ThreePart'] = eva_masked['ThreePart']
            sheet_map['EVA_FivePart'] = eva_masked['FivePart']
            
            # fft = [eva_masked['Forefoot'],'Forefoot',[[-0.5,7.5,7,4]]]
            # tpt = [eva_masked['ThreePart'],'ThreePart',[[-0.5,7.5,3,4],[2.5,7.5,2,4],[4.5,7.5,2,4]]]
            # fpt = [eva_masked['FivePart'],'FivePart',[[-0.5,7.5,3,4],[2.5,7.5,1,4],[3.5,7.5,1,4],[4.5,7.5,1,4],[5.5,7.5,1,4]]]
            # eva_mask = [fft,tpt,fpt]
        
        sheet_map['Flat'] = flat
        sheet_map['Flat_200'] = flat_200
        for mask in flat_masked:
            sheet_map['Flat_Forefoot'] = flat_masked['Forefoot']
            sheet_map['Flat_ThreePart'] = flat_masked['ThreePart']
            sheet_map['Flat_FivePart'] = flat_masked['FivePart']
        #     fft = [flat_masked['Forefoot'],'Forefoot',[[-0.5,7.5,7,4]]]
        #     tpt = [flat_masked['ThreePart'],'ThreePart',[[-0.5,7.5,3,4],[2.5,7.5,2,4],[4.5,7.5,2,4]]]
        #     fpt = [flat_masked['FivePart'],'FivePart',[[-0.5,7.5,3,4],[2.5,7.5,1,4],[3.5,7.5,1,4],[4.5,7.5,1,4],[5.5,7.5,1,4]]]
        #     flat_mask = [fft,tpt,fpt]        
    
        sheet_map['z'] = z
        sheet_map['thrsh'] = thr
        sheet_map['Dif'] = eva - flat
        sheet_map['Dif_200'] = eva_200 - flat_200
        
        
        ofile = opath + oname
        out_data(ofile,sheet_map)
        # print(sheet_map)
    
        # figname = _name+'_'+flat[2]
        # masks = [flat_mask,eva_mask]
        # compare_fmaps(DAT[1],masks[1],figname)

        
    
        
        



































################################################################################
################################################################################
################################################################################
################################################################################
##      OLD CODE
################################################################################
################################################################################
################################################################################
################################################################################


'''      
    
        dif_ = [eva[0]-flat[0],'EVA-Flat']
        dif_200 = [eva_200[0]-flat_200[0],'EVA-Flat 200kPa']
        case = [flat,eva,dif_,pspm_val,flat_200,eva_200,dif_200,pspm_thr]
        mask = [fft,tpt,fpt]
        figname = _name+'_'+flat[2]
        compare_fmaps(case,mask,figname)
        
            
        if 'EVA' in grid[1]:
            eva = [grid[0],grid[1],grid[2]]
            eva_200 = [grid[0].where(grid[0]>200,np.nan),grid[1]+'>200kPa']
        elif 'Flat' in grid[1]:
            flat = [grid[0],grid[1],grid[2]]
            flat_200 = [grid[0].where(grid[0]>200,np.nan),grid[1]+'>200kPa']                
        elif 'z' in grid[1]:
            pspm_val = [grid[0],grid[1]]
        elif 'thrsh' in grid[1]:
            pspm_thr = [grid[0],grid[1]]

    ### Difference generator ensuring direction is always the same (eva-flat)
    for case in sets[:2]: # set to 2 based on existing pspm comparisons (STOCK only)
        for grid in case:
            
            elif 'Masks' in grid[1]:
                fft = [grid[0]['Forefoot'],'Forefoot',[[-0.5,7.5,7,4]]]
                tpt = [grid[0]['ThreePart'],'ThreePart',[[-0.5,7.5,3,4],[2.5,7.5,2,4],[4.5,7.5,2,4]]]
                fpt = [grid[0]['FivePart'],'FivePart',[[-0.5,7.5,3,4],[2.5,7.5,1,4],[3.5,7.5,1,4],[4.5,7.5,1,4],[5.5,7.5,1,4]]]
                

    
    
    plt.figure()
    plt.subplot(211)
    for dat in Left:
        plt.plot(Left[dat])
    plt.subplot(212)
    for dat in Right:
        plt.plot(Right[dat])
    
    plt.show()
    
    
    

    for col in spm_dat:
        if spm_dat[col]['participant'].split('/')[-1].split('_')[0] == _name:
            if spm_dat[col]['participant'].split('/')[-1].split('_')[3] == 'Left.txt':
                pspm_dat[_name+'_Left_'+spm_dat[col].name] = spm_dat[col]
            else:
                pspm_dat[_name+'_Right_'+spm_dat[col].name] = spm_dat[col]
    pspm_dat = pspm_dat.iloc[0:99,:]
    pspm_map = generate_fmap(pspm_dat)
    pspm_keys = list(pspm_map.keys())
   
    ### Grouping loop to pull all direct comparisons together
    for key in keys:
        tag = key.split('_')[2]
        if key.split('_')[1] == 'Stock':
            if key.split('_')[3] == 'Left':
                masked = apply_masks(fmap[key],mdata[key])
                sets[0].append([masked,'Masks','Left'])
                sets[0].append([fmap[key],tag,'Left'])
            else:
                masked = apply_masks(fmap[key],mdata[key])
                sets[1].append([masked,'Masks','Left'])
                sets[1].append([fmap[key],tag,'Right'])
                
        elif key.split('_')[3] == 'Left':
            sets[2].append([fmap[key],tag,'Left'])
        else:
            sets[3].append([fmap[key],tag,'Right'])
    
    for key in pspm_keys:
        tag = key.split('.')[0].split('_')[-1]
        if 'Left' in key:
            sets[0].append([pspm_map[key],'pSPM_'+tag])
        else:
            sets[1].append([pspm_map[key],'pSPM_'+tag])
    
    
    ### Difference generator ensuring direction is always the same (eva-flat)
    for case in sets[:2]: # set to 2 based on existing pspm comparisons (STOCK only)
        for grid in case:
            if 'EVA' in grid[1]:
                eva = [grid[0],grid[1],grid[2]]
                eva_200 = [grid[0].where(grid[0]>200,np.nan),grid[1]+'>200kPa']
            elif 'Flat' in grid[1]:
                flat = [grid[0],grid[1],grid[2]]
                flat_200 = [grid[0].where(grid[0]>200,np.nan),grid[1]+'>200kPa']                
            elif 'z' in grid[1]:
                pspm_val = [grid[0],grid[1]]
            elif 'thrsh' in grid[1]:
                pspm_thr = [grid[0],grid[1]]
            elif 'Masks' in grid[1]:
                fft = [grid[0]['Forefoot'],'Forefoot',[[-0.5,7.5,7,4]]]
                tpt = [grid[0]['ThreePart'],'ThreePart',[[-0.5,7.5,3,4],[2.5,7.5,2,4],[4.5,7.5,2,4]]]
                fpt = [grid[0]['FivePart'],'FivePart',[[-0.5,7.5,3,4],[2.5,7.5,1,4],[3.5,7.5,1,4],[4.5,7.5,1,4],[5.5,7.5,1,4]]]
                
                
        dif_ = [eva[0]-flat[0],'EVA-Flat']
        dif_200 = [eva_200[0]-flat_200[0],'EVA-Flat 200kPa']
        case = [flat,eva,dif_,pspm_val,flat_200,eva_200,dif_200,pspm_thr]
        mask = [fft,tpt,fpt]
        figname = _name+'_'+flat[2]
        compare_fmaps(case,mask,figname)
        
      
            
            



for fname in filelist:
    df = pd.read_excel(fname, sheet_name='Max Peak')
    pdata = df.iloc[:,0:99].T    ### Data for whole insole with columb as condition and 1-99 sensors
    mdata = df.iloc[:,99:].T     ### Data for masks with columb as condition   
    sets = [[],[],[],[]]
    fmap = generate_fmap(pdata)
    keys = list(fmap.keys())
    
    # print(fmap[keys[0]])
    # print(mdata[keys[0]])
    # masked = apply_masks(fmap[keys[0]],mdata[keys[0]])
    # print(masked)
    
    
   
    
    ### Grouping loop to pull all direct comparisons together
    for key in keys:
        tag = key.split('_')[2]
        if key.split('_')[1] == 'Stock':
            if key.split('_')[3] == 'Left':
                masked = apply_masks(fmap[key],mdata[key])
                sets[0].append([masked,'Masks','Left'])
                sets[0].append([fmap[key],tag,'Left'])
            else:
                masked = apply_masks(fmap[key],mdata[key])
                sets[1].append([masked,'Masks','Left'])
                sets[1].append([fmap[key],tag,'Right'])
                
        elif key.split('_')[3] == 'Left':
            sets[2].append([fmap[key],tag,'Left'])
        else:
            sets[3].append([fmap[key],tag,'Right'])
    
    
    
    # compare_fmaps([stock[0][0],stock[0][1]])
    
    # ofile = mypath + os.path.splitext(os.path.basename(fname))[0] + '_Masked.xlsx'
    # out_data(ofile,sheet_map)
'''