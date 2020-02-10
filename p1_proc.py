###############################################################################
###
### p1_proc
### This file is part of CorePressure
### This file was created by Dr Daniel Parker 
### includes a set of fuctions to process walking data from pedar 
###   
### Copyright (C) 2018 University of Salford - All Rights Reserved
### You may use, distribute and modify this code under the terms of MIT Licence
### See <filename> or go to <url> for full licence details
###
###############################################################################

#p1_proc
import os
import numpy as np
import pandas as pd
import local
from matplotlib import pyplot as plt


def closeall(): # this closes all figures in reverse order
    l = plt.get_fignums()
    l.reverse()
    for a in l:
        plt.close(a)

def bld_flist(fpath,ftype='.asc'):
        ''' builds list of all files in directory+subs with given file type '''
        flist = [os.path.join(r,file) for r,d,f in os.walk(fpath) for file in f 
            if file.endswith(ftype)]
        return(flist)

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

def apply_masks(grid,Name):
    masked = {}
    peaks = pd.Series()
    aves = pd.Series()
    # tvals = pd.Series()
    #svals = pd.Series()

    ###Forefoot Mask    
    ff = grid.loc[8:11,:].copy(deep=True)
    peaks['Forefoot'] = max(ff.max())
    aves['Forefoot'] = ff.mean().mean()
    # tvals['Forefoot'] = ff[ff >200].mean().mean()
    #svals['Forefoot'] = ff[ff >200].sum().sum()
    forefoot = grid.copy(deep=True)
    forefoot.loc[8:11,:] = max(ff.max())
    masked['Forefoot'] = forefoot
  
    
    ### ThreePart Mask
    three_part = grid.copy(deep=True)
    
    TP_1 = grid.loc[8:11,0:2].copy(deep=True)
    three_part.loc[8:11,0:2] = max(TP_1.max())
    peaks['TP_1'] = max(TP_1.max())
    #thrs['TP_1'] = TP_1[TP_1 > 200].count().sum()
    aves['TP_1'] = TP_1.mean().mean()
    #tvals['TP_1'] = TP_1[TP_1 >200].mean().mean()
    #svals['TP_1'] = TP_1[TP_1 >200].sum().sum()
    
    TP_2 = grid.loc[8:11,3:4].copy(deep=True)
    three_part.loc[8:11,3:4] = max(TP_2.max())
    peaks['TP_2'] = max(TP_2.max())
    #thrs['TP_2'] = TP_2[TP_2 > 200].count().sum()
    aves['TP_2'] = TP_2.mean().mean()
    #tvals['TP_2'] = TP_2[TP_2 >200].mean().mean()
    #svals['TP_2'] = TP_2[TP_2 >200].sum().sum()
    
    TP_3 = grid.loc[8:11,5:6].copy(deep=True)
    three_part.loc[8:11,5:6] = max(TP_3.max())
    peaks['TP_3'] = max(TP_3.max())
    #thrs['TP_3'] = TP_3[TP_3 > 200].count().sum()
    aves['TP_3'] = TP_3.mean().mean()
    #tvals['TP_3'] = TP_3[TP_3 >200].mean().mean()
    #svals['TP_3'] = TP_3[TP_3 >200].sum().sum()
    
    masked['ThreePart'] = three_part
    
    
    ### FivePart Mask
    five_part = grid.copy(deep=True)

    FP_1 = grid.loc[8:11,0:2].copy(deep=True)
    five_part.loc[8:11,0:2] = max(FP_1.max())
    peaks['FP_1'] = max(FP_1.max())
    #thrs['FP_1'] = FP_1[FP_1 >200].count().sum()
    aves['FP_1'] = FP_1.mean().mean()
    #tvals['FP_1'] = FP_1[FP_1 >200].mean().mean()
    #svals['FP_1'] = FP_1[FP_1 >200].sum().sum()
        
    FP_2 = grid.loc[8:11,3].copy(deep=True)
    five_part.loc[8:11,3] = FP_2.max()
    peaks['FP_2'] = FP_2.max()
    #thrs['FP_2'] = FP_2[FP_2 >200].count().sum()
    aves['FP_2'] = FP_2.mean()
    #tvals['FP_2'] = FP_2[FP_2 >200].mean()
    #svals['FP_2'] = FP_2[FP_2 >200].sum()
        
    FP_3 = grid.loc[8:11,4].copy(deep=True)
    five_part.loc[8:11,4] = FP_3.max()
    peaks['FP_3'] = FP_3.max()
    #thrs['FP_3'] = FP_3[FP_3 >200].count().sum()
    aves['FP_3'] = FP_3.mean()
    #tvals['FP_3'] = FP_3[FP_3 >200].mean()
    #svals['FP_3'] = FP_3[FP_3 >200].sum()
        
    FP_4 = grid.loc[8:11,5].copy(deep=True)
    five_part.loc[8:11,5] = FP_4.max()
    peaks['FP_4'] = FP_4.max()
    #thrs['FP_4'] = FP_4[FP_4 >200].count().sum()
    aves['FP_4'] = FP_4.mean()
    #tvals['FP_4'] = FP_4[FP_4 >200].mean()
    #svals['FP_4'] = FP_4[FP_4 >200].sum()
        
    FP_5 = grid.loc[8:11,6].copy(deep=True)
    five_part.loc[8:11,6] = FP_5.max()
    peaks['FP_5'] = FP_5.max()
    #thrs['FP_5'] = FP_5[FP_5 >200].count().sum()
    aves['FP_5'] = FP_5.mean()
    #tvals['FP_5'] = FP_5[FP_5 >200].mean()
    #svals['FP_5'] = FP_5[FP_5 >200].sum()
    
    masked['FivePart'] = five_part
    return(ff,masked,peaks,aves)#,thrs,tvals,svals)


def out_data(ofile,sheet_map):
    writer = pd.ExcelWriter(ofile)
    for sheet in sheet_map:
        sheet_map[sheet].to_excel(writer,sheet)
    writer.save()


closeall()


fpath = 'C:/Users/hls376/Dropbox/UoS/Portals/[UoS] Price & Parker/Pressure_SPM/Mask&ROI_Processed/'


filelist = bld_flist(fpath,ftype='.xlsx')

fname = filelist[0]
m_data = {}

for fname in filelist:
    nkey = fname.split('/')[-1].split('.')[0]
    fdat = pd.read_excel(fname, sheet_name=None)
    ftyp = 'Mod'
    m_data[nkey] = [ftyp,fdat]


### Group by condition
conditions = []
for d in m_data[list(m_data.keys())[0]][1]:
    conditions.append(d)

c_data = {}
m_peaks = pd.DataFrame()
s_thrs = pd.DataFrame()
# mask_vals = pd.DataFrame()
# mask_sval = pd.DataFrame()
# mask_matched = pd.DataFrame()

# threshsensors = pd.DataFrame(index = ['Flat Count','Flat Sum', 'Flat Mean', 
#                                     'Flat Std', 'Cont Count', 'Cont Sum',
#                                     'Cont Mean', 'Cont Std', 'Change count',
#                                     'Change Sum', 'Change Mean', 'Change Sd'])

# std_data = {}


for m in m_data:
    _data = m_data[m][1]
    flat_ff, flat_masked, flat_peaks,flat_ave =apply_masks(_data['Flat'],'Flat')
    eva_ff, eva_masked, eva_peaks,eva_ave = apply_masks(_data['EVA'],'EVA')
    
    peaks = pd.DataFrame()
    peaks['flat_peak'] = flat_peaks
    peaks['flat_ave'] = flat_ave
    peaks['eva_peak'] = eva_peaks
    peaks['eva_ave'] = eva_ave
    peaks['flat >200'] = flat_peaks.where(flat_peaks >200)
    peaks['eva @>200'] = eva_peaks.where(flat_peaks >200)
    peaks['dif - abs'] = peaks['eva @>200'] - peaks['flat >200']
    peaks['dif - %'] = ((peaks['eva @>200'] - peaks['flat >200'])/peaks['flat >200'])
    stackpeak = peaks.transpose().stack(dropna=False)
    m_peaks[m] = stackpeak
    
    
    plt.figure(m)

    plt.subplot(3,5,1)
    set_grid(_data['Flat'],'PP Image')
    plt.subplot(3,5,2)
    set_grid(_data['Flat'].where(_data['Flat'] >200),'At-Risk')
    plt.subplot(3,5,3)
    set_grid(flat_masked['Forefoot'],'1Part ROI')
    plt.subplot(3,5,4)
    set_grid(flat_masked['ThreePart'],'3Part ROI')
    plt.subplot(3,5,5)
    set_grid(flat_masked['FivePart'],'5Part ROI')
    
    plt.subplot(3,5,6)
    set_grid(_data['EVA'],'')
    plt.subplot(3,5,7)
    set_grid(_data['EVA'].where(_data['Flat'] >200),'')
    plt.subplot(3,5,8)
    set_grid(eva_masked['Forefoot'],'')
    plt.subplot(3,5,9)
    set_grid(eva_masked['ThreePart'],'')
    plt.subplot(3,5,10)
    set_grid(eva_masked['FivePart'],'')
    
    plt.subplot(3,5,11)
    set_grid(_data['EVA']-_data['Flat'],'')
    plt.subplot(3,5,12)
    set_grid(_data['EVA'].where(_data['Flat'] >200)-_data['Flat'].where(_data['Flat'] >200),'')
    plt.subplot(3,5,13)
    set_grid(eva_masked['Forefoot']-flat_masked['Forefoot'],'')
    plt.subplot(3,5,14)
    set_grid(eva_masked['ThreePart']-flat_masked['ThreePart'],'')
    plt.subplot(3,5,15)
    set_grid(eva_masked['FivePart']-flat_masked['FivePart'],'')



    plt.savefig(fpath+m+'.png')
    
            
    
    ## Create Plots:
    # for m in m_data:
    #     _data = m_data[m][1]
    #     flat_masked = apply_masks(_data['Flat'])
    #     eva_masked = apply_masks(_data['EVA'])
    # 
    #     plt.figure(m)
    #     plt.subplot(3,5,1)
    #     set_grid(flat_['Flat'],'Flat')
    #     plt.subplot(3,5,6)
    #     set_grid(eva_masked['EVA'],'EVA')
    #     plt.subplot(3,5,3)
    #     set_mask(flat_masked['Forefoot'],'Flat_Forefoot',masksets('Forefoot'))
    #     plt.subplot(3,5,8)
    #     set_mask(eva_masked['Forefoot'],'EVA_Forefoot',masksets('Forefoot'))
    #     plt.subplot(3,5,4)
    #     set_mask(flat_masked['ThreePart'],'Flat_ThreePart',masksets('ThreePart'))
    #     plt.subplot(3,5,9)
    #     set_mask(eva_masked['ThreePart'],'EVA_ThreePart',masksets('ThreePart'))
    #     plt.subplot(3,5,5)
    #     set_mask(flat_masked['FivePart'],'Flat_FivePart',masksets('FivePart'))
    #     plt.subplot(3,5,10)
    #     set_mask(eva_masked['FivePart'],'EVA_FivePart',masksets('FivePart'))
    #     
    #     plt.subplot(3,5,2)
    #     set_grid(_data['Flat_200'],'200kPa')
    #     plt.subplot(3,5,7)
    #     set_grid(_data['EVA_200'],'200kPa')
    
    
    
    
    flat_thrs = pd.Series()
    FThr_200 = flat_ff.where(flat_ff >200)
    flat_thrs['Count'] = FThr_200.count().sum()
    flat_thrs['Peak'] = np.nanmax(np.asarray(FThr_200.values.flatten()))
    flat_thrs['Sum'] = np.nansum(np.asarray(FThr_200.values.flatten()))
    flat_thrs['Ave'] = np.nanmean(np.asarray(FThr_200.values.flatten()))
    flat_thrs['SD'] = np.nanstd(np.asarray(FThr_200.values.flatten()))
    
    eva_thrs = pd.Series()
    EThr_200 = eva_ff.where(flat_ff >200)
    eThr_200 = eva_ff.where(eva_ff >200)
    eva_thrs['Count'] = eThr_200.count().sum()
    eva_thrs['Peak'] = np.nanmax(np.asarray(EThr_200.values.flatten()))
    eva_thrs['Sum'] = np.nansum(np.asarray(EThr_200.values.flatten()))
    eva_thrs['Ave'] = np.nanmean(np.asarray(EThr_200.values.flatten()))
    eva_thrs['SD'] = np.nanstd(np.asarray(EThr_200.values.flatten()))
    
    thresh = pd.DataFrame()
    thresh['flat'] = flat_thrs
    thresh['eva'] = eva_thrs    
    thresh['diff - abs'] = eva_thrs - flat_thrs
    thresh['diff - %'] = ((eva_thrs - flat_thrs)/flat_thrs)
    
    stackthresh = thresh.transpose().stack(dropna=False)
    s_thrs[m] = stackthresh
    
    # peaks = pd.DataFrame(data=[flat_peaks,eva_peaks])
    # print(peaks)
    # print(thresh)
    
    # 
    # print(flat_peaks)
    # print(flat_peaks.where(flat_peaks > 200))
    #  
    # fp = flat_peaks.copy(deep=True).reset_index(drop=True)
    # print(fp)


# peak_plot = pd.DataFrame()
# peak_plot['Open'] = m_peaks.transpose()['flat']['Forefoot']
# peak_plot['Close'] = m_peaks.transpose()['eva']['Forefoot']
# 
# mins = peak_plot.min(axis=1)
# maxs = peak_plot.max(axis=1)
# peak_plot['Low'] = mins
# peak_plot['High'] = maxs
# 
# import matplotlib.finance as mpf
# 
# fig, ax = plt.subplots(figsize=(8,5))
# mpf.candlestick2_ochl(ax, peak_plot['Open'], peak_plot['Close'], peak_plot['High'], peak_plot['Low'], width=1, colorup='r', colordown='b', alpha=0.75)
# plt.show()
# 


sheet_map = {'Mask':m_peaks.transpose().sort_index(),'Threshold':s_thrs.transpose().sort_index()}
opath = 'C:/Users/hls376/Dropbox/UoS/Portals/[UoS] Price & Parker/Pressure_SPM/Mask&Sensor.xlsx'
out_data(opath,sheet_map)


# print(m_peaks)
# print(s_thrs)

'''
    
    
    fp = np.asarray(flat_peaks.values)
    ep = np.asarray(eva_peaks.values)
    print(float(fp))
    print(ep)
    
    print(np.where(ep>200,ep,'nan'))
    
    # FMsk_200 = fp.where(fp >200)
    # CMsk_200 = eva_peaks.where(fp >200)
    # print(fp)
    # print(FMsk_200)
    # print(eva_peaks)
    # print(CMsk_200)
    
    
    Fl = _data['Flat'].loc[8:11,:].copy(deep=True)
    Ct = _data['EVA'].loc[8:11,:].copy(deep=True)

    Fl200 = Fl.where(Fl > 200)
    Ct200 = Ct.where(Ct > 200)
    MCt200 = Ct.where(Fl > 200)

    FL = np.asarray(Fl200.values.flatten())
    CT = np.asarray(MCt200.values.flatten())
    
    Odata = []
    
    ##Flat
    Odata.append(Fl200.count().sum())
    Odata.append(np.nansum(FL))
    Odata.append(np.nanmean(FL))
    Odata.append(np.nanstd(FL))
    
    ##EVA
    Odata.append(Ct200.count().sum())
    Odata.append(np.nansum(CT))
    Odata.append(np.nanmean(CT))
    Odata.append(np.nanstd(CT))
    
    ##Percent Change
    Odata.append((((Fl200.count().sum() - Ct200.count().sum())/Fl200.count().sum())*100))
    Odata.append(((np.nansum(FL) - np.nansum(CT))/np.nansum(FL))*100 )
    Odata.append(((np.nanmean(FL) - np.nanmean(CT))/np.nanmean(FL))*100 )
    Odata.append(((np.nanstd(FL) - np.nanstd(CT))/np.nanstd(FL))*100 )
    
    threshsensors[m] = Odata
    
       
    mask_peaks[m] = pd.concat([flat_peaks, eva_peaks], axis=0)
    mask_thrs[m] = pd.concat([flat_thrs, eva_thrs], axis=0)
    mask_vals[m] = pd.concat([flat_tvals, eva_tvals], axis=0)
    mask_sval[m] = pd.concat([flat_svals, eva_svals], axis=0)   
    mask_matched[m] = pd.concat([FMsk_200, CMsk_200], axis=0)
    
threshsensors['Ave'] = threshsensors.mean(axis=1)
threshsensors['SD'] = threshsensors.std(axis=1)

                
ave_mask = mask_peaks.copy(deep=True)
ave_mask['Ave'] = mask_peaks.mean(axis=1)
ave_mask['SD'] = mask_peaks.std(axis=1)
ave_mask['n_200'] = mask_peaks[mask_peaks > 200.0].count(axis='columns')
ave_mask['n_per'] = (ave_mask['n_200']/mask_peaks.count(axis='columns'))*100

ave_thrs = mask_thrs.copy(deep=True)
ave_thrs['Ave'] = mask_thrs.mean(axis=1)
ave_thrs['SD'] = mask_thrs.std(axis=1)

ave_matched = mask_matched.copy(deep=True)
ave_matched['Ave'] = mask_matched.mean(axis=1)
ave_matched['SD'] = mask_matched.std(axis=1)

ave_vals = mask_vals.copy(deep=True)
ave_vals['Ave'] = mask_vals.mean(axis=1)
ave_vals['SD'] = mask_vals.std(axis=1)

ave_sval = mask_sval.copy(deep=True)
ave_sval['Ave'] = mask_sval.mean(axis=1)
ave_sval['SD'] = mask_sval.std(axis=1)


sheet_map = {'Mask':ave_mask,'Matched':ave_matched,'Thrs':ave_thrs,'TVals':ave_vals,'Svals':ave_sval,'Threshold':threshsensors}
opath = 'C:/Users/hls376/Dropbox/UoS/Portals/[UoS] Price & Parker/Pressure_SPM/MASK_AVE.xlsx'
out_data(opath,sheet_map)


# ave_mask.to_excel(opath)

# print(ave_mask)

    # print(flat_masked['FivePart'])
    
    # print(_data['Flat'])
    # print(flat_masked['ThreePart'])
    # print(max(_data['Flat'].loc[8:11,:]))

    
    
    # print(len(eva_masked))    



'''

'''



for cond in conditions:
    df_concat = pd.DataFrame()
    for f in m_data:
        pdata = m_data[f][1][cond]
        df_concat[f] = pdata
    c_data[cond] = df_concat


    
    
    # ff_data[cond] = df_forefoot
    
    ## Grand Average
    # by_row_index = df_concat.groupby(df_concat.index)
    mean_data[cond] = c_data[cond].mean()
    # std_data[cond] = by_row_index.std()
    
    print(mean_data['Flat'])

    # ff_grp = df_forefoot.groupby(df_forefoot.index)
    # ff_mean[cond] = ff_grp.mean()
    # ff_std[cond] = ff_grp.std()




###Mask Analysis
main = [c_data['EVA'],c_data['Flat']]

for f in c_data['Flat']:
    
    
    
    plt.figure()
    plt.subplot(221)
    set_grid(c_data['Flat'][f],f)
    plt.subplot(222)
    set_grid(c_data['EVA'][f],f)
    # set_grid(c_data['Flat'][f].loc[8:11,:],'Forefoot')
    
    plt.show()
    
'''
'''

## Create Plots:
# for m in m_data:
#     _data = m_data[m][1]
#     flat_masked = apply_masks(_data['Flat'])
#     eva_masked = apply_masks(_data['EVA'])
# 
#     plt.figure(m)
#     plt.subplot(3,5,1)
#     set_grid(flat_['Flat'],'Flat')
#     plt.subplot(3,5,6)
#     set_grid(eva_masked['EVA'],'EVA')
#     plt.subplot(3,5,3)
#     set_mask(flat_masked['Forefoot'],'Flat_Forefoot',masksets('Forefoot'))
#     plt.subplot(3,5,8)
#     set_mask(eva_masked['Forefoot'],'EVA_Forefoot',masksets('Forefoot'))
#     plt.subplot(3,5,4)
#     set_mask(flat_masked['ThreePart'],'Flat_ThreePart',masksets('ThreePart'))
#     plt.subplot(3,5,9)
#     set_mask(eva_masked['ThreePart'],'EVA_ThreePart',masksets('ThreePart'))
#     plt.subplot(3,5,5)
#     set_mask(flat_masked['FivePart'],'Flat_FivePart',masksets('FivePart'))
#     plt.subplot(3,5,10)
#     set_mask(eva_masked['FivePart'],'EVA_FivePart',masksets('FivePart'))
#     
#     plt.subplot(3,5,2)
#     set_grid(_data['Flat_200'],'200kPa')
#     plt.subplot(3,5,7)
#     set_grid(_data['EVA_200'],'200kPa')
#     
#     
#     plt.subplot(3,5,11)
#     set_grid(_data['Dif'],'Difference')
#     plt.subplot(3,5,12)
#     set_grid(_data['Dif_200'],'200kPa Diff')
#     plt.subplot(3,5,14)
#     set_grid(_data['thrsh'],'SPM')
# 
#     plt.savefig(fpath+m+'.png')


'''




# ff_grp = ff_data['EVA'].groupby(ff_data['EVA'].index)

# print(ff_grp)



# test = ff_mean['EVA']
# # test = test.loc[8:11,:]
# print(ff_grp)
# 
# for dat in main:

# for i in main:
#     trim_i = 

# EVA_200 = c_data['EVA_200']
# Forefoot_Eva,ThreePart_EVA,FivePart_EVA = 
# Flat_200 = c_data['Flat_200']



# 
# plt.figure()
# plt.subplot(3,5,1)
# set_grid(mean_data['Flat'],'Flat')
# plt.subplot(3,5,6)
# set_grid(mean_data['EVA'],'EVA')
# plt.subplot(3,5,3)
# set_mask(mean_data['Flat_Forefoot'],'Flat_Forefoot',masksets('Forefoot'))
# plt.subplot(3,5,8)
# set_mask(mean_data['EVA_Forefoot'],'EVA_Forefoot',masksets('Forefoot'))
# plt.subplot(3,5,4)
# set_mask(mean_data['Flat_ThreePart'],'Flat_ThreePart',masksets('ThreePart'))
# plt.subplot(3,5,9)
# set_mask(mean_data['EVA_ThreePart'],'EVA_ThreePart',masksets('ThreePart'))
# plt.subplot(3,5,5)
# set_mask(mean_data['Flat_FivePart'],'Flat_FivePart',masksets('FivePart'))
# plt.subplot(3,5,10)
# set_mask(mean_data['EVA_FivePart'],'EVA_FivePart',masksets('FivePart'))
# 
# plt.subplot(3,5,2)
# set_grid(mean_data['Flat_200'],'200kPa')
# plt.subplot(3,5,7)
# set_grid(mean_data['EVA_200'],'200kPa')
# 
# 
# plt.subplot(3,5,11)
# set_grid(mean_data['Dif'],'Difference')
# plt.subplot(3,5,12)
# set_grid(mean_data['Dif_200'],'200kPa Diff')
# plt.subplot(3,5,14)
# set_grid(mean_data['thrsh'],'SPM')
# 
# 
# 
# 
# plt.savefig(fpath+'GrandAverage.png')


# 
# print(mean_data['EVA'])
# print(std_data['EVA'])