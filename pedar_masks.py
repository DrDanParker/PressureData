
import os
import pandas as pd


def whole_forefoot(dframe):
    p_masks = {'Forefoot': [str(x) for x in range(55,83)]}
    maskdata = {}
    for mask in sorted(p_masks):
        maskdata[mask] = [p_masks[mask],dframe[p_masks[mask]].max(axis=1)]
        # dframe[mask] = dframe[p_masks[mask]].max(axis=1)
    return maskdata
def three_part_forefoot(dframe):
    p_masks = {'ThreePart_1':['55','56','62','63','69','70','76','77'],
                'ThreePart_2':['57','58','64','65','71','72','78','79'],
                'ThreePart_3':['59','60','61','66','67','68','73','74','75','80','81','82']}
    maskdata = {}
    for k,v in sorted(p_masks):
        print(k)
        maskdata[mask] = [p_masks[mask],dframe[p_masks[mask]].max(axis=1)]
        # dframe[mask] = dframe[p_masks[mask]].max(axis=1)
    return maskdata

def five_part_forefoot(dframe):
    p_masks = {'FivePart_1':['55','56','62','63','69','70','76','77'],
                'FivePart_2':['57','58','64','65','71','72','78','79'],
                'FivePart_3':['59','66','73','80'],
                'FivePart_4':['60','67','74','81'],
                'FivePart_5':['61','68','75','82']}
    maskdata = {}
    for key in sorted(p_masks):
        print(key)
        maskdata[mask] = {'mask':p_masks[mask],'value':dframe[p_masks[mask]].max(axis=1)}
        # dframe[mask] = dframe[p_masks[mask]].max(axis=1)
    return maskdata
    
def out_data(ofile,sheet_map):
    writer = pd.ExcelWriter(ofile)
    for sheet in sheet_map:
        sheet_map[sheet].to_excel(writer,sheet)
    writer.save()

# def roi_detect(grid):
    #https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array
    #http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_peak_local_max.html
    
################################################################################
##
##  Run Code
##
################################################################################
if __name__ == "__main__":
    mypath = 'C:/Users/hls376/Dropbox/UoS/Portals/[UoS] Price & Parker/Pressure_SPM/AMP_Pedar_Data/PreProcessed/To Analyse/Masking/'
    filelist = [os.path.join(r,file) for r,d,f in os.walk(mypath) for file in f]
    
    for fname in filelist:
        print(fname)
        sheet_map = pd.read_excel(fname, sheet_name=None)
        for sheet in sheet_map:
            df = sheet_map[sheet]
            whole_forefoot(df)
            three_part_forefoot(df)
            five_part_forefoot(df)
            sheet_map[sheet] = df
        ofile = mypath + os.path.splitext(os.path.basename(fname))[0] + '_Masked.xlsx'
        out_data(ofile,sheet_map)
        