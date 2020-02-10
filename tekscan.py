
import os, sys
from matplotlib import pyplot

def convert_tekscan(fname,f_tag):

	### read all lines:
	ifile = open(fname, 'r')
	ilines = ifile.readlines()
	ifile.close()

	### set constants:  ## Based on Tekscan insole system - may need adapted
	nCols = 21
	nRows = 60

	### find start of each frame:
	indFrames = [i for i,iline in enumerate(ilines) if (iline.startswith('Frame'))]
	II = []
	

	### Setup new file:
	outfile = f_tag + '.csv'
	f = open(outfile,'w')
	
	### Rebuild data in time series format:
	for i in range(len(indFrames)):
		# Cycle Frames
		I = ''
		for j in range(nRows):
			line = ilines[(indFrames[i]+j+1)]
			line = line.strip('\n')
			I = I + line
		I = I + '\n'	
		
		f.write(I)

	f.close()	


pyplot.close('all')

dir0   = 'C:/Users/hls376/Desktop/Tek/'    # Path to files
file_List   = os.listdir(dir0)                       ## Build file list
file_List.sort()

#Optional command line prompt
print('Processing ' + str(len(file_List)) + ' files:')  

### Start batch:
for i in range(0, len(file_List)):
	fname = os.path.join(dir0, file_List[i])
	(fileBaseName, fileExtension)=os.path.splitext(file_List[i])
	f_tab = os.path.join(dir0, fileBaseName)
	print('		......' + fileBaseName)
	p0     = convert_tekscan(fname, f_tab)





