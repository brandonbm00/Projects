import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import seaborn as sns
import pdb



def get_header(path):
	with open(path) as infile:
		contents = infile.read()
	return contents.strip('\r\n').split(',') 

'''
def read_data(hdr, pathstr):
	all_transaction_files = glob.glob(pathstr)
	header = get_header(hdr)
	all_transactions = pd.read_csv(all_transaction_files[0], header=None, usecols=[0,1,2,3,4,5,6], delimiter=',')
	all_transactions.columns = header
	return all_transactions
'''

def read_data(hdr, datafile):
	header = get_header(hdr)
	try:
		data = pd.read_csv(datafile, header=None, usecols=[0,1,2,3,4,5,6], delimiter=',')	
		data.columns = header
	except pd.io.common.EmptyDataError:
		print('No columns found for user, creating empty dataframe')
		data = pd.DataFrame(columns=header)	
	return data


##### Main Routine


all_trans = read_data('transactionheader.csv', 'transactiondata.csv')
all_trans['Posting Date'] = pd.to_datetime(all_trans['Posting Date'])

ATM = all_trans.where(all_trans['Description'].str.contains('ATM'))
ATM = ATM.dropna(how='all')

userdat = read_data('userheader.csv', 'userdata.csv')


desc_idx = all_trans.columns.get_loc('Description')


for row in all_trans.iterrows():
	if (userdat == row[1][desc_idx]).values.any():	
		continue
	else:
		category = raw_input('Enter category for %s: \n' % row[1][desc_idx])


#	for column in userdat:
#		if row[1][desc_idx] in userdat[column]: 
#			continue
#		else:
#			category = raw_input('Enter category for %s: \n' % row[1][desc_idx])



































































































						  
	



fig, ax = plt.subplots(2,2)
ax[0][0].plot(timeax, all_trans['Balance'])

plt.setp(ax[0][0].xaxis.get_majorticklabels(), rotation=45)
plt.tight_layout()
plt.show()
