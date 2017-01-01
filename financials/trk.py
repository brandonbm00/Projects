import pandas as pd
from difflib import SequenceMatcher
import pprint

pp = pprint.PrettyPrinter()
lowtol = 0.8

def dict_from_row(header, row):
    idx, data = row
    return {k:v for k,v in zip(header, data.tolist())} 

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def cat_from_input(dic):
    cats = ['rec','tnspt','food','weed','work','tmpl','other']
    cat = None
    while cat not in cats:
        print('Found unknown transaction *** \n')
        pp.pprint(dic)
        cat = raw_input('Enter category for from possible names rec, tnspt, food, weed, work, tmpl, other : \n')
        
    return cat 
        

def get_category(dic, hist):
    ''' check if new_desc is in hist, prompt user if not '''

    category = None
    new_desc = dic['desc'] # The new description
     
    for old_desc in hist['desc']:
        if similar(new_desc, old_desc) == 1.0:
            category = 'classified' # Already classified this transaction
            return category    
        elif similar(new_desc, old_desc) < 1.0 and similar(new_desc, old_desc) > lowtol:
            print('Similar match found for %s of %s' % (new_desc, old_desc))
            category = hist['class'][hist['desc'].tolist().index(old_desc)]
            return category
    if category is None:
        category = cat_from_input(dic)

    return category
    

new_header  = pd.read_csv('transactionheader.csv', sep=',')
hist_header = pd.read_csv('historicalheader.csv', sep=',')

new = pd.read_csv('transactiondata.csv', names=new_header, sep=',', quotechar='"')

try:
    hist = pd.read_csv('historicaldata.csv', names=hist_header, sep=',', quotechar='"')
except IOError:
    hist = hist_header

new.dropna(how='all', axis=1)

for row in new.iterrows():
    dic = dict_from_row(new.columns, row)
    category = get_category(dic, hist)
      
    if category == 'classified':
        print('Found preclassified transaction %s' % dic['desc'])
        continue

    dic['class'] = category
    write = pd.DataFrame(dic, index=[0])
    write.to_csv('historicaldata.csv', mode='a', header=False, index=False)
            






