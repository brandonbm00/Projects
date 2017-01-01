# Elastic barplot

import numpy as np
import matplotlib.pyplot as plt


def elastic_barplot(barheights, bardevs, barlabs, width, title, xlab, ylab):
    ''' Elastic Barplot ***
        barheights: list of numpy arrays representing N bar heights
        bardevs: list of numpy arrays representing N bar standard deviations 
        barlabs: list of numpy arrays representing N bar labels in order
    '''
 
    try:
        assert len(barheights) == len(bardevs)
    except AssertionError: 
        raise ValueError('Barheights and Bardevs not the same length') 
   
    N = len(barheights)
    ngroups = len(barheights[0]) 

    fig, ax = plt.subplots()
    ind = np.arange(ngroups)
    xlim_r = ind[-1]+width*N 
  

    # Currently only supports N=3 because I haven't figured out the color pallete 
    colorbank = ['#7ABA7A',
                 '#B76EB8',
                 '#028482']
    
    # Main loop 
    fig, ax = plt.subplots()
    plotdict = {}

    for i in range(N):
        plotdict[i] = ax.bar(ind+width*i,
                             barheights[i], 
                             width,
                             color=colorbank[i],
                             hatch='\\')
    
    ylim_r = max(map(max, zip(*barheights)))  


    # Space the tick labels correctly
    offset = width*(N/2.0) # distance between side and center of bar group    

    xticks_loc = np.linspace(offset, xlim_r-offset, ngroups)

    plt.xlim(0, xlim_r)
    plt.ylim(0, ylim_r)

    

    #ax.set_xticks(xticks_loc)
    plt.xticks(xticks_loc, barlabs, rotation=45, fontname='Times New Roman', fontsize=24)
    plt.yticks(fontname='Times New Roman', fontsize=24)
    plt.gca().yaxis.grid(True)
    plt.title(title, fontname='Times New Roman', fontsize=28)
    plt.xlabel(xlab, fontname='Times New Roman', fontsize=24)
    plt.ylabel(ylab, fontname='Times New Roman', fontsize=24)
    plt.tight_layout()


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')


data1 = np.array([1,2,3,4,5,6,9])
data2 = np.array([9,3,5,7,18,24,4])
data3 = np.array([9,10,11,12,13,4,14])

labl = ['one','two','three','four','five','six','seven']

elastic_barplot([data2, data1, data3],
                [1,2,3], 
                labl,
                0.2,
                'Example Elastic Barplot',
                'xlabel', 
                'ylabel')
plt.show()
