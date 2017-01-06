
def elastic_barplot_ptn(barheights, ratios, labels, width, xfsize):


    cbank = ['#759564', '#CDDB9D', '#F4F7EC', '#EFD9DC', '#DAB4C9']
    
    nbars = len(barheights)
    nratios = len(ratios)
   
    bar_idx = np.arange(nbars)
    
    try: 
        assert len(barheights) == len(ratios[0])
    except AssertionError:
        raise ValueError('Heights and ratios must be same length')


    # Main loop
    
    plotdict = {}
    bottom = 0
 
    fig, ax = plt.subplots()
    for ptn in range(nratios): # Loop over ratios
        heights = np.array(ratios[ptn]) * barheights
        
        plotdict[ptn] = plt.bar(bar_idx,
                                heights, 
                                width,
                                color=cbank[ptn], 
                                bottom=bottom)
        bottom += heights
    
    xlim_r = bar_idx[-1]+width
    ylim_r = max(bottom) 

    plt.xlim(0, xlim_r)
    plt.ylim(0, ylim_r)

    offset = width / 2.0
    xticks_loc = np.linspace(offset, xlim_r-offset, nbars)

    plt.xticks(xticks_loc, labels, rotation=60, fontname='Times New Roman', fontsize=xfsize)    

    return fig, ax


if __name__ == "__main__" :

    '''

    heights = [10,11,12,13,14]
    ratios = [[0.0, 0.25, 0.5, 0.75, 1.0],
              [1.0, 0.75, 0.5, 0.25, 0.0],
              [0.0, 0.33, 0.66,0.33, 0.0],
              [1.0, 0.75, 0.5, 0.25, 0.0],
              [0.0, 0.25, 0.5, 0.75, 1.0]]
    fig, ax = elastic_barplot_ptn(heights, ratios, 0.25) 
    plt.show()

    '''
    
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    cars = {0:'AVENTADOR', 1:'VEYRON', 2:'LAFERARRI', 3:'ZONDA', 4:'P1'} 
    years = {0:1991, 1:1994, 2:1955, 3:1987, 4:1950}


    samples = 100

    random_cars = [cars[rand] for rand in np.random.randint(0, 5, samples)]
    random_years = [years[rand] for rand in np.random.randint(0, 5, samples)]
    random_counts = np.random.randint(0,10000, samples)

 
    data_dict = {'cars':random_cars, 'year':random_years, 'number':random_counts}
    
    data_df = pd.DataFrame.from_dict(data_dict)

    print('First 10 rows of sample dataframe')
    print(data_df.head(10))

    # loop through unique values of cars

    unique_cars = list(np.unique(data_df['cars'].tolist()))

    print('\n')
    print('Unique cars: \n')
    print(unique_cars)
    print('\n')    
  
    partitions = [(1900, 1960), (1980, 1989), (1990, 2016)]  
    ratios = [[] for i in range(len(partitions))]

    rollup = data_df[['cars','number']].groupby('cars').sum().reset_index().sort('number')
    print(rollup.head(5))
    heights = {car:val for car, val in zip(rollup['cars'].tolist(), rollup['number'].tolist())}
    print(heights)

    total_heights = np.zeros(len(unique_cars))
 
 

    for car in unique_cars:
        subset = data_df.where(data_df['cars'] == car)
        for ptn in partitions: 
            between = subset[(ptn[0] < subset['year']) & (subset['year'] < ptn[1])].sum()
            ratio = between['number'] / np.sum(subset['number']) 
            ratios[partitions.index(ptn)].append(ratio)               
        
        total_heights[unique_cars.index(car)] = heights[car]


    print(ratios)
    fig, ax = elastic_barplot_ptn(total_heights, ratios, unique_cars, 0.5, 24) 
    plt.tight_layout()
    plt.show()    
  
        


