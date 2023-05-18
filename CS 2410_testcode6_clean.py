#!/usr/bin/env python
# coding: utf-8

# # CS 2410-01 Group Project
# # Group 5 - Project Code
# **Group Members**
# - Andrew Benavides
# - Marc Cruz
# - Scott Chang
# 

# In[1]:


get_ipython().run_cell_magic('html', '', '<style>\n  table {margin-left: 0 !important;}\n</style>')


# In[2]:


# import math
import pandas as pd
import numpy as np
import seaborn as sns
# import plotly.express as px # for data
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# import matplotlib.transforms as transforms

dataset = pd.read_csv('DataTable_US_Cleaned_3.csv')

#Dictionary that holds all the possible type of occupations and each are assigned a number
job_type = {
            0  : 'All Occupations' ,
            11 : 'Management Occupations' ,
            13 : 'Business and Financial Operations Occupations' ,
            15 : 'Computer and Mathematical Occupations' ,
            17 : 'Architecture and Engineering Occupations' ,
            19 : 'Life, Physical, and Social Science Occupations' ,
            21 : 'Community and Social Service Occupations' ,
            23 : 'Legal Occupations' ,
            25 : 'Educational Instruction and Library Occupations' ,
            27 : 'Arts, Design, Entertainment, Sports, and Media Occupations' ,
            29 : 'Healthcare Practitioners and Technical Occupations' ,
            31 : 'Healthcare Support Occupations' ,
            33 : 'Protective Service Occupations' ,
            35 : 'Food Preparation and Serving Related Occupations' ,
            37 : 'Building and Grounds Cleaning and Maintenance Occupations' ,
            39 : 'Personal Care and Service Occupations' ,
            41 : 'Sales and Related Occupations' ,
            43 : 'Office and Administrative Support Occupations' ,
            45 : 'Farming, Fishing, and Forestry Occupations' ,
            47 : 'Construction and Extraction Occupations' ,
            49 : 'Installation, Maintenance, and Repair Occupations' ,
            51 : 'Production Occupations' ,
            53 : 'Transportation and Material Moving Occupations' 
            }

# categories/features represented in the data
cat      = {
            0 : 'Age',
            1 : 'Industry' ,
            2 : 'State' ,
            3 : 'Work City' ,
            4 : 'Gender',
            5 : 'Race' ,
            6 : 'Years of Professional Work Experience' ,
            7 : 'Years Professional Work Experience in Field' ,
            8 : 'Highest Education Level' ,
            9 : 'Job Title'
            }
        

numcities = dataset[['Work City']].drop_duplicates().size


# In[3]:


def normalized_pay(df, the_order, income_cat, SOC_group, cat_num, the_median, c1, c2, c3, c4, label_annot, ptile25, ptile75):
    '''normalized_pay function is normalizing income based on data, list of desired list of cities,
    given occupation, income of the desired category, and several other parameters based on the data
    
    args:
    df           the data that holds all the features.
    the_order    list of cities
    income_cat   the category of income (ie normalized or raw income) from which we are caluclating.
    SOC_group    the specific occupation from the dictionary 'job_type'
    cat_num      category number from the dictionary 'cat' which specifies the grouping category.
    the_median   median of the data
    c1           Check 1, Checking whether a city started ABOVE the raw median and moved ABOVE 
                 the normalized median.
    c2           Check 2, Checking whether a city started BELOW the raw median and moved ABOVE 
                 the normalized median.
    c3           Check 3, Checking whether a city started ABOVE the raw median and moved BELOW
                 the normalized median.
    c4           Check 4, Checking whether a city started BELOW the raw median and moved BELOW
    label_annot  the list of labels of the included data
    ptile25      25th percentile of the data. Resize the area of the graph
    ptile75      75th percentile of the data Resize the area of the graph
    
    returns:
    produces a side ways box plot showcasing purchasing power among cities for a specified occupation.
    '''
    
    income_norm = income_cat
    job_class_num = SOC_group
    cat_name = cat[cat_num]
    
    statecount = df

    median_norm = the_median

    # Find the order
    my_order = the_order 

    fig, ax = plt.subplots(figsize=(25, statecount[cat_name].nunique()*.3))
    
    # checking which color scheme to use for the boxplot bars.  We use the default for any category
    # except for cat_num = 2 (for States) or cat_num=3 (for Cities)
    if (cat_num == 2) or (cat_num == 3):
        the_colors = color_palette(my_order, c1, c2, c3, c4)
    else:
        the_colors = 'Pastel2'
    
    subdata_norm = sns.boxplot(data = statecount,
                               y = cat_name,
                               x = income_norm,
                               whis = 0, 
                               showfliers = False,
                               width = 0.8,
                               ax = ax,
                               order = my_order,
                               palette = the_colors)
    
    
    # Graph Formatting Settings    
    
    plt.xlim( ptile25 * .9 , ptile75 * 1.1 ) 
    subdata_norm.axvline(median_norm, color = 'red')

    # additional formatting of the labels and axis
    subdata_norm = label_formatter(subdata_norm, cat_num, c1, c2, c3, c4, label_annot)
  
    # Formatting title and labels at head of graph.
    plt.title(f'Cost-of-Living Adjusted (Real) Annual Salary\n{job_type[job_class_num]}\n', loc = 'center', fontsize = 24)
    plt.title(f'Adjusted Median Salary: ${round(median_norm):,}', loc = 'left', fontsize = 18, color = 'red')
    plt.title(f'Each bar shows: [ <-- (   25-75 %tile   ) --> ]', loc = 'right', fontsize = 18, color = 'red')
    
def raw_pay(df, the_order, income_cat, SOC_group, cat_num, the_median, c1, c2, c3, c4, label_annot, ptile25, ptile75):
    '''raw_pay function is using the raw values derived from the data, list of desired list of cities,
    given occupation, income of the desired category, and several other parameters based on the data
    
    args:
    df          the data that holds all the features.
    the_order   list of cities
    income_cat  the category of income (ie normalized or raw income) from which we are caluclating.
    SOC_group   the specific occupation from the dictionary 'job_type'
    cat_num     category number from the dictionary 'cat' which specifies the grouping category.
    the_median  median of the data
    c1          Check 1, Checking whether a city started ABOVE the raw median and moved ABOVE 
                the normalized median.
    c2          Check 2, Checking whether a city started BELOW the raw median and moved ABOVE 
                the normalized median.
    c3          Check 3, Checking whether a city started ABOVE the raw median and moved BELOW
                the normalized median.
    c4          Check 4, Checking whether a city started BELOW the raw median and moved BELOW 
                        the normalized median.
    label_annot the list of labels of the included data
    ptile25     25th percentile of the data. Resize the area of the graph
    ptile75     75th percentile of the data Resize the area of the graph
    
    returns:
    produces a side ways box plot showcasing purchasing power among cities for a specified occupation.
    '''
    
    income_raw = income_cat  # income_cat is the category of income (ie normalized or raw income) from which we are caluclating.  
    job_class_num = SOC_group
    cat_name = cat[cat_num]

    
    statecount = df
    
    median_raw = the_median
    
    # Find the order
    my_order = the_order #statecount.groupby(by=[cat_name])[income_raw].median().sort_values(ascending=False).index
    
    fig, ax = plt.subplots(figsize=(25, statecount[cat_name].nunique()*.3))
    
    # checking which color scheme to use for the boxplot bars.  We use the default for any category
    # except for cat_num = 2 (for States) or cat_num=3 (for Cities)
    if (cat_num == 2) or (cat_num == 3):
        the_colors = color_palette(my_order, c1, c2, c3, c4)
    else:
        the_colors = 'Pastel2'
        
    subdata_raw = sns.boxplot(data = statecount,
                                y = cat_name,
                                x = income_raw,
                                whis = 0,
                                showfliers = False,
                                width = .8,
                                ax = ax,
                                order = my_order,
                                palette = the_colors)
                              
    # Graph Formatting Settings
    # Here we use income_norm so that both the norm and raw graphs are of the same scale
    
    plt.xlim( ptile25 * .9 , ptile75 * 1.1) 
    subdata_raw.axvline(median_raw, color='red')

    # additional formatting of the labels and axi
    subdata_raw = label_formatter(subdata_raw, cat_num, c1, c2, c3, c4, label_annot)
    
    
    # Formatting title and labels at head of graph.
    plt.title(f'Non-Adjusted (Nominal) Annual Salary\n{job_type[job_class_num]}\n', loc='center', fontsize = 24)
    plt.title(f'Non-Adjusted Median Salary: ${round(median_raw):,}', loc='left', fontsize = 18, color = 'red')
    plt.title(f'Each bar shows: [ <-- (   25-75 %tile   ) --> ]', loc='right', fontsize = 18, color = 'red')

    
def true_pay(SOC_group, cat_num = 3 , min_observe_count = 9, terms=''):
    '''true_pay function is used to sort the needed arguments/parameters for the above functions (normalized pay & raw 
    pay) and is what the user would be mainly using this function to output the side ways box plots.
    
    args:
    SOC_group           the specific occupation from the dictionary 'job_type'
    cat_num             category number from the dictionary 'cat' which specifies the grouping category.
    min_observe_count   minimum number of observations needed before plotting the sideway box plots.
    terms               specific occupations (need to be precise based on job_type set list)
    
    returns:
    two box plots for normalized pay and raw pay. Each highlighting based on the raw data and then normalized data.
    '''
    income_norm = 'Adjusted Annual'
    income_raw = 'Annual Salary'
    job_class_num = SOC_group
    cat_name = cat[cat_num]
    min_count = min_observe_count
    
    queryslug = '==' if SOC_group != 0 else '!='
    
    # setting up the dataframe for only those categories that we are interested from the raw dataset.
    statecount = dataset[['Job Class', 
                          income_norm, 
                          income_raw, 
                          cat[0],
                          cat[1], 
                          cat[2], 
                          cat[3], 
                          cat[4], 
                          cat[5], 
                          cat[6], 
                          cat[7], 
                          cat[8], 
                          cat[9]
                         ]].query(f'`Job Class` {queryslug} {job_class_num}')
    
    # Filter out those records that do not have a minimum number of entries of the category selected.
    # It is likely that too few entries would produce results that are not representative of actual trends.
    
    query_list = f" `Job Title`.str.contains('')"     # Passing an emtpy query term if there are no keywords to search
                                                      # Otherwise, we will build a query search string
    if terms !='':
        word_list = terms.split()
        query_list = ''
        for each in range(len(word_list)):
            query_list = query_list + f" `Job Title`.str.contains('{word_list[each]}' , case = False) "
            if (len(word_list) > 1) and (each < len(word_list)-1) : query_list = query_list + ' and '

    # filter out those records that have Annual Salary less that Fed poverty rate ($12880 in 2021).
    # Allow query of records if there are keywords to search.  
    # Finally, filter out records that do not have a minimum number of records (must be greater than number passed as argument)
    statecount = statecount.loc[statecount['Annual Salary'] > 12880].query(query_list, engine='python').groupby(cat_name).filter(lambda x: x[cat_name].count()>min_count)
    
    print('Total records matching query:', len(statecount) )
    
    if len(statecount) > 0:
        pass      
    else:
        print('No records found that met criteria.  Exiting...')
        return None
    
    # Initial data processing.
    median_norm = statecount[income_norm].median()
    median_raw = statecount[income_raw].median()

    statecount['Job Class'] = statecount['Job Class'].astype('int')    # change the data type of column 'Job Class' to int type for ease of filtering.

    # Find the order.  We are interested in determining which city falls below median in the raw income list
    # but is above the median in the normalized list.
    # We start by creating a df that contains the list of city names and whether they are above or below their respective
    # medians (True/False)
    order_list_norm = statecount.groupby(by=[cat_name])[income_norm].median().sort_values(ascending=False)
    order_list_raw = statecount.groupby(by=[cat_name])[income_raw].median().sort_values(ascending=False)
    
    # We find the 25% and 75% quantiles to pass on as arguments to use for setting the plot boundaries.
    # We are searching through both norm and raw values to find ideal horizontal scale for both graphs.
    ptile25 = min( min(statecount.groupby(by=[cat_name])[income_norm].quantile(.25)) ,  min(statecount.groupby(by=[cat_name])[income_raw].quantile(.25)) )
    ptile75 = max( max(statecount.groupby(by=[cat_name])[income_norm].quantile(.75)) , max(statecount.groupby(by=[cat_name])[income_raw].quantile(.75)) )

    med_norm_city_above = order_list_norm >= median_norm     # True/False whether normalized city is above normalized median
    med_raw_city_above = order_list_raw >= median_raw     # True/False whether raw city is below raw median
    
# calculating distance to median before adjustment and distance to median after adjustment.  We are
# calculating the relative percentage change of distance to median to gauge relative change in purchasing power.

    # We create a df from which to performce distance calculations from median of before vs after CoL adjustments.
    # We set them to new dataframes.
    med_dist_norm = order_list_norm[:]
    med_dist_raw = order_list_raw[:]

    # We are caonverting the distance to a percentage.
    med_dist_ptage_norm = (med_dist_norm - median_norm)/median_norm
    med_dist_ptage_raw = (med_dist_raw - median_raw)/median_raw

    # To make sure they are in the same order, we reindex the entries from the unadjusted medians to the same order of the CoL adjusted medians.
    med_dist_ptage_raw = med_dist_ptage_raw.reindex(med_dist_ptage_norm.index)

    # We now combine the calculated percentage movement with the to the same order of the CoL adjusted medians.  Later we will be searching through this
    # df to annotate the labels on the graph.
    # med_dist_delta = pd.DataFrame()
    # med_dist_delta = [] 
    med_dist_delta = list( f'{round( (med_dist_ptage_norm[row] - med_dist_ptage_raw[row]) *100 , 2 ): 06,.2f} %' for row in range(len(med_dist_ptage_norm)) ) 

    #This section of code is the Check1, Check2, Check3, Check4 content
    # Checking whether a city started above/below the raw median and moved above/below the normalized median.
    # These df lists will be passed to the functions to help format the axis and labels of the graphs
    city_hi2hi = (med_raw_city_above * med_norm_city_above) == True
    
    city_hi2hi = city_hi2hi.loc[city_hi2hi==True].index

    city_low2hi = (med_raw_city_above == False) * (med_norm_city_above == True) == True    
    city_low2hi = city_low2hi.loc[city_low2hi==True].index

    city_hi2low = (med_raw_city_above == True) * (med_norm_city_above == False) == True
    city_hi2low = city_hi2low.loc[city_hi2low==True].index

    city_low2low = (med_raw_city_above == False) * (med_norm_city_above == False) == True
    city_low2low = city_low2low.loc[city_low2low==True].index

    # highlight_check = highlight_check.loc[highlight_check==True].index             # and moved to above norm median after adjusting for CoL

    # Find the order.  We are interested in graphing results in descending order.
    # We need only the list of names, and can retrieve it from order_list_norm, which contains both city name and median value.
    # For the_order, we only need the name of the city.
    the_order = order_list_norm.index
    # print(highlight_check)

    print( 'Processing...' )

    # show outliers
    
    
    normalized_pay(statecount, the_order, income_norm, SOC_group, cat_num, median_norm, 
                   city_hi2hi, city_low2hi, city_hi2low, city_low2low, med_dist_delta, ptile25, ptile75)

    raw_pay(statecount, the_order, income_raw, SOC_group, cat_num, median_raw, 
            city_hi2hi, city_low2hi, city_hi2low, city_low2low, med_dist_delta, ptile25, ptile75)


# In[4]:


def color_palette(label_order, check_1, check_2, check_3, check_4):
    '''setting up the color palette for the boxplots.  Setting up the color so they are visually similar to 
    the colors on the label.  The exact colors used for the labels cannot be used here.  The color is either too dark/deep
    or it is too intense that the median line is difficult to see.  Similar colors are chosen, and it is visually easier to line up the label with the 
    representative bar.  
    
    The arguments passed in this function are similar to those for the label_formatter function.  The
    color order of the bars is the same order as the for the labels.  
    
    color code:
    #00bfff (similar to BLUE) means it started above in unadjusted, and remained above in adjusted.
    LIGHT GREEN (similar to DARK GREEN) means it started below in unadjusted, and ended above in adjusted.
    LIGHT CORAL (similar to DARK RED) means it started above in unadjusted, and ended below in adjusted.
    SILVER (similar to BLACK) means it started below in unadjusted, and remained below in adjusted.
    
    args:
    label_order would be the order of the labels
    check_1     Similar to c1 in the previous functions, Checking whether a city started ABOVE the raw median 
                and moved ABOVE the normalized median.
    check_2     Similar to c2 in the previous functions, Checking whether a city started LOW the raw median 
                and moved ABOVE the normalized median.
    check_3     Similar to c3 in the previous functions, Checking whether a city started ABOVE the raw median 
                and moved BELOW the normalized median.
    check_4     Similar to c4 in the previous functions, Checking whether a city started BELOW the raw median 
                and moved BELOW the normalized median.
    returns:
    list of the order of colors of bars in variable 'bar_colors'
    '''
    
    length = len(label_order)
   
    # initialize list variable to hold the colors.  This will be used to assign a matching color to the bar to the label.
    bar_colors = [None] * length

    # Check to see if a state or city_name name appears, if so highlight.
    # We are highlighting cities if they are above/below median in raw income and where they move above/below median in normalized income.
    # We then set the bar color the same as the label color.
    for i in range(length):
        for a in range(len(check_1)):
            if check_1[a] == label_order[i] : bar_colors[i] = '#00bfff'   #00bfff #70bbff
        for b in range(len(check_2)):
            if check_2[b] == label_order[i]: bar_colors[i] = 'lightgreen'   #90e77d #cfe6ca
        for c in range(len(check_3)):
            if check_3[c] == label_order[i]: bar_colors[i] = 'lightcoral'    #f1bcb8
        for d in range(len(check_4)):
            if check_4[d] == label_order[i]: bar_colors[i] = 'silver'    #f2f2f2

    return bar_colors


    
def label_formatter(plotdata, cat_num, check_1, check_2, check_3, check_4, label_annot):
    '''label_formatter function is to help format and color code the labels if the category 
    selected is state of city_name. When this condition is true, it is looking to see 
    if CA is in the label, and if so, mark it with asterisks to make it stand out.
    Similarly, if there is a CA city listed in the label, it will also mark it with asterisks 
    to make it stand out.
    
    Further, the labels are color coded.  It is checking for whether the state or the city listed started with a median value above/below 
    the aggregate median of all data in that category, and comparing with whether it is above/below the median value of the aggregate median of
    the CoL adjusted data.  
    
    color code:
    BLUE means it started above in unadjusted, and remained above in adjusted.
    DARK GREEN means it started below in unadjusted, and ended above in adjusted.
    DARK RED means it started above in unadjusted, and ended below in adjusted.
    BLACK means it started below in unadjusted, and remained below in adjusted.
    
    args:
    plotdata    would be the income data
    cat_num     would be the category number specified by the user
    check_1     Similar to c1 in the previous functions, Checking whether a city started ABOVE the raw median 
                and moved ABOVE the normalized median.
    check_2     Similar to c2 in the previous functions, Checking whether a city started LOW the raw median 
                and moved ABOVE the normalized median.
    check_3     Similar to c3 in the previous functions, Checking whether a city started ABOVE the raw median 
                and moved BELOW the normalized median.
    check_4     Similar to c4 in the previous functions, Checking whether a city started BELOW the raw median 
                and moved BELOW the normalized median.
    label_annot desired annotations for the boxplots
    
    returns:
    x-axis values to be used to plot the sideways box plots.
    '''
    
    ax = plotdata
    
    ax.set_xlabel('Income Level', fontsize=20)
    ax.set_ylabel(cat[cat_num], fontsize=30)

    ax.tick_params(axis='x', labelsize=14, rotation=45)
    ax.tick_params(axis='y', labelsize=14)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '$' + '{:,.0f}'.format(x)))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10000))
    
    if (cat_num == 2) or (cat_num == 3):
        
        # initialize list variable to hold the colors.  This will be used to assign a matching color to the bar to the label.
        bar_colors = []
        
        # Check to see if a state or city_name name appears, if so highlight.
        # We are highlighting cities if they are above/below median in raw income and where they move above/below median in normalized income.
        # We then set the bar color the same as the label color.
        for i in range(len(ax.get_yticklabels())):
            for a in range(len(check_1)):
                if f"'{check_1[a]}'" in str(ax.get_yticklabels()[i]): ax.get_yticklabels()[i].set_color('blue')
            for b in range(len(check_2)):
                if f"'{check_2[b]}'" in str(ax.get_yticklabels()[i]): ax.get_yticklabels()[i].set_color('darkgreen')
            for c in range(len(check_3)):
                if f"'{check_3[c]}'" in str(ax.get_yticklabels()[i]): ax.get_yticklabels()[i].set_color('darkred')
            for d in range(len(check_4)):
                if f"'{check_4[d]}'" in str(ax.get_yticklabels()[i]): ax.get_yticklabels()[i].set_color('black')
        
        legend_label = ['Blue: Stayed Above Median', 'Green: From Below to Above Median', 'Red: From Above to Below Median', 'Black: Stayed Below Median']
        textstr = '\n'.join((
                        # '   Bars Represents   ',
                        # '| <- 25-50 %tile -> |',
                        # '---------------------',
                        # '\n',
                         
                        'State/City_Name Color Code',
                        'ex: [State/City] | 50% ',
                        '----------------------------------',
                        'Blue: From Above, Stayed Above Median',
                        'Green: From Below to Above Median',
                        'Red: From Above to Below Median',
                        'Black: From Below, Stayed Below Median',
                        '\n',
                        '% indicates movement of',
                        'relative income from',
                        'initial unadjusted value',
                        'to after CoL value of income' ))
            
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(.79, .9, textstr, ha = 'left', transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
        
        # adding additional annotations
        labels = [item.get_text() for item in ax.get_yticklabels()] 
    
        # add relative percentage movement of median from before adjustment to after CoL adjustment
        for i in range(len(ax.get_yticklabels())):
            labels[i] = labels[i] + " | " + label_annot[i]
    
        # Highlight label text if CA or CA city is an entry. If true, set font adjustments.    
        if cat_num == 2:  # category number 2 is looking names of States

            state_names = []
            state_names = dataset[cat[2]].unique()
            # labels = [item.get_text() for item in ax.get_yticklabels()]      

            # Checking for CA in the labels.  If found, mark with *** to indicate.
            for i in range(len(ax.get_yticklabels())):
                if 'CA' in str(ax.get_yticklabels()[i]):  labels[i] = '*** ' + labels[i]           

        if cat_num == 3:    # category number of 3 is looking for names of cities. 
            # Compile list of CA cities
            CA_cities = []
            CA_cities = dataset.loc[dataset[ str(cat[2]) ].isin(['CA'])][cat[3]].unique()

            # Storing the labels in case we need to format or change properties of the label when shown on graph.
            # labels = [item.get_text() for item in ax.get_yticklabels()]

            for i in range(len(ax.get_yticklabels())):     # Check to see if CA city name appears, if so highlight.
                for j in range(len(CA_cities)):
                    if CA_cities[j] in str(ax.get_yticklabels()[i]): 
                        labels[i] = '*** ' + labels[i]  
    
        #re-inserting the labels with the changes
        ax.set_yticklabels(labels)                                      
    
    return ax


# 
# The job classification categories
# |SOC Code|Description|
# |---|---|
# | 0|All Occupations|
# |11|Management Occupations|
# |13|Business and Financial Operations Occupations|
# |15|Computer and Mathematical Occupations|
# |17|Architecture and Engineering Occupations|
# |19|Life, Physical, and Social Science Occupations|
# |21|Community and Social Service Occupations|
# |23|Legal Occupations|
# |25|Educational Instruction and Library Occupations|
# |27|Arts, Design, Entertainment, Sports, and Media Occupations|
# |29|Healthcare Practitioners and Technical Occupations|
# |31|Healthcare Support Occupations|
# |33|Protective Service Occupations|
# |35|Food Preparation and Serving Related Occupations|
# |37|Building and Grounds Cleaning and Maintenance Occupations|
# |39|Personal Care and Service Occupations|
# |41|Sales and Related Occupations|
# |43|Office and Administrative Support Occupations|
# |45|Farming, Fishing, and Forestry Occupations|
# |47|Construction and Extraction Occupations|
# |49|Installation, Maintenance, and Repair Occupations|
# |51|Production Occupations|
# |53|Transportation and Material Moving Occupations|
# 
# 
#     
# The grouping categories:
# |Num|Grouping Category|
# |---|---|
# |0|Age|
# |1|Industry|
# |2|State|
# |3|Work City|
# |4|Gender|
# |5|Race|
# |6|Years of Professional Work Experience|
# |7|Years Professional Work Experience in Field|
# |8|Highest Education Level|
# |9|Job Title|
# 
#     
#     
# ### Fuction has following format:
# **true_pay( $\color{red}{\text{job classification}}$, $\color{orange}{\text{grouping category}}$, $\color{blue}{\text{minimum # of observations needed before plotting}}$, optional: search terms)**
# 
# **Note:**  we can only plot and view data for those records where there was Cost of Living (CoL) information available.  Overall there were 22,881 records, but CoL info only applied to 17,025.  
# - Since the survey data was from 2021, the CoL info I took was only from 2021 from the database I found.  
# 
# - By default, the minumum number of observations needed is set to 10 or more records.  Otherwise, I didn't think there was enough data.  Some of them might only have 1 record.

# In[12]:


#def true_pay(SOC_group, cat_num = 3 , min_observe_count = 9, optional: keyword search (in quotes) ):

true_pay(25, 3, 1, 'prof' )

