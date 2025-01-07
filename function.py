"""
-------------------------------------------------------------------------------
                                function.py
-------------------------------------------------------------------------------
"""


import numpy as np

def count_unique(columns_name, columns_value):
    '''This function find the numbers of every unique value in all the columns of the datasets.
        Count the unique value and if the numbers '0' is > than 50000. Print a warning message.
         The columns with 0 > 50000 are 18.
         Next this columns are removed from the dataset and the KNN, and other model are calculated.'''
    over5 = 0
    list = []
    position = []
    for i, columns in enumerate(columns_name):
        print(f'\nValue of the columns:          {columns}: ')
        a, counts = np.unique(columns_value[:, i], return_counts=True)
        print(a,counts)

        for j, values in enumerate(a):
            if values == 0:
                zeros = counts[j]
                print(zeros)
                if zeros > 50000:
                    print("ATTENZIONE QUESTA COLONNA POTREBBE ESSERE ELIMINATA PER LA PCA")
                    over5 = over5 + 1
                    list.append(columns_name[i])
                    position.append(i)
    print(f"\nNumbers of columns with over 50000 zeros: {over5}")
    print(f"Position of the columns with more than 5000 zeros: {position}")
    return list, position
