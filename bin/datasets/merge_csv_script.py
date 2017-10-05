import glob
import pandas as pd

all_data = pd.DataFrame() #initializes DF which will hold aggregated csv files

#list of all df
dfs = []
for f in glob.glob("*.csv"): #for all csv files in pwd
    #add parameters to read_csv
    df = pd.read_csv(f, header=None) #create dataframe for reading current csv
    #print df
    dfs.append(df) #appends current csv to final DF
all_data = pd.concat(dfs, ignore_index=True)
print(all_data)
#       0      1      2
#0  test1  test1  test1
#1  test1  test1  test1
#2  test1  test1  test1
#3  test2  test2  test2
#4  test2  test2  test2
#5  test2  test2  test2
all_data.to_csv("merged.csv", index=None, header=None)
