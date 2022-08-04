
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from tkinter import filedialog
from tkinter import *


#### imports from LIam
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backend_bases as backend
import numpy as np
import statistics
from scipy import stats 
import os

datafile_folder_path = "/Users/bobmauck/devel/LHSP_MOM_GUI/main/Data_Files/"
datafiles = os.listdir(datafile_folder_path)
print("Datafiles in {}: {}".format(datafile_folder_path, datafiles))

# create the root window
root = tk.Tk()
root.title('Work with MOM datafile')
root.geometry('1000x1000')

#default for my computer - do an ASK to set default?
myDir = "/Users/bobmauck/Dropbox/BIG Science/MOMs/2022_Stuff"

my_font1 = ('courier', 10)

l1 = tk.Label(root,text='Read File & create DataFrame',
    width=60,font=my_font1)  

l1.grid(row=1,column=1)

### put data under this?
t1=tk.Text(root,width=40,height=50)
t1.grid(row=6,column=1,padx=5, pady = 100)

t2=tk.Text(root,width=40,height=50)
t2.grid(row=6,column=0,padx=5, pady = 100)

t3=tk.Text(root,width=40,height=50)
t3.grid(row=6,column=2,padx=5, pady = 100)

#### button for browsing files, but doing nothing
b1 = tk.Button(root, text='Browse Files', 
   width=20,command = lambda:upload_file("all"))
b1.grid(row=2,column=1) 

#### default values for entry boxes
my_entry_labels = ["Starting_Point", "End_Point", "Threshold"]
#### where to store entered values
my_entries = []

for x in range(3):
    my_entry = Entry(root)
    my_entry.grid(row = 4, column = x, pady = 20, padx = 20)
    my_entry.insert(0, "Enter_" + my_entry_labels[x])
    my_entries.append(my_entry)

def myClick():
    my_start = my_entries[0].get()
    my_end = my_entries[1].get()

    hello = "Cut from "+str(my_start) + " to " + str(my_end)

    my_Cut = upload_file("cut")
    my_Save_Dir = "~/devel/LHSP_MOM_GUI/main/Data_Files/"
    my_Cut.to_csv(my_Save_Dir + hello + ".TXT", headers=None, index=None)

    t2.insert(tk.END, hello + "\n") # add to Text widget






myButton = Button(root, text = "Cut File", command = myClick)
myButton.grid(row = 2, column = 0) #, pady = 20)







# #################
# # deteremine eligible values use din do_Mean...
# #####
def in_range_finder(a, myMin, myMax, myThreshold):
    if ((a > myMin) & (a < myMax) & (a < myThreshold)):
        return 1
    else:
        return 0


# #################
# # Do calcultions of the bird detected - assumes df contains only that section of the file - 
# #                                       must have cut it first to right length
# #####
def do_Mean_Bird_Calcs(multiple_files=False,f_name=None):
    ## open a cut file
    f_types = [
        ('CSV files',"*.csv"),
        ('TXT',"*.txt")
        ]

    ## for testing (False), I just use this file:
    if multiple_files:
        pass
        # f_name = f_name # fd.askopenfilename(initialdir = myDir, title = "Choose MOM File", filetypes = f_types)
    else:
        f_name = "/Users/bobmauck/devel/LHSP_MOM_GUI/main/Data_Files/185_One_Bird_70K_110K.TXT"
    

    
    l1.config(text=f_name) # display the path 
    my_df = pd.read_csv(f_name, header=None, names=["myIndex","Measure", "Datetime"])
    print(my_df["Measure"].describe())

    # into t3...
    displayname = f_name[len(myDir):len(f_name)] + "\n"
    my_Mean = round(my_df["Measure"].mean(),1)
    str1 = "Mean: \t" + str(my_Mean)+ "\n"
    t3.insert(tk.END, displayname + "\n")
    t3.insert(tk.END, "\t" + str1 + "\n") # add to Text widget
    # t3.insert(tk.END, str2) # add to Text widget
    # t3.insert(tk.END, str3) # add to Text widget
    
    # print("Min: \t" + str(my_df["Measure"].min())+ "\n")
    # print("Max: \t" + str(my_df["Measure"].max())+ "\n")

    # my_Bird_Frame = my_df(df.Measure < my_Threshold)
    # my_Bird_Frame.plot()
    # to_show = "calcs"
    # my_df.plot(x = "myIndex", y = "Measure")
    # plt.show()

    ### new
    # my_Threshold = 550000
    my_df = my_df[ 'Measure'].to_frame()
    # my_df['roll_mean'] = my_df['Measure'].rolling(5).mean()
    my_df['roll_std5'] = my_df['Measure'].rolling(5).std() 
    # my_df['roll_std10'] = my_df['Measure'].rolling(10).std()
    my_df['roll_mean5'] = my_df['Measure'].rolling(5).mean()
    # my_df['roll_mean10'] = my_df['Measure'].rolling(10).mean()
    # my_df['roll_meanDiff'] = my_df['roll_mean10'] - my_df['roll_mean5']

    my_df['pct_chg'] = my_df['roll_mean5'].pct_change()
    # my_df['pct_pct_chg'] = my_df['pct_chg'].pct_change() ### this doesn't do us any good at all

    print(my_df['pct_chg'].describe())
 

    my_df['pct_chg_abs'] = my_df['pct_chg'].abs()

    print(my_df['pct_chg_abs'].describe())

    print(my_df['pct_chg_abs'].quantile(.9))

    # my_df2 = my_df['pct_chg'].to_frame()
    my_df['pct_chg_abs'] = my_df['pct_chg'].abs()
    

    # my_df2 = my_df['pct_chg'].to_frame()

    
    ### make first derivative approx
    # my_df['diff_mean5'] = 0
    # for i in my_df['roll_mean5']
    #     my_df['diff_mean5']
    # ### new
    # print("Rolling mean MEAN: ")
    # print(str(my_df["roll_std"].mean()))

    ### for fun, make a numpy array
    my_numpy_array = my_df["pct_chg"].to_numpy()
    nmpy_median = np.nanmedian(my_numpy_array)
    
    nmpy_mean = np.nanmean(my_numpy_array)

    hi_point = np.nanargmax(my_numpy_array)
    # loc_hi_point = 

   # hi_point = max(my_numpy_array)

    lo_point = np.nanargmin(my_numpy_array)
    lo_value = my_df["pct_chg"].iloc[lo_point]
    hi_value = my_df["pct_chg"].iloc[hi_point]

    mom_Threshold = 0.019

    print("hi point: " + str(hi_point) + "\nlo point: " + str(lo_point))

    print("From numpy - mean: " + str(nmpy_mean))

    # my_numpy_array.apply_along_axis(in_range_finder, 0, my_numpy_array, [a, myMin, myMax, mom_Threshold]))
    
    # Select between hi and lo points
    target_df = my_df[hi_point:lo_point]

    # Reduce to dataframe with only below thresh for pct chg abs
    isWithinThreshold = target_df["pct_chg_abs"] < mom_Threshold
    target_df = target_df[isWithinThreshold]
    print(target_df)

    bird_df = my_df.iloc[target_df.index.values]
    print(bird_df["Measure"].describe())
    # target_df.plot()

    # my_df.plot(y = 'pct_chg')
    fig, ax = plt.subplots()
    # Make mean line
    ax.hlines(y=bird_df["Measure"].mean(), xmin=bird_df.index[0], xmax=bird_df.index[-1], linewidth=2, color='r')
    # Plot the data
    # my_df.iloc[target_df.index.values]["Measure"].plot(fig=fig)
    my_df["Measure"].plot(fig=fig)
    plt.show()

    ## now we do some calculations


## AUTOMATE 
# for datafile in datafiles:
    # print(os.path.join(datafile_folder_path,datafile))
    # sys.exit()
    # do_Mean_Bird_Calcs()
    # do_Mean_Bird_Calcs(True, os.path.join(datafile_folder_path,datafile))
    # sys.exit()


b3 = Button(root, text = "Cut Calc Mean", command = do_Mean_Bird_Calcs)
b3.grid(row = 2, column = 2) #, padx = 10, pady = 20) 

# #################
# # Do calcultions of the bird detected - assumes df contains only that section of the file - 
# #                                       must have cut it first to right length
# #####
# def do_Mean_Bird_Calcs():


# ### entry box ### doesn't work right now
# e = Entry(root, width = 30)
# # e.grid(row = 5, column = 3, padx = 5, pady = 10)
# e.pack()



def upload_file(to_show):
    f_types = [
        ('CSV files',"*.csv"),
        ('TXT',"*.txt")
        ]
    f_name = filedialog.askopenfilename(initialdir = myDir, 
        title = "Choose MOM File", 
        filetypes = f_types)

    
    l1.config(text=f_name) # display the path 
    # df=pd.read_csv(f_name) # create DataFrame
    df = pd.read_csv(f_name, header=None, names=["Measure", "Datetime"])

    #### show user info on the file chosen
    str1="\tRows:" + str(df.shape[0])+ "\t\tColumns:"+str(df.shape[1])+"\n"  #Minutes: "# +str(df.shape[0]/10.5/60)+"\n"
    str2="\tMinutes: " + str(round((df.shape[0])/10.5/60,2))+"\t"
    str3="(Hours: " + str(round((df.shape[0])/10.5/60/60,2))+" hours)\n"
    str3="(" + str(round((df.shape[0])/10.5/60/60,2))+" hours)\n"
    # str3="Hours: " + str(round(df.shape[0]/10.5/60/60,1))+"\n"

    # print(df["Measure"].describe())
    myDisplay_Mean = "\tMean Strain: " + str(round(df["Measure"].mean())) + "\n"
    # print(df["Measure"].quantile(0.99))

    #print(str1)
    displayname = f_name[len(myDir):len(f_name)] + "\n"
    t1.insert(tk.END, displayname)
    t1.insert(tk.END, str1) # add to Text widget
    t1.insert(tk.END, str2) # add to Text widget
    t1.insert(tk.END, str3) # add to Text widget
    t1.insert(tk.END, myDisplay_Mean) # add to Text widget
    #### end of showing info to user 

    # df2 = pd.read_csv(f_name, header=None, names=["Measure", "Datetime"])

  
       
    if (to_show == "cut"):
        the_start = int(my_entries[0].get())
        the_end = int(my_entries[1].get())
        # df3 = df2.iloc[20000:20600] #20000-20600
        df = df.iloc[the_start:the_end]
        df.plot()
    if (to_show == "all"):
        df.plot()
    else:
        df.plot() ## was setup for the calculation, but not now

    plt.show()

    return df


class Interval_Finder():
    def __init__(self, category, start_START, start_END, my_df):

        self.isGood = False
        self.category = category

        self.maybe_start = start_START  ## might be 0 if you want the whole DF
        self.maybe_end = start_END

        self.index_start = 0
        self.index_end = 0

        self.mom_df = my_df

        self.rolling_window = 5  ## could pass this as a parameter

        # self.ax = pyplot.gca()
        # self.lines=self.ax.lines
        # self.lines=self.lines[:]

        # self.tx = [self.ax.text(0,0,"") for l in self.lines]
        # self.marker = [self.ax.plot([startX],[startY], marker="o", color="red")[0]]

        self.currX = 0
        self.currY = 0

    def prep_df(self, type):
        self.mom_df['roll_mean5'] = self.mom_df['Measure'].rolling(self.rolling_window).mean()

        if(type == 'pct_chg'):
            self.mom_df['pct_chg'] = self.mom_df['roll_mean5'].pct_change()
            self.mom_df['pct_chg_abs'] = self.mom_df['pct_chg'].abs()

        if(type == 'std'):
            self.mom_df['roll_std'] = self.mom_df['Measure'].rolling(self.rolling_window).std() 

            
    def find_first_increase(self):  # assumes you have first done prep_df
        pass
        for i in self.mom_df:
            pass



    def find_last_decrease(self):
        pass
    
    def find_start(self):
        pass
    
    def find_end(self):
        pass







root.mainloop()