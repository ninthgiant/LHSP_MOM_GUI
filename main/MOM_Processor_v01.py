################################
#
#   MOM_Processor_v01
#
###########

########
# To do list:
#   plot known value on the figure
#   calculation based on average of the rolling average to lesson effect of edges
#   other calculation algorithms
#       calculation based on distance from midpoint
#   Batch process multiple files in one folder (Birds only)
#   Export range of values to CSV from one or batch of files
#       return those values as result of main function for now
#   Build class which has all this plus, known value, regression coeffs, export data
#################################

########
# import libraries, etc.
#################################

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
from setuptools import find_namespace_packages
import numpy as np
import statistics
from scipy import stats 
import os

###########
# set up desk window and other globals
###

##### if we want to automate processing of multiple files, use this
datafile_folder_path = "/Users/bobmauck/devel/LHSP_MOM_GUI/main/Data_Files/Cut_Bird_Only"
datafiles = os.listdir(datafile_folder_path)
# print("Datafiles in {}: {}".format(datafile_folder_path, datafiles))

# create the root window
root = tk.Tk()
root.title('Work with MOM datafile')
root.geometry('1200x1000')

#defaults for my computer - do an ASK to set defaults?
myDir = "/Users/bobmauck/Dropbox/BIG Science/MOMs/2022_Stuff" # where to open window for GUI file selection
default_window = 5
my_Save_Dir = "~/devel/LHSP_MOM_GUI/main/Data_Files/"  # where to put cut files

########
# put labels and buttons on the window
#####
my_font1 = ('courier', 10)
l1 = tk.Label(root,text='Read File & create DataFrame', width=60,font=my_font1)  
l1.grid(row=1,column=1)

### text boxes where the user input goes
t1=tk.Text(root,width=40,height=50)
t1.grid(row=6,column=1,padx=5, pady = 100)

t2=tk.Text(root,width=40,height=50)
t2.grid(row=6,column=0,padx=5, pady = 100)

t3=tk.Text(root,width=40,height=50)
t3.grid(row=6,column=2,padx=5, pady = 100)


#### default values for entry boxes labels
my_entry_labels = ["Starting_Point", "End_Point", "0.015", str(default_window)]
my_entry_labels_02 = ["Burrow", "Not Used", "Not Used", 'Not Used']
#### where to store entered values
my_entries = []
my_entries2 = []
for x in range(4):
    my_entry = Entry(root)
    my_entry.grid(row = 4, column = x, pady = 20, padx = 20)
    my_entry.insert(0, my_entry_labels[x])
    my_entries.append(my_entry)
## make one more row with one more entry
for x in range(4):
    my_entry2 = Entry(root)
    my_entry2.grid(row = 5, column = x, pady = 20, padx = 20)
    my_entry2.insert(0, my_entry_labels_02[x])
    my_entries2.append(my_entry2)

def mom_cut_button():
    my_start = my_entries[0].get()
    my_end = my_entries[1].get()
    my_label = my_entries2[0].get()


    hello = "Cut "+str(my_start) + " to " + str(my_end)

    my_fname, my_Cut = mom_open_file_dialog("cut")
    
    my_Cut.to_csv(my_Save_Dir + my_label + "_" + hello + ".TXT")

    t2.insert(tk.END, hello + "\n") # add to Text widget

def mom_calc_button(multiple_files):
    ### this is where you can get multiple files or one file; for now just get one file at a time with GUI
    my_rolling_window = int(my_entries[3].get())
    my_inclusion_threshold = float(my_entries[2].get())
    

    if multiple_files:
        pass  ## could cycle through files and rolling windows and thresholds
            ## AUTOMATE 
            # for datafile in datafiles:
            # print(os.path.join(datafile_folder_path,datafile))
            # sys.exit()
            # do_Mean_Bird_Calcs()
            # do_Mean_Bird_Calcs(True, os.path.join(datafile_folder_path,datafile))
            # sys.exit()
    else:
            ## get a file to work with, then send it here...
        bird_fname, bird_df = mom_open_file_dialog("not", False)  
            ## do the calculations
        bird_mean, bird_baseline = do_PctChg_Bird_Calcs(bird_df, bird_fname, my_rolling_window, my_inclusion_threshold)  ## last number is the points in the rolling windows
            #update the column wiht this calc info
        t3.insert(tk.END, "\tBird (" + str(my_rolling_window) + ", "+ str(my_inclusion_threshold)+ "): " + str(round(bird_mean,1)) + "\n")
        t3.insert(tk.END, "\tBird baseline: " + str(round(bird_baseline,1)) + "\n")
        t3.insert(tk.END, "\tStrain Change: " + str(round((bird_mean - bird_baseline),1)) + "\n")


#### buttons for browsing files, cutting files, or doing calculations
b1 = tk.Button(root, text='Browse Files', width=20,command = lambda: mom_open_file_dialog("all"))
b1.grid(row=2,column=1) 

b2 = Button(root, text = "Cut File", command = lambda:mom_cut_button())
b2.grid(row = 2, column = 0)

b3 = Button(root, text = "Cut Calc Mean", command = lambda:mom_calc_button(False))
b3.grid(row = 2, column = 2) #, padx = 10, pady = 20) 


def mom_format_dataframe(mydf):
    my_cols = mydf.shape[1]
    ### if there are 3, the first one was an axis, get rid of it on not the copy, but the original (inplace = True)
    if(my_cols == 3):
        mydf.drop(mydf.columns[0], axis=1, inplace = True)
    ### now name the columne
    mydf.columns = ['Measure', 'Datetime']
    return mydf


def mom_open_file_dialog(to_show, my_testing = False):
    f_types = [('CSV files',"*.csv"), ('TXT',"*.txt") ]
    if(not(my_testing)):  # make false for testing
        f_name = filedialog.askopenfilename(initialdir = myDir,  title = "Choose MOM File", filetypes = f_types)
    else:
        f_name = "/Users/bobmauck/devel/LHSP_MOM_GUI/main/Data_Files/185_One_Bird_70K_110K.TXT"

    l1.config(text=f_name) # display the path 
    ## read the chosen file
    # df = pd.read_csv(f_name, header=None, names=["Measure", "Datetime"])
    df = pd.read_csv(f_name, header=None, skiprows=1)
    df = mom_format_dataframe(df) #now 2 columns with proper names

    ## show info to user
    display_string = mom_get_file_info(df)
    display_string = f_name[len(myDir):len(f_name)] + "\n" + display_string + "\n"
    t1.insert(tk.END, display_string)
    #### end of showing info to user 

    if (to_show == "cut"):
        the_start = int(my_entries[0].get())
        the_end = int(my_entries[1].get())
        df = df.iloc[the_start:the_end]
        df.plot()
        plt.show()
    if (to_show == "all"):
        df.plot()
        plt.show()
    if(to_show == "not"):
        pass

    return f_name, df

def mom_get_file_info(my_df):
    #### show user info on the file chosen
    str1="\tRows:" + str(my_df.shape[0])+ "\t\tColumns:"+str(my_df.shape[1])+"\n"  #Minutes: "# +str(df.shape[0]/10.5/60)+"\n"
    str2="\tMinutes: " + str(round((my_df.shape[0])/10.5/60,2))+"\t"
    str3="(" + str(round((my_df.shape[0])/10.5/60/60,2))+" hours)\n"
    str4 = "\tMean Strain: " + str(round(my_df["Measure"].mean())) + "\n"

    return(str1 + str2 + str3 + str4)

# #################
#   do_PctChg_Bird_Calcs(my_df,f_name, my_window, my_threshold, my_update_screen)
#           Do calcultions of the bird detected - assumes df contains only that section of the file with bird, no other data 
#                                        must have cut it first to right length
#           my_update_screen shoudl be false if we are doing multiple birds in one batch
# #####
def do_PctChg_Bird_Calcs(my_df,f_name, my_window, my_threshold, my_update_screen=True):

        ## threshold to use for inclusion or exclusion from consideration between points
    mom_Threshold = my_threshold
 
    if(my_update_screen):
        # update the onscreen info into l1 and t3...
        l1.config(text=f_name) # display the path 
        display_string = mom_get_file_info(my_df)
        display_string = f_name[len(myDir):len(f_name)] + "\n" + display_string
        t3.insert(tk.END, display_string)


    lo_point, hi_point, my_df = mom_find_target_values(my_df, my_window)

    ####################
    # START of the treatment specific to the type of analysis this is (pct_chg~1st derivative)
    ########
    target_df = my_df[hi_point:lo_point]

    # Reduce to dataframe with only below thresh for pct chg abs
    isWithinThreshold = target_df["pct_chg_abs"] < mom_Threshold
    target_df = target_df[isWithinThreshold]
    # going back to the original df to get the values
    bird_df = my_df.iloc[target_df.index.values]
    bird_mean = bird_df["Measure"].mean()
    ####################
    # END of the treatment specific to the type of analysis this is (pct_chg~1st derivative)
    ########
 
    # get baseline from the same window, surounding the bird span
    bird_baseline_mean, bird_plot_df = mom_get_baseline(my_df, lo_point, hi_point)
    
    # now show focused plot and export that plot with info for later viewing
    # get the burrow number

    mom_do_birdplot(bird_df, bird_plot_df, display_string, my_window, my_threshold, "pctChg")
    
    return bird_mean, bird_baseline_mean

def mom_do_birdplot (bird_df, bird_plot_df, my_filename, my_window, my_threshold, my_type):
    ##### need to improve - send to specific folder for these plots
    # Now make a plot of the whole thing wtih a line over the points we will use to calculate the bird
    fig, ax = plt.subplots()
    # Make mean line
    ax.hlines(y=bird_df["Measure"].mean(), xmin=bird_df.index[0], xmax=bird_df.index[-1], linewidth=2, color='r')
    # Plot the data
    # my_df["Measure"].plot(fig=fig)
    bird_plot_df["Measure"].plot(fig=fig)
    # save the plot, name assumes your original files starts wtih 3-letter burrow number
    my_label = my_entries2[0].get()
    output_filename = my_label + "_" + my_type + "_" + str(my_window) + "  win_" + str(my_threshold) + "_thr.png"
    plt.savefig(output_filename)
    # show the plot
    # plot_text = str(bird_mean) # + str(bird_baseline_mean)
    plt.text(7.8, 12.5, "I am Adding Text To The Plot")
    plt.show()


def mom_get_baseline(my_df, lo_point, hi_point):
        # can be used any file as long as you adjust for how much room we can us
        ### now zoom into the focal area to show the plot and calculate the baseline mean
        # parameters for this. Should they be passed?
    total_len = my_df.shape[0]

    if(total_len > 300):
        my_padding = 100
    else:
        my_padding = 50

    baseline_max = 0.00002
        # get subset of data from my_df
    bird_plot_df = my_df[(hi_point - my_padding):(lo_point + my_padding)]
        # get a df with only baseline data
    bird_baseline_df = bird_plot_df.loc[bird_plot_df['pct_chg_abs'] < baseline_max]
    bird_baseline_mean = bird_baseline_df["Measure"].mean()

    return bird_baseline_mean, bird_plot_df


def mom_find_target_values(my_df, my_window):
        # gets a df to work with that has all the original columns, condense to one 'Measure/
        # gets my_window to determine the rolling window on which pct change is based
        # condense into one column to do the calculations, setup the data we need to make decisions 
        # uses pct change from a rolling window to decide what to include 
    
    my_df = my_df[ 'Measure'].to_frame()
    my_df['roll_std'] = my_df['Measure'].rolling(my_window, center = True).std() 
    my_df['roll_mean'] = my_df['Measure'].rolling(my_window, center = True).mean()
    my_df['pct_chg'] = my_df['roll_mean'].pct_change()
    my_df['pct_chg_abs'] = my_df['pct_chg'].abs()
    
    ### make a numpy array so that we can do quick math
    my_numpy_array = my_df["pct_chg"].to_numpy()

    ## find the interval of interest between the high and low points - assumes MOM "behavior"
    ##     hi poinr is positive change with stepping on MOM, low is negative change in exiting
    ##     do I need to adjust this in case opposite direction? which occurs first?
    hi_point = np.nanargmax(my_numpy_array)
    lo_point = np.nanargmin(my_numpy_array)
    ## get the actual values at those points - not really needed yet, so don't express
    # lo_value = my_df["pct_chg"].iloc[lo_point]
    # hi_value = my_df["pct_chg"].iloc[hi_point]

    return lo_point, hi_point, my_df


def mom_get_file_info(my_df):
    #### show user info on the file chosen
    str1="\tRows:" + str(my_df.shape[0])+ "\t\tColumns:"+str(my_df.shape[1])+"\n"  #Minutes: "# +str(df.shape[0]/10.5/60)+"\n"
    str2="\tMinutes: " + str(round((my_df.shape[0])/10.5/60,2))+"\t"
    str3="(" + str(round((my_df.shape[0])/10.5/60/60,2))+" hours)\n"
    str4 = "\tMean Strain: " + str(round(my_df["Measure"].mean())) + "\n"

    return(str1 + str2 + str3 + str4)




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