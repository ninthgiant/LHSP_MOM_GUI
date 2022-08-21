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
from tkinter import messagebox as mb
from tkinter import *

#### imports from LIam
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backend_bases as backendgi
from setuptools import find_namespace_packages
import numpy as np
import statistics
from scipy import stats 
import os

################
# Function Set_Globals to declare global variables all in one place - is this possible?
#    RAM 7/26/22
#    Parameters: NONE
#    Returns: NONE
#    About: run at startup, but not yet doing that, using definitions above only
#######
def Set_Globals():
    # general info about the fle
    global user_INPATH
    global user_BURROW
    global data_DATE

    global data ## this  is a dataframe to be defined later
    global calibrations

    global cal_gradient
    global cal_intercept
    global baseline_cal_mean
    global cal_r_squared

    global calibrations
    global birds

    # Lists to accumulate info for different birds - Make these globals at top of app?
    global birds_datetime_starts
    global birds_datetime_ends
    global birds_data_means
    global birds_cal_means
    global birds_details

    global myDir
    global default_window
    global my_Save_Dir
    global my_Save_Real_Dir
    global datafile_folder_path
    global datafiles

    global cal1_value
    global cal2_value
    global cal3_value

    ### these should be set in onscreen in second row, then grabbed before calibration, if changed
    cal1_value = 27.65
    cal2_value = 50.0
    cal3_value = 65.3

    global my_Continue

    ### now make them
    birds_datetime_starts = []
    birds_datetime_ends = []
    birds_data_means = []
    birds_cal_means = []
    birds_details = []

    # defaults for my computer - do an ASK to set defaults?
    myDir = "/Users/bobmauck/devel/LHSP_MOM_GUI/main/Data_Files/Cut_With_Calib" # where to open window for GUI file selection
    default_window = 5
    my_Save_Dir = "~/devel/LHSP_MOM_GUI/main/Data_Files/"  # where to put cut files
    my_Save_Real_Dir = "/Users/bobmauck/Dropbox/BIG Science/MOMs/2022-stuff/Daily_Calculated"

    ##### if we want to automate processing of multiple files, use this
    datafile_folder_path = "/Users/bobmauck/devel/LHSP_MOM_GUI/main/Data_Files/Cut_With_Calib"
    datafiles = os.listdir(datafile_folder_path)

############
# confirm_continue: a utility function to get response via click
####
def confirm_continue(my_Question):
    MsgBox = mb.askquestion ('Confirm', my_Question)
    if MsgBox == 'yes':
        return True
    else:
        return False

#############
# return_useful_name: takes a path string and returns just the name of the file
####
def return_useful_name(the_path):
    where = the_path.rfind("/")
    the_name = the_path[(where + 1):(len(the_path)-4)]
    return the_name

###########
# set up desk window and other globals
###
Set_Globals()



# create the root window
root = tk.Tk()
root.title('Work with MOM datafile')
root.geometry('1200x1000')



########
# put labels and buttons on the window
#####
my_font1 = ('courier', 10)
l1 = tk.Label(root,text='Read File & create DataFrame', width=60,font=my_font1)  
l1.grid(row=1,column=1)

### text boxes where the user input goes
t1=tk.Text(root,width=40,height=50)
t1.grid(row=6,column=0,padx=5, pady = 100)

t2=tk.Text(root,width=40,height=50)
t2.grid(row=6,column=2,padx=5, pady = 100)

t3=tk.Text(root,width=40,height=50)
t3.grid(row=6,column=1,padx=5, pady = 100)


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
    if(False):
        my_start = my_entries[0].get()
        my_end = my_entries[1].get()
        my_label = my_entries2[0].get()
        hello = "Cut "+str(my_start) + " to " + str(my_end)
        my_fname, my_Cut = mom_open_file_dialog("cut")  
        my_Cut.to_csv(my_Save_Dir + my_label + "_" + hello + ".TXT")
        t2.insert(tk.END, hello + "\n") # add to Text widget
    else:  ### the new way where we try to do the calibration
        pass
        ## get a file to work with, then send it here...
        # user_BURROW = my_entries2[0].get()
        bird_fname, bird_df = mom_open_file_dialog("not", False) 
        user_BURROW = return_useful_name(bird_fname) 
        my_Continue = True
        print(my_Continue)
        my_Continue == my_Do_Calibrations(bird_df)
        print(my_Continue)
        if(my_Continue):
            Do_Multiple_Birds(bird_df)
            


def mom_calc_button(multiple_files):
    ### for now just get one file at a time with GUI; have another button for multiple files
    my_rolling_window = int(my_entries[3].get())
    my_inclusion_threshold = float(my_entries[2].get())

        ## get a file to work with, then send it here...
    bird_fname, bird_df = mom_open_file_dialog("not", False)  
        ## do the calculations
    bird_mean, bird_baseline, n_points = do_PctChg_Bird_Calcs(bird_df, bird_fname, my_rolling_window, my_inclusion_threshold)  ## last number is the points in the rolling windows
        #update the column wiht this calc info
    t2.insert(tk.END, "\tBird (" + str(my_rolling_window) + ", "+ str(my_inclusion_threshold)+ "): " + str(round(bird_mean,1)) + "\n")
    t2.insert(tk.END, "\tBird baseline: " + str(round(bird_baseline,1)) + "\n")
    t2.insert(tk.END, "\tStrain Change: " + str(round((bird_mean - bird_baseline),1)) + "\t(N=" +str(n_points) + ")\n")

def mom_calc_multiple_files():
    raw_files_path = "/Users/bobmauck/devel/LHSP_MOM_GUI/main/Data_Files/Cut_Bird_Only"
    files_to_load = os.listdir(raw_files_path)
    files_to_load_absolute = [ os.path.join(raw_files_path,filename) for filename in files_to_load ] 
    print(files_to_load_absolute)
    dataframes_to_load = [ pd.read_csv(fpath, header=None, skiprows=1) for fpath in files_to_load_absolute]
    dataframes_to_load = [ mom_format_dataframe(df) for df in dataframes_to_load]
    my_Windows = [3, 5, 7]
    my_Thresholds = [0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03]
        ## could cycle through files and rolling windows and thresholds
        #       datafiles are located in: "/Users/bobmauck/devel/LHSP_MOM_GUI/main/Data_Files/Cut_Bird_Only"
    output_df = pd.DataFrame([], columns={ "bird_mean", "bird_baseline", "n_points", "rolling_window", "my_threshold","File_name", "type_of_calc"})
    type_of_calc = "pct_change"
    for bird_df, bird_fname in zip(dataframes_to_load, files_to_load):
        for my_rolling_window in my_Windows:
            for my_inclusion_threshold in my_Thresholds:
                print(bird_fname, my_rolling_window, my_inclusion_threshold)
                bird_mean, bird_baseline, n_points = do_PctChg_Bird_Calcs(bird_df, bird_fname, my_rolling_window, my_inclusion_threshold, False, False)
                output_df.loc[len(output_df.index)] = [bird_mean, bird_baseline, n_points, my_rolling_window, my_inclusion_threshold, bird_fname, type_of_calc]
    
    my_unique_name = " 002"
    output_path = "/Users/bobmauck/devel/LHSP_MOM_GUI/main/Data_Files/Cut_Bird_Only" + my_unique_name
    output_df.to_csv(output_path, sep = "\t", index=False)
    # print(output_df)

    return output_df

       

#### buttons for browsing files, cutting files, or doing calculations
b1 = tk.Button(root, text='Browse Files', width=20,command = lambda: mom_open_file_dialog("all"))
b1.grid(row = 2,column = 1) 

b2 = Button(root, text = "Process MOM", command = lambda:mom_cut_button())  
b2.grid(row = 2, column = 0)

b3 = Button(root, text = "Cut Calc Mean", command = lambda:mom_calc_button(False))
b3.grid(row = 2, column = 2) #, padx = 10, pady = 20) 

b4 = Button(root, text = "Multi Files", command = lambda:mom_calc_multiple_files())
b4.grid(row = 2, column = 3) #, padx = 10, pady = 20)



def mom_format_dataframe(mydf):
    my_cols = mydf.shape[1]
    ### if there are 3, the first one was an axis, get rid of it on not the copy, but the original (inplace = True)
    if(my_cols == 3):
        # NOTE: may need to format the Datetime column, but for now it is a string. Or test for type later 
        mydf.drop(mydf.columns[0], axis=1, inplace = True)
    ### now name the columns
    mydf.columns = ['Measure', 'Datetime']
    return mydf


def mom_open_file_dialog(to_show, my_testing = False):
    global data_DATE
    
    f_types = [('CSV files',"*.csv"), ('TXT',"*.txt") ]
    if(not(my_testing)):  # make false for testing
        f_name = filedialog.askopenfilename(initialdir = myDir,  title = "Choose MOM File", filetypes = f_types)
    else:
        f_name = "/Users/bobmauck/devel/LHSP_MOM_GUI/main/Data_Files/185_One_Bird_70K_110K.TXT"

    dispName = f_name[(len(f_name)-60):len(f_name)]
    l1.config(text=dispName) # display the path 
    df = pd.read_csv(f_name, header=None, skiprows=1)
    df = mom_format_dataframe(df) #make sure we have 2 columns with proper names

    ## show info to user
    display_string = mom_get_file_info(df)
    # display_string = f_name[len(myDir):len(f_name)] + "\n" + display_string + "\n"  ### this doesn't work if not in MyDir!!
    display_string = return_useful_name (f_name) + "\n" + display_string + "\n"
    t1.insert(tk.END, display_string)
    #### end of showing info to user 

    # set some globlas for later use
    global user_BURROW
    
    user_BURROW = return_useful_name (f_name) # f_name[len(myDir):len(f_name)]
    data_DATE = df.Datetime.iloc[-1] # .date()
    # data_DATE = df["Datetime"].iloc[-1].date()

    if (to_show == "cut"):
        the_start = int(my_entries[0].get())
        the_end = int(my_entries[1].get())
        df = df.iloc[the_start:the_end]

        # fig, ax = plt.subplots()
        # ax.plot(df.iloc[the_start:the_end])

        df.plot()
        plt.show()
    if (to_show == "all"):
        fig, ax = plt.subplots()
        ax.plot(df.loc[:,"Measure"])
        ## add_titlebox(ax, 'info here')
        ax.set_title(return_useful_name (f_name))
        ## df.plot() ## this was old way
        plt.show()
    if(to_show == "not"):
        pass

    return f_name, df

def add_titlebox(ax, text):
    ax.text(.02, .9, text,   ## proportion left to right, proportion bottom to top, the text
        horizontalalignment='left',
        transform=ax.transAxes,
        # bbox=dict(facecolor='white', alpha=0.6),
        fontsize=10)
    return ax

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
def do_PctChg_Bird_Calcs(my_df,f_name, my_window, my_threshold, my_update_screen = True, do_Plot = True):

        ## threshold to use for inclusion or exclusion from consideration between points
    mom_Threshold = my_threshold
 
    if(my_update_screen):
        # update the onscreen info into l1 and t3...make it fit into 80 chars
        dispName = f_name[(len(f_name)-60):len(f_name)]
        l1.config(text=dispName) # display the path 
        display_string = mom_get_file_info(my_df)
        display_string = return_useful_name (f_name) + "\n" + display_string  
        t2.insert(tk.END, display_string)


    lo_point, hi_point, my_df = mom_find_target_values(my_df, my_window)

    ####################
    # START of the treatment specific to the type of analysis this is (pct_chg~1st derivative)
    ########
    target_df = my_df[hi_point:lo_point]

    # Reduce to dataframe with only below thresh for pct chg abs
    isWithinThreshold = target_df["pct_chg_abs"] < mom_Threshold
    target_df = target_df[isWithinThreshold]
    # how many points are we using?
    n_points = target_df.shape[0]
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

    if(do_Plot):
        mom_do_birdplot(bird_df, bird_plot_df, display_string, my_window, my_threshold, "pctChg")
    
    return bird_mean, bird_baseline_mean, n_points

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
    plt.savefig(os.path.join("output_files",output_filename))
    # show the plot
    # plot_text = str(bird_mean) # + str(bird_baseline_mean)
    # plt.text(7.8, 12.5, "I am Adding Text To The Plot")
    # plt.show()


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

###################
#  getTracPointPair:  A function to set a pair of draggle points on an interactive trace plot
# RETURNS
#       mean, markers, isGood
#       mean -- mean strain measurement value between two marked points
#       markers -- dataframe of marker information, including start and end index on the trace
#       isGood -- boolean confirmation that the plot wasn't closed before both points were marked
#       newAxesLimits -- bounding box limits for the plot view that was shown right before exiting 
def getTracePointPair(my_df, category, markers=None, axesLimits=None):

    data = my_df
    # Print a message
    my_Push_Enter = "Add {category} start point, then press enter.".format(category=category)

    # Turn on the Matplotlib-Pyplot viewer - interactive mode
    # Shows the trace from the globally-defined data
    plt.ion()
    fig, ax = plt.subplots()

    fig.set_size_inches((20,4))  # (default_figure_width, default_figure_height)) 
 
    ax.plot(data.loc[:,"Measure"])

    if (axesLimits is not None):
        ax.set_xlim(left=axesLimits.xstart, right=axesLimits.xend)
        ax.set_ylim(bottom=axesLimits.ystart, top=axesLimits.yend)

    if (markers is not None):
        # Add any previous markers
        annotateCurrentMarkers(markers)

    # Initialize the draggable markers
    dm = DraggableMarker(category=category, startY=min(data["Measure"]))
    ax.set_title(my_Push_Enter)
    plt.show(block=True)

    plt.ioff()

    # Gather the marked points data
    index_start = min(dm.index_start, dm.index_end)
    index_end = max(dm.index_start, dm.index_end)
    time_start = data.loc[index_start,"Datetime"]
    time_end = data.loc[index_end,"Datetime"]
    measures = data.loc[index_start:index_end,"Measure"]
    mean = statistics.mean(measures)

    # Extract the axes limits for the final interactive plot view
    # in case the user wants to use those limits to restore the view on the next plot
    endView_xstart, endView_xend = ax.get_xlim()
    endView_ystart, endView_yend = ax.get_ylim()
    newAxesLimits = AxesLimits(endView_xstart, endView_xend, endView_ystart, endView_yend)

    # Confirm the plot was not exited before both points were marked
    isGood = dm.isGood

    print("""
    Measured {category} from {start} to {end}.
    Mean {category} measurement is {mean}.
    """.format(category=category, start=time_start, end=time_end, mean=round(mean,2)))

    # Create a dataframe with information about the marked points
    markers = pd.DataFrame({"Category":category,
                                "Point":["Start", "End"],
                                "Index":[index_start, index_end],
                                "Datetime":[time_start, time_end],
                                "Measure":[data.loc[index_start,"Measure"], data.loc[index_end,"Measure"]]})
    markers = markers.set_index("Index")

    return mean, markers, isGood, newAxesLimits

#########################
#   annotateCurrentMarkers: A function to plot all markers from a markers dataframe on the current plt viewer
#   (to be used for the markers dataframe as returned by getTracePointPair)
########
def annotateCurrentMarkers(markers):
    ax = plt.gca() # this assumes a current figure object? Is this the only external assumption?

    # Plot the pairs of marker points separately, so lines aren't drawn betwen them
    for l, df in markers.groupby("Category"):
        ax.plot(df.loc[:,"Measure"], marker="o", color="black", ms=8)
        for index, row in df.iterrows():
            label = "{category} {point}".format(category=df.loc[index,"Category"], point=df.loc[index, "Point"])
            ax.annotate(label, (index, df.loc[index, "Measure"]), rotation=60)

################
# Function my_Do_Calibrations get the calibration for the MOM on this burrow-night
#    RAM 7/25/22
#    parameters: my_dataframe -> data to work with
#       - 
#    returns tuple with information about the result of the calculations
#######
def my_Do_Calibrations(my_dataframe):
        
    data = my_dataframe
    
    global calibrations

    global cal_gradient
    global cal_intercept
    global baseline_cal_mean
    global cal_r_squared

    global cal1_value
    global cal2_value
    global cal3_value

    good_to_go = True

  
    # Add baselines
    baseline_cal_mean, baseline_cal_markers, baseline_cal_Good, axesLimits = getTracePointPair(data, "Baseline")
    markers = baseline_cal_markers

    # Add calibrations as 3 separate pairs of points
    cal1_mean, cal1_markers, cal1_Good, axesLimits = getTracePointPair(data, "Cal1[{}]".format(cal1_value), markers, axesLimits)
    markers = pd.concat([markers, cal1_markers])

    cal2_mean, cal2_markers, cal2_Good, axesLimits = getTracePointPair(data, "Cal2[{}]".format(cal2_value), markers, axesLimits)
    markers = pd.concat([markers, cal2_markers])

    cal3_mean, cal3_markers, cal3_Good, axesLimits = getTracePointPair(data, "Cal3[{}]".format(cal3_value), markers, axesLimits)
    markers = pd.concat([markers, cal3_markers])


    # Clean up the marked calibration points data
    calibrations = pd.DataFrame({"Category":["Cal1", "Cal2", "Cal3"],
                                    "Value_True":[cal1_value, cal2_value, cal3_value],
                                    "Value_Measured":[cal1_mean, cal2_mean, cal3_mean]})
    calibrations["Value_Difference"] = abs(calibrations["Value_Measured"] - baseline_cal_mean)

    # Get the linear regression information across the three calibration points
    cal_gradient, cal_intercept, cal_r_value, cal_p_value, cal_std_err = stats.linregress(calibrations["Value_Difference"], calibrations["Value_True"])
    cal_r_squared = cal_r_value**2

    # A tiny function to confirm if we want to continue
    #   after showing the calibration plot results. Used just below.
    def continueKey(doit):
        if(doit == 'y'):
            good_to_go = True
        else:
            good_to_go = False


    # print("Showing calibration results.\nPress 'y' to proceed or 'n' to exit.")
    fig, ax = plt.subplots()
    # fig.canvas.mpl_connect('key_press_event', continueKey)

    ax.plot(calibrations["Value_Difference"], calibrations["Value_True"], marker="o", color="black", linestyle="None")
    ax.plot(calibrations["Value_Difference"], calibrations["Value_Difference"]*cal_gradient+cal_intercept, color="gray", linestyle="dashed")
    plt.xlabel("Measured value (strain difference from baseline)")
    plt.ylabel("True value (g)")
    plt.title("Calibration regression\n(R^2={r}, Inter={i}, Slope={s})".format(r=round(cal_r_squared,5), i=round(cal_intercept,5), s=round(cal_gradient,5)))
    
    # #################
    # axes = plt.axes([0.81, 0.005, 0.1, 0.055])
    # bnext = Button(axes, 'Proceed')
    # bnext.on_clicked(continueKey("y"))

    # axes2 = plt.axes([0.1, 0.005, 0.1, 0.055])
    # bnext2 = Button(axes2, 'Stop') # , color="red")
    # bnext2.on_clicked(continueKey("n"))
    # #################
    
    plt.show()
    ### show user in real time
    my_cal_result = "\tCalibration regression\n\tR^2={r}\n\tIntcpt={i}, Slope={s}".format(r=round(cal_r_squared,5), i=round(cal_intercept,5), s=round(cal_gradient,5))
    t3.insert(tk.END, "File: " + user_BURROW + "\n") # add to Text widget
    t3.insert(tk.END, my_cal_result + "\n") # add to Text widget

     # Check all the calibrations were marked successfully
    # if (not baseline_cal_Good or not cal1_Good or not cal2_Good or not cal3_Good or (cal_r_squared<0.9)):
    if (abs(cal_r_squared) < 0.9):
        print("bad r2")
        good_to_go = False

    return good_to_go

#############################
# Function Do_Birds: get the calibration and bird weight data for a single bird in a MOM file
#    RAM 7/25/22
#    parameters: NONE 
#    returns: NONE
#######
def Do_Bird(my_DataFrame):
    
        # If the user wants to a bird, first get the calibration points for that bird
        bird_cal_mean, bird_cal_markers, bird_cal_good, bird_cal_axesLimits = getTracePointPair(my_DataFrame, "Calibration[Bird]")

        bird_data_mean, bird_data_markers, bird_data_good, bird_data_axesLimits = getTracePointPair(my_DataFrame, "Bird Data", bird_cal_markers, bird_cal_axesLimits)
        measure_start = bird_data_markers[bird_data_markers["Point"]=="Start"].Datetime.iloc[0]
        measure_end = bird_data_markers[bird_data_markers["Point"]=="End"].Datetime.iloc[0]

        # Allow the user to input extra details for a "Notes" column
        # bird_details = input("Enter any details about the bird:     ")
        bird_details = "None"

        # Add the info about this bird to the accumulating lists
        birds_datetime_starts.append(measure_start)
        birds_datetime_ends.append(measure_end)
        birds_data_means.append(bird_data_mean)
        birds_cal_means.append(bird_cal_mean)
        birds_details.append(bird_details)
        print("Bird Mass: ")
        my_result = round(((bird_data_mean - bird_cal_mean) * cal_gradient + cal_intercept),2)
        print(my_result)
        t2.insert(tk.END, "File: " + user_BURROW + "\n") # add to Text widget
        t2.insert(tk.END, "\tBird Mass: \t" + str(my_result) + "\n") # add to Text widget

#############################
# Function Do_Multiple_Birds: to id mltiple birds in one file
#    RAM 7/26/22
#    Parameters: NONE
#    Returns: NONE
#    
#######
def Do_Multiple_Birds(my_DataFrame):
    global birds
    Set_Globals()  # reset the saved birds
    # assumes have lists declared as global
    # Allow the user to continue entering birds for as many times as she wants
    while (True):
        if(confirm_continue("Enter bird data?")):
            Do_Bird(my_DataFrame)
        else:
            break

    # Done entering bird data
    #   Make the accumulated bird info into a clean dataframe for exporting
    
    birds = pd.DataFrame({"Burrow":user_BURROW,
                            "Date":data_DATE,
                            "Datetime_Measure_Start":birds_datetime_starts,
                            "Datetime_Measure_End":birds_datetime_ends,
                            "Mean_Data_Strain":birds_data_means,
                            "Mean_Calibration_Strain":birds_cal_means,
                            "Details":birds_details})

    # # Convert the Datetime columns back to character strings for exporting
    # birds["Datetime_Measure_Start"] = birds["Datetime_Measure_Start"].dt.strftime("%Y-%m-%d %H:%M:%S")
    # birds["Datetime_Measure_End"] = birds["Datetime_Measure_End"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Calculate the baseline and regressed mass estimates for the birds
    birds["Baseline_Difference"] = abs(birds["Mean_Data_Strain"] - birds["Mean_Calibration_Strain"]) 
    birds["Regression_Mass"] = birds["Baseline_Difference"] * cal_gradient + cal_intercept

    print("Bird calculated masses: ")
    print(birds["Regression_Mass"])

    print("Bird entries complete.")

    if(confirm_continue("Export Results?")):
        Output_MOM_Data()

    return birds

################
# Function Output_MOM_Data to declare global variables all in one place - is this possible?
#    RAM 7/26/22
#    Parameters: NONE
#    Returns: NONE
#    About: sends accumulated data to a csv file
#######
def Output_MOM_Data():
        # Ask the user for any last summary details for the summary info sheet
    #summaryDetails = input("Enter any summary details:\n")
    summaryDetails = "NONE"

    # Export summary info, including calibration info, to file
    path_summary = "Burrow_{burrow}_{date}_SUMMARY.txt".format(burrow=user_BURROW, date=data_DATE)

    #saveFilePath = fileDialog.asksaveasfile(mode='w', title="Save the file", defaultextension=".txt")

    path_summary = filedialog.asksaveasfile(initialdir = my_Save_Dir, initialfile = user_BURROW,
                                    defaultextension= '.csv',
                                    filetypes=[
                                        ("Text file",".txt"),
                                        ("CSV file", ".csv"),
                                        ("All files", ".*"),
                                    ])

    

    # with open(path_summary, 'w') as f:
    #     f.write("M.O.M. Results\n")
    #     f.write("Burrow: {burrow}\n".format(burrow=user_BURROW))
    #     f.write("Deployment date: {date}\n".format(date=data_DATE))
    #     f.write("Number of birds recorded: {numBirds}\n".format(numBirds=len(birds_data_means)))
    #     f.write("\n\n\nCalibration details:\n")
    #     f.write("Mean value from baseline for calibration: {}\n\n".format(baseline_cal_mean))
    #     f.write(calibrations.to_csv(sep="\t", index=False))
    #     f.write("\nCalibration regression\nR^2={r}, Intercept={i}, Slope={s}".format(r=round(cal_r_squared,5), i=round(cal_intercept,5), s=round(cal_gradient,5)))
    #     f.write("\n\n\n")
    #     f.write("Summary details:\n")
    #     f.write(summaryDetails)
    #     f.close()

    # print("Wrote summary details, including calibration info, to\n\t\t{spath}".format(spath=path_summary))

    # Export bird info (if any was added)
    path_bird = "Burrow_{burrow}_{date}_BIRDS.txt".format(burrow=user_BURROW, date=data_DATE)
    path_bird = path_summary

    if (len(birds_data_means) > 0):
        birds.to_csv(path_bird, index=False)
        mb.showinfo("Bird data saved")
        print("Wrote bird details to\n\t\t{bpath}".format(bpath=path_bird))
    else:
        mb.showinfo("No birds recorded.")
        print("No birds recorded.")




#############################
#   DraggableMarker:  A class for a set of draggable markers on a Matplotlib-plt line plot
#   Designed to record data from two separate markers, which the user confirms with an "enter key"
#   Adapted by Liam Taylor from https://stackoverflow.com/questions/43982250/draggable-markers-in-matplotlib
#############
class DraggableMarker():
    def __init__(self, category, startY, startX=0):
        self.isGood = False
        self.category = category

        self.index_start = 0
        self.index_end = 0

        self.buttonClassIndex = 0
        self.buttonClasses = ["{category} start".format(category=category), "{category} end".format(category=category)]

        self.ax = plt.gca()  # this assumes a current figure object? Is this the only external assumption?
        self.lines=self.ax.lines
        self.lines=self.lines[:]

        self.tx = [self.ax.text(0,0,"") for l in self.lines]
        self.marker = [self.ax.plot([startX],[startY], marker="o", color="red")[0]]

        self.draggable = False

        self.isZooming = False
        self.isPanning = False

        self.currX = 0
        self.currY = 0

        self.c0 = self.ax.figure.canvas.mpl_connect("key_press_event", self.key)
        self.c1 = self.ax.figure.canvas.mpl_connect("button_press_event", self.click)
        self.c2 = self.ax.figure.canvas.mpl_connect("button_release_event", self.release)
        self.c3 = self.ax.figure.canvas.mpl_connect("motion_notify_event", self.drag)

    def click(self,event):
        if event.button==1 and not self.isPanning and not self.isZooming:
            #leftclick
            self.draggable=True
            self.update(event)
            [tx.set_visible(self.draggable) for tx in self.tx]
            [m.set_visible(self.draggable) for m in self.marker]
            self.ax.figure.canvas.draw_idle()        
                
    def drag(self, event):
        if self.draggable:
            self.update(event)
            self.ax.figure.canvas.draw_idle()

    def release(self,event):
        self.draggable=False
        
    def update(self, event):
        try:        
            line = self.lines[0]
            x,y = self.get_closest(line, event.xdata) 
            self.tx[0].set_position((x,y))
            self.tx[0].set_text(self.buttonClasses[self.buttonClassIndex])
            self.marker[0].set_data([x],[y])
            self.currX = x
            self.currY = y
        except TypeError:
            pass

    def get_closest(self,line, mx):
        x,y = line.get_data()
        try: 
            mini = np.argmin(np.abs(x-mx))
            return x[mini], y[mini]
        except TypeError:
            pass

    ##############
    # Not sure we even want these - would rather deal with buttons on the window
    #####
    def key(self,event):
        if (event.key == 'o'):
            self.isZooming = not self.isZooming
            self.isPanning = False
        elif(event.key == 'p'):
            self.isPanning = not self.isPanning
            self.isZooming = False
        elif(event.key == 't'):
            # A custom re-zoom, now that 'r' goes to 
            # the opening view (which might be retained from a previous view)
            line = self.lines[0]
            full_xstart = min(line.get_xdata())
            full_xend = max(line.get_xdata())
            full_ystart = min(line.get_ydata())
            full_yend = max(line.get_ydata())
            self.ax.axis(xmin=full_xstart, xmax=full_xend, ymin=full_ystart, ymax=full_yend)
        elif (event.key == 'enter'):  #### these are the event keys we need for now
            if(self.buttonClassIndex==0):
                self.ax.plot([self.currX],[self.currY], marker="o", color="yellow")
                self.buttonClassIndex=1
                self.index_start = self.currX
                plt.title("Add {category} end point, then press enter.".format(category=self.category))
            elif(self.buttonClassIndex==1):
                self.index_end = self.currX
                self.isGood = True
                plt.close()
            self.update(event)

###################
#   AxesLimits: A class defining an object that stores axes limits for
#       pyplot displays
#######
class AxesLimits():
    def __init__(self, xstart, xend, ystart, yend):
        self.xstart = xstart
        self.xend = xend
        self.ystart = ystart
        self.yend = yend


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