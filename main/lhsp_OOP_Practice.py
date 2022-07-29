#######
# app setup tkinter, libraries
####

from curses import BUTTON1_CLICKED
from logging import setLogRecordFactory
import tkinter as tk
# from tkinter import ttk
from tkinter import Tk, Button
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


# ###########
# #
# #   File to build classes, but not to run
# #
# ##########

# # #########
# #   Handle_Point_Pair
# #       A class for pairs of points to be used whenever that is necessary
# #       Designed to record data from two separate markers, which the user confirms with an "enter key"
# # Adapted from https://stackoverflow.com/questions/43982250/draggable-markers-in-matplotlib
# class Handle_Point_Pair():
#     def __init__(self, category):

#         self.bird_data_mean = 0.0
#         self.bird_data_markers
#         self.bird_data_good
#         self.bird_data_axesLimits 
#         self.measure_start = 0 
#         self.measure_end = 0
#         self.data_type = category  ### bird, baseline, known_wt (for calibration)



#         bird_data_mean, bird_data_markers, bird_data_good, bird_data_axesLimits = getTracePointPair("Bird Data", bird_cal_markers, bird_cal_axesLimits)
#         measure_start = bird_data_markers[bird_data_markers["Point"]=="Start"].Datetime.iloc[0]
#         measure_end = bird_data_markers[bird_data_markers["Point"]=="End"].Datetime.iloc[0]


# #########
#   MOM_Viewer
#       A class for pairs of points to be used whenever that is necessary
#       Designed to record data from two separate markers, which the user confirms with an "enter key"
# Adapted from https://stackoverflow.com/questions/43982250/draggable-markers-in-matplotlib
class MOM_Viewer(Tk):
    def __init__(self):  ### setup gui window, present user with button to open a file, create all the necesary variables
        
        ### inherit from tkinter
        super().__init__()

        #### get the root window ready
            # configure the root window
        self.title('MOM Viewer App')
        self.geometry('600x600+150+200')
        self.frame = LabelFrame(self, padx = 5, pady = 5)
        self.frame.pack(padx = 10, pady = 10)

        # labels
        self.l0 = Label(self.frame, text='Move or Change this Text!')
        self.l0.pack()
        # self.label.grid(row=0,column=0)

        self.l1 = Label(self,text='Read File & create DataFrame',width=60) # ,font=my_font1)  
        # self.l1.grid(row=1,column=1)
        self.l1.pack(side = TOP)

       # self.l2= Label(setLogRecordFactory,text='Instructions', width=60)  
       # self.l2.grid(row=5,column=1)

        # buttons
        self.button = Button(self, text='Calibration', command = self.button_clicked)
        self.button.pack(side = LEFT)


        #### button for browsing files, but doing nothing
        self.b1 = Button(self, text='Browse Files', width=20, command = self.b1_clicked)
        self.b1.pack(side = RIGHT)

         #### button for browsing files, but doing nothing
        self.b2 = Button(self, text='Cut Files', width=20, command = self.b2_clicked)
        self.b2.pack(side = RIGHT)
    

        ###########
        # input interface
        ##
        #### default values for entry boxes
        self.my_entry_labels = ["Starting_Point", "End_Point"]
        #### where to store entered values
        self.my_entries = []

        for x in range(2):
            my_entry = Entry(self)
            my_entry.pack()  # grid(row = 4, column = x, pady = 20, padx = 20)
            my_entry.insert(0, "Enter_" + self.my_entry_labels[x])
            self.my_entries.append(my_entry)

 
        ################### 
        #  declare some  variables that will be used throughout
        ####

        #### the main datafile
        self.MOM_data = []
        
        self.user_INPATH = ""
        self.user_BURROW = ""
        self.data_DATE = ""

        self.data = [] ## for dataframes to be defined later
        self.MOM_data = []
        self.calibrations = []
        self.birds = []

        # Lists to accumulate info for different birds - Make these globals at top of app?
        # self.birds_datetime_starts
        # self.birds_datetime_ends
        # self.birds_data_means
        # self.birds_cal_means
        # self.birds_details

        ### now make them
        self.birds_datetime_starts = []
        self.birds_datetime_ends = []
        self.birds_data_means = []
        self.birds_cal_means = []
        self.birds_details = []
        
        self.cal_gradient = 0.0
        self.cal_intercept = 0.0
        self.baseline_cal_mean = 0.0
        self.cal_r_squared = 0.0

        ## globals from exec opne
        self.cal1_value = 0.0
        self.cal2_value = 0.0
        self.cal3_value = 0.0

        # Default figure view
        self.default_figure_width = 0
        self.default_figure_height = 0

        #### the datafile we will work with
        self.my_datafile = []

        ### the default datafile
        self.myDir = ""

        ## other defaults
        # font for welcome screen 
        self.my_font1 = ('courier', 10)

        # what files will we open?
        self.f_types = [ ('CSV files',"*.csv"), ('TXT',"*.txt")]
        self.f_name = ""

 


        self.set_defaults() #populate the default class variables

        print("here is directory" + self.myDir)

        

    def b1_clicked(self):
        # showinfo(title='MOM info', message='open the file', command = self.open_MOM_file)
        self.my_datafile = self.open_MOM_file("all")
        

    def button_clicked(self):
        # showinfo(title='Information', message='General Info')
        self.Calib_Handler()

    
    def b2_clicked(self):
        showinfo(title='Cut up a file', message='This will allow you to cut someday')
            
        # my_start = self.my_entries[0].get()
        # my_end = self.my_entries[1].get()
        # self.my_datafile = self.open_MOM_file("to_show")
        # hello = "Cut from "+str(my_start) + " to " + str(my_end)


      
    def set_defaults(self):
        # Assign saved user-specific values from the saved file
        #    which assigns values to cal1...value, default_figure.., and myDir
        #    these values are then assigned to class variables of same name
        exec(open("set_user_values.py").read())

        # From saved File: Default figure view
        self.cal1_value = cal1_value
        self.cal2_value = cal2_value
        self.cal3_value = cal3_value

        # From saved File: Default figure size
        self.default_figure_width = default_figure_width # 15
        self.default_figure_height = default_figure_height #10

        # From saved File: where to start looking
        self.myDir = myDir

    # def open_start_window(self):
        
    #     l1 = tk.Label(root,text='Read File & create DataFrame', width=60,font = self.my_font1)  
    #     l1.grid(row=1,column=1)

    #     ### put data under this?
    #     t1=tk.Text(root,width=40,height=50)
    #     t1.grid(row=5,column=1,padx=5, pady = 100)

    #     t2=tk.Text(root,width=40,height=50)
    #     t2.grid(row=5,column=0,padx=5, pady = 100)

    #     #### button for browsing files, but doing nothing
    #     b1 = tk.Button(root, text='Browse Files', 
    #     width=20,command = lambda:upload_file())
    #     b1.grid(row=2,column=1) 

    #     

    #     myButton = Button(root, text = "Cut File", command = myClick)
    #     myButton.grid(row = 2, column = 0, pady = 20)



    def open_MOM_file(self, to_show):

        f_types = [
            ('CSV files',"*.csv"),
            ('TXT',"*.txt")
            ]
        f_name = filedialog.askopenfilename(initialdir = self.myDir, 
            title = "Choose MOM File", 
            filetypes = f_types)

        self.user_INPATH = f_name
        
        # df=pd.read_csv(f_name) # create DataFrame
        df = pd.read_csv(f_name, header=None, names=["Measure", "Datetime"])
        str1="Rows:" + str(df.shape[0])+ "\nColumns:"+str(df.shape[1])+"\n"  #Minutes: "# +str(df.shape[0]/10.5/60)+"\n"
        str2="Minutes: " + str((df.shape[0])/10.5/60)+"\n"
        str3="Hours: " + str((df.shape[0])/10.5/60/60)+"\n"
        print("Working with: " + self.user_INPATH)
        print(str1)
        print(str2)
        print(str3)
        # t1.insert(tk.END, str1) # add to Text widget
        # t1.insert(tk.END, str2) # add to Text widget
        # t1.insert(tk.END, str3) # add to Text widget

        ##############don't do all of this, just get the datafile
        # df2 = pd.read_csv(f_name, header=None, names=["Measure", "Datetime"])

        if (to_show == "cut"):
            the_start = int(my_entries[0].get())
            the_end = int(my_entries[1].get())
            # df3 = df2.iloc[20000:20600] #20000-20600
            df = df.iloc[the_start:the_end]
            # df.plot()

        df.plot()
        plt.show()

        self.MOM_data = df

    def Calib_Handler(self):

        if(self.user_INPATH==""):  ## we have data to work with df.shape[0]
            showinfo(title='No data file', message='You have not chosen a data file!')
            
        else:
            str1= str(self.MOM_data.shape[0])  # self.l0.set(self.f_name)
            ### ToDo: update user on the window which file we are working with
            showinfo(title='Data file ready!', message= str1)
     

    def Bird_Handler(self):
        if(self.user_INPATH==""):  ## we have data to work with df.shape[0]
            showinfo(title='No data file', message='You have not chosen a data file!')
            
        else:
            str1= str(self.MOM_data.shape[0])  # self.l0.set(self.f_name)
            ### ToDo: update user on the window which file we are working with
            showinfo(title='Ready for choosing bird!', message= str1)

class Calibration_Widget(self, the_data):
    def __init__(self):

        #name the datafile
        self.data = self.the_data

        #setup dataframe
        self.calibrations = []

        #store the results
        self.cal_gradient = 0.0
        self.cal_intercept = 0.0
        self.baseline_cal_mean = 0.0
        self.cal_r_squared = 0.0

    def calib_add_baselines(self):
        baseline_cal_mean, baseline_cal_markers, baseline_cal_Good, axesLimits = getTracePointPair("Baseline")
        markers = baseline_cal_markers

if __name__ == "__main__":
  app = MOM_Viewer()
  app.mainloop()

