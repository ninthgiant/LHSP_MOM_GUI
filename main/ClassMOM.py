

import tkinter as tk
from tkinter import *
# from tkinter import ttk
# from tkinter import filedialog as fd
# from tkinter.messagebox import showinfo
from tkinter import filedialog

#### imports from LIam
import sys
import pandas as pd
import matplotlib.pyplot as pyplot
import matplotlib.backend_bases as backend
import numpy as np
import statistics
from scipy import stats 

# create the root window
root = tk()
root.title('Work with MOM datafile')
# root.resizable(True, True)
root.geometry('500x1000')
#default for my computer - do an ASK to set default?


class MOM_Night:

   
        

    def __init__(self, master):
        
        my_frame = Frame(master)
        my_frame.pack()

        self.myButton = Button(master, text = "OK", command = self.clicker)
        self.myButton.pack(pady = 20)

    def clicker(self):
        print("hey")

    def set_Default(self):
        self.myDir = "/Users/bobmauck/Dropbox/BIG Science/MOMs/2022_Stuff"

    def upload_file(self):
        f_types = [
            ('CSV files',"*.csv"),
            ('TXT',"*.txt")
            ]

        f_name = filedialog.askopenfilename(initialdir = myDir, 
            title = "Choose MOM File", 
            filetypes = f_types)

        self.set_Default()
        
        l1.config(text=f_name) # display the path 
        df=pd.read_csv(f_name) # create DataFrame
        str1="Rows:" + str(df.shape[0])+ "\nColumns:"+str(df.shape[1])+"\n"  #Minutes: "# +str(df.shape[0]/10.5/60)+"\n"
        str2="Minutes: " + str((df.shape[0])/10.5/60)+"\n"
        str3="Hours: " + str((df.shape[0])/10.5/60/60)+"\n"
        #print(str1)
        t1.insert(tk.END, str1) # add to Text widget
        t1.insert(tk.END, str2) # add to Text widget
        t1.insert(tk.END, str3) # add to Text widget

        data = pd.read_csv(f_name, header=None, names=["Measure", "Datetime"])
        # data["Datetime"] = pd.to_datetime(data["Datetime"])

        # str4 = "Min: "+ statistics.mean(data["Measures"])
        print(data["Measure"].describe())
        print(data["Measure"].mean())
        print(data["Measure"].quantile(0.99))
        # df.field_A.quantile(0.5) # same as median

        start_END = int(10.5 * 60 * 60)
        print(start_END)
        stop_END = int(data.shape[0]-start_END)
        print(stop_END)
        print("subset mean: ")
        # df2 = data.iloc[37800:400000]
        df2 = data.iloc[start_END:stop_END]
        print(df2["Measure"].describe())
        print("subset mean: ")
        print(df2["Measure"].mean())

        #data.plot(x = data["Measure"], y = data["Datetime"])
        #data.plot(y = 'Measure')
        # df2.plot()
        data.plot()
        pyplot.show()

        # for i in data["Measure"]:
        #    print(i)
        #   break


    
        # measures = data.loc[0:END,"Measure"]

        # str2 = sum(measures)/df.shape[0]

        # t1.insert(tk.end, str2)





Night = MOM_Night(root)



root.mainloop()
