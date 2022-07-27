
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


# create the root window
root = tk.Tk()
root.title('Work with MOM datafile')
# root.resizable(True, True)
root.geometry('700x1000')
#default for my computer - do an ASK to set default?
myDir = "/Users/bobmauck/Dropbox/BIG Science/MOMs/2022_Stuff"

my_font1 = ('courier', 10)

l1 = tk.Label(root,text='Read File & create DataFrame',
    width=60,font=my_font1)  

l1.grid(row=1,column=1)

b1 = tk.Button(root, text='Browse Files', 
   width=20,command = lambda:upload_file())

b1.grid(row=2,column=1) 

### put data under this?
t1=tk.Text(root,width=40,height=50)
t1.grid(row=3,column=1,padx=5, pady = 100)

### entry box ### doesn't work right now
e = Entry(root, width = 30)
# e.grid(row = 5, column = 3, padx = 5, pady = 10)
e.pack()

def upload_file():
    f_types = [
        ('CSV files',"*.csv"),
        ('TXT',"*.txt")
        ]
    f_name = filedialog.askopenfilename(initialdir = myDir, 
        title = "Choose MOM File", 
        filetypes = f_types)

    
    l1.config(text=f_name) # display the path 
    df=pd.read_csv(f_name) # create DataFrame
    str1="Rows:" + str(df.shape[0])+ "\nColumns:"+str(df.shape[1])+"\n"  #Minutes: "# +str(df.shape[0]/10.5/60)+"\n"
    str2="Minutes: " + str((df.shape[0])/10.5/60)+"\n"
    str3="Hours: " + str((df.shape[0])/10.5/60/60)+"\n"
    #print(str1)
    t1.insert(tk.END, str1) # add to Text widget
    t1.insert(tk.END, str2) # add to Text widget
    t1.insert(tk.END, str3) # add to Text widget

    df2 = pd.read_csv(f_name, header=None, names=["Measure", "Datetime"])


    df3 = df2.iloc[20000:20600] #20000-20600

    df3.plot()
    plt.show()



root.mainloop()