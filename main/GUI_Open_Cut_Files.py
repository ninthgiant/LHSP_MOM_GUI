
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
root.geometry('700x1000')

#default for my computer - do an ASK to set default?
myDir = "/Users/bobmauck/Dropbox/BIG Science/MOMs/2022_Stuff"

my_font1 = ('courier', 10)

l1 = tk.Label(root,text='Read File & create DataFrame',
    width=60,font=my_font1)  

l1.grid(row=1,column=1)

### put data under this?
t1=tk.Text(root,width=40,height=50)
t1.grid(row=5,column=1,padx=5, pady = 100)

t2=tk.Text(root,width=40,height=50)
t2.grid(row=5,column=0,padx=5, pady = 100)

#### button for browsing files, but doing nothing
b1 = tk.Button(root, text='Browse Files', 
   width=20,command = lambda:upload_file("all"))
b1.grid(row=2,column=1) 

#### default values for entry boxes
my_entry_labels = ["Starting_Point", "End_Point"]
#### where to store entered values
my_entries = []

for x in range(2):
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
    my_Cut.to_csv(my_Save_Dir + hello + ".TXT")

    t2.insert(tk.END, hello + "\n") # add to Text widget

myButton = Button(root, text = "Cut File", command = myClick)
myButton.grid(row = 2, column = 0, pady = 20)



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
    else:
        df.plot()

    plt.show()

    return df



root.mainloop()