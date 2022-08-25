# LHSP MOM GUI
#
#   8/25/2022 RAM
#       Transitioned original lhsp_mom_viewer to a full GUI version
#       
#   MOM_Processor_v02.py is the version to use as of 8/25
#       Opens window with 4 buttons. 
#           Browse button:  lets you view any file then close it and nothing esle is done
#           Process MOM button: duplicates functionality of original mom_viewer program, but all via GUI
#           Cut_Calc_Mean button: runs script that automates choice of window to use to calculate bird weight-
#                   uses the input boxes on the right side (defaults: 0.015 for threshold change and 5 for window) which can be changed
#                   Don't use this unless you know have looked at the code and see what it is doing
#           Multi Files button: applies the calculations in Cut_Calc_Mean across multiple files and multiple thresholds and windows
#                   Don't use this unless you have looked at the code

