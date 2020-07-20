# CMR-Tools
This Python / VTK program calculates Global Circumferential Strain and Ejection Fraction from left ventricle CMR segmentations.
Plot curves of the values across time are displayed and saved as .png files, along with these, raw data is saved as .csv files. 
Finally a 3D render is displayed, the layers of the left ventricle  are color coded for the Strain case.

# How to run?
The user selects the folder containing all the 3D files in the heartbeat timeline to be analyzed (in .nii or .nii.gz), this selection is done on a menu containing all the folders 
inside the current folder or inside the specified one on the command line.

# Example command lines to run the script

    python CMRtools.py
    python CMRtools.py /home/User/myFolder/
