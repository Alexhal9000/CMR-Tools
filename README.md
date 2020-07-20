# CMR-Tools
This Python / VTK program calculates Global Circumferential Strain and Ejection Fraction from left ventricle CMR segmentations.
Plot curves of the values across time are displayed and saved as .png files, along with these, raw data is saved as .csv files. 
Finally a 3D render is displayed, the layers of the left ventricle  are color coded for the Strain case.

# How to run?
The user selects the folder containing all the 3D segmented files in the heartbeat timeline to be analyzed (in .nii or .nii.gz), this selection is done on a menu containing all the folders 
inside the current folder or inside the specified one on the command line.

# Example command lines to run the script

    python CMRtools.py
    python CMRtools.py /home/User/myFolder/

# Data example:
## Raw file
<img src="https://github.com/Alexhal9000/CMR-Tools/blob/master/P75_4D.gif?raw=true" width="600" style="text-align: center; display: block;">
## Left ventricle segmented and rendered in 3D
<img src="https://github.com/Alexhal9000/CMR-Tools/blob/master/p75_112_aug.gif?raw=true" width="600" style="text-align: center;">
## Left ventricle circumferential strains visualized in 3D (red = high strain)
<img src="https://github.com/Alexhal9000/CMR-Tools/blob/master/strain112P75.gif?raw=true" width="600" style="text-align: center;">
## Left ventricle ejection fraction output curve
<img src="https://github.com/Alexhal9000/CMR-Tools/blob/master/Left_Ventricle_p75_112_aug_ejectionF_plot.png?raw=true" width="600" style="text-align: center;">
## Left ventricle circumferential strain output curves
<img src="https://github.com/Alexhal9000/CMR-Tools/blob/master/Myocardium_p75_112_aug_strain_plot.png?raw=true" width="600" style="text-align: center;">
