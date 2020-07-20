# MDSC 689.03
# Advanced Medical Image Processing
#
# Final Project - CMR Left Ventricle Evaluation Tool.
# Alejandro_Gutierrez
# April 07, 2020
# ----------------------------------------------------------------------------------------
#
# This program calculates Global Circumferential Strain and Ejection Fraction from
# left ventricle CMR segmentations. Plot curves of the values across time are displayed
# and saved as .png files, along with these, raw data is saved as .csv files.
# Finally a 3D render is displayed, the layers of the left ventricle  are color coded
# for the Strain case.
#
# The user selects the folder containing the files to be analyzed, this selection is done
# on a menu containing all the folders inside the current folder or inside the specified
# one on the command line.
#
# Example command lines to run the script
#
#    python CMRtools.py
#    python CMRtools.py /home/User/myFolder/
#
# -----------------------------------------------------------------------------------------

# Import libraries.
import vtk
import os
import sys
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
from skimage import feature
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
import time

myFileName = ""
myDirName = ""
frameIndex = 0


# ----------------------------------------------------------
#   Define read_medical_file_4d() function.
#   Reads files and folders from folder and displays them in a menu.
#   The selected folder is scanned and all items inside are
#   returned as an array of vtk reader objects.
# ----------------------------------------------------------

def read_medical_file_4d(**kwargs):
    if "msg" in kwargs:
        damsg = kwargs.get("msg")
        print(damsg)

    global myDirName
    # Scan files in current folder or in assigned one if argument is present.
    if len(sys.argv) > 1:
        myDir = sys.argv[1]
        myDirName = myDir
    else:
        myDir = os.curdir

    items = os.listdir(myDir)  # Fetch all items.
    fileList = []  # Initialize empty files array.

    # Discriminate files of unrelated formats.
    for names in items:
        if "." not in names:
            fileList.append(names)

    # Make a visual list of the files with assigned numbers to be displayed in terminal.
    cnt = 0
    for fileName in fileList:
        print(str(cnt) + " - " + fileName)
        cnt = cnt + 1

    # Wait for user input, number selects file from displayed menu.
    selection = input("Enter file number and hit enter: ")
    selectedFile = fileList[int(selection)]
    global myFileName
    myFileName = selectedFile

    items = os.listdir(myDirName + selectedFile)  # Fetch all items.
    fileList = []  # Initialize empty files array.

    # Discriminate files of unrelated formats.
    for names in items:
        if names.endswith(".nii") or names.endswith(".nii.gz"):
            fileList.append(myDirName + selectedFile + "/" + names)
    fileList = sorted(fileList)
    print(str(fileList))

    readerArray = []
    n = 0
    for dafile in fileList:
        # Read selected image from its file
        readerArray.append(vtk.vtkNIFTIImageReader())
        readerArray[n].SetFileName(dafile)
        readerArray[n].Update()
        n += 1

    return readerArray


# ----------------------------------------------------------------------------------
#   Define marching_cubes(source) function.
#   Takes a vtk image and performs a series of filters to render marching cubes.
#   It returns a 3D polygon vtk object.
# ----------------------------------------------------------------------------------

def marching_cubes(source):

    # Render in 3D with marching cubes.
    marchingCubes = vtk.vtkMarchingCubes()
    marchingCubes.SetInputData(source)
    marchingCubes.ComputeNormalsOn()
    marchingCubes.SetValue(0, 1.0)

    # Smooth the generated surface.
    smoothFilter = vtk.vtkSmoothPolyDataFilter()
    smoothFilter.SetInputConnection(marchingCubes.GetOutputPort())
    smoothFilter.SetNumberOfIterations(14)
    smoothFilter.SetRelaxationFactor(0.4)
    smoothFilter.FeatureEdgeSmoothingOff()
    smoothFilter.BoundarySmoothingOn()
    smoothFilter.Update()

    # Update normals on newly smoothed polyData.
    normalGenerator = vtk.vtkPolyDataNormals()
    normalGenerator.SetInputConnection(smoothFilter.GetOutputPort())
    normalGenerator.ComputePointNormalsOn()
    normalGenerator.ComputeCellNormalsOn()
    normalGenerator.Update()

    return normalGenerator


# ----------------------------------------------------------
#   Define change_frame(obj, ev) function.
#   Browses back and forth through the volumes belonging
#   to different time points with the Left and Right arrows.
# ----------------------------------------------------------

def change_frame(obj, ev):
    # Get the pressed key.
    key = obj.GetKeySym()

    global frameIndex

    if key == "Left" and frameIndex > 0:
        if modeSelection == "1":
            renderer.RemoveVolume(volume[frameIndex])
            renderer.AddVolume(volume[frameIndex - 1])
        else:
            renderer.RemoveActor(smoothedActor[frameIndex])
            renderer.AddActor(smoothedActor[frameIndex - 1])
        textActor.SetInput("Frame: " + str(frameIndex - 1))
        frameIndex = frameIndex - 1
    elif key == "Right" and frameIndex < frameNum - 1:
        if modeSelection == "1":
            renderer.RemoveVolume(volume[frameIndex])
            renderer.AddVolume(volume[frameIndex + 1])
        else:
            renderer.RemoveActor(smoothedActor[frameIndex])
            renderer.AddActor(smoothedActor[frameIndex + 1])
        textActor.SetInput("Frame: " + str(frameIndex + 1))
        frameIndex = frameIndex + 1
    else:
        if modeSelection == "1":
            renderer.RemoveVolume(volume[frameIndex])
            frameIndex = 0
            renderer.AddVolume(volume[frameIndex])
        else:
            renderer.RemoveActor(smoothedActor[frameIndex])
            frameIndex = 0
            renderer.AddActor(smoothedActor[frameIndex])
        textActor.SetInput("Frame: " + str(frameIndex))

    renderWindow.Render()


# ----------------------------------------------------------
#   Define vtk_medical_to_numpy(input_reader) function.
#   Takes a vtk file reader as input.
#   The converted image is returned as a numpy array.
# ----------------------------------------------------------

def vtk_medical_to_numpy(input_reader):
    vtkImage = input_reader.GetOutput()
    myPoints = vtkImage.GetPointData().GetScalars()
    npArray = vtk_to_numpy(myPoints)
    dim = vtkImage.GetDimensions()
    # The array's shape is the same as the Dimensions returned by VTK but inverted.
    npArray = npArray.reshape(dim[::-1])
    return npArray


# ----------------------------------------------------------
#   Define numpy_to_medical_vtk(myarray, params) function.
#   Takes a numpy array and a list of the spacing and
#   origin parameters as input.
#   The converted image is returned as a vtkImageData object.
# ----------------------------------------------------------

def numpy_to_medical_vtk(myarray, params):
    imageData = vtk.vtkImageData()
    depthArray = numpy_to_vtk(myarray.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
    imageData.SetDimensions(myarray.shape[2], myarray.shape[1], myarray.shape[0])
    imageData.SetSpacing(params[0][0], params[0][1], params[0][2])
    imageData.SetOrigin(params[1][0], params[1][1], params[1][2])
    imageData.GetPointData().SetScalars(depthArray)
    return imageData


# ----------------------------------------------------------
#   Define countBorderPixels(image) function.
#   Takes a numpy array as input.
#   For every slice a skeletonize function extracts a single
#   pixel wide edge from the result of a canny edge detection
#   function ran at a gaussian filtered slice.
#   Then only the smallest structure from each slice is
#   extracted, this is the inner myocardial wall.
#   The pixels are counted for each slice and an array of
#   these numbers is returned.
# ----------------------------------------------------------

def countBorderPixels(image):
    strains = []
    for slice in image:
        edges = skeletonize(feature.canny(ndi.gaussian_filter(slice, 1), 4))
        # edgesmid = skeletonize(slice)
        # plt.figure(figsize=(8, 8))
        # plt.imshow(edges, cmap=plt.cm.gray)
        # plt.show()
        # Smallest structure extraction
        s = ndimage.generate_binary_structure(2, 2)
        labeled_array, num_features = ndimage.label(edges, structure=s)
        if num_features == 2:
            smallestN = 0
            sizeN = 999999
            for n in range(num_features):
                component = np.where(labeled_array==n+1, 1.0, 0.0)
                dasum = component.sum()
                if dasum < sizeN:
                    smallestN = n+1
                    sizeN = dasum
            edges = np.where(labeled_array==smallestN, 1.0, 0.0)
            # plt.figure(figsize=(8, 8))
            # plt.imshow(edges, cmap=plt.cm.gray)
            # plt.show()
            strains.append(edges.sum())
        else:
            strains.append(0.0)
    return strains


# ----------------------------------------------------------
#   Define GCStrain(readerArray) function.
#   Takes an array of vtk reader objects, extracts them and
#   converts them into numpy arrays. These are fed to the
#   countBorderPixels function and stores the resulting
#   arrays of measurements into a larger array containing
#   all the time frames. It cleans the array from aberrant
#   data and calculates the strain values for each slice
#   separately. Results are plotted and saved.
#   Original files are overwritten with strain values
#   and returned for visualization as vtkImageData arrays.
# ----------------------------------------------------------

def GCStrain(readerArray):
    outArray = []
    outStrains = []
    global maxStrain

    # Read files and compute inner wall measurements.
    for myReader in readerArray:
        img = vtk_medical_to_numpy(myReader)
        frame = countBorderPixels(img)
        outStrains.append(frame)

    # Set slices to 0 if any item in the time series is aberrant.
    outStrains = np.array(outStrains)
    bannedSlices = []
    for timePoint in outStrains:
        nn = 0
        for slicePoint in timePoint:
            if slicePoint == 0.0:
                bannedSlices.append(nn)
            if nn in bannedSlices:
                outStrains[:, nn] = 0.0
            nn += 1

    # Transpose to calculate Strain values from each slice across time.
    outStrainsB = np.copy(np.transpose(outStrains))
    ss = 0
    for slicePoint in outStrainsB:
        tt = 0
        maxSize = slicePoint.max()
        if slicePoint[0] > 0.0:
            for timePointb in slicePoint:
                outStrainsB[ss, tt] = float(((timePointb-maxSize)/maxSize)*100.0)
                tt += 1
        ss += 1

    # Find the largest deformation and mean, print results.
    idx = np.unravel_index(np.nanargmin(outStrainsB, axis=None), outStrainsB.shape)
    idxb = np.unravel_index(np.nanargmax(outStrainsB[idx[0]], axis=None), outStrainsB[idx[0]].shape)
    maxStrain = outStrainsB[idx]
    averageMax = np.nanmin(outStrainsB, axis=1)
    print("\nThe largest deformation was found on slice " + str(idx[0]))
    print("\tEnd Diastolic: " + str(outStrains[idxb[0], idx[0]]) + " pixels on frame " + str(idxb[0]))
    print("\tEnd Systolic: " + str(outStrains[idx[1], idx[0]]) + " pixels on frame " + str(idx[1]))
    print("\tStrain of " + str(round(maxStrain, 2)) + "%")
    print("\nAverage Strain " + str(round(averageMax.mean(), 2)) + "%")

    # Plot and save the results as .png and save the raw data to a .csv file.
    plt.style.use('dark_background')
    plt.figure(figsize=(11, 6))
    ax = plt.subplot2grid((1, 5), (0, 0), rowspan=1, colspan=4)
    sn = 0
    for daplot in outStrainsB:
        if np.mean(daplot) < (-0.5):
            plt.plot(ndi.gaussian_filter1d(daplot, 1), label="Slice " + str(sn))
        sn += 1
    global myFileName
    print("\nPlot and strain data saved:")
    print("\t"+myFileName + "_strain_plot.png")
    print("\t"+myFileName + "_strain.csv")
    np.savetxt(myFileName + '_strain.csv', outStrainsB, delimiter=',', fmt='%f')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Strain (deformation ratio %)")
    ax.set_title("Computed Strains of " + myFileName)
    plt.savefig(myFileName + "_strain_plot.png")
    plt.show()

    # Fetch the source images and overwrite them with the strain values.
    nn = 0
    for myReaderb in readerArray:
        img = vtk_medical_to_numpy(myReaderb)
        parameters = [(myReaderb.GetNIFTIHeader().GetPixDim(1), myReaderb.GetNIFTIHeader().GetPixDim(2),
                       myReaderb.GetNIFTIHeader().GetPixDim(3)), (0, 0, 0)]
        # replace data with strains
        newimg = np.zeros(img.shape)
        ss = 0
        for newslice in newimg:
            newimg[ss] = np.where(img[ss] > 0.5, float(np.abs(outStrainsB[ss, nn])), -100.0)
            ss += 1
        outArray.append(numpy_to_medical_vtk(newimg, parameters))
        nn += 1

    return outArray


# ----------------------------------------------------------
#   Define ejectionFraction(readerArray) function.
#   Takes an array of vtk reader objects, extracts them and
#   converts them into numpy arrays. The first slices are
#   ignored assuming the mitral valve may cause artifacts,
#   then voxels are counted as a volume measurement and
#   stored in an array. Largest and smallest volumes
#   are found to compute ejection fractions. Results
#   are plotted and stored.
# ----------------------------------------------------------

def ejectionFraction(readerArray):
    # Fetch files and convert to numpy for sum.
    outEF = []
    for myReader in readerArray:
        img = vtk_medical_to_numpy(myReader)
        vol = img[2:img.shape[0] - 1].sum()
        outEF.append(vol)
    outEF = np.array(outEF)

    # Find maximum and minimum volumes to calculate Ejection Fraction.
    volmax = outEF.max()
    maxIndex = np.unravel_index(np.nanargmax(outEF, axis=None), outEF.shape)
    volmin = outEF.min()
    minIndex = np.unravel_index(np.nanargmin(outEF, axis=None), outEF.shape)
    EF = ((volmax - volmin)/volmax)*100
    print("The maximum volume is " + str(round(volmax, 2)) + " found at frame " + str(maxIndex[0]))
    print("The minimum volume is " + str(round(volmin, 2)) + " found at frame " + str(minIndex[0]))
    print("The Ejection fraction is " + str(round(EF, 2)) + "%")
    outEFB = []
    for eachvol in outEF:
        outEFB.append(((volmax - eachvol)/volmax)*100)
    outEFB = np.array(outEFB)

    # Plot and save results as .png and .csv.
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(outEFB)
    plt.plot(ndi.gaussian_filter1d(outEFB, 1))
    global myFileName
    print("\nPlot and ejection fraction data saved:")
    print("\t"+myFileName + "_ejectionF_plot.png")
    print("\t"+myFileName + "_ejectionF.csv")
    np.savetxt(myFileName + '_ejectionF.csv', outEFB, delimiter=',', fmt='%f')
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Ejection Fraction (volume ratio %)")
    ax.set_title("Computed Ejection Fraction of " + myFileName)
    plt.savefig(myFileName + "_ejectionF_plot.png")
    plt.show()


# ------------------------Main Code---------------------------------------

maxStrain = 1.0
# Initialize renderer and window.
renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindow.SetSize(1000, 1000)
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(renderWindow)

# Ask user to choose which calculation needs to be executed.
modeSelection = input("Enter 1 for Myocardial Wall Strain or 2 for Ejection Fraction: ")

if modeSelection == "1":
    # User selects data and feeds it to the Strain function.
    myReaderArray = read_medical_file_4d(msg="Select a folder with myocardial wall segmentations:")
    volume = []
    rayCastMapper = []
    n = 0
    strainArray = GCStrain(myReaderArray)

    # Create a Volume Property object.
    volumeProperty = vtk.vtkVolumeProperty()

    # Control opacity with a Piece wise function.
    compositeOpacity = vtk.vtkPiecewiseFunction()
    compositeOpacity.AddPoint(-0.1, 0.0)
    compositeOpacity.AddPoint(0.0, 0.05)
    compositeOpacity.AddPoint(abs(maxStrain), 0.3)
    volumeProperty.SetScalarOpacity(compositeOpacity)

    # Control the color table with Color Transfer Function.
    color = vtk.vtkColorTransferFunction()
    color.AddRGBPoint(0.0, 0.0, 0.0, 1.0)
    color.AddRGBPoint(abs(maxStrain), 1.0, 0.0, 0.0)
    volumeProperty.SetColor(color)

    for myReader in strainArray:
        # Render in 3D Ray Casting.
        # Create Ray Cast Mapper.
        rayCastMapper.append(vtk.vtkGPUVolumeRayCastMapper())
        rayCastMapper[n].SetInputData(myReader)

        # Create the volume.
        volume.append(vtk.vtkVolume())
        volume[n].SetMapper(rayCastMapper[n])
        volume[n].SetProperty(volumeProperty)

        n += 1

    frameNum = len(volume)

else:
    # Ejection Fraction
    myReaderLVArray = read_medical_file_4d(msg="Select a folder with Left Ventricle segmentations:")
    smoothedMapper = []
    smoothedActor = []
    ejectionFraction(myReaderLVArray)
    n = 0
    for myLVReader in myReaderLVArray:

        marchingCubesPrep = marching_cubes(myLVReader.GetOutput())
        # First Mapper.
        smoothedMapper.append(vtk.vtkPolyDataMapper())
        smoothedMapper[n].SetInputConnection(marchingCubesPrep.GetOutputPort())
        smoothedMapper[n].ScalarVisibilityOff()

        # First Actor
        smoothedActor.append(vtk.vtkActor())
        smoothedActor[n].SetMapper(smoothedMapper[n])
        smoothedActor[n].GetProperty().SetColor(0.64, 0.11, 0.18)
        smoothedActor[n].GetProperty().SetOpacity(1.0)

        n += 1

    frameNum = len(smoothedActor)

# Add volume to renderer.
renderer.ResetCamera()
cam = renderer.GetActiveCamera()
cam.SetPosition(-223.234, 309.115, -174.292)
cam.SetViewAngle(30)
cam.SetViewUp(0.506049, 0.00223801, -0.862502)
cam.SetFocalPoint(151.079, 183.345, 45)
renderer.Render()
if modeSelection == "1":
    renderer.AddVolume(volume[frameIndex])
else:
    renderer.AddActor(smoothedActor[frameIndex])

# Add current frame text actor.
textActor = vtk.vtkTextActor()
textActor.SetInput("Frame: " + str(frameIndex))
textActor.SetPosition(30, 30)
textActor.GetTextProperty().SetFontSize(24)
textActor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
textActor.GetTextProperty().SetFontFamilyToCourier()
renderer.AddActor2D(textActor)

# Render
renderWindow.Render()
interactor.AddObserver('KeyPressEvent', change_frame, 1.0)
interactor.Start()

windowToImageFilter = vtk.vtkWindowToImageFilter()
windowToImageFilter.SetInput(renderWindow)
windowToImageFilter.SetInputBufferTypeToRGBA()
windowToImageFilter.ReadFrontBufferOff()
windowToImageFilter.Update()

writer = vtk.vtkPNGWriter()
writer.SetFileName(myFileName+".png")
writer.SetInputConnection(windowToImageFilter.GetOutputPort())
writer.Write()


