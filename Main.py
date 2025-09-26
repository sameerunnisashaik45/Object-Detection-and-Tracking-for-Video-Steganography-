import cv2 as cv
from numpy import matlib
from ElGamal_Arnold import Elgamal_Arnold
from FANO import FANO
from GAO import GAO
from Global_Vars import Global_Vars
from Image_Results import *
from LOA import LOA
from Model_DWT import DWT
from Model_EfficientDet import Model_EfficientDet
from Plot_Results import *
from Proposed import Proposed
from SFOA import SFOA
from objfun_feat import objfun
import os
import warnings

warnings.filterwarnings('ignore')


def ReadImage(Filename):
    image = cv.imread(Filename)
    image = np.uint8(image)
    if len(image.shape) > 2:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image = cv.resize(image, (256, 256))
    return image


# Read Dataset
an = 0
if an == 1:
    Video = []
    path = './Dataset/train'
    out_dir = os.listdir(path)
    c = 0
    for i in range(len(out_dir)):
        folder = path + '/' + out_dir[i]
        in_dir = os.listdir(folder)
        for j in range(len(in_dir)):
            Filename = folder + '/' + in_dir[j]
            for f in range(2):
                data = cv.VideoCapture(Filename)
                ret, frame = data.read()
                if ret:
                    width = 512
                    height = 512
                    dim = (width, height)
                    resized_video = cv.resize(frame, dim)
                    Video.append(resized_video)
                    c += 1

    np.save('Image.npy', np.asarray(Video))

# Object Tracking and Detection
an = 0
if an == 1:
    Image = np.load('Image.npy', allow_pickle=True)
    Frames = Model_EfficientDet(Image)
    np.save('Detected_Image.npy', np.asarray(Frames))

# Optimization for Region Selection
an = 0
if an == 1:
    Image = np.load('Detected_Image.npy', allow_pickle=True)
    Global_Vars.Image = Image
    Npop = 10
    Chlen = 5
    xmin = matlib.repmat(1 * np.ones((1, Chlen)), Npop, 1)
    xmax = matlib.repmat(10 * np.ones((1, Chlen)), Npop, 1)
    initsol = np.zeros(xmin.shape)
    for i in range(xmin.shape[0]):
        for j in range(xmin.shape[1]):
            initsol[i, j] = np.random.uniform(xmin[i, j], xmax[i, j])
    fname = objfun
    Max_iter = 50

    print("FANO...")
    [bestfit1, fitness1, bestsol1, time1] = FANO(initsol, fname, xmin, xmax, Max_iter)  # FANO

    print("LOA...")
    [bestfit2, fitness2, bestsol2, time2] = LOA(initsol, fname, xmin, xmax, Max_iter)  # LOA

    print("GAO...")
    [bestfit4, fitness4, bestsol4, time3] = GAO(initsol, fname, xmin, xmax, Max_iter)  # GAO

    print("SFOA...")
    [bestfit3, fitness3, bestsol3, time4] = SFOA(initsol, fname, xmin, xmax, Max_iter)  # SFOA

    print("Proposed...")
    [bestfit5, fitness5, bestsol5, time5] = Proposed(initsol, fname, xmin, xmax, Max_iter)  # Improved SFOA

    BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
    np.save('BestSol.npy', BestSol)

# Read Secrete Image
an = 0
if an == 1:
    path = './Secret Image/Org_1.jpg'
    Image = ReadImage(path)
    np.save('Secret_Image.npy', Image)

# Encryption
an = 0
if an == 1:
    Image = np.load('Secret_Image.npy', allow_pickle=True)
    En_Img, De_Img = Elgamal_Arnold(Image)
    np.save('Encrypted_Image.npy', En_Img)

# Cryptography and Steganography
an = 0
if an == 1:
    Image = np.load('Image.npy', allow_pickle=True)
    Encey = np.load('Encrypted_Image.npy', allow_pickle=True)
    Stego, Reconstruct = DWT(Image, Encey)
    np.save('Stego_image.npy', Stego)
    np.save('Reconstruct.npy', Reconstruct)

plotConvResults()
plot_Results()
Table()
Encryt_Results()
Object_Plots_Results()
MAP()
Tracking_Image()
Image_Results()
Sample_Images()
