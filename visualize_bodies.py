import sys, os 
import numpy as np
import pandas as pd
import time
from multiprocessing.pool import Pool
import itertools
import pickle
import glob
from itertools import combinations
from itertools import product
import collections
import matplotlib.pyplot as plt



def array_col_to_bin(arr):
    # inputs= arr: binary array whose columns are to be turned into binary strings, 
    binarray = []
    for i1 in range(np.shape(arr)[1]):
        x = arr[:,i1]
        binarray += [''.join([str(xi) for xi in x])]
    return binarray


def assign_tissue_colors(celltypes):
    # output a dictionary that assigns different colors to different tissues
    palette = {}
    tissuetypes = [np.binary_repr(x,width=celltypes) for x in np.arange(1,2**celltypes,1)]
    p1 = np.random.randint(0,255,(len(tissuetypes),3))
    for i,t in enumerate(tissuetypes):
        palette[t] = p1[i]
    palette[np.binary_repr(0,width=celltypes)] = np.array([0,0,0])# empty locations are black
    return palette


def assign_tissue_colors_5celltypes():
    # output a dictionary that assigns different colors to different tissues
    palette = {}
    tissuetypes = [np.binary_repr(x,width=5) for x in np.arange(1,2**5,1)]
    p1 = np.random.randint(0,255,(len(tissuetypes),3))
    col = 255 * np.array([[0.5,0,0],[0,0.5,0],[0,0,0.5],[0.25,0.25,0],[0.25,0,0.25]])
    for i,t in enumerate(tissuetypes):
        t1 = np.array([int(i1) for i1 in t])
        palette[t] = t1@col#p1[i]
    palette[np.binary_repr(0,width=5)] = np.array([0,0,0])# empty locations are black
    return palette


def get_body_colors(body, loc, palette):
    loc_tissues = array_col_to_bin((body>0).astype(int))
    bodycol = np.ndarray(shape=(loc,loc,3),dtype=int)
    allloc = np.array([(i,j) for i in range(loc) for j in range(loc)])# location indices --> row and column on 2d grid
    for i0 in range(np.shape(body)[1]):
        # what is the corresp row and column in the 2-d grid for this location?
        bodycol[allloc[i0,0],allloc[i0,1],:] = palette[loc_tissues[i0]]
    return bodycol
    

def visualize_body_tissuetypes(BodyUpdates, celltypes, loc):
    # assign a different color to each tissuetype
    palette = assign_tissue_colors_5celltypes() #can change this to 'assign_tissue_colors(celltypes)' for other celltype numbers
    BodyCol = []
    for i1 in range(np.shape(BodyUpdates)[2]):
        # find out which location has what tissuetype at each timestep
        body = (BodyUpdates[:,:,i1]>0).astype(int)
        bodycol = get_body_colors(body, loc, palette)
        BodyCol += [bodycol]
        # save images
        #plt.figure();
        #plt.imshow(bodycol)
    return BodyCol




if __name__ == "__main__":
    palette = assign_tissue_colors_5celltypes()
