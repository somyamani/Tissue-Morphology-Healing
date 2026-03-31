import sys, os
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import time
import itertools
from multiprocessing import Pool
import pickle
import glob


dict1 = pickle.load(open('Dat_3_from_cslm/finalbody_richness_disperseness.pkl','rb'))
df = dict1['finalbody_df']

top_samples = df['sector_assignments'] == 'top'
center_samples = df['sector_assignments'] == 'center'
bottom_samples = df['sector_assignments'] == 'bottom'
tail_samples = df['sector_assignments'] == 'tail'

sector_boundaries = dict1['sectordict']
c_tail = sector_boundaries['c_tail']
m_top = sector_boundaries['m_top']
c_top = sector_boundaries['c_top']
m_center = sector_boundaries['m_center']
c_center = sector_boundaries['c_center']


topcolor = 'xkcd:sand yellow'
tailcolor = 'xkcd:dusty purple'
centercolor = 'xkcd:greenish beige'
bottomcolor = 'xkcd:squash'


c1 = 'xkcd:vivid blue'
c2 = 'xkcd:burnt red'
c3 = 'k'
linecol = [c1,c2,c3]

startcolor = 'xkcd:sky blue'
endcolor = 'xkcd:pale red'



def plot_tissue_growth_examples(files):
    
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(5,4))

    plt.scatter(df['PC1'][top_samples]+np.random.uniform(-0.005,0.005,(sum(top_samples))),df['PC2'][top_samples]+np.random.uniform(-0.005,0.005,(sum(top_samples))),s=0.5,c=topcolor, zorder=0)
    plt.scatter(df['PC1'][tail_samples]+np.random.uniform(-0.005,0.005,(sum(tail_samples))),df['PC2'][tail_samples]+np.random.uniform(-0.005,0.005,(sum(tail_samples))),s=0.5,c=tailcolor, zorder=0)
    plt.scatter(df['PC1'][center_samples]+np.random.uniform(-0.005,0.005,(sum(center_samples))),df['PC2'][center_samples]+np.random.uniform(-0.005,0.005,(sum(center_samples))),s=0.5,c=centercolor, zorder=0)
    plt.scatter(df['PC1'][bottom_samples]+np.random.uniform(-0.005,0.005,(sum(bottom_samples))),df['PC2'][bottom_samples]+np.random.uniform(-0.005,0.005,(sum(bottom_samples))),s=0.5,c=bottomcolor, zorder=0)

    plt.plot([-3,c_tail],[m_top*(-3) + c_top, m_top*c_tail + c_top], linewidth=1,c='xkcd:dark grey',linestyle='-', zorder=1)#top
    plt.plot([c_tail,c_tail],[-3,3],linewidth=1,c='xkcd:dark grey',linestyle='-', zorder=1)#tail
    plt.plot([-3,c_tail],[m_center*(-3) + c_center, m_center*c_tail + c_center],linewidth=1,c='xkcd:dark grey',linestyle='-', zorder=1)#center
    plt.xlim((-3,7.2)); plt.ylim((-2.7,3))
    
    for i,filename in enumerate(files):
        growdf = pickle.load(open(filename,'rb'))
        X_PC = growdf[['PC1','PC2']].to_numpy()

        plt.plot(X_PC[:,0], X_PC[:,1],linecol[i], linewidth=1.5, linestyle='-',alpha=0.7, zorder=2)

        plt.scatter(X_PC[0,0].flatten(), X_PC[0,1].flatten(), s=25, c = startcolor, zorder=3, edgecolor='k')
        plt.scatter(X_PC[-1,0].flatten(), X_PC[-1,1].flatten(), s=25, c = endcolor, zorder=3, edgecolor='k')

    plt.subplots_adjust(left=0.05,right=0.99,top=0.99,bottom=0.06)
    return 0


def plot_toptraj(numfiles):
#   allfiles = np.array(glob.glob('/home/somya/signals_migration/Dat_2_from_cslm/BU_files/top_BU_files/*.pkl'))
#   chosen_files = allfiles[np.random.randint(0,len(allfiles),numfiles)]
    
#   chosen_files = np.array(['/home/somya/signals_migration/Dat_2_from_cslm/BU_files/top_BU_files/AV_rep099_param00065.pkl',
#      '/home/somya/signals_migration/Dat_2_from_cslm/BU_files/top_BU_files/AV_rep019_param00116.pkl'])

    chosen_files = ['Dat_3_from_cslm/BU_files/top/AV_rep078_param00041.pkl',
            'Dat_3_from_cslm/BU_files/top/AV_rep049_param00077.pkl',
            'Dat_3_from_cslm/BU_files/top/AV_rep058_param00077.pkl']

    plot_tissue_growth_examples(chosen_files)
    plt.savefig('Dat_3_from_cslm/fig2/top_traj.png')
    return chosen_files


def plot_centertraj(numfiles):
#   allfiles = np.array(glob.glob('/home/somya/signals_migration/Dat_2_from_cslm/BU_files/center_BU_files/*.pkl'))
#   chosen_files = allfiles[np.random.randint(0,len(allfiles),numfiles)]
    

    chosen_files = ['Dat_3_from_cslm/BU_files/center/AV_rep044_param00092.pkl',
            'Dat_3_from_cslm/BU_files/center/AV_rep077_param00059.pkl',
            'Dat_3_from_cslm/BU_files/center/AV_rep085_param00107.pkl']

    plot_tissue_growth_examples(chosen_files)
    plt.savefig('Dat_3_from_cslm/fig2/center_traj.png')
    return chosen_files


def plot_bottomtraj(numfiles):
#   allfiles = np.array(glob.glob('/home/somya/signals_migration/Dat_2_from_cslm/BU_files/bottom_BU_files/*.pkl'))
#   chosen_files = allfiles[np.random.randint(0,len(allfiles),numfiles)]
    

    chosen_files = ['Dat_3_from_cslm/BU_files/bottom/AV_rep096_param00009.pkl',
            'Dat_3_from_cslm/BU_files/bottom/AV_rep015_param00052.pkl',
            'Dat_3_from_cslm/BU_files/bottom/AV_rep089_param00007.pkl']


    plot_tissue_growth_examples(chosen_files)
    plt.savefig('Dat_3_from_cslm/fig2/bottom_traj.png')
    return chosen_files


def plot_tailtraj(numfiles):
#   allfiles = np.array(glob.glob('/home/somya/signals_migration/Dat_2_from_cslm/BU_files/tail_BU_files/*.pkl'))
#   chosen_files = allfiles[np.random.randint(0,len(allfiles),numfiles)]

    chosen_files = ['Dat_3_from_cslm/BU_files/tail/AV_rep047_param00091.pkl',
            'Dat_3_from_cslm/BU_files/tail/AV_rep098_param00058.pkl',
            'Dat_3_from_cslm/BU_files/tail/AV_rep081_param00008.pkl']

    plot_tissue_growth_examples(chosen_files)
    plt.savefig('Dat_3_from_cslm/fig2/tail_traj.png')
    return chosen_files




