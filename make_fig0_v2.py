import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import glob
import matplotlib.gridspec as gridspec



FOLDERNAME = '/Volumes/T7/SigMig/from_bigram2/'
PATHNAME = 'Dat_3_from_cslm/finalbody_richness_disperseness.pkl'

dict1 = pickle.load(open(os.path.join(FOLDERNAME,PATHNAME),'rb'))
FB_cellpotential_df = dict1['FB_cellpotential_df']



def get_cells_pot_hist_gridspec():
    fig = plt.figure(figsize=(9,3))
    gs = gridspec.GridSpec(1,44)
    gs.update(wspace=1,hspace=0.0, left=0.09,right=0.98,bottom=0.2)

    ax = [fig.add_subplot(gs[0:10]), fig.add_subplot(gs[15:24]), fig.add_subplot(gs[28:40]), fig.add_subplot(gs[40:42])]

    bins = np.arange(1,7,1)
    X = ['num_tissuecells','avpot_tissuecells']
    Xlabels = ['number of cell-types','average cell-fate potential']
    Xticks = [1,2,3,4,5]
    Xticklabels = [1,2,3,4,5]
    for i,x in enumerate(X):
        histx = np.histogram(FB_cellpotential_df[x],bins=bins,density=False)[0]
        ax[i].bar(bins[:-1],histx, width=1, color='xkcd:cerulean')
        ax[i].set_xlabel(Xlabels[i],fontsize=12)
        ax[i].set_xticks(Xticks)
        ax[i].set_xticklabels(Xticklabels,fontsize=12)
        ax[i].set_ylabel('number of tissues',fontsize=12)


    numcell_avpot_hist = np.zeros((5,5))

    for numcells in range(5):
        for avpot in range(5):
            numcell_avpot_hist[4-numcells,avpot] = sum((FB_cellpotential_df['num_tissuecells']==numcells+1)&
                    (FB_cellpotential_df['avpot_tissuecells']>avpot)&(FB_cellpotential_df['avpot_tissuecells']<=(avpot+1)))
        numcell_avpot_hist[4-numcells] = numcell_avpot_hist[4-numcells]/sum(numcell_avpot_hist[4-numcells])
    h2d = ax[2].imshow(numcell_avpot_hist,cmap='Blues',vmin=0,vmax=0.55)
    ax[2].set_xlabel('average cell-fate potential',fontsize=12)
    ax[2].set_ylabel('number of cell-types',fontsize=12)
    ax[2].set_xticks([0,1,2,3,4])
    ax[2].set_yticks([0,1,2,3,4])
    ax[2].set_xticklabels([1,2,3,4,5],fontsize=12)
    ax[2].set_yticklabels([5,4,3,2,1],fontsize=12)
    
    plt.colorbar(h2d,ax[3])
    ax[3].set_ylabel('proportion of tissues',fontsize=12)
    plt.savefig('Users/somya/Downloads/signals_migration/fig0_v3.png')
    return 0

def template_for_gridspec():

    fig = plt.figure(figsize=(5,5))
    gs = gridspec.GridSpec(2,26, height_ratios=[1.0, 1.1])
    gs.update(wspace=0.0 ,hspace=0.50)

    ax0 = [fig.add_subplot(gs[0,0:8]), fig.add_subplot(gs[0,9:17]), fig.add_subplot(gs[0,18:26])]

    ax1 = [fig.add_subplot(gs[1,1:7]), fig.add_subplot(gs[1,8:16]),
           fig.add_subplot(gs[1,17:22]), fig.add_subplot(gs[1,23:24])]

    CFR_df2 = CFR_df.loc[CFR_df['p_diff']>0]
    N = len(CFR_df2)
    X = [CFR_df2[x] + np.random.uniform(-0.4,0.4,N) for x in ['av_cell_potential', 'potential_disparity', 'num_stable_nbhd']]
    Y = CFR_df2['num_stem_cell_nbhd'] + np.random.uniform(-0.4,0.4,N)
    Xlabels = ['average cell-fate \npotential','cell-fate potential \ndisparity','number of \nstable neighborhoods']
    Xticks = [[0,2.5,5],[0,1,2,3],[0,10,20,30]] 
    for i, x in enumerate(X):
        ax0[i].scatter(x, Y, c='xkcd:faded blue',s=1, alpha=0.2)
        ax0[i].set_xlabel(Xlabels[i])
        ax0[i].set_xticks(Xticks[i])

    ax0[0].set_ylabel('number of neighborhoods\n with stem-like cells')
    ax0[1].set_yticklabels([])
    ax0[2].set_yticklabels([])

    stemcellniche_pstable, stemcellniche_pdiff, stemcellniche_density = get_stemcellniche_param_histograms()
    max3 = max(np.max(stemcellniche_density,axis=0))
    max2 = max(np.max(stemcellniche_pdiff,axis=0))
    max1 = max(np.max(stemcellniche_pstable,axis=0))
    figmax = np.ceil(max([max1,max2,max3]))

    ax1[0].imshow(stemcellniche_pstable,vmin=0,vmax=figmax)
    #ax10.set_yticks([0,1,2,3,4,5,6]);
    #ax10.set_yticklabels([17.5,15,12.5,10,7.5,5,2.5])
    ax1[0].set_yticks([1,3,5])
    ax1[0].set_yticklabels([15,10,5])
    
    ax1[0].set_xticks([0,1,2]);ax1[0].set_xticklabels([0.3,0.5,0.7])
    ax1[0].set_ylabel('number of neighborhoods\n with stem-like cells')
    ax1[0].set_xlabel('P-stable')

    ax1[1].imshow(stemcellniche_pdiff,vmin=0,vmax=figmax)
    ax1[1].set_yticklabels([])
    ax1[1].set_xticks([0,1,2,3,4]);ax1[1].set_xticklabels([0,0.3,0.5,0.7,1])
    ax1[1].set_xlabel('P-diff')

    im = ax1[2].imshow(stemcellniche_density,vmin=0,vmax=figmax)
    ax1[2].set_yticklabels([])
    ax1[2].set_xticks([0,1,2]);ax1[2].set_xticklabels([0.3,0.5,0.7])
    ax1[2].set_xlabel('P-den')
    fig.colorbar(im,cax=ax1[3])

    return 0




