import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


dict1 = pickle.load(open('/Users/somya/Documents/SIGMIG_from_Bigram2/Dat_3/finalbody_richness_disperseness.pkl','rb'))
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

def plot_recovery():
    fig,ax = plt.subplots(nrows=3,ncols=3,figsize=(12,8))#,sharex='col',sharey='row')
    
    ax[0,0].scatter(df['PC1']+np.random.uniform(-0.005,0.005,(len(df))),df['PC2']+np.random.uniform(-0.005,0.005,(len(df))),s=0.5,c=df['s1ac_red'],cmap='viridis', zorder=0, vmin=0,vmax=1)
    ax[0,1].scatter(df['PC1']+np.random.uniform(-0.005,0.005,(len(df))),df['PC2']+np.random.uniform(-0.005,0.005,(len(df))),s=0.5,c=df['s1ac_unch'],cmap='viridis', zorder=0,vmin=0,vmax=1)
    im1 = ax[0,2].scatter(df['PC1']+np.random.uniform(-0.005,0.005,(len(df))),df['PC2']+np.random.uniform(-0.005,0.005,(len(df))),s=0.5,c=df['s1ac_inc'],cmap='viridis',zorder=0,vmin=0,vmax=1)
    
    ax[1,0].scatter(df['PC1']+np.random.uniform(-0.005,0.005,(len(df))),df['PC2']+np.random.uniform(-0.005,0.005,(len(df))),s=0.5,c=df['nsc1_red_thresh'],cmap='viridis', zorder=0, vmin=0,vmax=1)
    ax[1,1].scatter(df['PC1']+np.random.uniform(-0.005,0.005,(len(df))),df['PC2']+np.random.uniform(-0.005,0.005,(len(df))),s=0.5,c=df['nsc1_insuf'],cmap='viridis', zorder=0,vmin=0,vmax=1)
    im2 = ax[1,2].scatter(df['PC1']+np.random.uniform(-0.005,0.005,(len(df))),df['PC2']+np.random.uniform(-0.005,0.005,(len(df))),s=0.5,c=df['nsc1_inc'],cmap='viridis',zorder=0,vmin=0,vmax=1)

    ax[2,0].scatter(df['PC1']+np.random.uniform(-0.005,0.005,(len(df))),df['PC2']+np.random.uniform(-0.005,0.005,(len(df))),s=0.5,c=df['n5sc1_red_thresh'],cmap='viridis', zorder=0, vmin=0,vmax=1)
    ax[2,1].scatter(df['PC1']+np.random.uniform(-0.005,0.005,(len(df))),df['PC2']+np.random.uniform(-0.005,0.005,(len(df))),s=0.5,c=df['n5sc1_insuf'],cmap='viridis', zorder=0,vmin=0,vmax=1)
    im3 = ax[2,2].scatter(df['PC1']+np.random.uniform(-0.005,0.005,(len(df))),df['PC2']+np.random.uniform(-0.005,0.005,(len(df))),s=0.5,c=df['n5sc1_inc'],cmap='viridis',zorder=0,vmin=0,vmax=1)
    
    linecol = 'xkcd:cherry red'
    linestyl = '--'
    linwidth =1.5
    for i in range(3):
        for j in range(3):
            ax[i,j].plot([-3,c_tail],[m_top*(-3) + c_top, m_top*c_tail + c_top], linewidth=linwidth,c=linecol,linestyle=linestyl, zorder=1)#top
            ax[i,j].plot([c_tail,c_tail],[-3,3],linewidth=linwidth,c=linecol,linestyle=linestyl, zorder=1)#tail
            ax[i,j].plot([-3,c_tail],[m_center*(-3) + c_center, m_center*c_tail + c_center],linewidth=linwidth,c=linecol,linestyle=linestyl, zorder=1)#center
            ax[i,j].set_xlim((-2.5,7)); ax[i,j].set_ylim((-2.2,2.8))
           #ax[i,j].set_box_aspect(0.7)


    ax[0,0].set_xticklabels([])
    ax[0,1].set_xticklabels([])
    ax[0,2].set_xticklabels([])
    ax[1,0].set_xticklabels([])
    ax[1,1].set_xticklabels([])
    ax[1,2].set_xticklabels([])

    ax[0,1].set_yticklabels([])
    ax[0,2].set_yticklabels([])
    ax[1,1].set_yticklabels([])
    ax[1,2].set_yticklabels([])
    ax[2,1].set_yticklabels([])
    ax[2,2].set_yticklabels([])

    fig.subplots_adjust(left=0.05, right=0.9, top=0.99, bottom=0.05, wspace=0.1,hspace=0.1)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    fig.colorbar(im3, cax=cbar_ax)
    fig.savefig('/Users/somya/Documents/SIGMIG_from_Bigram2/Dat_3_from_cslm/fig3/recovery5_scatters.png')
    print('saved fig')
    plt.close('all')
    return 0
