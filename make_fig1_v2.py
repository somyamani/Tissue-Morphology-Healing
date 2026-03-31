import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import glob



dict1 = pickle.load(open('/Users/somya/Documents/SIGMIG_from_Bigram2/Dat_3_from_cslm/finalbody_richness_disperseness.pkl','rb'))
df = dict1['finalbody_df']
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



def make_fig1_gs1():
    fig = plt.figure(figsize=(9,15))
    gs = gridspec.GridSpec(8,30,height_ratios=[0.90,0.90,0.90,0.90,1,1,1,1])
    gs.update(wspace=0.2 ,hspace=1.0, left=0.05,right=0.99,bottom=0.03,top=0.99)

    axA = fig.add_subplot(gs[0:4,0:18])
    axB = [fig.add_subplot(gs[0:2,20:28]),fig.add_subplot(gs[2:4,20:28]),fig.add_subplot(gs[0:4,28])]
    axC1 = [fig.add_subplot(gs[4,0:6]),fig.add_subplot(gs[4,8:14]),fig.add_subplot(gs[4,16:22]),fig.add_subplot(gs[4,24:30])]
    axC2 = [fig.add_subplot(gs[5,0:6]),fig.add_subplot(gs[5,8:14]),fig.add_subplot(gs[5,16:22]),fig.add_subplot(gs[5,24:30])]
    axC3 = [fig.add_subplot(gs[6,0:6]),fig.add_subplot(gs[6,8:14]),fig.add_subplot(gs[6,16:22]),fig.add_subplot(gs[6,24:30])]
    axC4 = [fig.add_subplot(gs[7,0:6]),fig.add_subplot(gs[7,8:14]),fig.add_subplot(gs[7,16:22]),fig.add_subplot(gs[7,24:30])]

    make_axA(axA)
    make_axB(axB)
    make_axC1(axC1)
    make_axC2(axC2)
    make_axC3(axC3)
    make_axC4(axC4)
    plt.savefig('/Users/somya/Documents/SIGMIG_from_Bigram2/Dat_3_from_cslm/Fig1_scatters_hists.png') 
    return 0


def make_axA(axA):
    top_samples = df['sector_assignments'] == 'top'
    tail_samples = df['sector_assignments'] == 'tail'
    center_samples = df['sector_assignments'] == 'center'
    bottom_samples = df['sector_assignments'] == 'bottom'

    XYdat = [top_samples, tail_samples, center_samples, bottom_samples]
    X = [df['PC1'][x] + np.random.uniform(-0.005,0.005,(sum(x))) for x in XYdat]
    Y = [df['PC2'][y] + np.random.uniform(-0.005,0.005,(sum(y))) for y in XYdat]
    colorlist = [topcolor, tailcolor, centercolor, bottomcolor]
    for i in range(len(XYdat)):
        axA.scatter(X[i],Y[i],s=0.5,c=colorlist[i],zorder=0)

    axA.plot([-3,c_tail],[m_top*(-3) + c_top, m_top*c_tail + c_top], linewidth=1,c='xkcd:dark grey',linestyle='-', zorder=1)#top
    axA.plot([c_tail,c_tail],[-3,3],linewidth=1,c='xkcd:dark grey',linestyle='-', zorder=1)#tail
    axA.plot([-3,c_tail],[m_center*(-3) + c_center, m_center*c_tail + c_center],linewidth=1,c='xkcd:dark grey',linestyle='-', zorder=1)#center
    axA.set_xlim((-2.5,7)); axA.set_ylim((-2.2,2.8))

    # mark points for representative images in each sector
    # top, tail,center,tail
    paramids = [125,77,6,57,110,62,54,98,10,54,45,82]
    reps = [28,60,40,34,25,6,32,38,58,64,92,73]
    points_x = []
    points_y = []
    for i in range(len(paramids)):
        points_x += list(np.array(df.loc[(df['paramid']==paramids[i])&(df['rep']==reps[i])][['PC1']])[0])
        points_y += list(np.array(df.loc[(df['paramid']==paramids[i])&(df['rep']==reps[i])][['PC2']])[0])
    axA.scatter(points_x,points_y,s=15,c='xkcd:brick red', edgecolor='xkcd:pale red',zorder=2)
    axA.spines['right'].set_visible(False)
    axA.spines['top'].set_visible(False)
    return 0


def make_axB(axB):
    colors = ['av_dispersity','body_area']
    linecol = 'xkcd:cherry red'
    linestyl = '--'
    linwidth =1.5

    for i in range(2):
        fig1 = axB[i].scatter(df['PC1']+np.random.uniform(-0.005,0.005,(len(df))),df['PC2']+np.random.uniform(-0.005,0.005,(len(df))),s=0.5,c=df[colors[i]],cmap='viridis', zorder=0, vmin=0,vmax=1)
        axB[i].plot([-3,c_tail],[m_top*(-3) + c_top, m_top*c_tail + c_top], linewidth=linwidth,c=linecol,linestyle=linestyl, zorder=1)#top
        axB[i].plot([c_tail,c_tail],[-3,3],linewidth=linwidth,c=linecol,linestyle=linestyl, zorder=1)#tail
        axB[i].plot([-3,c_tail],[m_center*(-3) + c_center, m_center*c_tail + c_center],linewidth=linwidth,c=linecol,linestyle=linestyl, zorder=1)#center
        axB[i].set_xlim((-2.5,7)); axB[i].set_ylim((-2.2,2.8))
        axB[i].spines['right'].set_visible(False)
        axB[i].spines['top'].set_visible(False)
    plt.colorbar(fig1,axB[2])
    return 0

def make_axC1(axC1):
    #intden
    intden_y = np.zeros((4,3))
    for i,sec in enumerate(['top','center','bottom','tail']):
        df_sec = df.loc[df['sector_assignments']==sec]
        for j,val in enumerate([3,5,7]):
            intden_y[i,j] = sum(df_sec['int_den']==val)/len(df_sec)
    
    Col = [topcolor, centercolor, bottomcolor, tailcolor]
    for i in range(4):
        axC1[i].bar([1,2,3], intden_y[i], facecolor=Col[i])
        axC1[i].set_xticks([1,2,3],[0.3,0.5,0.7])
        axC1[i].spines['right'].set_visible(False)
        axC1[i].spines['top'].set_visible(False)
    return 0

def make_axC2(axC2):
    #p_stable
    pstable_y = np.zeros((4,3))
    for i,sec in enumerate(['top','center','bottom','tail']):
        df_sec = df.loc[df['sector_assignments']==sec]
        for j,val in enumerate([3,5,7]):
            pstable_y[i,j] = sum(df_sec['p_stable']==val)/len(df_sec)
    
    Col = [topcolor, centercolor, bottomcolor, tailcolor]
    for i in range(4):
        axC2[i].bar([1,2,3], pstable_y[i], facecolor=Col[i])
        axC2[i].set_xticks([1,2,3],[0.3,0.5,0.7])
        axC2[i].spines['right'].set_visible(False)
        axC2[i].spines['top'].set_visible(False)
    return 0


def make_axC3(axC3):
    #p_diff
    pdiff_y = np.zeros((4,5))
    for i,sec in enumerate(['top','center','bottom','tail']):
        df_sec = df.loc[df['sector_assignments']==sec]
        for j,val in enumerate([0,0.3,0.5,0.7,1]):
            pdiff_y[i,j] = sum(df_sec['p_diff']==val)/len(df_sec)
    
    Col = [topcolor, centercolor, bottomcolor, tailcolor]
    for i in range(4):
        axC3[i].bar([1,2,3,4,5], pdiff_y[i], facecolor=Col[i])
        axC3[i].set_xticks([1,2,3,4,5],[0,0.3,0.5,0.7,1])
        axC3[i].spines['right'].set_visible(False)
        axC3[i].spines['top'].set_visible(False)
    return 0

def make_axC4(axC4):
    #adj_dtrs
    adjdtrs_y = np.zeros((4,3))
    for i,sec in enumerate(['top','center','bottom','tail']):
        df_sec = df.loc[df['sector_assignments']==sec]
        for j,val in enumerate([3,5,7]):
            adjdtrs_y[i,j] = sum(df_sec['adj_dtrs']== (10-val))/len(df_sec)
    
    Col = [topcolor, centercolor, bottomcolor, tailcolor]
    for i in range(4):
        axC4[i].bar([1,2,3], adjdtrs_y[i], facecolor=Col[i])
        axC4[i].set_xticks([1,2,3],[0.3,0.5,0.7])
        axC4[i].spines['right'].set_visible(False)
        axC4[i].spines['top'].set_visible(False)
    return 0




def make_fig1_gs3():
    fig = plt.figure(figsize=(3,10))
    gs = gridspec.GridSpec(9,10, height_ratios=[1,0.25,1,0.25,1,1,0.25,1,1])
    gs.update(wspace=0.0 ,hspace=0.03, left=0,right=1,bottom=0.03,top=1)

    axD = []
    rows_with_figs = [0,2,4,5,7,8]
    for i in range(6):
        r1 = rows_with_figs[i]
        axD+=[fig.add_subplot(gs[r1,0:5]),fig.add_subplot(gs[r1,5:10])]


    make_axD(axD)
    plt.savefig('Dat_3_from_cslm/fig1/Fig1_sectortissue_images.png')
    return 0


def make_axD(axD):
    for i in range(len(axD)):
        axD[i].set_xticks([])
        axD[i].set_yticks([])
    draw_topfiles(axD[0],axD[1])
    draw_tailfiles(axD[2],axD[3])
    draw_bottomfiles(axD[4],axD[5],axD[6],axD[7])
    draw_centerfiles(axD[8],axD[9],axD[10],axD[11])
    
    return 0

def draw_topfiles(ax1,ax2):
    topfiles = ['Dat_3_from_cslm/finalbody_mat/top/FBF_rep028_param00125.npy','Dat_3_from_cslm/finalbody_mat/top/FBF_rep060_param00077.npy']
    axs = [ax1,ax2]
    for i in range(len(topfiles)):
        topmat = np.load(topfiles[i])
        topmat = (topmat*250/max(topmat.flatten())).astype(int)# adjust brightness 
        axs[i].imshow(topmat)
    return 0

def draw_tailfiles(ax1,ax2):
    tailfiles = ['Dat_3_from_cslm/finalbody_mat/tail/FBF_rep034_param00057.npy','Dat_3_from_cslm/finalbody_mat/tail/FBF_rep040_param00006.npy']
    axs = [ax1,ax2]
    for i in range(len(tailfiles)):
        tailmat = np.load(tailfiles[i])
        tailmat = (tailmat*250/max(tailmat.flatten())).astype(int)# adjust brightness 
        axs[i].imshow(tailmat)
    return 0

def draw_bottomfiles(ax1,ax2,ax3,ax4):
    bottomfiles = ['Dat_3_from_cslm/finalbody_mat/bottom/FBF_rep073_param00082.npy','Dat_3_from_cslm/finalbody_mat/bottom/FBF_rep092_param00045.npy',
            'Dat_3_from_cslm/finalbody_mat/bottom/FBF_rep064_param00054.npy','Dat_3_from_cslm/finalbody_mat/bottom/FBF_rep058_param00010.npy']
    axs = [ax1,ax2,ax3,ax4]
    for i in range(len(bottomfiles)):
        bottommat = np.load(bottomfiles[i])
        bottommat = (bottommat*250/max(bottommat.flatten())).astype(int)# adjust brightness 
        axs[i].imshow(bottommat)
    return 0

def draw_centerfiles(ax1,ax2,ax3,ax4):
    centerfiles = ['Dat_3_from_cslm/finalbody_mat/center/FBF_rep025_param00110.npy','Dat_3_from_cslm/finalbody_mat/center/FBF_rep006_param00062.npy',
            'Dat_3_from_cslm/finalbody_mat/center/FBF_rep038_param00098.npy','Dat_3_from_cslm/finalbody_mat/center/FBF_rep032_param00054.npy']
    axs = [ax1,ax2,ax3,ax4]
    for i in range(len(centerfiles)):
        centermat = np.load(centerfiles[i])
        centermat = (centermat*250/max(centermat.flatten())).astype(int)# adjust brightness 
        axs[i].imshow(centermat)
    return 0



