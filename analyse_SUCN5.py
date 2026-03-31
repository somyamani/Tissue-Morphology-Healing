import sys, os
import numpy as np
import pandas as pd
import glob
import stable_unstable_cells_neighbourhoods5 as SUCN5
import pickle
import time
import math
from multiprocessing import Pool
import argparse
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import optimal_sector_finder_SUCN5 as OSFS
import itertools
import visualize_bodies as VB


# code to look at homeostasis

# for each final body, perform 10 independent random perturbations (delete all cells of a cell-type from a random grid-location)
# check: does it return to the body return to its original composition [compute difference in composition across grid-locations over time.. is this decreasing? increasing?]


if os.path.isfile('/home/somyamn/Signals_migration/Dat_3/finalbody_richness_disperseness.pkl'):
    chIUAGFAL=0
    dict1 = pickle.load(open('/home/somyamn/Signals_migration/Dat_3/finalbody_richness_disperseness.pkl','rb'))
    df = dict1['finalbody_df']
    x = dict1['sectordict']
    c_tail = x['c_tail']; m_top=x['m_top']; c_top=x['c_top']; m_center=x['m_center']; c_center=x['c_center']


def get_finalbody_richness_disperseness_df(paramid):
    # create empty array of the right size [100 rows, 13 columns]
    res = np.zeros((100,13))# store paramid(1), rep(1), all parameters(6) and values of variables(5)
    
    # load param values
    params = pickle.load(open('/home/somyamn/Signals_migration/model4_param.p','rb'))[int(paramid)]
    p_stable, p_diff, intden, celltypes, loc, T, adj_dtrs, no_div_threshold, reps = params

    adj_dtrs = int(adj_dtrs * 10)
    p_stable = int(p_stable * 10)
    intden = int(intden * 10)#param-6
    
    # load av_tissue_prop, get final_body_prop, store in res
    for rep in range(100):
        f = '/home/somyamn/Signals_migration/Dat_3/Adjdtrs{a}/pstable{s}/intden{d}/Analysis/av_tissue_prop/AV_rep{r:03d}_param{p:05d}.pkl'.format(a=adj_dtrs, s=p_stable, d=intden, p=paramid,r=rep)# CHANGE THIS PATH TO THE CORRECT PATH ON YOUR MACHINE
        if os.path.isfile(f):
            if os.path.getsize(f)>0:
                df_final = pickle.load(open(f,'rb')).iloc[-1]
                res[rep] = [paramid, rep, celltypes, loc, adj_dtrs, p_stable, p_diff, intden, df_final['frac_occ_body'], df_final['num_distinct_niches'], df_final['coverage_equality'], df_final['av_dispersity'], df_final['av_mixness']]
            else:
                os.remove(f)
                print('removed',f)
    full_ros = np.sum(res, axis=1)>0
    res = res[full_ros]
    print('processed',paramid)
    return res


def save_richness_dispersity_parallel():
    int_param_id = np.arange(0,135,1).astype(int)
    with Pool(28) as pool:# CHANGE NUMBER OF CORES ACCORDING TO YOUR MACHINE 
        X = np.vstack(pool.map(get_finalbody_richness_disperseness_df,int_param_id))
    # make df
    DF = pd.DataFrame(X,columns=['paramid', 'rep', 'celltypes','grid_squares', 'adj_dtrs', 'p_stable', 'p_diff', 'int_den', 'body_area', 'num_distinct_niches', 'coverage_equality', 'av_dispersity', 'av_mixness'])
    # save df
    f_out = '/home/somyamn/Signals_migration/Dat_3/finalbody_richness_disperseness.pkl'
    dict1 = {}
    dict1['finalbody_df'] = DF
    pickle.dump(dict1,open(f_out,'wb'))
    print('processed and saved')
    return 0
#RUN PREV CODE IN PARALLEL


def save_finalbody_tissue_images(paramid):
    params = pickle.load(open('/home/somyamn/Signals_migration/model4_param.p','rb'))[int(paramid)]
    p_stable, p_diff, intden, celltypes, loc, T, adj_dtrs, no_div_threshold, reps = params

    adj_dtrs = int(adj_dtrs * 10)
    p_stable = int(p_stable * 10)
    intden = int(intden * 10)#param-6
    
    palette = VB.assign_tissue_colors_5celltypes()
    for rep in range(100):
        f = '/home/somyamn/Signals_migration/Dat_3/Adjdtrs{a}/pstable{s}/intden{d}/Data/FinalBody/FB_rep{r:03d}_param{p:05d}.npy'.format(a=adj_dtrs, s=p_stable, d=intden, p=paramid,r=rep)
        body = np.load(f)
        bodycol = VB.get_body_colors(body, loc, palette)
        
        f_out = '/home/somyamn/Signals_migration/Dat_3/Adjdtrs{a}/pstable{s}/intden{d}/Analysis/FinalBodyFigs/FBF_rep{r:03d}_param{p:05d}.npy'.format(a=adj_dtrs, s=p_stable, d=intden, p=paramid,r=rep)
        np.save(f_out,bodycol)
    print('processed paramid',paramid)
    return 0

def save_finalbody_tissue_images_parallel():
    with Pool(28) as pool:
        pool.map(save_finalbody_tissue_images,np.arange(0,135,1))
    return 0


def pca_final_body():
    df_arr = df.to_numpy()

    param_val = df[['adj_dtrs', 'p_stable', 'p_diff', 'int_den']].to_numpy()

    graph_prop_df = df[['body_area', 'num_distinct_niches','coverage_equality', 'av_dispersity', 'av_mixness']]

    graph_prop_means = graph_prop_df.mean()
    graph_prop_std = graph_prop_df.std()

    #graph_prop_scaled = StandardScaler().fit_transform(graph_prop)#normalize graph_prop
    graph_prop_scaled = (graph_prop_df - graph_prop_means)/graph_prop_std
    graph_prop_scaled = graph_prop_scaled.to_numpy()
    pca_graph = PCA(n_components=len(graph_prop_df.columns))#perform PCA
    PC_graph = pca_graph.fit_transform(graph_prop_scaled)# graph properties along principle components [scaled]

    allPC = pca_graph.components_# each row: what linear combination of graph_prop columns will give each PC
    expvar = pca_graph.explained_variance_ratio_# what proportion of variance does each PC explain?

    sigvar = expvar > (1/len(graph_prop_df.columns))# significant PCs explain more than 1/(number of graph_prop) of the variance
    sigPC = allPC[sigvar,:]# graph properties along the significant PCs

    # What do the significant PCs capture? get correlations of sigPC with graph_prop_scaled
    # which model parameters govern which significant PCs? get correlations of sigPC with param_val

    Corr_graphprop = np.zeros((len(graph_prop_df.columns),len(graph_prop_df.columns)))
    Corr_param = np.zeros((len(graph_prop_df.columns),np.shape(param_val)[1]))

    for i0 in range(np.shape(sigPC)[1]):

        for i1 in range(len(graph_prop_df.columns)):
            Corr_graphprop[i0,i1] = list(stats.pearsonr(graph_prop_scaled[:,i1],PC_graph[:,i0]))[0]

        for i1 in range(np.shape(param_val)[1]):
            Corr_param[i0,i1] = list(stats.pearsonr(param_val[:,i1],PC_graph[:,i0]))[0]

    # interpretation of Corr_graphprop: correlation above a certain reasonable threshold --- correspondence. Choose a threshold such that there is minimum overlap across PCs, and maximum coverage of graph properties.

    # same for Corr_param
    df['PC1']=PC_graph[:,0]
    df['PC2'] = PC_graph[:,1]
    # elbow plot for supplementary information
    plt.plot(np.cumsum(expvar),c='k',linewidth=2)
    plt.plot([0,4],[0.86,0.86],c='xkcd:cerulean blue',linewidth=2,linestyle='--')
    plt.xticks([0,1,2,3,4],[1,2,3,4,5])
    plt.xlabel('PCs');plt.ylabel('explained variance (cumulative)')
    plt.savefig('/Users/somya/Documents/SIGMIG_from_IBS_desktop/SigMig/Dat_3/suppfigs/elbowplot.png')
    plt.close()
    # make a dictionary with the df, and other properties of the PCA (esp. graph_prop_means and std) and store this dictionary. This is used further in analysing tissue update trajectories in terms of the PCs
    dict1 = {'finalbody_df':df, 'graph_prop_means':graph_prop_means, 'graph_prop_std':graph_prop_std, 'sigPC':sigPC, 'explained_variance':expvar, 'PC_graphprop_corr':Corr_graphprop, 'PC_param_corr':Corr_param} 

    pickle.dump(dict1, open('/home/somyamn/Signals_migration/Dat_3/finalbody_richness_disperseness.pkl','wb'))
    return 0
#RUN THIS CODE ON A SINGLE CORE


def characterize_steady_state():
    #new columns
    time_to_comp_ss = np.zeros((len(df)))
    time_to_overall_ss = np.zeros((len(df)))
    avnumcells = np.zeros((len(df)))# for occupied gridloc in final body
    stdnumcells = np.zeros((len(df)))# for occupied gridloc in final body

    # load BU in a for loop
    for i in range(len(df)):
        print(i)
        dfi = df.iloc[i]
        fname = '/home/somyamn/Signals_migration/Dat_3/Adjdtrs{a}/pstable{s}/intden{d}/Data/BodyUpdates/BU_rep{r:03d}_param{p:05d}.npy'.format(a=int(dfi['adj_dtrs']), s=int(dfi['p_stable']), d=int(dfi['int_den']), p=int(dfi['paramid']),r=int(dfi['rep']))
        BU = np.load(fname)
        finalbody = BU[:,:,-1]
    # steady state: have final bodies reached stady state -- compositional, overall (incl. cell numbers)
    # how many time-steps to each type of steady state [append this to df columns]
        finalbody_rep = np.repeat(finalbody[:,:,np.newaxis],500,axis=2)
        overalldiff = BU - finalbody_rep
        compdiff = (BU>0).astype(int) - (finalbody_rep>0).astype(int)
        
        compss = np.sum(np.sum((compdiff !=0).astype(int),axis=0),axis=0)
        if any(compss !=0):
            time_to_comp_ss[i] = np.where(compss !=0)[0][-1]

        overallss = np.sum(np.sum((overalldiff !=0).astype(int),axis=0),axis=0)
        if any(overallss!=0):
            time_to_overall_ss[i] = np.where(overallss !=0)[0][-1]
    # across occupied grid locations, what is the average (std) number of cells in the final body? [append columns to df]
        gridcells = np.sum(finalbody,axis=0)
        avnumcells[i] = np.mean(gridcells[gridcells>0])
        stdnumcells[i] = np.std(gridcells[gridcells>0])

    df['time_to_comp_ss'] = time_to_comp_ss
    df['time_to_overall_ss'] = time_to_overall_ss
    df['avnumcells'] = avnumcells
    df['stdnumcells'] = stdnumcells

    dict1['finalbody_df'] = df
    pickle.dump(dict1, open('/home/somyamn/Signals_migration/Dat_3/finalbody_richness_disperseness.pkl','wb'))
    return 0
# RUN PREV CODE ON ONE CORE

# now the finalbody_df has got PC1 and PC2, and info about time to steady states 
# find optimal sector boundaries -- save these in the Analysis directory in the finalbody_richness_disperseness.pkl dictionary

def get_sectors():
    x = OSFS.optimal_PC_sectors_parallel()
    c_tail = x[0]; m_top=x[1]; c_top=x[2]; m_center=x[3]; c_center=x[4]
    sectordict = OSFS.optimal_PC_sectors_fine(c_tail, m_top, c_top, m_center, c_center)
    return sectordict
#RUN ON A SINGLE CORE.. OSFS runs on multiple processors (CHANGE NUMBER OF PARALLEL CORES IN OSFS CODE)




#RESULTS-2

# update dfs for all body_updates with their locations in PC-space and sector assignments
# create transition matrices

def append_PCloc_BodyUpdate_dfs(paramid):
    # load dictionary from 'finalbody_richness_disperseness.pkl' to get information about graph_prop_means, SigPC
    params = pickle.load(open('/home/somyamn/Signals_migration/model4_param.p','rb'))[int(paramid)]
    p_stable, p_diff, intden, celltypes, loc, T, adj_dtrs, no_div_threshold, reps = params
    
    adj_dtrs = int(adj_dtrs * 10)
    p_stable = int(p_stable * 10)
    intden = int(intden * 10)#param-6

    graph_prop_means = dict1['graph_prop_means']
    graph_prop_std = dict1['graph_prop_std']
    sigPC = dict1['sigPC']

    for rep in range(100):
        f = '/home/somyamn/Signals_migration/Dat_3/Adjdtrs{a}/pstable{s}/intden{d}/Analysis/av_tissue_prop/AV_rep{r:03d}_param{p:05d}.pkl'.format(a=adj_dtrs, s=p_stable, d=intden, p=paramid,r=rep)# CHANGE PATHS ACCORDING TO YOUR MACHINE
        if os.path.isfile(f):
            df2 = pickle.load(open(f,'rb'))    
            growdf = df2.rename(columns={'frac_occ_body':'body_area'})#match column names
            # scale growing tissue properties using the means and stds from final body properties
            growdf_scaled = (growdf[['body_area','num_distinct_niches','coverage_equality','av_dispersity','av_mixness']] - graph_prop_means)/graph_prop_std
            X = growdf_scaled.to_numpy()
            X_PC = X @ sigPC.T# recast growing tissue properties into PCs
            df2['PC1'] = X_PC[:,0]
            df2['PC2'] = X_PC[:,1]
            # check consistency with final body df
            dif1 = np.around(X_PC[-1,1],4) != np.around(np.array(df.loc[(df['paramid']==paramid)&(df['rep']==rep)]['PC2'])[0],4)
            dif0 = np.around(X_PC[-1,0],4) != np.around(np.array(df.loc[(df['paramid']==paramid)&(df['rep']==rep)]['PC1'])[0],4)
            if (dif1 | dif0):
                print((paramid,rep),'inconsistent',dif0,dif1)
            pickle.dump(df2,open(f,'wb'))
            print('processed',f)
    return 0

def runparallel_append_PCloc_BodyUpdate_dfs():
    with Pool(28) as pool:
        pool.map(append_PCloc_BodyUpdate_dfs,np.arange(0,135,1))
    return 0


# make lists of sectorwise filenames for final_body_figures and av_tissue_properties
def make_sectorwise_filename_lists():
    for sector in ['center']:#['top','center','bottom','tail']:
        df_sector = df.loc[df['sector_assignments']==sector].reset_index()
        print(sector,len(df_sector))
        # reduce this df to those rows that have properties close to the average (< 1/2 sd away)
        df_means = np.mean(df_sector[['num_distinct_niches','coverage_equality', 'av_dispersity', 'av_mixness']])
        df_std = np.std(df_sector[['num_distinct_niches','coverage_equality', 'av_dispersity', 'av_mixness']])
        df_diff = (abs(df_sector[['num_distinct_niches','coverage_equality', 'av_dispersity', 'av_mixness']]-df_means) < 0.5*df_std).to_numpy()
        df_diff2 = np.sum(df_diff,axis=1)==0
        df_sector = df_sector[df_diff2].reset_index()
        # CHANGE BELOW PATHS ACCORDING TO YOUR MACHINE
        fname_fbf = '/home/somyamn/Signals_migration/finalbody_fig_{}.txt'.format(sector)
        fname_av = '/home/somyamn/Signals_migration/bodyupdates_{}.txt'.format(sector)
        av_list=[];fbf_list=[]
        print(sector,sum(df_diff2))
        for i0 in range(len(df_sector)):
            paramid = int(df_sector.loc[i0]['paramid'])
            rep = int(df_sector.loc[i0]['rep'])
            adj_dtrs = int(df_sector.loc[i0]['adj_dtrs'])
            p_stable = int(df_sector.loc[i0]['p_stable'])
            intden = int(df_sector.loc[i0]['int_den'])
            if sector != np.array(df_sector.loc[(df_sector['paramid']==paramid)&(df_sector['rep']==rep)]['sector_assignments'])[0]:
                print(paramid,rep)
                break;
                
            av_list += ['/home/somyamn/Signals_migration/Dat_3/Adjdtrs{a}/pstable{s}/intden{d}/Analysis/av_tissue_prop/AV_rep{r:03d}_param{p:05d}.pkl'.format(a=adj_dtrs, s=p_stable, d=intden, p=paramid,r=rep)]
            fbf_list += ['/home/somyamn/Signals_migration/Dat_3/Adjdtrs{a}/pstable{s}/intden{d}/Analysis/FinalBodyFigs/FBF_rep{r:03d}_param{p:05d}.npy'.format(a=adj_dtrs, s=p_stable, d=intden, p=paramid,r=rep)]
        # choose 200 random files for av_list. Throw away the rest
        x = np.random.randint(0,len(av_list),200)
        av_list = [av_list[i1] for i1 in x]
        with open(fname_fbf,'w') as f1:
                f1.write('\n'.join(fbf_list))

        with open(fname_av,'w') as f2:
                f2.write('\n'.join(av_list))
    return 0


def count_sector_crossings_oscillations(filename):
    growdf = pickle.load(open(filename,'rb'))
    X_PC = growdf[['PC1','PC2']].to_numpy()

    # label each point as 'bottom', 'middle', 'head', 'tail'
    top_samples = (X_PC[:,1]>(m_top*X_PC[:,0] + c_top))&(X_PC[:,0]<c_tail)

    tail_samples = X_PC[:,0]>=c_tail
    
    center_samples = (X_PC[:,1]>(m_center*X_PC[:,0] + c_center))&(X_PC[:,1]<(m_top*X_PC[:,0] + c_top))&(X_PC[:,0]<c_tail)
    
    bottom_samples = (X_PC[:,1]<=(m_center*X_PC[:,0] + c_center))&(X_PC[:,1]<(m_top*X_PC[:,0] + c_top))&(X_PC[:,0]<c_tail)
    
    sector_seq = 1*bottom_samples + 2*center_samples + 3*tail_samples + 4*top_samples
    sector_steps = np.array([sector_seq[0:-1],sector_seq[1:]])

    crossings = sector_steps[0] - sector_steps[1] !=0
    num_crossings = sum(crossings) 
    final_sector = sector_seq[-1]
    
    penultimate_sector = final_sector
    time_to_final=0
    if any(crossings):
        time_to_final = np.where(crossings!=0)[0][-1]
        penultimate_sector = sector_steps[0][time_to_final]
    time_to_penultimate=0
    if any((sector_seq != final_sector)&(sector_seq != penultimate_sector)) & any(crossings):
        time_to_penultimate = np.where((sector_seq != final_sector)&(sector_seq != penultimate_sector))[0][-1]
    #number of crossings after reaching penultimate sector but before the final crossing
    num_crossings_after_penultimate = sum(crossings[time_to_penultimate:time_to_final])
    is_oscillations = num_crossings_after_penultimate > 2

    print('processed',filename)
    return final_sector, num_crossings, time_to_final, num_crossings_after_penultimate, is_oscillations


def check_traj_init_sector():
    bottom_init = 0
    center_init = 0
    top_init = 0
    tail_init = 0

    params = pickle.load(open('/home/somyamn/Signals_migration/model4_param.p','rb'))
    sector_param_rep = df[['paramid','rep']].to_numpy()
    # list of all BodyUpdate files for tissues that end up in the required sector
    for [paramid, rep] in sector_param_rep:
        p_stable, p_diff, intden, celltypes, loc, T, adj_dtrs, no_div_threshold, reps = params[int(paramid)]

        adj_dtrs = int(adj_dtrs * 10)
        p_stable = int(p_stable * 10)
        intden = int(intden * 10)#param-6

        fname = '/home/somyamn/Signals_migration/Dat_3/Adjdtrs{a}/pstable{s}/intden{d}/Analysis/av_tissue_prop/AV_rep{r:03d}_param{p:05d}.pkl'.format(a=adj_dtrs, s=p_stable, d=intden, p=int(paramid),r=int(rep))
        growdf = pickle.load(open(fname,'rb'))
        X_PC = growdf[['PC1','PC2']].to_numpy()

        # label each point as 'bottom', 'middle', 'head', 'tail'
        top_init += (X_PC[0,1]>(m_top*X_PC[0,0] + c_top))&(X_PC[0,0]<c_tail)

        tail_init += X_PC[0,0]>=c_tail

        center_init += (X_PC[0,1]>(m_center*X_PC[0,0] + c_center))&(X_PC[0,1]<(m_top*X_PC[0,0] + c_top))&(X_PC[0,0]<c_tail)

        bottom_init += (X_PC[0,1]<=(m_center*X_PC[0,0] + c_center))&(X_PC[0,1]<(m_top*X_PC[0,0] + c_top))&(X_PC[0,0]<c_tail)
        print(fname)
    return top_init, tail_init, center_init, bottom_init




def get_sector_crossings_oscillation_array():
    sectors = np.array(['bottom','center','tail','top'])
    crossings_oscillations_df = pd.DataFrame(columns = ['final_sector','num_crossings','time_to_final', 'num_crosses_oscillations','is_oscillations'])
    params = pickle.load(open('/home/somyamn/Signals_migration/model4_param.p','rb'))
    sector_param_rep = df[['paramid','rep']].to_numpy()
    # list of all BodyUpdate files for tissues that end up in the required sector
    files = []
    for [paramid, rep] in sector_param_rep:
        p_stable, p_diff, intden, celltypes, loc, T, adj_dtrs, no_div_threshold, reps = params[int(paramid)]

        adj_dtrs = int(adj_dtrs * 10)
        p_stable = int(p_stable * 10)
        intden = int(intden * 10)#param-6

        fname = '/home/somyamn/Signals_migration/Dat_3/Adjdtrs{a}/pstable{s}/intden{d}/Analysis/av_tissue_prop/AV_rep{r:03d}_param{p:05d}.pkl'.format(a=adj_dtrs, s=p_stable, d=intden, p=int(paramid),r=int(rep))
        files += [fname]

    with Pool(28) as pool:
        crossings_oscillations_mat = pool.map(count_sector_crossings_oscillations,files)
    crossings_oscillations_df['final_sector'] = sectors[np.array([list(crossings_oscillations_mat[i])[0] for i in range(13500)])-1]
    crossings_oscillations_df['num_crossings'] = np.array([list(crossings_oscillations_mat[i])[1] for i in range(13500)])
    crossings_oscillations_df['time_to_final'] = np.array([list(crossings_oscillations_mat[i])[2] for i in range(13500)])
    crossings_oscillations_df['num_crosses_oscillations'] = np.array([list(crossings_oscillations_mat[i])[3] for i in range(13500)])
    crossings_oscillations_df['is_oscillations'] = np.array([list(crossings_oscillations_mat[i])[4] for i in range(13500)])
    dict1['crossings_oscillations_df'] = crossings_oscillations_df

    return crossings_oscillations_df

def get_dwelltime_transition_matrix(filename):
    growdf = pickle.load(open(filename,'rb'))
    X_PC = growdf[['PC1','PC2']].to_numpy()

    dwelltime_transition_matrix = np.zeros((4,4))
    
    # label each point as 'bottom', 'middle', 'head', 'tail'
    top_samples = (X_PC[:,1]>(m_top*X_PC[:,0] + c_top))&(X_PC[:,0]<c_tail)
    tail_samples = X_PC[:,0]>=c_tail
    center_samples = (X_PC[:,1]>(m_center*X_PC[:,0] + c_center))&(X_PC[:,1]<(m_top*X_PC[:,0] + c_top))&(X_PC[:,0]<c_tail)
    bottom_samples = (X_PC[:,1]<=(m_center*X_PC[:,0] + c_center))&(X_PC[:,1]<(m_top*X_PC[:,0] + c_top))&(X_PC[:,0]<c_tail)
    sector_steps = 1*bottom_samples + 2*center_samples + 3*tail_samples + 4*top_samples
    sector_steps = np.array([sector_steps[0:-1],sector_steps[1:]])
    # dwells and jumps
    dwelltime_transition_matrix[0,1] = sum((sector_steps[0]==1)&(sector_steps[1]==2))
    dwelltime_transition_matrix[0,2] = sum((sector_steps[0]==1)&(sector_steps[1]==3))
    dwelltime_transition_matrix[0,3] =  sum((sector_steps[0]==1)&(sector_steps[1]==4))
    dwelltime_transition_matrix[1,0] =  sum((sector_steps[0]==2)&(sector_steps[1]==1))
    dwelltime_transition_matrix[1,2] =  sum((sector_steps[0]==2)&(sector_steps[1]==3))
    dwelltime_transition_matrix[1,3] =  sum((sector_steps[0]==2)&(sector_steps[1]==4))
    dwelltime_transition_matrix[2,0] =  sum((sector_steps[0]==3)&(sector_steps[1]==1))
    dwelltime_transition_matrix[2,1] =  sum((sector_steps[0]==3)&(sector_steps[1]==2))
    dwelltime_transition_matrix[2,3] =  sum((sector_steps[0]==3)&(sector_steps[1]==4))
    dwelltime_transition_matrix[3,0] =  sum((sector_steps[0]==4)&(sector_steps[1]==1))
    dwelltime_transition_matrix[3,1] =  sum((sector_steps[0]==4)&(sector_steps[1]==2))
    dwelltime_transition_matrix[3,2] =  sum((sector_steps[0]==4)&(sector_steps[1]==3))

    #TOP
    if sector_steps[0,0] == 4:
        dwelltime_transition_matrix[3,3] = sum(top_samples)/(1+sum(dwelltime_transition_matrix[:,3]))
    elif any(top_samples):
        dwelltime_transition_matrix[3,3] = sum(top_samples)/(sum(dwelltime_transition_matrix[:,3]))
    #TAIL
    if sector_steps[0,0] == 3:
        dwelltime_transition_matrix[2,2] = sum(tail_samples)/(1+sum(dwelltime_transition_matrix[:,2]))
    elif any(tail_samples):
        dwelltime_transition_matrix[2,2] = sum(tail_samples)/(sum(dwelltime_transition_matrix[:,2]))
    #CENTER
    if sector_steps[0,0] == 2:
        dwelltime_transition_matrix[1,1] = sum(center_samples)/(1+sum(dwelltime_transition_matrix[:,1]))
    elif any(center_samples):
        dwelltime_transition_matrix[1,1] = sum(center_samples)/(sum(dwelltime_transition_matrix[:,1]))
    #BOTTOM
    if sector_steps[0,0] == 1:
        dwelltime_transition_matrix[0,0] = sum(bottom_samples)/(1+sum(dwelltime_transition_matrix[:,0]))
    elif any(bottom_samples):
        if sum(dwelltime_transition_matrix[:,0]) == 0:
            print(filename)
        dwelltime_transition_matrix[0,0] = sum(bottom_samples)/(sum(dwelltime_transition_matrix[:,0]))

    return dwelltime_transition_matrix


def get_transition_matrices(sector_assignment):
    params = pickle.load(open('/home/somyamn/Signals_migration/model4_param.p','rb'))
    sector_param_rep = df[['paramid','rep']].loc[df['sector_assignments'] == sector_assignment].to_numpy()
    # list of all BodyUpdate files for tissues that end up in the required sector
    files = []
    for [paramid, rep] in sector_param_rep:
        p_stable, p_diff, intden, celltypes, loc, T, adj_dtrs, no_div_threshold, reps = params[int(paramid)]
        
        adj_dtrs = int(adj_dtrs * 10)
        p_stable = int(p_stable * 10)
        intden = int(intden * 10)#param-6

        fname = '/home/somyamn/Signals_migration/Dat_3/Adjdtrs{a}/pstable{s}/intden{d}/Analysis/av_tissue_prop/AV_rep{r:03d}_param{p:05d}.pkl'.format(a=adj_dtrs, s=p_stable, d=intden, p=int(paramid),r=int(rep)) 
        files += [fname]

    with Pool(28) as pool:
        dwelltime_transition_matrix = pool.map(get_dwelltime_transition_matrix,files)
   #DTM = dwelltime_transition_matrix
    # average transition matrix over all trajectories:
    DTM = np.sum(dwelltime_transition_matrix,axis=0)
    for i in range(np.shape(DTM)[0]):
        x = sum(DTM[i,:])
        if x>0:
            DTM[i,:] = DTM[i,:]/x
    return DTM

def save_transition_matrices():
    x = ['top','center','tail','bottom']
    for sectorassign in x:
        DTM = get_transition_matrices(sectorassign)
        f_out = '/home/somyamn/Signals_migration/Dat_3/transitionmat_{}'.format(sectorassign) 
        np.save(f_out,DTM)
        print(sectorassign,'transition matrix calculated and saved')
    return 0


#RESULTS-3

def get_body_recovery(init_body, CellFateRules, celltypes, loc, T, adj_daughters, no_div_threshold):
    Adj = SUCN5.get_adjacency_matrix(loc)
    Adj = ((Adj + np.eye(loc**2))>0).astype(int)

    BodyRecovery = np.zeros((celltypes, loc**2, T))# cellular composition of each location
    BodyRecovery[:,:,0] = init_body

    for i in range(T-1):
        body_divdeath, num_deadcells, num_divcells = SUCN5.update_body_division_death(BodyRecovery[:,:,i], Adj, adj_daughters, no_div_threshold)
        body_t = SUCN5.update_body_cellfate(body_divdeath, Adj, CellFateRules)
        BodyRecovery[:,:,i+1] = body_t
    return BodyRecovery


def simulate_tissue_recovery(inp):
    paramid,rep = inp
    params = pickle.load(open('/home/somyamn/Signals_migration/model4_param.p','rb'))[paramid]
    p_stable, p_diff, density, celltypes, loc, T, adj_daughters, no_div_threshold, reps = params
    
    filedir = '/home/somyamn/Signals_migration/Dat_3/Adjdtrs{a}/pstable{s}/intden{d}'.format(a=int(adj_daughters*10), s=int(10*p_stable), d = int(10*density))
    
    # all required filenames
    FBfilename = os.path.join(filedir,'Data/FinalBody/FB_rep{r1:03d}_param{p1:05d}.npy'.format(r1=int(rep), p1=int(paramid)))
    CFRfilename = os.path.join(filedir,'Rules/CellFateRules/CFR_rep{r1:03d}_param{p1:05d}.pkl'.format(r1=int(rep), p1=int(paramid)))

    # load final body
    FB = np.load(FBfilename).astype(int)
    # load rules
    CFR = pickle.load(open(CFRfilename,'rb'))

    # perform 10 perturbations
    T=100 #check how the body responds to perturbation for T timesteps 
    numperturb=25
    RecoveryDist = np.zeros((numperturb,T))
    RDfilename = os.path.join(filedir,'Analysis/RecoveryDist/RD_rep{r1:03d}_param{p1:05d}.pkl'.format(r1=int(rep), p1=int(paramid)))
#   if os.path.isfile(RDfilename):
#       print(RDfilename,'exists')
#   else:
    for i0 in range(numperturb):
        print('starting',i0)
        # set rng seed which takes paramid and rep into factor, so its different across all cores
        seed = int(str(time.time()).split('.')[1]) + paramid * rep
        np.random.seed(seed)

        # perturb
        init_body = np.copy(FB)
        # pick a random non-empty grid-location and a random cell-type (which it contains) to delete
        nonemptygrid = np.where(np.sum(FB,axis=0)>0)[0]
        r1 = nonemptygrid[np.random.randint(len(nonemptygrid))]
        presentcells = np.where(FB[:,r1]>0)[0]
        r2 = presentcells[np.random.randint(len(presentcells))]
        init_body[r2,r1] = 0

    # NOTE :removing a single cell-type from a single location in a tissue that is full and contiguous should do nothing (it does reduce the number of cells in the square, so maybe some of the cells here are dividing, but it doesn't change compositions), since it does not change the compositional neighborhood for any cell, unless the chosen grid-square is on the edge of some niche. So I should find that in the bottom sector, most of the time the perturbation does nothing.
    # a more meaningful perturbation would be to remove all cells of the same type from all neighboring grid-squares. This will change the numbers of cells in all these grid squares, as well as compositional neighborhoods for cells in at least one grid square.

        # update body
        BodyRecovery = get_body_recovery(init_body, CFR, celltypes, loc, T, adj_daughters, no_div_threshold) 
        # calculate difference between final_body and recovering_body across these T time-steps
        FBrep = (np.repeat(FB[:,:,np.newaxis],T,axis=2)>0).astype(int)
        dif1 = (((BodyRecovery>0).astype(int) - FBrep) !=0).astype(int)/celltypes # divide by celltypes to normalize
        RecoveryDist[i0,:] = np.mean(np.sum(dif1,axis=0),axis=0)# average hamming distance across all grid-locations along the T recovery time-steps

    pickle.dump(RecoveryDist,open(RDfilename,'wb'))
    print('processed and saved',paramid,rep)
    return RecoveryDist

def simulate_tissue_recovery_BigPerturbation(inp):
    paramid,rep = inp
    params = pickle.load(open('/home/somyamn/Signals_migration/model4_param.p','rb'))[paramid]
    p_stable, p_diff, density, celltypes, loc, T, adj_daughters, no_div_threshold, reps = params
    
    filedir = '/home/somyamn/Signals_migration/Dat_3/Adjdtrs{a}/pstable{s}/intden{d}'.format(a=int(adj_daughters*10), s=int(10*p_stable), d = int(10*density))
    
    # all required filenames
    FBfilename = os.path.join(filedir,'Data/FinalBody/FB_rep{r1:03d}_param{p1:05d}.npy'.format(r1=int(rep), p1=int(paramid)))
    CFRfilename = os.path.join(filedir,'Rules/CellFateRules/CFR_rep{r1:03d}_param{p1:05d}.pkl'.format(r1=int(rep), p1=int(paramid)))

    # load final body
    FB = np.load(FBfilename).astype(int)
    # load rules
    CFR = pickle.load(open(CFRfilename,'rb'))

    # perform 10 perturbations
    T=100 #check how the body responds to perturbation for T timesteps 
    numperturb=25
    RecoveryDist = np.zeros((numperturb,T))
    RDfilename = os.path.join(filedir,'Analysis/RecoveryDistBigPerturb/RD_rep{r1:03d}_param{p1:05d}.pkl'.format(r1=int(rep), p1=int(paramid)))
#   if os.path.isfile(RDfilename):
#       print(RDfilename,'exists')
#   else:
    for i0 in range(numperturb):
        # set rng seed which takes paramid and rep into factor, so its different across all cores
        seed = int(str(time.time()).split('.')[1]) + paramid * rep
        np.random.seed(seed)

        # perturb
        init_body = np.copy(FB)
        # pick a random non-empty grid-location and a random cell-type (which it contains) to delete
        nonemptygrid = np.where(np.sum(FB,axis=0)>0)[0]
        r1 = nonemptygrid[np.random.randint(len(nonemptygrid))]
        # find all grid squares adjacent to r1
        adj = SUCN5.get_adjacency_matrix(loc)
        adjr1 = np.append(np.where(adj[r1])[0],r1)
        
        presentcells = np.where(FB[:,r1]>0)[0]
        r2 = presentcells[np.random.randint(len(presentcells))]
        init_body[r2][adjr1] = 0

        # update body
        BodyRecovery = get_body_recovery(init_body, CFR, celltypes, loc, T, adj_daughters, no_div_threshold) 
        # calculate difference between final_body and recovering_body across these T time-steps
        FBrep = (np.repeat(FB[:,:,np.newaxis],T,axis=2)>0).astype(int)
        dif1 = (((BodyRecovery>0).astype(int) - FBrep) !=0).astype(int)/celltypes # divide by celltypes to normalize
        RecoveryDist[i0,:] = np.mean(np.sum(dif1,axis=0),axis=0)# average hamming distance across all grid-locations along the T recovery time-steps
    pickle.dump(RecoveryDist,open(RDfilename,'wb'))
    print('processed and saved',RDfilename)
    return RecoveryDist

def simulate_tissue_recovery_Perturb3(inp):
    paramid,rep = inp
    params = pickle.load(open('/home/somyamn/Signals_migration/model4_param.p','rb'))[paramid]
    p_stable, p_diff, density, celltypes, loc, T, adj_daughters, no_div_threshold, reps = params
    
    filedir = '/home/somyamn/Signals_migration/Dat_3/Adjdtrs{a}/pstable{s}/intden{d}'.format(a=int(adj_daughters*10), s=int(10*p_stable), d = int(10*density))
    
    # all required filenames
    FBfilename = os.path.join(filedir,'Data/FinalBody/FB_rep{r1:03d}_param{p1:05d}.npy'.format(r1=int(rep), p1=int(paramid)))
    CFRfilename = os.path.join(filedir,'Rules/CellFateRules/CFR_rep{r1:03d}_param{p1:05d}.pkl'.format(r1=int(rep), p1=int(paramid)))

    # load final body
    FB = np.load(FBfilename).astype(int)
    # load rules
    CFR = pickle.load(open(CFRfilename,'rb'))

    # perform 10 perturbations
    T=100 #check how the body responds to perturbation for T timesteps 
    numperturb=25
    RecoveryDist = np.zeros((numperturb,T))
    RDfilename = os.path.join(filedir,'Analysis/RecoveryDistEmptySquare/RD_rep{r1:03d}_param{p1:05d}.pkl'.format(r1=int(rep), p1=int(paramid)))
#   if os.path.isfile(RDfilename):
#       print(RDfilename,'exists')
#   else:
    for i0 in range(numperturb):
        # set rng seed which takes paramid and rep into factor, so its different across all cores
        seed = int(str(time.time()).split('.')[1]) + paramid * rep
        np.random.seed(seed)

        # perturb
        init_body = np.copy(FB)
        # pick a random non-empty grid-location and remove all cells in it
        nonemptygrid = np.where(np.sum(FB,axis=0)>0)[0]
        r1 = nonemptygrid[np.random.randint(len(nonemptygrid))]
        init_body[:,r1] = 0

        # update body
        BodyRecovery = get_body_recovery(init_body, CFR, celltypes, loc, T, adj_daughters, no_div_threshold) 
        # calculate difference between final_body and recovering_body across these T time-steps
        FBrep = (np.repeat(FB[:,:,np.newaxis],T,axis=2)>0).astype(int)
        dif1 = (((BodyRecovery>0).astype(int) - FBrep) !=0).astype(int)/celltypes # divide by celltypes to normalize
        RecoveryDist[i0,:] = np.mean(np.sum(dif1,axis=0),axis=0)# average hamming distance across all grid-locations along the T recovery time-steps
    pickle.dump(RecoveryDist,open(RDfilename,'wb'))
    print('processed and saved',RDfilename)
    return RecoveryDist


def get_tissue_recovery_BigPerturb_EmptySquare(I):
    
    files = glob.glob('/home/somyamn/Signals_migration/Dat_3/Adjdtrs*/pstable*/intden*/Analysis/RecoveryDistEmptySquare/*.pkl')
    # check how many paramids have been processed
    a = [(int(files[i][-9:-4]),int(files[i][-18:-15])) for i in range(len(files))]
    b = list(itertools.product(np.arange(0,135,1),np.arange(0,100,1)))
    if any(a):
        remaining_param = set(b) - set(a)
    else:
        remaining_paramid = b
    Inps = list(remaining_param)

#   for reps in range(100):
#       inp = (I,reps)
    inp = Inps[I]
    simulate_tissue_recovery(inp)
    simulate_tissue_recovery_BigPerturbation(inp)
    simulate_tissue_recovery_Perturb3(inp)
    return 0


def append_recovery_stats_to_df():
    # 1 square - 1 celltype (s1c1)
    # neighboring squares - 1 celtype (nsc1)
    # 1 square - all cells (s1ac)

    s1c1_unch = np.zeros((13500))
    s1c1_red = np.zeros((13500))
    s1c1_inc = np.zeros((13500))

    nsc1_unch = np.zeros((13500))
    nsc1_red = np.zeros((13500))
    nsc1_inc = np.zeros((13500))

    s1ac_unch = np.zeros((13500))
    s1ac_red = np.zeros((13500))
    s1ac_inc = np.zeros((13500))


    i=-1
    for paramid in range(135):
        params = pickle.load(open('/home/somyamn/Signals_migration/model4_param.p','rb'))[paramid]
        p_stable, p_diff, density, celltypes, loc, T, adj_daughters, no_div_threshold, reps = params
        filedir = '/home/somyamn/Signals_migration/Dat_3/Adjdtrs{a}/pstable{s}/intden{d}'.format(a=int(adj_daughters*10), s=int(10*p_stable), d = int(10*density))
        for rep in range(100):
            i+=1
            s1c1_filename = os.path.join(filedir,'Analysis/RecoveryDist/RD_rep{r1:03d}_param{p1:05d}.pkl'.format(r1=int(rep), p1=int(paramid)))
            nsc1_filename = os.path.join(filedir,'Analysis/RecoveryDistBigPerturb/RD_rep{r1:03d}_param{p1:05d}.pkl'.format(r1=int(rep), p1=int(paramid)))
            s1ac_filename = os.path.join(filedir,'Analysis/RecoveryDistEmptySquare/RD_rep{r1:03d}_param{p1:05d}.pkl'.format(r1=int(rep), p1=int(paramid)))
            
            s1c1_recoverydist = pickle.load(open(s1c1_filename,'rb'))
            nsc1_recoverydist = pickle.load(open(nsc1_filename,'rb'))
            s1ac_recoverydist = pickle.load(open(s1ac_filename,'rb'))
            
            s1c1_init_fin_dist = s1c1_recoverydist[:,-1] - s1c1_recoverydist[:,0]
            nsc1_init_fin_dist = nsc1_recoverydist[:,-1] - nsc1_recoverydist[:,0]
            s1ac_init_fin_dist = s1ac_recoverydist[:,-1] - s1ac_recoverydist[:,0]

            s1c1_unch[int(i)] = sum(s1c1_init_fin_dist == 0)/len(s1c1_init_fin_dist)
            s1c1_red[int(i)] = sum(s1c1_init_fin_dist < 0)/len(s1c1_init_fin_dist)
            s1c1_inc[int(i)] = sum(s1c1_init_fin_dist > 0)/len(s1c1_init_fin_dist)
    
            nsc1_unch[int(i)] = sum(nsc1_init_fin_dist == 0)/len(nsc1_init_fin_dist)
            nsc1_red[int(i)] = sum(nsc1_init_fin_dist < 0)/len(nsc1_init_fin_dist)
            nsc1_inc[int(i)] = sum(nsc1_init_fin_dist > 0)/len(nsc1_init_fin_dist)

            s1ac_unch[int(i)] = sum(s1ac_init_fin_dist == 0)/len(s1ac_init_fin_dist) 
            s1ac_red[int(i)] = sum(s1ac_init_fin_dist < 0)/len(s1ac_init_fin_dist)
            s1ac_inc[int(i)] = sum(s1ac_init_fin_dist > 0)/len(s1ac_init_fin_dist)
        print('processed param',paramid)

    df['s1c1_unch'] = s1c1_unch
    df['s1c1_red'] = s1c1_red
    df['s1c1_inc'] = s1c1_inc

    df['nsc1_unch'] = nsc1_unch
    df['nsc1_red'] = nsc1_red
    df['nsc1_inc'] = nsc1_inc

    df['s1ac_unch'] = s1ac_unch
    df['s1ac_red'] = s1ac_red
    df['s1ac_inc'] = s1ac_inc

    dict1['finalbody_df'] = df
    pickle.dump(dict1, open('/home/somyamn/Signals_migration/Dat_3/finalbody_richness_disperseness.pkl','wb'))
    return 0


# NEW ANALYSIS: CHECK IN CFR: how many times does an intrinsically stable cell become unstable in a neighborhood, and how many times does the opposite happen? 


def count_extrinsic_cell_influence():
    # for each CellFateRule, find out how many times an intrinsically stable/unstable cell was unstable/stable in some neighborhood
    # report two numbers for each sample-point: 'num_stabilized', 'num_destabilized'
    num_stabilized = np.zeros((13500))
    num_destabilized = np.zeros((13500))
    i=0
    CFR_df2 = pd.DataFrame(columns=['num_stabilized','num_destabilized'])
    for paramid in range(135):
        params = pickle.load(open('/home/somyamn/Signals_migration/model4_param.p','rb'))[paramid]
        p_stable, p_diff, density, celltypes, loc, T, adj_daughters, no_div_threshold, reps = params
        filedir = '/home/somyamn/Signals_migration/Dat_3/Adjdtrs{a}/pstable{s}/intden{d}/Rules'.format(a=int(adj_daughters*10), s=int(10*p_stable), d = int(10*density))
        for rep in range(100):
            cfr_fname = os.path.join(filedir,'CellFateRules/CFR_rep{r1:03d}_param{p1:05d}.pkl'.format(r1=int(rep), p1=int(paramid)))
            cfr = pickle.load(open(cfr_fname,'rb'))
            
            si_fname = os.path.join(filedir,'stable_interactions/SI_rep{r1:03d}_param{p1:05d}.npy'.format(r1=int(rep), p1=int(paramid)))
            si = np.diag(np.load(si_fname))# 2 if stable, 1 if unstable, 0 if it doesnt care. Flipped to keep with the ordering of cells in cfr
            
            keylist = list(cfr.keys())
            destab = 0
            stab = 0
            for k in keylist:
                cells_present = np.flipud(np.array([int(k[j]) for j in range(len(k))])>0)

                dm = np.diag(cfr[k]['differentiation_matrix'])
                mm = cfr[k]['migration_vector']
                    

                if any((si==1)&cells_present):
                    x1 = (si==1)&((dm==1)&(mm==0)).astype(int) # stabilized
                    stab += sum(x1[cells_present])/sum(cells_present[si==1].astype(int))
                if any((si==2)&cells_present):
                    x2 = (si==2)&((dm==0)|(mm>0)).astype(int) #destabilized
                    destab += sum(x2[cells_present])/sum(cells_present[si==2].astype(int))

            num_stabilized[i] = stab/len(keylist)     
            num_destabilized[i] = destab/len(keylist)
            i+=1
            print(i)

    CFR_df2['num_stabilized'] = num_stabilized
    CFR_df2['num_destabilized'] = num_destabilized
    dict1['stabilized_destabilized_cells_df'] = CFR_df2
    pickle.dump(dict1, open('/home/somyamn/Signals_migration/Dat_3/finalbody_richness_disperseness.pkl','wb'))
    return num_stabilized, num_destabilized


# how do model parameters control features of cell fate rules?
# interesting features -- cell fate potential, migration potential [hierarchical or uniform]
# infer -- do some cells look like stem cells?


def calculate_cell_fate_migration_potential(cfr_fname):
    cfr = pickle.load(open(cfr_fname,'rb'))
    cell_fate_potentials = np.zeros((5))# how many other cell-types can cell-type i become across neighborhoods
    cell_migration_potential = np.zeros((5))# number of neighborhoods in which cell-type-i migrates
    num_stable_nbhd = 0# number of stable neighborhoods encoded by this rule
    av_stable_nbhd_size = 0# average number of cell-types across stable neighborhoods
    num_stable_nbhd_per_cell = np.zeros((5))# number of stable neighborhoods that any cell-type is a part of

    keylist = list(cfr.keys())
    all_neighborhoods = np.array([np.array([int(k[j]) for j in range(len(k))])>0 for k in keylist])
    is_stable = np.zeros((len(all_neighborhoods)))
    all_diff = np.zeros((5,5))
    for ki,k in enumerate(keylist):
        cells_present = all_neighborhoods[ki].astype(int)
        dm = cfr[k]['differentiation_matrix']
        cells_update = cells_present @ dm

        all_diff += dm

        mm = cfr[k]['migration_vector']
    # how many neighborhoods is a cell-type able to migrate in
        cell_migration_potential += (mm>0).astype(int)
        #is_stable[ki] = (np.sum(np.diag(dm))==5)&(np.sum(mm)==0)# no differentiation and no migration
    # the above criterion is too strong: a nbhd can be stable despite differentiation given that the set of cell-types remains constant
        is_stable[ki] = np.all((cells_present == cells_update)==True)#&(np.sum(mm)==0)
    # across all neighborhoods, add up the fate potential of each cell-type
    cell_fate_potentials = np.sum((all_diff>0).astype(int),axis=1)
    num_stable_nbhd = np.sum(is_stable.astype(int))
    stable_nbhds = all_neighborhoods[is_stable>0]
    av_stable_nbhd_size = np.mean(np.sum(stable_nbhds,axis=1))
    num_stable_nbhd_per_cell = np.mean(np.sum(stable_nbhds,axis=0))

    # OLD: Interesting question: does the rule possess stem-like cell-types?
    
    # presence of stem-cells in a stable neighborhood = presence of one and only one cell-type whose fate potential includes and at least 80% of the cell-types composing this neighborhood 
    # how many cells in a stable neighborhood have fate potential overlapping with half the cell-types in the neighborhood?-- average number across stable neighborhoods
    #nbhd_potential_overlap = stable_nbhds @ (all_diff>0).astype(int).T# number of cell-types in a nbhd (row) that is within the fate-potential of a cell-type (column)
    #nbhd_within_cell_potential_overlap = nbhd_potential_overlap * stable_nbhds# we only want the overlap with fate potentials of cell-types present in the nbhd
    #stem_like_cells = nbhd_within_cell_potential_overlap > np.tile(np.sum(stable_nbhds,axis=1)*0.95,(5,1)).T
    
    # number of stable neighborhoods with stem-like cell-type
    # ignore singleton neighborhoods --- keep track of which neighborhoods are singleton, and don't consider them for the final calculation
    #num_nbhd_with_stem_cell = np.sum(((np.sum(stem_like_cells,axis=1)==1)&(np.sum(stable_nbhds,axis=1)>1)).astype(int))
   
    av_cell_pot = np.mean(cell_fate_potentials)
    av_cell_mig = np.mean(cell_migration_potential)
    av_stable_nbhd_per_cell = np.mean(num_stable_nbhd_per_cell)
    potential_disparity = max(cell_fate_potentials) - np.mean(cell_fate_potentials)
    is_stemcells = np.any(cell_fate_potentials==5)
    print('processed',cfr_fname)
    return av_cell_pot, av_cell_mig, num_stable_nbhd, av_stable_nbhd_size, av_stable_nbhd_per_cell, potential_disparity, is_stemcells


def get_cell_fate_migration_potential():
    CFR_array = np.zeros((13500,11))
    i=0
    for paramid in range(135):
        params = pickle.load(open('/home/somyamn/Signals_migration/model4_param.p','rb'))[paramid]
        p_stable, p_diff, density, celltypes, loc, T, adj_daughters, no_div_threshold, reps = params
        filedir = '/home/somyamn/Signals_migration/Dat_3/Adjdtrs{a}/pstable{s}/intden{d}/Rules'.format(a=int(adj_daughters*10), s=int(10*p_stable), d = int(10*density))
        for rep in range(100):
            filename = os.path.join(filedir,'CellFateRules/CFR_rep{r1:03d}_param{p1:05d}.pkl'.format(r1=int(rep), p1=int(paramid)))
            X = calculate_cell_fate_migration_potential(filename)
            CFR_array[i,0] = p_stable
            CFR_array[i,1] = p_diff
            CFR_array[i,2] = density
            CFR_array[i,3] = adj_daughters
            CFR_array[i,4] = X[0]
            CFR_array[i,5] = X[1]
            CFR_array[i,6] = X[2]
            CFR_array[i,7] = X[3]
            CFR_array[i,8] = X[4]
            CFR_array[i,9] = X[5]
            CFR_array[i,10] = X[6]
            i+=1
    CFR_df = pd.DataFrame(data = CFR_array, columns = ['p_stable','p_diff','density','adj_dtrs','av_cell_potential','av_cell_mig', 'num_stable_nbhd', 'av_stable_nbhd_size', 'av_stable_nbhd_per_cell','potential_disparity','is_stemcells'])
    dict1['CFR_df'] = CFR_df
    pickle.dump(dict1, open('/home/somyamn/Signals_migration/Dat_3/finalbody_richness_disperseness.pkl','wb'))
    return CFR_df



# what fraction of domains in the final tissue contains stem-like cells?
# for each paramid-rep, load final body, cell-fate rules, stable_neighborhoods
# find unique domain cell-compositions for the final body
# find out how many of the domains are stable neighborhoods
# how many of the domains contain stem-like cells


def count_final_body_cells_potential(filenames):
    CFR_file = filenames[0]
    FB_file = filenames[1]
    cfr = pickle.load(open(CFR_file,'rb'))
    final_body = np.load(FB_file)
    final_body_cells = np.sum(final_body,axis=1)>0
    
    keylist = list(cfr.keys())
    all_neighborhoods = np.array([np.array([int(k[j]) for j in range(len(k))])>0 for k in keylist])
    all_diff = np.zeros((5,5))
    for ki,k in enumerate(keylist):
        cells_present = all_neighborhoods[ki].astype(int)
        dm = cfr[k]['differentiation_matrix']
        all_diff += dm
    all_diff = (all_diff>0).astype(int)
    cell_potentials =  np.sum(all_diff,axis=1)
    cell_potentials_ultimate = np.sum((np.linalg.matrix_power(all_diff,5)>0).astype(int),axis=1)

    avpot_tissuecells = np.mean(cell_potentials[final_body_cells])
    maxpot_tissuecells = max(cell_potentials[final_body_cells])
    maxpot_tissuecells_ultimate = max(cell_potentials_ultimate[final_body_cells])
    num_tissuecells = sum(final_body_cells.astype(int))
    print('processed',CFR_file)
    return num_tissuecells, avpot_tissuecells, maxpot_tissuecells, maxpot_tissuecells_ultimate



def get_FB_cells_potential():
    FBcellpotential_array = np.zeros((13500, 10))
    i=0
    for paramid in range(135):
        params = pickle.load(open('/home/somyamn/Signals_migration/model4_param.p','rb'))[paramid]
        p_stable, p_diff, density, celltypes, loc, T, adj_daughters, no_div_threshold, reps = params
        filedir = '/home/somyamn/Signals_migration/Dat_3/Adjdtrs{a}/pstable{s}/intden{d}'.format(a=int(adj_daughters*10), s=int(10*p_stable), d = int(10*density))
        for rep in range(100):
            CFR_file = os.path.join(filedir,'Rules/CellFateRules/CFR_rep{r1:03d}_param{p1:05d}.pkl'.format(r1=int(rep), p1=int(paramid)))
            FB_file = os.path.join(filedir,'Data/FinalBody/FB_rep{r1:03d}_param{p1:05d}.npy'.format(r1=int(rep), p1=int(paramid)))
            filenames = [CFR_file,FB_file]
            num_tissuecells, avpot_tissuecells, maxpot_tissuecells, maxpot_tissuecells_ultimate = count_final_body_cells_potential(filenames)
            FBcellpotential_array[i,0] = p_stable
            FBcellpotential_array[i,1] = p_diff
            FBcellpotential_array[i,2] = density
            FBcellpotential_array[i,3] = adj_daughters
            FBcellpotential_array[i,4] = num_tissuecells
            FBcellpotential_array[i,5] = avpot_tissuecells
            FBcellpotential_array[i,6] = paramid
            FBcellpotential_array[i,7] = rep
            FBcellpotential_array[i,8] = maxpot_tissuecells
            FBcellpotential_array[i,9] = maxpot_tissuecells_ultimate
            i+=1

    FB_cellpotential_df = pd.DataFrame(data = FBcellpotential_array, columns = ['p_stable','p_diff','density','adj_dtrs','num_tissuecells','avpot_tissuecells','paramid','rep', 'maxpot_tissuecells', 'maxpot_tissuecells_ultimate'])
    dict1['FB_cellpotential_df'] = FB_cellpotential_df
    pickle.dump(dict1, open('/home/somyamn/Signals_migration/Dat_3/finalbody_richness_disperseness.pkl','wb'))
    return FB_cellpotential_df



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-I', help = 'parameter index', type=int)
    args = parser.parse_args()



#   save_tissue_richness_disperseness_dev(args.I)
    get_tissue_recovery_BigPerturb_EmptySquare(args.I -1)
