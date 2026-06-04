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
import itertools

# TO SET UP
# code to look at homeostasis

# for each final body, perform 10 independent random perturbations (delete all cells of a cell-type from a random grid-location)
# check: does it return to the body return to its original composition [compute difference in composition across grid-locations over time.. is this decreasing? increasing?]
# test for 50 time-steps

dict1 = pickle.load(open('/Users/somya/Documents/SIGMIG_from_Bigram2/Dat_3/finalbody_richness_disperseness.pkl','rb'))
df = dict1['finalbody_df']
x = dict1['sectordict']
c_tail = x['c_tail']; m_top=x['m_top']; c_top=x['c_top']; m_center=x['m_center']; c_center=x['c_center']




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

def simulate_tissue_recovery_VeryLargePert(inp):
    tim1 = time.time()
    pertsize=5# the size of the neighborhood of a randomly chosen cell from which a cell-type will be removed 
    paramid,rep = inp
    params = pickle.load(open('/Users/somya/Documents/SIGMIG_from_Bigram2/model4_param.p','rb'))[paramid]
    p_stable, p_diff, density, celltypes, loc, T, adj_daughters, no_div_threshold, reps = params
    
    filedir = '/Users/somya/Documents/SIGMIG_from_Bigram2/Dat_3/Adjdtrs{a}/pstable{s}/intden{d}'.format(a=int(adj_daughters*10), s=int(10*p_stable), d = int(10*density))
    
    # all required filenames
    FBfilename = os.path.join(filedir,'Data/FinalBody/FB_rep{r1:03d}_param{p1:05d}.npy'.format(r1=int(rep), p1=int(paramid)))
    CFRfilename = os.path.join(filedir,'Rules/CellFateRules/CFR_rep{r1:03d}_param{p1:05d}.pkl'.format(r1=int(rep), p1=int(paramid)))

    # load final body
    FB = np.load(FBfilename).astype(int)
    # load rules
    CFR = pickle.load(open(CFRfilename,'rb'))
    
    # perform 10 perturbations
    T=100 #check how the body responds to perturbation for T timesteps 
    numperturb=10
    RecoveryDist = np.zeros((numperturb,T))
    RDfilename = os.path.join(filedir,'Analysis/RecoveryDistVLPerturb/Adj{a}/RD_rep{r1:03d}_param{p1:05d}.pkl'.format(a=int(pertsize), r1=int(rep), p1=int(paramid)))
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
        adjplus = np.linalg.matrix_power(adj+np.eye(len(adj)),int(pertsize))
        adjr1 = np.append(np.where(adjplus[r1])[0],r1)
        
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
    print('processed and saved',RDfilename,'in',(time.time()-tim1)/60,'min')
    return 0


def get_tissue_recovery_VLP():
    pertsize=5
  # b = list(itertools.product(np.arange(0,135,1),np.arange(0,100,1)))
  # Inps = list(b)
  # for inp in Inps:
    for r1 in range(100):
        for p1 in np.arange(0,5,1):
            n = simulate_tissue_recovery_VeryLargePert((p1,r1))
  # with Pool(6) as p:
  #     p.map(simulate_tissue_recovery_VeryLargePert,Inps)
    return 0



def append_recovery_stats_to_df():
    # 1 square - 1 celltype (s1c1)
    # neighboring squares - 1 celtype (nsc1)
    # 1 square - all cells (s1ac)
    # 
    s1c1_unch = np.zeros((13500))
    s1c1_red = np.zeros((13500))
    s1c1_inc = np.zeros((13500))

    nsc1_unch = np.zeros((13500))
    nsc1_red = np.zeros((13500))
    nsc1_red_thresh = np.zeros((13500))
    nsc1_inc = np.zeros((13500))
    nsc1_insuf = np.zeros((13500))

    s1ac_unch = np.zeros((13500))
    s1ac_red = np.zeros((13500))
    s1ac_inc = np.zeros((13500))
    s1ac_insuf = np.zeros((13500))

    n5sc1_unch = np.zeros((13500))
    n5sc1_red = np.zeros((13500))
    n5sc1_red_thresh = np.zeros((13500))
    n5sc1_inc = np.zeros((13500))
    n5sc1_insuf = np.zeros((13500))

    i=-1
    for paramid in range(135):
        params = pickle.load(open('/Users/somya/Documents/SIGMIG_from_Bigram2/model4_param.p','rb'))[paramid]
        p_stable, p_diff, density, celltypes, loc, T, adj_daughters, no_div_threshold, reps = params
        filedir = '/Users/somya/Documents/SIGMIG_from_Bigram2/Dat_3/Adjdtrs{a}/pstable{s}/intden{d}'.format(a=int(adj_daughters*10), s=int(10*p_stable), d = int(10*density))
        for rep in range(100):
            i+=1
            s1c1_filename = os.path.join(filedir,'Analysis/RecoveryDist/RD_rep{r1:03d}_param{p1:05d}.pkl'.format(r1=int(rep), p1=int(paramid)))
            nsc1_filename = os.path.join(filedir,'Analysis/RecoveryDistBigPerturb/RD_rep{r1:03d}_param{p1:05d}.pkl'.format(r1=int(rep), p1=int(paramid)))
            s1ac_filename = os.path.join(filedir,'Analysis/RecoveryDistEmptySquare/RD_rep{r1:03d}_param{p1:05d}.pkl'.format(r1=int(rep), p1=int(paramid)))
            n5sc1_filename = os.path.join(filedir,'Analysis/RecoveryDistVLPerturb/Adj5/RD_rep{r1:03d}_param{p1:05d}.pkl'.format(r1=int(rep), p1=int(paramid)))

            s1c1_recoverydist = pickle.load(open(s1c1_filename,'rb'))
            nsc1_recoverydist = pickle.load(open(nsc1_filename,'rb'))
            s1ac_recoverydist = pickle.load(open(s1ac_filename,'rb'))
            n5sc1_recoverydist = pickle.load(open(n5sc1_filename,'rb'))
            
            s1c1_init_fin_dist = s1c1_recoverydist[:,-1] - s1c1_recoverydist[:,0]
            nsc1_init_fin_dist = nsc1_recoverydist[:,-1] - nsc1_recoverydist[:,0]
            s1ac_init_fin_dist = s1ac_recoverydist[:,-1] - s1ac_recoverydist[:,0]
            n5sc1_init_fin_dist = n5sc1_recoverydist[:,-1] - n5sc1_recoverydist[:,0]

            s1c1_init_fin_ratio = s1c1_recoverydist[:,-1] / s1c1_recoverydist[:,0]
            nsc1_init_fin_ratio = nsc1_recoverydist[:,-1] / nsc1_recoverydist[:,0]
            s1ac_init_fin_ratio = s1ac_recoverydist[:,-1] / s1ac_recoverydist[:,0]
            n5sc1_init_fin_ratio = n5sc1_recoverydist[:,-1] / n5sc1_recoverydist[:,0]
            

            s1c1_unch[int(i)] = sum(s1c1_init_fin_ratio == 1)/len(s1c1_init_fin_dist)
            s1c1_red[int(i)] = sum(s1c1_init_fin_ratio < 1)/len(s1c1_init_fin_dist)
            s1c1_inc[int(i)] = sum(s1c1_init_fin_ratio > 1)/len(s1c1_init_fin_dist)

            s1ac_unch[int(i)] = sum(s1ac_init_fin_ratio == 1)/len(s1ac_init_fin_dist) 
            s1ac_red[int(i)] = sum(s1ac_init_fin_ratio < 1)/len(s1ac_init_fin_dist)
            s1ac_inc[int(i)] = sum(s1ac_init_fin_ratio > 1)/len(s1ac_init_fin_dist)
            s1ac_insuf[int(i)] = sum(s1ac_init_fin_ratio == 1)/len(s1ac_init_fin_dist) 
    
            nsc1_unch[int(i)] = sum(nsc1_init_fin_ratio == 1)/len(nsc1_init_fin_dist)
            nsc1_red[int(i)] = sum(nsc1_init_fin_ratio < 1)/len(nsc1_init_fin_dist)
            nsc1_red_thresh[int(i)] = sum(nsc1_init_fin_ratio <= 0.8)/len(nsc1_init_fin_dist)
            nsc1_insuf[int(i)] = sum((nsc1_init_fin_ratio>0.8)&(nsc1_init_fin_ratio<=1))/len(nsc1_init_fin_dist)
            nsc1_inc[int(i)] = sum(nsc1_init_fin_ratio > 1)/len(nsc1_init_fin_dist)
    
            n5sc1_unch[int(i)] = sum(n5sc1_init_fin_ratio == 1)/len(n5sc1_init_fin_dist)
            n5sc1_red[int(i)] = sum(n5sc1_init_fin_ratio < 1)/len(n5sc1_init_fin_dist)
            n5sc1_red_thresh[int(i)] = sum(n5sc1_init_fin_ratio <= 0.6)/len(n5sc1_init_fin_dist)
            n5sc1_insuf[int(i)] = sum((n5sc1_init_fin_ratio>0.6)&(n5sc1_init_fin_ratio<=1))/len(n5sc1_init_fin_dist)
            n5sc1_inc[int(i)] = sum(n5sc1_init_fin_ratio > 1)/len(n5sc1_init_fin_dist)
        print('processed param',paramid)

    df['s1c1_unch'] = s1c1_unch
    df['s1c1_red'] = s1c1_red
    df['s1c1_inc'] = s1c1_inc

    df['nsc1_unch'] = nsc1_unch
    df['nsc1_red'] = nsc1_red
    df['nsc1_red_thresh'] = nsc1_red_thresh
    df['nsc1_inc'] = nsc1_inc
    df['nsc1_insuf'] = nsc1_insuf

    df['s1ac_unch'] = s1ac_unch
    df['s1ac_red'] = s1ac_red
    df['s1ac_inc'] = s1ac_inc
    df['s1ac_insuf'] = s1ac_insuf

    df['n5sc1_unch'] = n5sc1_unch
    df['n5sc1_red'] = n5sc1_red
    df['n5sc1_red_thresh'] = n5sc1_red_thresh
    df['n5sc1_inc'] = n5sc1_inc
    df['n5sc1_insuf'] = n5sc1_insuf

    dict1['finalbody_df'] = df
    pickle.dump(dict1, open('/Users/somya/Documents/SIGMIG_from_Bigram2/Dat_3/finalbody_richness_disperseness.pkl','wb'))
    return 0

def append_average_recovery_to_df():
    # 1 square - 1 celltype (s1c1)
    # neighboring squares - 1 celltype (nsc1)
    # 1 square - all cells (s1ac)
    # 5 neighboring squares - 1 celltype (n5sc1)

    s1c1_avred = np.zeros((13500))
    nsc1_avred = np.zeros((13500))
    s1ac_avred = np.zeros((13500))
    n5sc1_avred = np.zeros((13500))

    i=-1
    for paramid in range(135):
        params = pickle.load(open('/Users/somya/Documents/SIGMIG_from_Bigram2/model4_param.p','rb'))[paramid]
        p_stable, p_diff, density, celltypes, loc, T, adj_daughters, no_div_threshold, reps = params
        filedir = '/Users/somya/Documents/SIGMIG_from_Bigram2/Dat_3/Adjdtrs{a}/pstable{s}/intden{d}'.format(a=int(adj_daughters*10), s=int(10*p_stable), d = int(10*density))
        for rep in range(100):
            i+=1
            s1c1_filename = os.path.join(filedir,'Analysis/RecoveryDist/RD_rep{r1:03d}_param{p1:05d}.pkl'.format(r1=int(rep), p1=int(paramid)))
            nsc1_filename = os.path.join(filedir,'Analysis/RecoveryDistBigPerturb/RD_rep{r1:03d}_param{p1:05d}.pkl'.format(r1=int(rep), p1=int(paramid)))
            s1ac_filename = os.path.join(filedir,'Analysis/RecoveryDistEmptySquare/RD_rep{r1:03d}_param{p1:05d}.pkl'.format(r1=int(rep), p1=int(paramid)))
            n5sc1_filename = os.path.join(filedir,'Analysis/RecoveryDistVLPerturb/Adj5/RD_rep{r1:03d}_param{p1:05d}.pkl'.format(r1=int(rep), p1=int(paramid)))

            s1c1_recoverydist = pickle.load(open(s1c1_filename,'rb'))
            nsc1_recoverydist = pickle.load(open(nsc1_filename,'rb'))
            s1ac_recoverydist = pickle.load(open(s1ac_filename,'rb'))
            n5sc1_recoverydist = pickle.load(open(n5sc1_filename,'rb'))
            
            # ratios of injury magnitudes after healing vs right after injury
            s1c1_init_fin_ratio = s1c1_recoverydist[:,-1] / s1c1_recoverydist[:,0]
            nsc1_init_fin_ratio = nsc1_recoverydist[:,-1] / nsc1_recoverydist[:,0]
            s1ac_init_fin_ratio = s1ac_recoverydist[:,-1] / s1ac_recoverydist[:,0]
            n5sc1_init_fin_ratio = n5sc1_recoverydist[:,-1] / n5sc1_recoverydist[:,0]

            s1c1_avred[int(i)] = np.median(1 - s1c1_init_fin_ratio)
            nsc1_avred[int(i)] = np.median(1 - nsc1_init_fin_ratio) 
            s1ac_avred[int(i)] = np.median(1 - s1ac_init_fin_ratio)
            n5sc1_avred[int(i)] = np.median(1 - n5sc1_init_fin_ratio)
        print('processed param',paramid)

    df['s1c1_avred'] = s1c1_avred
    df['nsc1_avred'] = nsc1_avred
    df['s1ac_avred'] = s1ac_avred
    df['n5sc1_avred'] = n5sc1_avred

    dict1['finalbody_df'] = df
    pickle.dump(dict1, open('/Users/somya/Documents/SIGMIG_from_Bigram2/Dat_3/finalbody_richness_disperseness.pkl','wb'))
    return 0


if __name__=='__main__':

  # perturbsizes = [2,3,4,5]
  # with Pool(5) as pool:
  #     pool.map(get_tissue_recovery_VLP, perturbsizes)
  # get_tissue_recovery_VLP()
    append_recovery_stats_to_df()
  # append_average_recovery_to_df()
