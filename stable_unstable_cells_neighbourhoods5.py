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
import networkx as nx
import argparse
import math
# 3 kinds of cells: stable cells (differentiate into themselves and do not migrate), differentiating cells, migrating cells

# 2 kinds of neighbourhoods: stable, unstable-diff, unstable-mig
    # stable neighbourhoods: all cells differentiate into one of the cell-types present in the neighbourhood
    # unstable-diff neighbourhoods: all cells differentiate into some cell-type not present here 
    # unstable-mig neighbourhoods: all cells migrate a fixed, randomly generated distance away


# decisions based on signals + state:

# stable cell in stable neighbourhood: remains the same
# unstable cell in stable neighbourhood: remains the same

# stable cell in unstable-diff neighbourhood: differentiates into a random cell-type not present in this neighbourhood 
# differentiating cell in unstable-diff neighbourhood: differentiates into a random cell-type not present in the neighbourhood
# migrating cell in unstable-diff neighbourhood: migrates and differentiates

# stable cell in unstable-mig neighbourhood: migrates to random locaion 
# differentiating cell in unstable-mig neighbourhood: differentiates and migrates
# migrating cell in unstable-mig neighbourhood: migrates to random location


# in systems where cells only listen to cell-intrinsic cues, cells are expected to be scattered all across the body
# in systems where cells only listen to cell-extrinsic cues, cells are expected to form segregated regions
# in systems where cells listen to both intrinsic and extrinsic cues, it becomes possible to form hierarchical organizations


    # this truthtable contains ALL rules for cases where cell only listens to cell internal cues OR only listens to external signals OR both internal and external cues

    # when cells listen only to instrinsic cues:
    # (cells follow these rules also when all cells in the neighbourhood are of their own type)
    # there are stable cells and unstable cells
    # stable cells do not migrate, and they differentiate into themselves
    # unstable cells differentiate into other types or migrate

    # when cells listen to both intrinsic and extrinsic cues:
    # pairwaise interactions:- cells can stabilize or destabilize each other
    # stable neighbourhoods: all cells interact to stabilize each other
    # unstable neighbourhoods: at least one cell is unstable due to (innate instability + lack of stabilizing interactions) OR (a destabilizing interaction)

    # when cells listen to only extrinsic cues:
    # all cells in the neighbourhood make the same decision.
    # they all either differentiate into the same cell type/ migrate the same number of steps away
    # stable neighbourhoods: all cells differentiate into one of the celltypes within the neighbourhood
    # unstable neighbourhood: all cells differentiate into a random cell-type/ jump a random number of steps

def array_col_to_bin(arr):
    # inputs= arr: binary array whose columns are to be turned into binary strings, 
    binarray = []
    for i1 in range(np.shape(arr)[1]):
        x = arr[:,i1]
        binarray += [''.join([str(xi) for xi in x])]
    return binarray

# cell-types are either stable or unstable on their own (acc p_stable), and interactions between pairs of cell-types can be stable/unstable/none (acc p_stable, den)

def generate_random_interactions(celltypes, p_stable, den):
    # pairwise cellular interactions
    # when cells receive destabilizing inputs, do cells differentiate or migrate?
    # what do cells differentiate into? how many steps do they migrate?
    stable_interactions = (np.random.uniform(0,1,(celltypes,celltypes)) < p_stable).astype(int) + 1# 2= stabilizing interaction, 1 = destabilizing
    is_interact = np.random.uniform(0,1,(celltypes,celltypes)) < den# 1=does interact, 0 = doesn't interact
    is_interact = (is_interact + np.eye(celltypes))>0
    stable_interactions = stable_interactions * is_interact # 0 = doesn't interact, 1 = destabilizing, 2= stabilizing 

    return stable_interactions

# cells in neighborhoods where they don't interact with any other cell-type interact with only intrinsic cues
# cells in neighbohoods where they do interact with other cell-types listen only to extrinsic cues
# therefore, 'state_dep' and 'sig_dep' depend on the parameter 'den'


def generate_intrinsic_extrinsic_dependence_masks(stable_interactions, celltypes):
    # for each cellular neighborhood, define two vectors: v1 = does a cell depend on intrinsic cues?, v2 = does a cell depend on extrinsic cues?
    # make a mask for pairwise cellular interactions based on v1, v2
    # Output: the set of masks for each neighborhood (dictionary)
    # mask[i,j] = 1 if fate of cell j depends on interaction with cell i

    Masks = {}
    All_neighborhoods_bin = [np.binary_repr(x,width=celltypes) for x in np.arange(1,2**celltypes,1)]# convert into np array of 0/1 ints
    for n in All_neighborhoods_bin:
        # generate v1, v2 (which cells depend on intrinsic, extrinsic cues)
        # constraints: each cell must depend on at least one type of cue
        v1=np.zeros((celltypes)); v2=np.zeros((celltypes))
        cells = np.array([int(i) for i in n])
        cells_present = np.where(cells)[0]

        v1[cells_present] = [sum(stable_interactions[cells>0,i])==0 for i in cells_present]# cells listen to intrinsic cues when there are no available external cues
        v2[cells_present] = 1-v1[cells_present]# cells listen to external cues whenever some are available

        # interaction_mask: does a cell care about some other cell in the neighborhood
        interaction_mask = np.ones((celltypes, celltypes))

        interaction_mask[:,cells>0] = interaction_mask[:,cells>0] @ np.diag(v2[cells>0])
        interaction_mask[cells>0,cells>0] = v1[cells>0]# cells that care about intrinsic cues

        # if the neighborhood has only a single celltype, it depends on intrinsic cues
        if sum(cells)==1:
            interaction_mask[cells>0, cells>0]=1
        
        # cell doesn't care about cells not in the neighborhood
        interaction_mask[cells==0,:]=0; interaction_mask[:,cells==0] = 0
        
        Masks[n] = {'cells':cells, 'mask':interaction_mask}
    return Masks


def generate_cellfate_rules(celltypes, loc, Masks, stable_interactions, p_diff):
    # outputs: dictionary[identifiers of different cellular neighborhoods]
    # what to store: (a) set of celltypes in neighborhood, (b) masks, (c) binary cell differentiation matrix, (d) jump size vector

    # rule: interactions from other cell-types modify intrinsic bahaviors. instrinsically stable cells with destabilizing neighbor = unstable
    # intrinsically unstable cell with stabilizing neighbor = stable

    CellFateTable={}
    N = Masks.keys()# list of cellular neighborhoods

    # if a cell is intrinsically unstable, and it receives no inputs from neighbors, it always differentiates into a particular other cell-type/ migrates by a particular jump-size
    intrinsic_diff = np.zeros((celltypes)); intrinsic_mig = np.zeros((celltypes))
    differentiating_cells = np.random.uniform(0,1,celltypes) <= p_diff
    intrinsic_diff[differentiating_cells] = np.random.randint(0,celltypes,sum(differentiating_cells)).astype(int)# sometimes, these turn out to be the same celltype
    intrinsic_mig[differentiating_cells==0] = np.random.randint(0,loc,sum(differentiating_cells==0))+1

    for n in N:
        cells = Masks[n]['cells']
        mask = Masks[n]['mask']
        
        interactions = np.ones((celltypes, celltypes))*0# no celltype cares about any inputs. all celltypes are stable.
        interactions[np.where(mask)] = stable_interactions[np.where(mask)]# use the mask to derive inputs from relevant intrinsic and extrinsic cues   

        nonself_interactions = np.copy(interactions)
        nonself_interactions = nonself_interactions - np.diag(np.diag(nonself_interactions))

        is_destab_extrinsic = (np.sum(nonself_interactions==1,axis=0)>0)# checks whether there are any nonself destabilizing interactions
        is_stab_extrinsic = (np.sum(nonself_interactions==2,axis=0)>0)&(is_destab_extrinsic==0)# checks whether there are cells with stabilizing but no destabilizing inputs
        is_destab_intrinsic = np.diag(interactions==1)&(is_destab_extrinsic==0)&(is_stab_extrinsic==0)# checks whether the cell is inherently unstable and doesn't have any other inputs

        # cells are unstable either if they are inherently unstable and receive no inputs OR if they receive external destabilizing inputs
        # if unstable cells receive no inputs, they follow 'intrinsic_diff' and 'intrinsic_mig'
        # cells that are unstable in this neighborhood either differentiate into some celltype or migrate. 
        # cellular decisions are expressed as a binary differentiation matrix, and a vector of jump-sizes

        differentiation_matrix = np.eye(celltypes)# rows=old cell identity, column = new cell identity. This matrix is a binary, diagonal matrix if there are no differentiating cells.
        
        differentiating_cells = is_destab_extrinsic & (np.random.uniform(0,1,celltypes) <= p_diff)# cells with external destabilizing inputs that differentiate
        # intrinsically differentiating cells
        intrinsic_destab_diff = (intrinsic_diff>0)&(is_destab_intrinsic)
        differentiation_matrix[intrinsic_destab_diff, intrinsic_destab_diff]=0
        differentiation_matrix[intrinsic_destab_diff, (intrinsic_diff[intrinsic_destab_diff]).astype(int)] = 1
        # cells differentiating due to extrinsic inputs (random differentiation)
        differentiation_matrix[differentiating_cells, differentiating_cells] = 0 # differentiating cells change their identity 
        differentiation_matrix[differentiating_cells, np.random.randint(0,celltypes,sum(differentiating_cells))] = 1
        differentiation_matrix = np.fliplr(np.flipud(differentiation_matrix))# in order to keep the rows of this matrix consistent with how cell-types are arranged in n

        migration_list = np.zeros((celltypes))
        
        migrating_cells = is_destab_extrinsic & (differentiating_cells==0)# unstable cells that migrate
        # intrinsically migrating cells
        intrinsic_destab_mig = (intrinsic_mig>0)&(is_destab_intrinsic)
        migration_list[intrinsic_destab_mig] = intrinsic_mig[intrinsic_destab_mig]
        # cells migrating due to extrinsic inputs
        migration_list[migrating_cells] = np.random.randint(0,loc,sum(migrating_cells))+1
        migration_list = np.flip(migration_list)# to keep this consistent with how cell-types are arranged in n
        CellFateTable[n] = {'cells':cells, 'differentiation_matrix':differentiation_matrix, 'migration_vector':migration_list}
    return CellFateTable


def stable_unstable_neighborhoods(CellFateTable):
    # identify list of stable neighborhoods
    # differentiation_matrix = identity, migration_vector = all zeros
    stable_neighborhoods = []
    N = CellFateTable.keys()
    for n in N:
        diff = np.fliplr(np.flipud(CellFateTable[n]['differentiation_matrix']))
        stable_diff = (sum(sum(np.equal(diff,np.eye(len(diff)))==False))==0)
        mig = CellFateTable[n]['migration_vector']
        stable_mig = sum(mig)==0
        if stable_diff & stable_mig:
            stable_neighborhoods += [n]
    return stable_neighborhoods


# Adjacency matrix for body locations
def get_adjacency_matrix(loc):
    allloc = np.array([[(i,j) for i in range(loc)] for j in range(loc)])
    allloc = allloc.reshape(-1,*allloc.shape[-1:])
    Adjacency = np.zeros((len(allloc),len(allloc)))
    for i,r1 in enumerate(allloc):
        adjacent_rows = np.array([r1[0]+1, r1[0], r1[0]-1])
        adjacent_cols = np.array([r1[1]+1, r1[1], r1[1]-1])

        #periodic boundary conditions
        adjacent_rows[adjacent_rows==-1] = loc-1
        adjacent_rows[adjacent_rows==loc] = 0
        adjacent_cols[adjacent_cols==-1] = loc-1
        adjacent_cols[adjacent_cols==loc] = 0

        for adjrow in adjacent_rows:
            j = np.where((allloc[:,0]==adjrow)&(allloc[:,1]==r1[1]))[0][0]
            Adjacency[i,j] = 1
        for adjcol in adjacent_cols:
            j = np.where((allloc[:,0]==r1[0])&(allloc[:,1]==adjcol))[0][0]
            Adjacency[i,j] = 1
    return Adjacency





##### CODES FOR DYNAMICS #########
# Initial condition versions
# adjacency matrix
# versions of code to update body by one time step: (A) with division and death, (B) without division and death
# code for cell migration


# Randomly distributed initial cells
def initialize_body_randomscatter(loc, celltypes, density = 0.6):
    init = (np.random.uniform(0,1,(celltypes,loc**2))>density).astype(int)
    return init


# all initial cells located in a small region of 5 adjacent locations
def initialize_body_5adj(loc, celltypes, Adj, embryo_size = 5, num_init_cells=20):
    init = np.zeros((celltypes, loc**2))
    # pick 'embryo_size' adjacent locations
    Adj_n = np.linalg.matrix_power(Adj,embryo_size)>0
    embryo_loc = np.where(Adj_n[0,:])[0][0:embryo_size]
    # place 'num_init_cells' celltypes randomly in these 5 locations
    cells_loc = embryo_loc[np.random.randint(0,embryo_size,num_init_cells)]
    cells_iden = np.random.randint(0,celltypes,num_init_cells)
    for i in range(num_init_cells):
        init[cells_iden[i], cells_loc[i]] += 1 
    return init



# generate a random seed. store this seed. initialize RNG with this seed
# Other kinds of init: only a single embryonic cell/ only a single type of embryonic cell/ a single location with random collection of cells


# Function used by cells to decide whether they die or divide

# a maximum of 50\% of the cells can decide to die at any point
# use sigmoid: at very low [?] cell number, prob_death~0, at very high [?] cell number, prob_death~0.5

# prob of division is 0.8-prob_death (~ opposite function for division):
# at very low cell number, 0.8 of the cells divide and at very high cell number, 0 of the cells divide

def get_prob_death(numcells):

    no_death_threshold = 10# number of cells below which there is no death
    death_slope = 0.02

    if numcells <= no_death_threshold:
        prob_death = 0
    else:
        prob_death = 0.02 * (numcells - no_death_threshold)

    # no more than 80% cells can die at the same time-step
    prob_death = 0.8 * (prob_death>0.8) + prob_death * (prob_death<=0.8)

    return prob_death


def get_prob_div(numcells, no_div_threshold=10):
    # cells >= 10 --> no div
    # cells >0, <10 --> div
    # at cells = 1, all cells divide
    
    no_div_threshold = 10# number of cells above which there is no cell division
    # for no_div_threshold = 5, if all the body's locations are completely filled, there should be no further division
    div_slope = -1/(no_div_threshold-1)
    div_intercept = no_div_threshold/(no_div_threshold-1)
    if numcells == 0:
        prob_div=0# no division if there are no cells to divide
    elif numcells>=no_div_threshold:
        prob_div=0
    else:
        prob_div = div_slope * numcells + div_intercept
    return prob_div



def cell_death_decision(cell_vector, prob_death):
    celldeath = np.zeros((len(cell_vector)))
    for i in range(len(cell_vector)):
        d1=0
        if cell_vector[i]>0:
            d1 = np.random.uniform(0,1,int(cell_vector[i]))<prob_death
        celldeath[i] = np.sum(d1)# number of cells of type i that die
    return celldeath


def cell_div_decision(cell_vector, prob_div):
    celldiv = np.zeros((len(cell_vector)))
    for i in range(len(cell_vector)):
        d1=0
        if cell_vector[i]>0:
            d1 = np.random.uniform(0,1,int(cell_vector[i]))<prob_div
        celldiv[i] = np.sum(d1)# number of cells of type i that divide in this time step
    return celldiv


def get_divcell_loc(divcells, cell_loc, Adj, adj_daughters=0.1):
    # adj_daughters = fraction of new cells that stay in the same location
    numloc = int(sum(divcells))
    adjloc = np.where(Adj[cell_loc,:])[0]# identities of adjacent locations

    divcell_loc = adjloc[np.random.randint(0,len(adjloc),(numloc))]
    # the first ('adj_daughters' * numloc) cells stay in their own location
    divcell_loc[0:round(adj_daughters*numloc)] = np.array([cell_loc]*round(adj_daughters*numloc))
    return divcell_loc


def update_body_division_death(body, Adj, adj_daughters=0.1, no_div_threshold=10):
    new_body = np.zeros(np.shape(body))
    
    # unoccupied locations have incoming edges, but no outgoing edges
    # this makes sure that cells only migrate to locations adjacent to occupied locations
    unoccupied_loc = np.sum(body,axis=0)==0
    Adj_occ = np.copy(Adj)
    Adj_occ[unoccupied_loc,:] = Adj_occ[unoccupied_loc,:] * 0

    # go through all locations in the body, and calculate probability of division and death for each cell
    L = np.arange(0,np.shape(body)[1],1)
    num_deadcells = 0
    num_divcells = 0

    for l in L:
        cells = body[:,l].astype(int)
        if any(cells):
            # number of cells in the location
            numcells_loc = sum(cells)
            # number of cells in the neighborhood
            numcells_nbhd = np.sum(np.sum(body[:,Adj[l,:]>0]))# sum of number of cells in all locations adjacent to l
            # first, each cell decides whether to die or not
            prob_death = get_prob_death(numcells_loc)
            prob_div = get_prob_div(numcells_nbhd, no_div_threshold)
            
            #print(l, [numcells_loc, prob_death], [numcells_nbhd,prob_div])

            deadcells = cell_death_decision(cells, prob_death)
            cells_rem = cells - deadcells
            new_body[:,l] = new_body[:,l] + cells_rem
            # for the rest of the cells, they decide whether to divide or not
            divcells = cell_div_decision(cells_rem, prob_div).astype(int)
            divcell_loc = get_divcell_loc(divcells, l, Adj_occ, adj_daughters)
            # for each dividing cell:
            # one daughter cell remains in the location, and the other daughter cell decides whether to stay in the same location, or go to an adjacent location
            c1=0
            for i,c in enumerate(divcells):
                new_body[i,divcell_loc[c1:c1+c]] += 1
                c1 += c
            num_deadcells += sum(deadcells)
            num_divcells += sum(divcells)
    return new_body, num_deadcells, num_divcells


def calculate_neighborhood(body,Adj,loc):
    all_adjacent_loc = Adj[loc,:]
    N1 = (np.sum(body[:,all_adjacent_loc>0],axis=1)>0).astype(int)
    n=''.join([str(n1) for n1 in N1])
    return n


def get_new_loc(Adj, jump_size, pos, numcells):
    # ADD: cells only move to locations adjacent to occupied locations 
    if jump_size==0:
        new_pos = pos
    else:
        JumpAdj = np.linalg.matrix_power(Adj + np.eye(len(Adj)),int(jump_size))
        x = np.where(JumpAdj[pos]>0)[0]
        new_pos = x[np.random.randint(0,len(x),numcells)]
    return new_pos


def update_body_cellfate(body, Adj, CellFateRules):
    # each cell in the body, calculate the neighborhood
    # update the state of each cell according to CellFateRules
    new_body = np.zeros(np.shape(body))
    
    # unoccupied locations have incoming edges, but no outgoing edges
    # this makes sure that cells only migrate to locations adjacent to occupied locations
    unoccupied_loc = np.sum(body,axis=0)==0
    Adj_occ = np.copy(Adj)
    Adj_occ[unoccupied_loc,:] = Adj_occ[unoccupied_loc,:] * 0

    # go through all locations in the body, and calculate cellular neighborhoods
    # each cell in this location has the same cellular neighborhood -- calculate their fates and fill in new_body
    L = np.arange(0,np.shape(body)[1],1)

    for l in L:
        cells = body[:,l].astype(int)
        if any(cells):
            n = calculate_neighborhood(body, Adj, l)
            diff = CellFateRules[n]['differentiation_matrix']
            mig = CellFateRules[n]['migration_vector']

            migcells = np.where(cells * (mig>0))[0]
            diffcells = cells * (mig==0)

            # first find out what these cells differntiate into
            new_cells = diffcells.T @ diff 
            new_body[:,l] += new_cells
            # then make the cells that migrate jump to new locations
            if len(migcells)>0:
                for i1, m in enumerate(migcells):
                    new_loc = get_new_loc(Adj_occ, mig[m], l, cells[m])
                    loc_counter = collections.Counter(new_loc)
                    new_body[m, np.array(list(loc_counter.keys()))] += np.array(list(loc_counter.values())) 
    return new_body
    

def get_body_neighborhoods(body,adj):
    # for each location, instead of the tissue-type, report the neighborhood.
    # in a sense, this is a more functional representation of the body -- more informative of fate + gets rid of 'trivial tissues' and reduces apparent complexity
    # trivial tissues: this tissue looks like a small island within a bigger tissue whose cellular composition is a superset of the 'island', and every single location within the 'island' tissue has the same neighborhood (= superset tissue)
    reduced_body = body @ adj
    return reduced_body


# the sum of all distances is some measure of 'energy' of the body's current config
# it is possible that the final body is actually stable (the reduced-tissue-graph does not change upon update), but not all neighborhoods are stable [frustrated]
def body_stability(reduced_body, stable_neighborhoods):
    #definition - an empty neighborhood is unstable

    # hammingdistance of tissue neighborhoods in the body from stable neighborhoods
    # report minimum distance for each location
    celltypes = np.shape(reduced_body)[0]
    # stable neighborhoods are binary strings. convert them into binary arrays
    if any(stable_neighborhoods):
        SN = np.array([[int(x) for x in stable_neighborhoods[i]] for i in range(len(stable_neighborhoods))])# each row is a stable neighborhood
        SN = np.vstack((SN,np.zeros((celltypes))))# empty space is stable
        # each column of reduced_body is the cellular neighborhood of that location
        D1 = SN @ (reduced_body==0)
        D2 = (SN==0) @ reduced_body
        D = D1 + D2
        # minimum distance of each location's neighborhood from some stable neighborhood
        Dmin = np.amin(D,axis=0)
    else:
        Dmin = np.sum((reduced_body>0).astype(int),axis=0)
    return Dmin


def get_body_updates(CellFateRules, loc, celltypes, T, adj_daughters, no_div_threshold, stable_neighborhoods):
    Adj = get_adjacency_matrix(loc)
    Adj = ((Adj + np.eye(loc**2))>0).astype(int)

    #init_body = initialize_body_randomscatter(loc, celltypes)
    init_body = initialize_body_5adj(loc, celltypes, Adj)

    BodyUpdates = np.zeros((celltypes, loc**2, T))# cellular composition of each location
    
    BodyUpdates[:,:,0] = init_body
    Num_deadcells = []
    Num_divcells = []

    Numcelltypes = np.zeros((T))
    Frac_occ = np.zeros((T))
    Num_md = np.zeros((T))
    Cov_eq = np.zeros((T))
    Av_dispersity = np.zeros((T))
    Av_mixness = np.zeros((T))

    Numcelltypes[0] = np.sum((np.sum(init_body,axis=1)>0).astype(int))
    Frac_occ[0], Num_md[0], Cov_eq[0], Av_dispersity[0], Av_mixness[0] = get_tissue_richness_disperseness(init_body, loc)


    for i in range(T-1):
       # time1 = time.time()
        body_divdeath, num_deadcells, num_divcells = update_body_division_death(BodyUpdates[:,:,i], Adj, adj_daughters, no_div_threshold)
        Num_deadcells += [num_deadcells]
        Num_divcells += [num_divcells]
       # body_divdeath = np.copy(BodyUpdates[:,:,i])
        body_t = update_body_cellfate(body_divdeath, Adj, CellFateRules)
       # print('time to update body',(time.time()-time1)/60, 'min')
        BodyUpdates[:,:,i+1] = body_t
       # Body properties 
        Numcelltypes[i+1] = np.sum((np.sum(body_t,axis=1)>0).astype(int))
        Frac_occ[i+1], Num_md[i+1], Cov_eq[i+1], Av_dispersity[i+1], Av_mixness[i+1] = get_tissue_richness_disperseness(body_t, loc)
    
    df = pd.DataFrame(columns = ['numcelltypes', 'frac_occ_body', 'num_distinct_niches', 'coverage_equality', 'av_dispersity', 'av_mixness'])
    df['numcelltypes'] = Numcelltypes
    df['frac_occ_body'] = Frac_occ
    df['num_distinct_niches'] = Num_md
    df['coverage_equality'] = Cov_eq
    df['av_dispersity'] = Av_dispersity
    df['av_mixness'] = Av_mixness

    return BodyUpdates, Num_deadcells, Num_divcells, df


# Analysis

def get_tissue_richness_disperseness(body, loc):
    # var1: how much coverage of total body area
    ar = np.sum((np.sum(body,axis=0)>0).astype(int))/loc**2
    adj = get_adjacency_matrix(loc)

    # var2: how many distinct microdomains (normalized by min[ 2^celltypes, grid**2 ])
    md = np.unique((body>0).astype(int),axis=1)
    md = md[:,np.sum(md,axis=0)>0]
    num_md = np.shape(md)[1]

    if num_md>1:
        # var3: area-coverage equality (euclidean distance from all equal)
        ar_md = np.zeros((num_md))
        ar_eq = np.ones((num_md)) * ar/num_md # if area coverage by each microdomain is perfectly equal
        ar_uneq = np.zeros((num_md)); ar_uneq[0] = ar # if one of the microdomains covers all of the body [perfectly unequal]

        # var4: no. of disconnected components with the same cellular composition (use adj to count number of connected components. normalization: (num_components-1)/(area_covered - 1). If the tissue is maximally dispersed, this will give 1. If the tissue is clustered into a single component, this gives a 0.)
        Difloc = np.zeros((loc**2))
        dispersity = np.zeros((num_md))

        for i in range(num_md):
            dif = (body>0).astype(int)-np.tile(md[:,i],(loc**2,1)).T !=0
            difloc = np.sum(dif,axis=0)==0# all grid squares with cellular composition same as that of the ith microdomain
            Difloc[difloc] = i+1 #assign niche-label to leach location (0 is for empty square)
            ar_md[i] = sum(difloc)# number of locations occupied by each niche-type

            # number of distinct components
            adji = adj[difloc][:,difloc]
            comp = np.unique((np.linalg.matrix_power(adji,loc**2)>0).astype(int), axis=0)
            numcomp = np.shape(comp)[0]
            if ar_md[i]==1:
                dispersity[i] = 0
            else:
                dispersity[i] = (numcomp-1)/(ar_md[i]-1)

        cov_eq = 1 - math.dist(ar_eq, ar_md/loc**2)/math.dist(ar_eq, ar_uneq)# =1 when coverage is perfectly equal, =0 when perfectly unequal
        av_dispersity = np.mean(dispersity)# =1 when perfectly dispersed microdomains, =0 when perfectly clustered microdomains

        # var5: mixness: how intertwined are the different niches? it is possible for a niche to occupy a single component, and yet have a rich neighborhood (noodle-like).
        # look at the average number of distinct niche-types adjacent to each square (since this is a square grid, max=5 and min=1). mixness = (num_distinct_adj_nichetypes-1)/5
        mixness = np.zeros((loc**2))# 0 = perfectly separated niches, 1 = perfectly mixed body
        nonempty_adj = adj[Difloc>0][:,Difloc>0]
        nonempty_Difloc = Difloc[Difloc>0]
        for i in range(len(nonempty_Difloc)):
            nonempty_adj_squares = np.unique(nonempty_Difloc[nonempty_adj[i]>0])
            mixness[i] = (len(nonempty_adj_squares)-1)/5
        av_mixness = np.mean(mixness)

    else:
        cov_eq=1; av_dispersity=0; av_mixness=0
    return ar, num_md, cov_eq, av_dispersity, av_mixness






def get_dat(param_id):
    print('index=',param_id)
    fname = '/home/somyamn/Signals_migration/model4_param.p'
    param = pickle.load(open(fname,'rb'))[param_id]

    p_stable, p_diff, density, celltypes, loc, T, adj_daughters, no_div_threshold, reps = param

    for rep in range(reps):
        seed = int(str(time.time()).split('.')[1]) + param_id * rep
        np.random.seed(seed)

        filename = '/home/somyamn/Signals_migration/Dat_3/Adjdtrs{a}/pstable{s}/intden{d}/Analysis/av_tissue_prop/AV_rep{r1:03d}_param{p1:05d}.pkl'.format(a=int(adj_daughters*10), s=int(10*p_stable), d = int(10*density), r1=int(rep), p1=int(param_id))
       
        if os.path.isfile(filename):
            print(filename,'exists')
        else:

            print('celltypes:',celltypes, 'loc:',loc, 'rep:',rep,' p_stable = ', p_stable, 'p_diff = ', p_diff, 'den = ',density, 'adj_daughters=',adj_daughters, 'no_div_threshold =',no_div_threshold)

            stable_interactions = generate_random_interactions(celltypes, p_stable, density)
            Masks = generate_intrinsic_extrinsic_dependence_masks(stable_interactions, celltypes)
            CellFateRules = generate_cellfate_rules(celltypes, loc, Masks, stable_interactions, p_diff)
            stable_neighborhoods = stable_unstable_neighborhoods(CellFateRules)         
            BodyUpdates, Num_deadcells, Num_divcells, df = get_body_updates(CellFateRules, loc, celltypes, T, adj_daughters, no_div_threshold, stable_neighborhoods)
            
            # Rules files
            rulefile1 = '/home/somyamn/Signals_migration/Dat_3/Adjdtrs{a}/pstable{s}/intden{d}/Rules/Masks/M_rep{r1:03d}_param{p1:05d}.pkl'.format(a=int(adj_daughters*10), s=int(10*p_stable), d = int(10*density), r1=int(rep), p1=int(param_id))
            rulefile2 = '/home/somyamn/Signals_migration/Dat_3/Adjdtrs{a}/pstable{s}/intden{d}/Rules/CellFateRules/CFR_rep{r1:03d}_param{p1:05d}.pkl'.format(a=int(adj_daughters*10), s=int(10*p_stable), d = int(10*density), r1=int(rep), p1=int(param_id))
            rulefile3 = '/home/somyamn/Signals_migration/Dat_3/Adjdtrs{a}/pstable{s}/intden{d}/Rules/stable_interactions/SI_rep{r1:03d}_param{p1:05d}.npy'.format(a=int(adj_daughters*10), s=int(10*p_stable), d = int(10*density), r1=int(rep), p1=int(param_id))
            
            pickle.dump(Masks,open(rulefile1,'wb'))
            pickle.dump(CellFateRules,open(rulefile2,'wb'))
            np.save(rulefile3, stable_interactions)

            # Data files. CHANGE PATHS ACCORDINGLY ####
            datafile1 = '/home/somyamn/Signals_migration/Dat_3/Adjdtrs{a}/pstable{s}/intden{d}/Data/BodyUpdates/BU_rep{r1:03d}_param{p1:05d}.npy'.format(a=int(adj_daughters*10), s=int(10*p_stable), d = int(10*density), r1=int(rep), p1=int(param_id))
            datafile2 = '/home/somyamn/Signals_migration/Dat_3/Adjdtrs{a}/pstable{s}/intden{d}/Data/FinalBody/FB_rep{r1:03d}_param{p1:05d}.npy'.format(a=int(adj_daughters*10), s=int(10*p_stable), d = int(10*density), r1=int(rep), p1=int(param_id))
            datafile3 = '/home/somyamn/Signals_migration/Dat_3/Adjdtrs{a}/pstable{s}/intden{d}/Data/stable_neighborhoods/SN_rep{r1:03d}_param{p1:05d}.npy'.format(a=int(adj_daughters*10), s=int(10*p_stable), d = int(10*density), r1=int(rep), p1=int(param_id))
            
            np.save(datafile1,BodyUpdates)
            np.save(datafile2,BodyUpdates[:,:,-1])
            np.save(datafile3,stable_neighborhoods)

            # Analysis files
            analysisfile1 = '/home/somyamn/Signals_migration/Dat_3/Adjdtrs{a}/pstable{s}/intden{d}/Analysis/av_tissue_prop/AV_rep{r1:03d}_param{p1:05d}.pkl'.format(a=int(adj_daughters*10), s=int(10*p_stable), d = int(10*density), r1=int(rep), p1=int(param_id))
            pickle.dump(df,open(analysisfile1,'wb'))

            print('saved: rep:',rep,'p_stable = ', p_stable, 'p_diff = ', p_diff, 'den = ',density, 'adj_dtrs', adj_daughters, 'no_div_threshold',no_div_threshold)
    return 0



def write_params_tofile2():
    celltypes = [5]
    loc=[15]
    T=[500]
    P_stable = [0.3, 0.5, 0.7]#3
    P_diff = [0, 0.3, 0.5, 0.7, 1]#5
    den = [0.3, 0.5, 0.7]#3
    Adj_daughters = [0.3,0.5,0.7]#3
    No_div = [10]
    reps = [100]
    # around 100,000 * 100  points
    Params = list(product(P_stable, P_diff, den, celltypes, loc, T, Adj_daughters, No_div, reps))
    pickle.dump(Params,open('/home/somyamn/Signals_migration/model4_param.p','wb'))
    return 0





if __name__ == "__main__":


#   write_params_tofile()

    parser = argparse.ArgumentParser()
    parser.add_argument('-I', help = 'parameter index', type=int)
    args = parser.parse_args()
    print('starting job',args.I)
    get_dat(args.I-1)

