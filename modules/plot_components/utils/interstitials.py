import numpy as np

def get_neighbors(i,j,shape,which_neighs=np.arange(6)):
    
    """
    Returns node neighbors coordinates
    
    which_neighs = which neighbors to return 

    shape:  shape of original map 
    i, j:   coords of node X whose neighbors we want
    
    Given node X, the neighboring nodes are: [Left, Right] * [Top, Mid, Bot]
         LT   RT
        LM  X  RM
         LB   RB
         
    For odd-i hex:
         0 +1         +1 +1
        -1  0   i j   +1  0
         0 -1         +1 -1
    For even-i hex:
        -1 +1          0 +1
        -1  0   i j   +1  0
        -1 -1          0 -1

    coords  = [[ [ RT_i, RT_j ], 
                 [ RM_i, RM_j ],
                 [ RB_i, RB_j ],
                 [ LB_i, LB_j ],
                 [ LM_i, LM_j ],
                 [ LT_i, LT_j ] ]  ## Even hex 
                 
               [ [ RT_i, RT_j ],   ## Odd hex
                 [ RM_i, RM_j ],
                 [ RB_i, RB_j ],
                 [ LB_i, LB_j ],
                 [ LM_i, LM_j ],
                 [ LT_i, LT_j ]  ]]
    

    coords[eo_idx][neigh_dir][ij]
    coords[0][3][0] == even hex, Left Bottom neighbor, i coord
    
    link_dir = [ / - \ / - \ ]

    Returns: 
        neighbors (coords in map)
        interstitial neighbors (coords of interstitial neighbors in expanded map)
        links_dir (which neighbor direction each entry is)
    
    """
    
    s = shape
    
    ## Coords of neighbors in original map
    neighs = []
    
    ## Coords of interstitial neighbors (node between i,j and neighbor)
    interstitial_neighs = []
    
    ## 
    links_dir = []
    
    ## RT, RM, RB, LB, LM, LT
    coords = [ [ [  0 ,  1 ],    # RT
                 [  1 ,  0 ],    # RM
                 [  0 , -1 ],    # RB
                 [ -1 , -1 ],    # LB
                 [ -1 ,  0 ],    # LM
                 [ -1 ,  1 ] ] , # LT  # End even
               [ [  1 ,  1 ],    # RT  # Start odd
                 [  1 ,  0 ],    # RM
                 [  1 , -1 ],    # RB
                 [  0 , -1 ],    # LB
                 [ -1 ,  0 ],    # LM
                 [  0 ,  1 ] ] ] # LT
    
    eo_idx = -1
    
    
    eo_idx = 0 if j%2==0 else 1 # Even or Odd

    for pos_idx in which_neighs:
        delta_i = coords[eo_idx][pos_idx][0]
        delta_j = coords[eo_idx][pos_idx][1]
        delta_ij = np.array([delta_i, delta_j])
        
        if ((delta_i+i)>=0) and ((delta_j+j)>=0) and ((delta_i+i)<s[0]) and ((delta_j+j)<s[1]):
            ij = np.array([i,j])
            new_ij = ij+delta_ij
            neighs.append(new_ij)
            
            new_inters_ij = (ij*2) + delta_ij
            interstitial_neighs.append(new_inters_ij)
            
            links_dir.append(pos_idx)

            
    return np.array(neighs), np.array(interstitial_neighs), links_dir

def calculate_interstitials(weights):
    
    im_m = (weights.shape[0]*2)-1
    im_n = (weights.shape[1]*2)-1
    
    interstitial_matrix = np.full((im_m, im_n), np.nan)
    interstitial_dirs = np.full((im_m, im_n), np.nan)
    s = weights.shape[:2]
    
    def get_dist(node,neigh):
        dist = np.linalg.norm(weights[neigh]-weights[node])
        return dist
    
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            neighs, inters_neighs, links_dir = get_neighbors(i,j,s)
            dist_tot = 0
            for n_i in range(inters_neighs.shape[0]) : # for each neighbor n_i,
                n_ii = inters_neighs[n_i][0]
                n_ij = inters_neighs[n_i][1]
                
                # if nan, this dist pair hasn't been calculated yet
                if np.isnan(interstitial_matrix[n_ii, n_ij]): # get distance from n_i to node at i,j
                    dist_i = get_dist( (i,j), (neighs[n_i][0], neighs[n_i][1]))
                    interstitial_matrix[n_ii, n_ij] = dist_i
                    dist_tot += dist_i
                    
                    interstitial_dirs[n_ii, n_ij] = links_dir[n_i]
                    
                # else already calculated; add to dist_tot for ave calc
                else:
                    dist_tot += interstitial_matrix[n_ii, n_ij]
            
            interstitial_matrix[i*2, j*2] = dist_tot/inters_neighs.shape[0]
                
    return interstitial_matrix, interstitial_dirs

#interstitial_matrix, interstitial_dirs = calculate_interstitials(small_weights)
#print(interstitial_matrix.shape)