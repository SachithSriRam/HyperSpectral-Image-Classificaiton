import scipy.io as sio
from util import *

#Set parameters

#Number of nearest neighbours to choose for p0
k = 20

#Number of nearest neighbours to choose for diffusion distance
k_diffusion = 100

#Sigma for weight matrix
sig = 1

#time-steps
t = 30

#Cut-off for eigen-vectors
M = 10

#Number of labelled classes
num_classes = 3

#Radius to consider for spatial consensus
r_s = 3.0

if __name__ == '__main__':   
    
    X = load_mat_data('Indian_pines_corrected.mat', 'indian_pines_corrected')
    Y = load_mat_data('Indian_pines_gt.mat', 'indian_pines_gt')
    print X.shape, Y.shape
    zero_coords = (Y==0)
    plt.imshow(Y)
    plt.show()
    
    y_kmeans = do_Kmeans(X.reshape((1250,200)), 3, Y, False)
    y_kmeans_PCA = do_Kmeans(X.reshape((1250,200)), 3, Y, True)
    rows, cols = X.shape[0], X.shape[1]
    
    #Compute Emperical densities
    NN_k = get_nearest_density_neighbors(X, k)
    p = compute_empirical_density(X, NN_k)
    print ("p computed..\n")
    plt.imshow(np.reshape(p, (25,50)))
    plt.show()
    
    #p = sio.loadmat('p0.mat')['Density'].squeeze()
    plt.imshow(np.reshape(p, (25,50)))
    plt.show()
    #Compute diffusion distances
    
    #For diffusion distance , we take k_diff 
    NN_k_diffusion = get_nearest_euclid_neighbors(X, k_diffusion)
    W, P = compute_W(X, NN_k_diffusion, k_diffusion, sig)
#     plt.imshow(P)
#     plt.show()
    
    #If using eigh , eigen values are in ascending order. So flip them
    #Eigen vectors are columns , so flip them also

#     eig_val, eig_vec = np.linalg.eigh(P)
#     eig_val = eig_val[::-1]
#     eig_vec = np.fliplr(eig_vec)
     
    eig_val, eig_vec = np.linalg.eig(P)
    
#     plt.plot(eig_val)
#     plt.show()
    
    # dt = compute_diffusion_dist(eig_val, eig_vec, M, P, t)
    # np.save('Indian_Pines_diffusion_matrix.npy', dt)

    #dt2 = sio.loadmat('dt.mat')['PWdist'].squeeze()
    dt2 = np.load('Indian_Pines_diffusion_matrix.npy')    
    # plt.imshow(dt)
    # plt.show()
    pro = compute_pro(p, dt2)
    
    assert pro.shape == p.shape
    print ("pro computed..\n")
        
    plt.imshow(np.reshape(pro, (25,50)), cmap = 'gray')
    plt.show()
    
    
    
    pro2 = compute_pro(p, dt2)
    
    #Function to optimize
    Opt = p * pro2
    print Opt.shape
    plt.imshow(np.reshape(Opt, (25,50)), cmap = 'gray')
    plt.show()
    
    indices = np.argsort(Opt)[::-1]
    #print indices
    
    #Get modes
    MODES =  list(indices[0:num_classes].flatten())
    #MODES = np.array([77,1127,461])
    print "Modes are ", MODES
    print Opt[MODES]
    
    #Define labels
    LABELS = np.zeros(Opt.shape)
    #0 denotes no label 
    for mode_label,mode_coords in enumerate(MODES):
        LABELS[mode_coords] = mode_label+1
    
    #A 2D array where Row_idx is the index of pixel and the row is the pixel_idx's in increasing order of closeness
    sorted_neighbours = get_sorted_euclid_dist_matrix(X)
    
    print "Performing DLSS clustering"
    LABELS = DLSS_clustering(LABELS, sorted_neighbours, r_s, p, X, dt2, MODES, zero_coords)
    LABELS2 = LABELS + 1
    LABELS2[Y.flatten() ==0] = 0
    print metrics.adjusted_rand_score(Y.flatten(), LABELS2.flatten())
    
    LABELS = DS_clustering(LABELS, sorted_neighbours, r_s, p, X, dt2, MODES, zero_coords)
    LABELS = LABELS + 1
    LABELS[Y.flatten() ==0] = 0
    print metrics.adjusted_rand_score(Y.flatten(), LABELS.flatten())