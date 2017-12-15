import scipy.io as sio
import math
import operator
import numpy as np
from mpmath import matrix
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics

def tmp(X):
    r,c = X.shape
    y = np.zeros((r,c,3))
    for i in range(r):
        for j in range(c):
            if X[i,j] == 0:
                y[i,j] = (0,0,0)
            elif X[i,j] == 1:
                y[i,j] = (1,0,0)
            elif X[i,j] == 2:
                y[i,j] = (0,1,0)
            elif X[i,j] == 3:
                y[i,j] = (0,0,1)
    return y
def do_Kmeans(X, k, y, do_PCA):
    """ Performs standard K-means clusering
    
    """
    if do_PCA:
        pca = PCA(n_components = 5)
        X =  pca.fit_transform(X)
    
    print X.shape
    kmeans = KMeans(n_clusters = k, max_iter=100)
    kmeans.fit(X)
    res =  kmeans.predict(X)
    #To start labelling from 1
    res = res + 1
    res[y.flatten()==0] = 0
    plt.imshow(tmp(res.reshape(25,50)))
    plt.show()
    print metrics.adjusted_rand_score(y.flatten(), res.flatten())
    return y

#Works for Indian-Pines
def load_mat_data(filename, str):
    """ Loads data from .mat file into numpy array.
    
    Currently, set to Indian-Pines.
    """
    mat = sio.loadmat(filename)
    data = mat[str]
    
    #This slicing is only for Indian Pine
    data = data[0:50, 0:25]
    
    #Flip for viewing
    #If X
    if len(data.shape) == 3:
        data = data.transpose(1,0,2)
    #If Y
    else:
        data = data.transpose()
    
    return data

def euclid_dist_matrix(rows, cols):
    """ Returns a eucledian distance matrix between every pixel and every other pixel
    
    Must be a faster way, but not found it yet
    """
    out = {}
    
    for i1 in xrange(rows):
        for j1 in xrange(cols):
            out[(i1,j1)] = {}
            
            for i2 in xrange(rows):
                for j2 in xrange(cols):
                    out[(i1,j1)][(i2,j2)] = math.sqrt((i1 - i2)**2 + (j1 - j2)**2)
    return out

def density_dist_matrix(X):
    """ Returns a eucledian distance matrix between every pixel and every other pixel.
    
    It is eucledian distance over their spectral coordinates , not pixel coordinates
    """
    out = {}
    
    rows, cols = X.shape[0], X.shape[1]
    for i1 in xrange(rows):
        for j1 in xrange(cols):
            out[(i1,j1)] = {}
            
            for i2 in xrange(rows):
                for j2 in xrange(cols):
                    out[(i1,j1)][(i2,j2)] = np.linalg.norm(X[i1,j1] - X[i2,j2])
    return out


def get_sorted_euclid_dist_matrix(X):
    """ For every row_idx denoting a pixel, the row is pixel-indices surrounding it in inc. order
     
     Indices are falttened instead of tuples"""
    
    row, col = X.shape[0], X.shape[1]
    D = euclid_dist_matrix(row, col)
    sorted_D = np.zeros((row * col, row * col))
    
    for cur_point in D:
        
        sorted_neighbors = sorted(D[cur_point].items(), key=operator.itemgetter(1))
        i,j = cur_point
        cur_idx = i*col + j
        cords = [int(tmp[0][0]*col + tmp[0][1]) for tmp in sorted_neighbors]
        sorted_D[cur_idx] = cords
    
    return sorted_D.astype(int)

def get_nearest_euclid_neighbors(X, k):
    """ For every point(2D-pixel coord), returns the K-nearest neighbors acc. to Eucledian distance
    
    """
    D = euclid_dist_matrix(X.shape[0], X.shape[1])
    NN = {}
    
    for cur_point in D:
        #sort the list
        sorted_neighbors = sorted(D[cur_point].items(), key=operator.itemgetter(1))
        
        #Take top-k nearest points (exclude the point itself)
        NN[cur_point] = sorted_neighbors[1 : k+1]
    return NN

def get_nearest_density_neighbors(X, k):
    """ For every point, returns the K-nearest neighbors acc. to Eucledian distance of their spectral dimension
    
    """
    D = density_dist_matrix(X)
    NN = {}
    
    for cur_point in D:
        #sort the list
        sorted_neighbors = sorted(D[cur_point].items(), key=operator.itemgetter(1))
        
        #Take top-k nearest points (exclude the point itself)
        NN[cur_point] = sorted_neighbors[1 : k+1]
    return NN

def convert_to_numpy(rows, cols, x):
    """ Converting a dict-of-dicts to a 2D numpy array. 
    
    Changes tuple indexing to single number indexing
    """
    n = rows * cols
    out = np.zeros((n, n))
    for tmp1 in x:
        i,j = tmp1[0], tmp1[1]
        idx1 = i*cols + j
        
        for tmp2  in x[tmp1]:
            i2,j2 = tmp2[0], tmp2[1]
            idx2 = i2*cols + j2
            out[idx1][idx2] = x[tmp1][tmp2]
            
    return out
def compute_W(X, NN_k, k, sig):
    """Given an image(X) and nearest-neighbors(NN_k) for each pixel, computes W matrix
    
    """
    rows, cols, channels = X.shape
    W = np.zeros((rows*cols, rows*cols))
    
    
    #First find value of sigma to be half of average of all distances
    sum = 0.0
    count = 0.0
    
    for i1 in xrange(rows):
        for j1 in xrange(cols):
            for ((i2, j2),dist) in NN_k[(i1, j1)]:
                sum += np.linalg.norm(X[i1,j1]-X[i2,j2])**2
                count += 1

    sigma = 0.5 * sum/count
    print "Sigma for compute_W is ",sigma
    
    for i1 in xrange(rows):
        for j1 in xrange(cols):
            idx1 = i1 * cols + j1
            
            for ((i3,j3),dist) in NN_k[(i1, j1)]:
                idx3 = i3 * cols + j3
                
                #Normalized Eucledian distance
                #tmp = ( (X[i1,j1]/np.linalg.norm(X[i1,j1]) - X[i3,j3]/np.linalg.norm(X[i3,j3]))**2 ).sum()
                #W[idx3][idx1] = math.exp((-1.0/sig) * tmp)
                tmp = np.linalg.norm(X[i1, j1] - X[i2, j2])**2
                W[idx1][idx3] = math.exp((-1.0/sigma) * tmp)
                
    #Normalize W row-wise to get P
    P = W/W.sum(axis=1,keepdims=True)
    return W, P

def compute_diffusion_dist(eig_val, eig_vec, M, P, t):
    """ Given the eigen values and eigen vectors of P, it computes the diffusion distance matrix
    
    """
    #Size of each eigen-vec denotes row*col
    n = len(eig_vec[0])
    
    #Diffusion distance metric to return
    dt = np.zeros((n,n))
    
    #Eigen vectors are column vectors
    for x in xrange(n):
        for y in xrange(n):
            tmp =  sum( (math.pow(eig_val[i], 2*t) * math.pow((eig_vec[x][i] - eig_vec[y][i]), 2)) for i in xrange(M))
            dt[x,y] = math.sqrt(tmp)
    
    return dt

def compute_empirical_density(X, NN_k):
    """ Compute p0 and p for each point
    
    """
    
    rows, cols = X.shape[0], X.shape[1]
    p0 = np.zeros((rows*cols))
    p = np.zeros((rows*cols))
    
    #First find value of sigma to be half of average of all distances
    sum = 0.0
    count = 0.0
    
    for i1 in xrange(rows):
        for j1 in xrange(cols):
            for ((i2, j2),dist) in NN_k[(i1, j1)]:
                sum += np.linalg.norm(X[i1,j1]-X[i2,j2])
                count += 1

    sigma = 0.05 * sum/count
    
    for i1 in xrange(rows):
        for j1 in xrange(cols):           
            idx1 = i1*cols + j1
            
            for ((i2, j2), dist) in NN_k[(i1, j1)]:
                tmp = math.pow(np.linalg.norm(X[i1,j1]-X[i2,j2])/sigma, 2)
                p0[idx1] += math.exp(-tmp)
    
    #Normalize p0 to get p  
    p = p0/p0.sum()
    return p
    
def compute_pro(p, dt):
    """Given p and diffusion distances, computes pro
    
    """
    pro = np.zeros(p.shape)
    
    for idx1 in xrange(len(p)):
        cur_p = p[idx1]
        min_dist = float('inf')
        
        #Boolean flag would be true only for global max
        is_max = True
        
        for idx2 in xrange(len(p)):
            if idx2 != idx1 and p[idx2] >=  p[idx1] and dt[idx1,idx2] < min_dist:
                is_max = False
                min_dist = dt[idx1, idx2]
                pro[idx1] = min_dist
            
        #If it is maximum density, then assign it the distance to furthest point
        if is_max:
            print "For max shape is", dt[idx1].shape
            pro[idx1] = np.amax(dt[idx1])
    
    #Normalize pro so that max is 1
    pro = pro/np.amax(pro)
    return pro

def euclid_distance(idx1, idx2, row, col):
    r1 = idx1/col
    c1 = idx1%col
    
    r2 = idx2/col
    c2 = idx2%col
    
    return math.sqrt( (r1-r2)**2 + (c1-c2)**2 )

def get_spatial_consensus_label_1(cur_idx, LABELS, sorted_nn, r_s, X):
    """ For the given point, checks the labels of all points within r_s and assigns the label occuring > 0.5 times. else 0.
    
    """
    
    rows, cols = X.shape[0], X.shape[1]
    label_count = {}
    count = 0
    
    for nn in sorted_nn[cur_idx]:
        if nn != cur_idx and LABELS[nn] != 0 and euclid_distance(cur_idx, nn, rows, cols) <= r_s:
            count += 1
            if LABELS[nn] not in label_count:
                label_count[LABELS[nn]] = 1
            else:
                label_count[LABELS[nn]] += 1
    
    if len(label_count) == 0:
        return 0
    else:
        max_freq = 0; fin_label = 0;
        for tmp in label_count:
            if label_count[tmp] > max_freq:
                max_freq = label_count[tmp]
                fin_label = tmp
        return fin_label
    
def get_spatial_consensus_label_2(cur_idx, LABELS, sorted_nn, r_s, X):
    """ For the given point, checks the labels of all points within r_s and assigns the label occuring > 0.5 times. else 0.
    
    """
    
    rows, cols = X.shape[0], X.shape[1]
    label_count = {}
    count = 0
    
    for nn in sorted_nn[cur_idx]:
        if nn != cur_idx and LABELS[nn] != 0 and euclid_distance(cur_idx, nn, rows, cols) <= r_s:
            count += 1
            if LABELS[nn] not in label_count:
                label_count[LABELS[nn]] = 1
            else:
                label_count[LABELS[nn]] += 1
    
    for tmp in label_count:
        if label_count[tmp] * 1.0/count > 0.5:
            return int(tmp)
                
    #If no label is greater than 0.5 return 0
    return 0

def get_spectral_label(cur_idx, d_t, p, LABELS):
    
    #Sort other points in inc. order of diffusion distance to this point and return their indices
    sorted_dists = np.argsort(d_t[cur_idx])
    
    for cur_n in sorted_dists:
        if cur_n != cur_idx and LABELS[cur_n] != 0 and p[cur_n] > p[cur_idx]:
            return LABELS[cur_n]

def DLSS_clustering(LABELS, sorted_nn, r_s, p, X, d_t, center_coords, zero_coords):
    """ Performs 2-pass DLSS clustering as mentioned in the paper
    
    """
    sorted_idxs = np.argsort(p)[::-1]
    
    #1st pass
    for cur_idx in sorted_idxs:
        if LABELS[cur_idx] == 0:
           l_spatial = get_spatial_consensus_label_1(cur_idx, LABELS, sorted_nn, r_s, X)
           l_spectral = get_spectral_label(cur_idx,d_t , p, LABELS)
           
           if l_spatial == 0 or l_spatial == l_spectral:
               LABELS[cur_idx] = l_spectral
    
    display_results(LABELS, X, center_coords, zero_coords)
    
    #2nd pass
    for cur_idx in sorted_idxs:
        if LABELS[cur_idx] == 0:
            l_spatial = get_spatial_consensus_label_2(cur_idx, LABELS, sorted_nn, r_s, X)
            l_spectral = get_spectral_label(cur_idx,d_t , p, LABELS)
            
            if l_spatial != 0:
                LABELS[cur_idx] = l_spatial
            else:
                LABELS[cur_idx] = l_spectral
    
    display_results(LABELS, X, center_coords, zero_coords)
    return LABELS

def DS_clustering(LABELS, sorted_nn, r_s, p, X, d_t, center_coords, zero_coords):
    """ Assign point to its learest highest spectral neighbour
    
    """
    sorted_idxs = np.argsort(p)[::-1]
    
    #1st pass
    for cur_idx in sorted_idxs:
        if LABELS[cur_idx] == 0:
           l_spectral = get_spectral_label(cur_idx,d_t , p, LABELS)
           LABELS[cur_idx] = l_spectral
    
    display_results(LABELS, X, center_coords, zero_coords)
    return LABELS

def display_results(LABELS, X, center_coords, zero_coords):
    rows, cols = X.shape[0], X.shape[1]
    print X.shape
    fig = np.zeros((rows, cols,3))
    color_dict = {}
    color_dict[0] = (1,1,1)
    color_dict[1] = (1,0,0)
    color_dict[2] = (0,1,0)
    color_dict[3] = (0,0,1)
    
    for idx, val in enumerate(LABELS):
        r1 = idx/cols
        c1 = idx%cols
        #fig[r1,c1] = (20*val, 50*val, 50*val)
        fig[r1,c1] = color_dict[int(val)]
    
    fig[zero_coords] = (0,0,0)
    
    for tmp in center_coords:
        r = tmp/cols
        c = tmp%cols
        cv2.circle(fig, (c,r), 1, (0,0,0))
    
    plt.imshow(fig)
    plt.show()
#     fig = fig.astype('uint8')
#     cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#     cv2.imshow('image', fig)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()