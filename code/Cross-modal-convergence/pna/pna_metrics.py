import torch
import numpy as np
from calibrated_similarity import calibrate, calibrate_layers
import CCA

def hsic_biased(A, B):
    n = A.shape[0]
    H = torch.eye(n, dtype=A.dtype, device=A.device) - 1 / n
    return torch.trace(A @ H @ B @ H)

def hsic_unbiased(A, B):
    '''
    From: Adapted from Koepke, https://github.com/minyoungg/platonic-rep/blob/main/metrics.py#L111
    Eqn 5 from: https://jmlr.csail.mit.edu/papers/volume13/song12a/song12a.pdf
    '''
    m = A.shape[0]
    # Zero out the diagonal elements of K and L
    A_tilde = A.clone().fill_diagonal_(0)
    B_tilde = B.clone().fill_diagonal_(0)
    
    # Compute HSIC using the formula in Equation 5
    HSIC_value = (
        (torch.sum(A_tilde * B_tilde.T))
        + (torch.sum(A_tilde) * torch.sum(B_tilde) / ((m - 1) * (m - 2)))
        - (2 * torch.sum(torch.mm(A_tilde, B_tilde)) / (m - 2))
        )
    
    HSIC_value /= m * (m - 3)
    return HSIC_value

def hsic(A, B, unbiased=False):
    if unbiased:
        return hsic_unbiased(A, B)
    else:
       return hsic_biased(A, B)

def compute_cka(feats_A, feats_B, kernel="linear", rbf_sigma=1.0, unbiased=False):
    '''
    From: Adapted from Koepke, https://github.com/minyoungg/platonic-rep/blob/main/metrics.py#L111
    '''
    feats_A = feats_A.to(torch.float64)
    feats_B = feats_B.to(torch.float64)

    if kernel == "linear":
        kernel_A = torch.mm(feats_A, feats_A.T)
        kernel_B = torch.mm(feats_B, feats_B.T)

    elif kernel == "rbf":
        kernel_A = torch.exp(-torch.cdist(feats_A, feats_A) ** 2 / (2 * rbf_sigma ** 2))
        kernel_B = torch.exp(-torch.cdist(feats_B, feats_B) ** 2 / (2 * rbf_sigma ** 2))

    H_AA = hsic(kernel_A, kernel_A, unbiased=unbiased)
    H_BB = hsic(kernel_B, kernel_B, unbiased=unbiased)
    H_AB = hsic(kernel_A, kernel_B, unbiased=unbiased)

    cka_value = H_AB / (torch.sqrt(H_AA * H_BB) + 1e-6)  
    # cka_value = H_AB / (torch.sqrt(H_AA * H_BB))  
    return cka_value.item()



def compute_cknna(feats_A, feats_B, kernel="linear", rbf_sigma=1.0, unbiased=False, topk=10, distance_agnostic=False):
    '''
        From: Adapted from Koepke, https://github.com/minyoungg/platonic-rep/blob/main/metrics.py#L111
        '''
    feats_A = feats_A.to(torch.float64)
    feats_B = feats_B.to(torch.float64)
        
    if kernel == "linear":
            kernel_A = torch.mm(feats_A, feats_A.T)
            kernel_B = torch.mm(feats_B, feats_B.T)
    
    elif kernel == "rbf":
            kernel_A = torch.exp(-torch.cdist(feats_A, feats_A) ** 2 / (2 * rbf_sigma ** 2))
            kernel_B = torch.exp(-torch.cdist(feats_B, feats_B) ** 2 / (2 * rbf_sigma ** 2))

    def similarity(kernel_A, kernel_B, topk):
        if unbiased:
            K_hat = kernel_A.clone().fill_diagonal_(float("-inf"))
            L_hat = kernel_B.clone().fill_diagonal_(float("-inf"))
        else:
             K_hat, L_hat = kernel_A, kernel_B

        _, topk_K_indices = torch.topk(K_hat, topk, dim=1)
        _, topk_L_indices = torch.topk(L_hat, topk, dim=1)

        n = kernel_A.shape[0]
        mask_K = torch.zeros(n, n).scatter_(1, topk_K_indices, 1)
        mask_L = torch.zeros(n, n).scatter_(1, topk_L_indices, 1)
        mask = mask_K * mask_L

        if distance_agnostic:
                sim = mask * 1.0
        else:
            if unbiased:
                    sim = hsic_unbiased(mask * kernel_A, mask * kernel_B)
            else:
                    sim = hsic_biased(mask * kernel_A, mask * kernel_B)
        return sim

    sim_kl = similarity(kernel_A, kernel_B, topk)
    sim_kk = similarity(kernel_A, kernel_A, topk)
    sim_ll = similarity(kernel_B, kernel_B, topk)

    return sim_kl.item() / (torch.sqrt(sim_kk * sim_ll) + 1e-6).item()

def mutual_knn(knn_A, knn_B):
    '''
    mKNN(l,l) = 1/N sum_i^N (|KNN_A(i,l) intersect KNN_B(i,l)| / k)
    '''
    assert knn_A.shape == knn_B.shape
    k = knn_A.shape[1]
    matches = knn_A.unsqueeze(2) == knn_B.unsqueeze(1)
    overlap = matches.any(dim=2).sum(dim=1)
    per_sample_score = overlap.float() / k

    return per_sample_score.mean().item()

def compute_mutual_knn(feats_A, feats_B, topk):
    knn_A = knn_layers(feats_A.unsqueeze(1), topk)[:, 0, :]  # [N, k]
    knn_B = knn_layers(feats_B.unsqueeze(1), topk)[:, 0, :]  # [N, k]

    return mutual_knn(knn_A, knn_B)


# TODO --------------------------------------

def compute_rsa(X,Y):
    """How dissimilar the representations are"""

    return None

def center_and_scale(act):
    act = act - torch.mean(act, axis=0)
    act = act / (torch.std(act, axis=0) + 1e-8)
    return act

def compute_svcca(X,Y,cca_dim=10):
    ''' From the platonic paper'''
    c_X = center_and_scale(X)
    c_Y = center_and_scale(Y)

    # SVD
    U1, _, _ = torch.svd_lowrank(c_X, q=cca_dim)
    U2, _, _ = torch.svd_lowrank(c_Y, q=cca_dim)

    U1 = U1.cpu().detach().numpy()
    U2 = U2.cpu().detach().numpy()

    cca = CCA(n_components=cca_dim)
    cca.fit(U1, U2)
    U1_c, U2_c = cca.transform(U1, U2)

    # sometimes it goes to nan, this is just to avoid that
    U1_c += 1e-10 * np.random.randn(*U1_c.shape)
    U2_c += 1e-10 * np.random.randn(*U2_c.shape)

     # Compute SVCCA similarity
    svcca_similarity = np.mean(
            [np.corrcoef(U1_c[:, i], U2_c[:, i])[0, 1] for i in range(cca_dim)]
    )
    return svcca_similarity

def compute_pwcca(X,Y):

    return None



# --------------------------------------------

def compare_layers(feats_A, feats_B, metric_fn, metric_kwargs):
    n_layers_A = feats_A.shape[1]
    n_layers_B = feats_B.shape[1]

    scores = torch.empty(n_layers_A,n_layers_B,dtype=torch.float32,)

    for i in range(n_layers_A):
        X = feats_A[:, i, :]
        for j in range(n_layers_B):
            Y = feats_B[:, j, :]
            scores[i, j] = metric_fn(X,Y,**metric_kwargs,)

    return scores

# TODO: improve this code: --------------------------------

def knn_layers(embeddings, k):
    if torch.cuda.is_available(): use_gpu = True
    else: use_gpu = False

    # sklearn needs CPU NumPy arrays
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().float().cpu().numpy()

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    n_samples, n_layers, dim = embeddings.shape

    all_indices = torch.empty(
        (n_samples, n_layers, k),
        dtype=torch.long,
    )
    for layer in range(n_layers):
        layer_embeddings = embeddings[:, layer, :].copy()

        # XXX
        if use_gpu :
            import faiss
            if faiss.get_num_gpus() > 0:
                faiss.normalize_L2(layer_embeddings)
                index = faiss.IndexFlatL2(dim)
                
                resources = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(resources,0,index,) #move to GPU
                index.add(layer_embeddings)
                _, indices = index.search(layer_embeddings, k + 1)
        else: 
            """ Pure AI implementation for CPU (no faiss) """
            from sklearn.neighbors import NearestNeighbors
            from sklearn.preprocessing import normalize

            layer_embeddings = normalize(
                layer_embeddings,
                norm="l2",
                axis=1
            )

            if k >= n_samples:# XXX
                k = n_samples - 1
                print(f"Warning: k ({k}) is greater than or equal to the number of samples ({n_samples}). Adjusting k to {k}.")

            index = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean")
            index.fit(layer_embeddings)

            _, indices = index.kneighbors(layer_embeddings)

        clean_indices = np.empty((n_samples, k), dtype=np.int64)
        for i in range(n_samples):
            neighbours = indices[i][indices[i] != i]
            clean_indices[i] = neighbours[:k]

        all_indices[:, layer, :] = torch.from_numpy(clean_indices.copy())

    return all_indices


