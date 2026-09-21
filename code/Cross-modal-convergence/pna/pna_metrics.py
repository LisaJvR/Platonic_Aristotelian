import torch
import numpy as np
from calibrated_similarity import calibrate, calibrate_layers
import faiss
# import CCA
_FAISS_RESOURCES = None
from tqdm import tqdm

def hsic_biased(A, B):
    """
    Adapted to skip calculating H
    """
    A_centered = (
        A
        - A.mean(dim=0, keepdim=True)
        - A.mean(dim=1, keepdim=True)
        + A.mean()
    )
    return torch.sum(A_centered * B)

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
        # - (2 * torch.sum(torch.mm(A_tilde, B_tilde)) / (m - 2)) #XXX might be slower
        - (2 * torch.dot(A_tilde.sum(dim=0), B_tilde.sum(dim=1)) / (m - 2))
        )
    
    HSIC_value /= m * (m - 3)
    return HSIC_value

def compute_biased_linear_cka(feats_A, feats_B):
    '''
    Compute in kernel space
    '''
    X = feats_A - feats_A.mean(dim=0, keepdim=True)
    Y = feats_B - feats_B.mean(dim=0, keepdim=True)

    XTX = X.T @ X
    YTY = Y.T @ Y

    denominator = (
            torch.norm(XTX) *
            torch.norm(YTY)
            + 1e-12
        )

    YTX = Y.T @ X

    numerator = torch.norm(YTX) ** 2

    cka_value = numerator / denominator

    return cka_value.item()

def compute_cka_kernel(feats, kernel="linear", rbf_sigma=1.0, unbiased=False):
    if kernel == "linear":
        kernel_matrix = torch.mm(feats, feats.T)
    elif kernel == "rbf":
        kernel_matrix = torch.exp(-torch.cdist(feats, feats) ** 2 / (2 * rbf_sigma ** 2))

    return kernel_matrix


def compute_cka(kernel_A, kernel_B, kernel="linear", rbf_sigma=1.0, unbiased=False):
    '''
    From: Adapted from Koepke, https://github.com/minyoungg/platonic-rep/blob/main/metrics.py#L111
    '''
    if unbiased: hsic_fn = hsic_unbiased
    else: hsic_fn = hsic_biased

    H_AA = hsic_fn(kernel_A, kernel_A)
    H_BB = hsic_fn(kernel_B, kernel_B)
    H_AB = hsic_fn(kernel_A, kernel_B)

    del kernel_A, kernel_B

    cka_value = H_AB / (torch.sqrt(H_AA * H_BB) + 1e-6)  
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
    Calculate the mutual knn between 2 sets embeddings (each from 1 layer)
    knn_A and knn_B : [N, k]
    '''
    assert knn_A.shape == knn_B.shape
    k = knn_A.shape[1] # number of neighbors

    matches = knn_A.unsqueeze(2) == knn_B.unsqueeze(1) # [N, k, k] boolean tensor indicating matches
    overlap = matches.any(dim=2).sum(dim=1)
    per_sample_score = overlap.float() / k

    return per_sample_score.mean().item()

def compute_mutual_knn(layer_feats_A, layer_feats_B, topk):
    '''
    layer_feats_A and layer_feats_B : [N, 1, D]
    Must recieve one pair or layers to compare.
    '''
    
    knn_A = knn_layer(layer_feats_A, topk) #[N, k]
    knn_B = knn_layer(layer_feats_B, topk) #[N, k]

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
    '''
    feats_A and feats_B : [N, L, D]
    '''
    n_layers_A = feats_A.shape[1]
    n_layers_B = feats_B.shape[1]

    device = feats_A.device

    scores = torch.empty(n_layers_A,n_layers_B,dtype=torch.float32,device=device)

    if metric_fn == compute_cka:
        # if metric_kwargs.get("unbiased", False):
            # compute all cka simultaneously in kernel space
        Y_kernels = []
        for i in tqdm(range(n_layers_A), desc=f"Computing CKA scores {n_layers_A} layers A vs {n_layers_B} layers B"):
            X = feats_A[:, i, :] 
            # compute kernel
            X_ker = compute_cka_kernel(X, **metric_kwargs)

            for j in range(n_layers_B):
                if len(Y_kernels) > j:
                    Y_ker = Y_kernels[j]
                else:
                    Y = feats_B[:, j, :]
                    # compute Y kernel
                    Y_ker = compute_cka_kernel(Y, **metric_kwargs)
                    Y_kernels.append(Y_ker.cpu())

                scores[i, j] = metric_fn(X_ker,Y_ker,**metric_kwargs,)
    else:
        for i in tqdm(range(n_layers_A), desc=f"Computing: {metric_fn.__name__} scores {n_layers_A} layers A vs {n_layers_B} layers B"):
            X = feats_A[:, i, :] 
            for j in range(n_layers_B):
                Y = feats_B[:, j, :]
                scores[i, j] = metric_fn(X,Y,**metric_kwargs,)

    return scores

def get_faiss_resources():
    global _FAISS_RESOURCES

    if _FAISS_RESOURCES is None:
        _FAISS_RESOURCES = faiss.StandardGpuResources()

    return _FAISS_RESOURCES

def knn_layer(layer_embeddings, k):
    '''
    Computes the nearest neigbors of each sample in the embeddigns for 1 layer
    Embeddings shape: [N, D]
    '''
    if torch.cuda.is_available(): use_gpu = True
    else: use_gpu = False

    n_samples, dim = layer_embeddings.shape

    if use_gpu: 
        resources = get_faiss_resources()

    if use_gpu :
        layer_embeddings = (layer_embeddings
            .detach().float().cpu().numpy())

        layer_embeddings = np.ascontiguousarray(
            layer_embeddings,
            dtype=np.float32
        )

        if faiss.get_num_gpus() > 0:
            faiss.normalize_L2(layer_embeddings)
            index = faiss.GpuIndexFlatL2(resources, dim, )
            index.add(layer_embeddings)
            _, indices = index.search(layer_embeddings, k + 1)
    else: 
            """ Pure AI implementation for CPU (no faiss) """
            from sklearn.neighbors import NearestNeighbors
            from sklearn.preprocessing import normalize

             # sklearn needs CPU NumPy arrays
            if isinstance(layer_embeddings, torch.Tensor):
                layer_embeddings = layer_embeddings.detach().float().cpu().numpy()
            layer_embeddings = np.ascontiguousarray(layer_embeddings, dtype=np.float32)

            layer_embeddings = normalize(layer_embeddings,
                norm="l2",axis=1)

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

    return torch.from_numpy(clean_indices.copy()) # shape: [N, k]

def compute_mknn_caption_density(layer_feats_A, layer_feats_B, topk):
    '''Must recieve all ca'''
    
    knn_A = knn_layer(layer_feats_A, topk) #[N, k]
    knn_B = knn_layer(layer_feats_B, topk) #[N, k]

    return mutual_knn(knn_A, knn_B)
    


