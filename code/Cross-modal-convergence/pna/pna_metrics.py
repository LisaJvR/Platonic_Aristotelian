import torch
import numpy as np

def hsic(A, B, unbiased=False):
    '''
        From: Adapted from Koepke, https://github.com/minyoungg/platonic-rep/blob/main/metrics.py#L111
        Eqn 5 from: https://jmlr.csail.mit.edu/papers/volume13/song12a/song12a.pdf
    '''

    if unbiased:
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
    
    else:
        n = A.shape[0]
        H = torch.eye(n, dtype=A.dtype, device=A.device) - 1 / n
        return torch.trace(A @ H @ B @ H)

def compute_cka(feats_A, feats_B, type="linear", rbf_sigma=1.0, u=False):
    '''
    From: Adapted from Koepke, https://github.com/minyoungg/platonic-rep/blob/main/metrics.py#L111
    '''
    if type == "linear":
        kernel_A = torch.mm(feats_A, feats_A.T)
        kernel_B = torch.mm(feats_B, feats_B.T)

    elif type == "rbf":
        kernel_A = torch.exp(-torch.cdist(feats_A, feats_A) ** 2 / (2 * rbf_sigma ** 2))
        kernel_B = torch.exp(-torch.cdist(feats_B, feats_B) ** 2 / (2 * rbf_sigma ** 2))

    H_AA = hsic(kernel_A, kernel_A, unbiased=u)
    H_BB = hsic(kernel_B, kernel_B, unbiased=u)
    H_AB = hsic(kernel_A, kernel_B, unbiased=u)

    cka_value = H_AB / (torch.sqrt(H_AA * H_BB) + 1e-6)  
    return cka_value.item()

def cka_layers(feats_A, feats_B, type="linear", rbf_sigma=1.0, biased=False):
    n_layers_A = feats_A.shape[1]
    n_layers_B = feats_B.shape[1]

    scores = torch.empty((n_layers_A, n_layers_B), dtype=torch.float32)

    for layer_A in range(n_layers_A):
        for layer_B in range(n_layers_B):
            score = compute_cka(feats_A[:, layer_A, :], feats_B[:, layer_B, :], type=type, rbf_sigma=rbf_sigma, u=not biased)
            scores[layer_A, layer_B] = score

    return scores

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

def mutual_knn_layers(knnA_indices, knnB_indices,num_image_layers, num_text_layers, topk):
    n_layers_A = num_image_layers
    n_layers_B = num_text_layers

    scores = torch.empty((n_layers_A, n_layers_B), dtype=torch.float32)

    for layer_A in range(n_layers_A):
        for layer_B in range(n_layers_B):

            knn_A = knnA_indices[:, layer_A, :]
            knn_B = knnB_indices[:, layer_B, :]

            score = mutual_knn(knn_A,knn_B,)
            scores[layer_A, layer_B] = score

    return scores