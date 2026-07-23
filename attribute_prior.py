import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize


class StructureAttributePrior(nn.Module):
    def __init__(
        self,
        T_attr=10,
        alpha_attr=0.15,
        beta_attr=0.2,
        edge_top_k=10,
        edge_threshold=0.45,
        knn_k=10,
        knn_threshold=0.05,
        svd_max_dim=100,
        knn_batch_size=2048,
    ):
        super().__init__()
        self.T_attr = T_attr
        self.alpha_attr = alpha_attr
        self.beta_attr = beta_attr
        self.edge_top_k = edge_top_k
        self.edge_threshold = edge_threshold
        self.knn_k = knn_k
        self.knn_threshold = knn_threshold
        self.svd_max_dim = svd_max_dim
        self.knn_batch_size = knn_batch_size
        self.register_buffer("A_attr", None)
        self.register_buffer("Z_prior", None)
        self.svd_dim = None

    def estimate_attribute_aware_H(self, x, H_indices, H_values, shape):
        num_nodes, num_edges = shape
        device = x.device
        H = torch.sparse_coo_tensor(H_indices, H_values, size=(num_nodes, num_edges)).to(device)

        edge_deg = torch.sparse.mm(H.t(), torch.ones(num_nodes, 1).to(device)).clamp(min=1e-12)
        edge_sum = torch.sparse.mm(H.t(), x)
        edge_centroids = edge_sum / edge_deg

        row_idx, col_idx = H_indices
        sim = F.cosine_similarity(x[row_idx], edge_centroids[col_idx], dim=1)
        sim = F.relu(sim)
        sim = sim + 1e-6

        return torch.sparse_coo_tensor(H_indices, sim, size=(num_nodes, num_edges)).to(device)

    def compute_semantic_edge_similarity(self, H_weighted, x):
        num_nodes, num_edges = H_weighted.shape
        device = x.device
        edge_deg = torch.sparse.mm(H_weighted.t(), torch.ones(num_nodes, 1).to(device)).clamp(min=1e-12)
        edge_feat = F.normalize(torch.sparse.mm(H_weighted.t(), x) / edge_deg, p=2, dim=1)

        sim_dense = torch.mm(edge_feat, edge_feat.t())
        top_k = min(max(1, int(self.edge_top_k)), num_edges)
        vals, inds = torch.topk(sim_dense, k=top_k, dim=1)

        row_inds = torch.arange(num_edges, device=device).unsqueeze(1).expand(-1, top_k).flatten()
        col_inds = inds.flatten()
        values = vals.flatten()

        mask = values > float(self.edge_threshold)

        return torch.sparse_coo_tensor(
            torch.stack([row_inds[mask], col_inds[mask]]),
            values[mask],
            (num_edges, num_edges),
        ).coalesce()

    def get_sparse_knn_graph(self, Z):
        N = Z.shape[0]
        device = Z.device
        Z = F.normalize(Z, p=2, dim=1)
        indices_list = []
        values_list = []
        k = min(max(1, int(self.knn_k)), N)
        batch_size = max(1, int(self.knn_batch_size))

        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            batch_Z = Z[i:end]
            sim_batch = torch.mm(batch_Z, Z.t())
            vals, inds = torch.topk(sim_batch, k=k, dim=1)
            row_ids = torch.arange(i, end, device=device).unsqueeze(1).expand(-1, k).flatten()
            col_ids = inds.flatten()
            val_dat = vals.flatten()
            mask = (col_ids != row_ids) & (val_dat > 0.0)
            indices_list.append(torch.stack([row_ids[mask], col_ids[mask]]))
            values_list.append(val_dat[mask])

        all_indices = torch.cat(indices_list, dim=1)
        all_values = torch.cat(values_list)

        A = torch.sparse_coo_tensor(all_indices, all_values, (N, N)).coalesce()
        A_t = A.t()

        A_sym = (A + A_t) / 2.0
        A_sym = A_sym.coalesce()

        indices = A_sym.indices()
        values = A_sym.values()
        mask = values > float(self.knn_threshold)

        return torch.sparse_coo_tensor(indices[:, mask], values[mask], (N, N)).coalesce()

    def build_prior_affinity(self, x, H_raw, n_clusters):
        device = x.device
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float().to(device)

        if torch.is_tensor(H_raw):
            if H_raw.shape[0] != x.shape[0]:
                H_raw = H_raw.t()
            if not H_raw.is_sparse:
                H_raw = H_raw.to_sparse()
            if not H_raw.is_coalesced():
                H_raw = H_raw.coalesce()
            indices, values, shape = H_raw.indices(), H_raw.values(), H_raw.shape
        else:
            import scipy.sparse as sp
            if H_raw.shape[0] != x.shape[0]:
                H_raw = H_raw.T
            if sp.issparse(H_raw):
                H_raw = H_raw.tocoo()
                indices = torch.from_numpy(np.vstack((H_raw.row, H_raw.col)).astype(np.int64))
                values = torch.from_numpy(H_raw.data).float()
                shape = H_raw.shape
            else:
                H_t = torch.from_numpy(H_raw).to_sparse()
                indices, values, shape = H_t.indices(), H_t.values(), H_t.shape

        indices, values = indices.to(device), values.to(device)

        H_struct = self.estimate_attribute_aware_H(x, indices, values, shape)
        S_HH = self.compute_semantic_edge_similarity(H_struct, x)

        ones_n = torch.ones(shape[0], 1).to(device)

        deg_e_struct = torch.sparse.mm(H_struct.t(), ones_n).squeeze()
        deg_e_struct_inv = torch.pow(deg_e_struct + 1e-12, -1.0).unsqueeze(1)
        deg_v_struct = torch.sparse.mm(
            H_struct,
            deg_e_struct_inv * torch.sparse.mm(H_struct.t(), ones_n),
        ).squeeze()
        deg_v_struct_inv = torch.pow(deg_v_struct + 1e-12, -1.0).unsqueeze(1)

        h_t_ones = torch.sparse.mm(H_struct.t(), ones_n)
        s_hh_ones = torch.sparse.mm(S_HH, h_t_ones)
        sem_deg_v = torch.sparse.mm(H_struct, s_hh_ones).squeeze()
        sem_deg_v_inv = torch.pow(sem_deg_v + 1e-12, -1.0).unsqueeze(1)

        z_smooth = x.clone()
        for _ in range(int(self.T_attr)):
            msg = torch.sparse.mm(H_struct.t(), z_smooth) * deg_e_struct_inv
            feat_struct = torch.sparse.mm(H_struct, msg) * deg_v_struct_inv

            msg_s = torch.sparse.mm(S_HH, torch.sparse.mm(H_struct.t(), z_smooth))
            feat_sem = torch.sparse.mm(H_struct, msg_s) * sem_deg_v_inv

            z_agg = (1 - self.beta_attr) * feat_struct + self.beta_attr * feat_sem
            z_smooth = (1 - self.alpha_attr) * z_agg + self.alpha_attr * x

        z_np = z_smooth.cpu().detach().numpy()
        z_centered = z_np - np.mean(z_np, axis=0)

        buffer_k = min(min(z_centered.shape) - 1, int(self.svd_max_dim))
        if buffer_k <= 0:
            self.svd_dim = n_clusters
            emb_prior = normalize(z_centered, axis=1)
        else:
            u, s, _ = svds(z_centered, k=buffer_k, random_state=0)
            u = np.flip(u, axis=1)
            s = np.flip(s)

            search_start = max(1, n_clusters)
            if len(s) > search_start + 1:
                ratios = s[search_start:-1] / (s[search_start + 1:] + 1e-12)
                best_idx = np.argmax(ratios) + search_start
                optimal_k = best_idx + 1
            else:
                optimal_k = min(buffer_k, n_clusters * 2 + 5)

            self.svd_dim = optimal_k
            u_selected = u[:, :optimal_k]
            emb_prior = normalize(u_selected, axis=1)

        self.Z_prior = torch.from_numpy(emb_prior).float().to(device)
        self.A_attr = self.get_sparse_knn_graph(self.Z_prior)

    def forward(self):
        return self.A_attr
