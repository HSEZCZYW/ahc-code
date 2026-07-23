import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score, silhouette_score
from scipy.optimize import linear_sum_assignment
import random
import os
import time

import argparse
import yaml

import data
from hgnn_model import HGNN
from MembershipDecoder import MembershipDecoder
from attribute_prior import StructureAttributePrior


def compute_hyperedge_representations(H_incidence, node_z):

    H = H_incidence.coalesce()
    indices = H.indices()
    values = H.values()

    num_edges = H.size(1)
    feat_dim = node_z.size(1)

    edge_sum = torch.zeros(num_edges, feat_dim, device=node_z.device)
    edge_sum.index_add_(0, indices[1], node_z[indices[0]] * values.unsqueeze(1))

    deg_e = torch.zeros(num_edges, device=node_z.device)
    deg_e.index_add_(0, indices[1], values)

    edge_reps = edge_sum / (deg_e.unsqueeze(1) + 1e-12)
    return edge_reps

def learn_weighted_incidence(
    H_incidence,
    node_z,
    decoder,
    mask_floor=0.05,
    prune_threshold=0.1,
    A_attr=None,
    H_prune_reference=None,
    prune_lambda_mask=1.0,
    prune_lambda_attr=0.5,
    attr_prune_threshold=None,
):

    H = H_incidence.coalesce()
    indices = H.indices()
    values = H.values()

    edge_reps = compute_hyperedge_representations(H_incidence, node_z)

    node_feat = node_z[indices[0]]
    edge_feat = edge_reps[indices[1]]

    score = decoder(node_feat, edge_feat)
    mask = torch.sigmoid(score)

    soft_weight = mask_floor + (1.0 - mask_floor) * mask
    new_values = values * soft_weight

    attr_prune_active = (
        A_attr is not None
        and float(prune_lambda_attr) != 0.0
    )
    if attr_prune_threshold is None:
        attr_prune_threshold = prune_threshold

    if attr_prune_active:
        H_ref = H_prune_reference.coalesce() if H_prune_reference is not None else H
        ref_indices = H_ref.indices()
        H_ref_binary = torch.zeros(
            H.shape,
            dtype=mask.dtype,
            device=node_z.device
        )
        H_ref_binary[ref_indices[0], ref_indices[1]] = 1.0

        A_attr = A_attr.detach()
        if A_attr.is_sparse:
            A_attr = A_attr.coalesce()
            attr_sum = torch.sparse.mm(A_attr, H_ref_binary)
        else:
            attr_sum = torch.mm(A_attr, H_ref_binary)

        edge_counts = H_ref_binary.sum(dim=0)
        pair_attr_sum = attr_sum[indices[0], indices[1]]
        pair_ref_membership = H_ref_binary[indices[0], indices[1]]
        pair_counts = (edge_counts[indices[1]] - pair_ref_membership).clamp(min=1e-12)
        attr_keep_score = (pair_attr_sum / pair_counts).to(mask.dtype)
        keep_score = float(prune_lambda_mask) * mask + float(prune_lambda_attr) * attr_keep_score
    else:
        keep_score = mask

    threshold = float(attr_prune_threshold) if attr_prune_active else float(prune_threshold)
    keep_mask = keep_score >= threshold
    kept_indices = indices[:, keep_mask]
    kept_values = new_values[keep_mask]

    H_dyn = torch.sparse_coo_tensor(
        kept_indices, kept_values, H.shape, device=node_z.device
    ).coalesce()

    return H_dyn


def compute_incidence_normalization(H_incidence, device=None):
    H = H_incidence.coalesce()
    if device is None:
        device = H.device

    indices = H.indices()
    values = H.values().float()
    num_nodes, num_edges = H.shape

    deg_v = torch.zeros(num_nodes, device=device, dtype=values.dtype)
    deg_e = torch.zeros(num_edges, device=device, dtype=values.dtype)
    deg_v.index_add_(0, indices[0], values)
    deg_e.index_add_(0, indices[1], values)

    dv_inv_sqrt = torch.pow(deg_v + 1e-12, -0.5)
    de_inv = torch.pow(deg_e + 1e-12, -1.0)
    return dv_inv_sqrt, de_inv


def blend_incidence_for_encoder(H_base, H_aug, rho_enc):
    H_base = H_base.coalesce()
    H_aug = H_aug.coalesce()
    rho_enc = float(rho_enc)

    indices = torch.cat([H_base.indices(), H_aug.indices()], dim=1)
    values = torch.cat([
        (1.0 - rho_enc) * H_base.values(),
        rho_enc * H_aug.values(),
    ], dim=0)

    return torch.sparse_coo_tensor(
        indices,
        values,
        H_base.shape,
        device=H_base.device
    ).coalesce()


def build_augmented_incidence(
    H_incidence,
    node_z,
    A_attr=None,
    k_cand=1,
    lambda_cand=0.1,
    tau_cand=0.0,
    lambda_z=1.0,
    lambda_a=0.0,
    device=None,
):
    if device is None:
        device = node_z.device

    H = H_incidence.coalesce()
    num_nodes, num_edges = H.shape
    k_cand = max(0, int(k_cand))

    if k_cand == 0 or num_nodes == 0 or num_edges == 0 or lambda_cand == 0:
        return H

    with torch.no_grad():
        edge_reps = compute_hyperedge_representations(H, node_z)
        node_norm = F.normalize(node_z.detach(), p=2, dim=1)
        edge_norm = F.normalize(edge_reps.detach(), p=2, dim=1)
        emb_sim = torch.mm(node_norm, edge_norm.t())

        existing_mask = torch.zeros(
            (num_nodes, num_edges),
            dtype=torch.bool,
            device=device
        )
        h_indices = H.indices()
        existing_mask[h_indices[0], h_indices[1]] = True

        attr_active = (
            A_attr is not None
            and float(lambda_a) != 0.0
        )
        if attr_active:
            H_binary_dense = torch.zeros(
                (num_nodes, num_edges),
                dtype=emb_sim.dtype,
                device=device
            )
            H_binary_dense[h_indices[0], h_indices[1]] = 1.0

            A_attr = A_attr.detach()
            if A_attr.is_sparse:
                A_attr = A_attr.coalesce()
                attr_sum = torch.sparse.mm(A_attr, H_binary_dense)
            else:
                attr_sum = torch.mm(A_attr, H_binary_dense)

            edge_counts = H_binary_dense.sum(dim=0).clamp(min=1e-12)
            attr_score = attr_sum / edge_counts.unsqueeze(0)
            score = float(lambda_z) * emb_sim + float(lambda_a) * attr_score
        else:
            score = emb_sim

        candidate_sim = score.masked_fill(existing_mask, float("-inf"))
        topk = min(k_cand, num_nodes)
        top_scores, top_nodes = torch.topk(candidate_sim, k=topk, dim=0)

        edge_ids = torch.arange(num_edges, device=device).unsqueeze(0).expand_as(top_nodes)
        keep_mask = torch.isfinite(top_scores) & (top_scores >= float(tau_cand))

        if keep_mask.sum().item() == 0:
            return H

        added_nodes = top_nodes[keep_mask]
        added_edges = edge_ids[keep_mask]

        cand_base_values = torch.ones(
            added_nodes.numel(),
            dtype=H.values().dtype,
            device=device
        )

        cand_indices = torch.stack([added_nodes, added_edges], dim=0)
        cand_values = cand_base_values * float(lambda_cand)

        h_aug_indices = torch.cat([H.indices(), cand_indices], dim=1)
        h_aug_values = torch.cat([H.values(), cand_values], dim=0)

        H_aug = torch.sparse_coo_tensor(
            h_aug_indices,
            h_aug_values,
            H.shape,
            device=device
        ).coalesce()

    return H_aug


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cluster_metrics(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    map_dict = {row: col for row, col in zip(row_ind, col_ind)}
    y_pred_aligned = np.array([map_dict.get(x, x) for x in y_pred])
    acc = w[row_ind, col_ind].sum() / y_pred.size
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred_aligned, average='macro')
    return acc, f1, nmi, ari


def balance_loss(probs):
    n_nodes = probs.shape[0]
    n_clusters = probs.shape[1]
    c_t_c = torch.matmul(probs.t(), probs)
    c_t_c = c_t_c / n_nodes
    norm_fro = torch.norm(c_t_c, p='fro')
    loss = norm_fro - (1.0 / n_clusters ** 0.5)
    return loss


def sharpen_target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def incidence_to_affinity(H_incidence, device):
    H = H_incidence.to_dense() if H_incidence.is_sparse else H_incidence


    deg_e = H.sum(0)
    de_inv = torch.pow(deg_e.float() + 1e-12, -1.0)
    W_e = torch.diag(de_inv)

    A_dyn = torch.mm(torch.mm(H, W_e), H.t())


    A_dyn.fill_diagonal_(0.0)


    indices = torch.nonzero(A_dyn).t()
    values = A_dyn[indices[0], indices[1]]


    A_sparse = torch.sparse_coo_tensor(indices, values, A_dyn.shape).to(device).coalesce()

    return A_sparse


def fuse_incidence_affinity(A_base, A_dyn, eta):

    A_base = A_base.coalesce()
    A_dyn = A_dyn.coalesce()


    A_base_dense = A_base.to_dense()
    A_dyn_dense = A_dyn.to_dense()

    A_blend = eta * A_base_dense + (1.0 - eta) * A_dyn_dense
    A_blend.fill_diagonal_(0.0)

    indices = torch.nonzero(A_blend).t()
    values = A_blend[indices[0], indices[1]]

    A_blend_sparse = torch.sparse_coo_tensor(
        indices, values, A_blend.shape, device=A_blend.device
    ).coalesce()

    return A_blend_sparse

def ncut_loss(Y_hat, A_dyn, d_pi_stable):
    device = Y_hat.device
    num_clusters = Y_hat.shape[1]

    DS = Y_hat * d_pi_stable.unsqueeze(1)
    vol_mat = torch.mm(Y_hat.t(), DS)

    if A_dyn.is_sparse:
        AS = torch.sparse.mm(A_dyn, Y_hat)
    else:
        AS = torch.mm(A_dyn, Y_hat)

    StAS = torch.mm(Y_hat.t(), AS)
    cut_mat = vol_mat - StAS
    identity = torch.eye(num_clusters).to(device)
    loss_dyn = torch.trace(torch.linalg.pinv(vol_mat + 1e-4 * identity) @ cut_mat)
    return loss_dyn


def train_hysinc(args, dataset_override=None, print_result=True):
    seed = int(getattr(args, "seed", 42))
    setup_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dataset_override is None:
        dataset = data.load(args.dataset)
    else:
        dataset = dataset_override

    features = dataset['features']
    if sp.issparse(features): features = features.toarray()
    H_raw = dataset['adj']
    labels = dataset['labels']
    y_true = np.argmax(labels, axis=1) if labels.ndim == 2 else labels

    if H_raw.shape[0] != features.shape[0]:
        if H_raw.shape[1] == features.shape[0]: H_raw = H_raw.T

    x = torch.from_numpy(features).float().to(device)
    n_clusters = len(np.unique(y_true))
    x_norm = F.normalize(x, p=2, dim=1)

    if not sp.issparse(H_raw):
        H_scipy = sp.csc_matrix(H_raw)
    else:
        H_scipy = H_raw.tocsc()
    H_coo = H_scipy.tocoo()
    indices = torch.from_numpy(np.vstack((H_coo.row, H_coo.col)).astype(np.int64))
    values = torch.from_numpy(H_coo.data.astype(np.float32))
    H_incidence = torch.sparse_coo_tensor(indices, values, H_coo.shape).to(device)

    deg_v = np.array(H_scipy.sum(1)).flatten()
    deg_e = np.array(H_scipy.sum(0)).flatten()
    dv_inv_sqrt = torch.from_numpy(np.power(deg_v + 1e-12, -0.5)).float().to(device)
    de_inv = torch.from_numpy(np.power(deg_e + 1e-12, -1.0)).float().to(device)

    if torch.cuda.is_available(): torch.cuda.synchronize()
    start_time = time.time()


    model = HGNN(
        x.shape[1],
        args.hidden,
        n_clusters,
        svd_dim=n_clusters,
        dropout=args.dropout
    ).to(device)

    dynamic_decoder = MembershipDecoder(
        dim=n_clusters,
        hidden_dim=128
    ).to(device)

    optimizer = optim.Adam(
        list(model.parameters()) + list(dynamic_decoder.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )
    A_base = incidence_to_affinity(H_incidence, device)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    lambda_cand = getattr(args, "lambda_cand", 0.1)
    rho_enc = getattr(args, "rho_enc", 0.1)
    encoder_blend_start_epoch = getattr(args, "encoder_blend_start_epoch", 30)
    tau_p = getattr(args, "tau_p", 0.1)
    lambda_z = getattr(args, "lambda_z", 1.0)
    lambda_a = getattr(args, "lambda_a", 0.0)
    gamma_attr = getattr(args, "gamma_attr", 0.0)
    T_attr = getattr(args, "T_attr", 1)
    alpha_attr = getattr(args, "alpha_attr", 0.15)
    beta_attr = getattr(args, "beta_attr", 0.2)

    prior_learner = StructureAttributePrior(
        T_attr=T_attr,
        alpha_attr=alpha_attr,
        beta_attr=beta_attr,
        edge_top_k=10,
        edge_threshold=0.45,
        knn_k=10,
        knn_threshold=0.05,
        svd_max_dim=100,
        knn_batch_size=2048,
    ).to(device)
    prior_learner.build_prior_affinity(
        x,
        H_incidence,
        n_clusters,
    )
    A_attr = prior_learner().detach().coalesce()

    if A_attr.is_sparse:
        d_attr = torch.sparse.mm(
            A_attr,
            torch.ones(x.shape[0], 1).to(device)
        ).squeeze()
    else:
        d_attr = A_attr.sum(dim=1)
    d_attr_stable = d_attr + 1e-4 * d_attr.mean()

    T_target = None

    best_acc, best_f1, best_nmi, best_ari = 0.0, 0.0, 0.0, 0.0
    best_epoch = -1


    for epoch in range(args.epochs):
        model.train()
        dynamic_decoder.train()
        optimizer.zero_grad()


        logits, final_emb, x_rec, z_contrast = model(x, H_incidence, dv_inv_sqrt, de_inv)


        if epoch >= 30:
            H_aug = build_augmented_incidence(
                H_incidence=H_incidence,
                node_z=final_emb,
                A_attr=A_attr,
                k_cand=1,
                lambda_cand=lambda_cand,
                tau_cand=0.0,
                lambda_z=lambda_z,
                lambda_a=lambda_a,
                device=device,
            )
        else:
            H_aug = H_incidence


        encoder_delta_active = (
            epoch >= 30
            and epoch >= encoder_blend_start_epoch
            and float(rho_enc) > 0.0
        )
        if encoder_delta_active:
            H_encoder = blend_incidence_for_encoder(
                H_base=H_incidence,
                H_aug=H_aug,
                rho_enc=rho_enc
            )
            dv_encoder, de_encoder = compute_incidence_normalization(H_encoder, device=device)
            logits, final_emb, x_rec, z_contrast = model(x, H_encoder, dv_encoder, de_encoder)

        curr_temp = max(0.1, 1.0 * (0.995 ** epoch))
        Y_hat = F.softmax(logits / curr_temp, dim=1)

        if epoch % 10 == 0 or T_target is None:
            with torch.no_grad():
                T_target = sharpen_target_distribution(Y_hat)


        H_dyn = learn_weighted_incidence(
            H_incidence=H_aug,
            node_z=final_emb,
            decoder=dynamic_decoder,
            mask_floor=0.05,
            prune_threshold=tau_p,
            A_attr=A_attr,
            H_prune_reference=H_incidence,
            prune_lambda_mask=1.0,
            prune_lambda_attr=0.5,
            attr_prune_threshold=0.1,
        )


        A_dyn = incidence_to_affinity(H_dyn, device)


        if epoch < args.fusion_start_epoch:
            A_used = A_base
        else:
            A_used = fuse_incidence_affinity(
                A_base=A_base,
                A_dyn=A_dyn,
                eta=args.eta
            )


        if A_used.is_sparse:
            d_pi = torch.sparse.mm(A_used, torch.ones(x.shape[0], 1).to(device)).squeeze()
        else:
            d_pi = A_used.sum(dim=1)
        d_pi_stable = d_pi + 1e-4 * d_pi.mean()


        loss_dyn = ncut_loss(Y_hat, A_used, d_pi_stable)
        loss_attr = ncut_loss(Y_hat, A_attr, d_attr_stable)

        loss_bal = balance_loss(Y_hat)
        loss_rec = 1.0 - F.cosine_similarity(x_rec, x_norm, dim=1).mean()
        loss_kl = F.kl_div(Y_hat.log(), T_target, reduction='batchmean')

        loss = args.gamma_dyn * loss_dyn + \
               gamma_attr * loss_attr + \
               args.gamma_bal * loss_bal + \
               args.gamma_rec * loss_rec + \
               args.gamma_kl * loss_kl

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(dynamic_decoder.parameters()),
            max_norm=2.0
        )
        optimizer.step()
        scheduler.step()


        with torch.no_grad():
            y_pred = torch.argmax(Y_hat, dim=1).cpu().numpy()
            train_acc, train_f1, train_nmi, train_ari = cluster_metrics(y_true, y_pred)


            acc, f1, nmi, ari = train_acc, train_f1, train_nmi, train_ari

            if train_acc > best_acc:
                best_acc = train_acc
                best_f1 = train_f1
                best_nmi = train_nmi
                best_ari = train_ari
                best_epoch = epoch + 1


    if torch.cuda.is_available(): torch.cuda.synchronize()
    end_time = time.time()

    if print_result:
        print(
            f"Dataset: {args.dataset} | ACC: {best_acc:.4f} | "
            f"NMI: {best_nmi:.4f} | F1: {best_f1:.4f} | ARI: {best_ari:.4f}"
        )
        print(f"Effective Time: {end_time - start_time:.2f}s")

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "acc": best_acc,
        "nmi": best_nmi,
        "f1": best_f1,
        "ari": best_ari,
        "time": end_time - start_time,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HySINC")
    parser.add_argument('--dataset', type=str, default='cocitation/cora', help='Name of the dataset to run')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML configuration file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for a single run')
    parser.add_argument('--seeds', type=str, default=None, help='Comma-separated seeds for repeated runs, e.g. 42,43,44,45,46')
    cmd_args = parser.parse_args()

    if not os.path.exists(cmd_args.config):
        raise FileNotFoundError(f"Config file not found: {cmd_args.config}")

    with open(cmd_args.config, 'r', encoding='utf-8') as f:
        all_configs = yaml.safe_load(f)

    if cmd_args.dataset not in all_configs:
        raise ValueError(f"Dataset '{cmd_args.dataset}' is not found in {cmd_args.config}.")

    dataset_config = all_configs[cmd_args.dataset]


    class ConfigArgs:
        def __init__(self, dataset_name, config_dict):
            self.dataset = dataset_name
            for key, value in config_dict.items():
                if key in ['lr']:
                    value = float(value)
                setattr(self, key, value)


    args = ConfigArgs(cmd_args.dataset, dataset_config)

    if cmd_args.seeds:
        seeds = [int(seed.strip()) for seed in cmd_args.seeds.split(',') if seed.strip()]
    else:
        seeds = [int(cmd_args.seed)]

    results = []
    for seed in seeds:
        args.seed = seed
        results.append(train_hysinc(args, print_result=(len(seeds) == 1)))

    if len(results) > 1:
        acc = np.array([item["acc"] for item in results], dtype=float)
        nmi = np.array([item["nmi"] for item in results], dtype=float)
        f1 = np.array([item["f1"] for item in results], dtype=float)
        ari = np.array([item["ari"] for item in results], dtype=float)
        times = np.array([item["time"] for item in results], dtype=float)
        std_ddof = 1 if len(results) > 1 else 0

        print(f"Dataset: {args.dataset} | {len(results)}-Seed Summary")
        print(f"ACC: {acc.mean():.4f} +/- {acc.std(ddof=std_ddof):.4f}")
        print(f"NMI: {nmi.mean():.4f} +/- {nmi.std(ddof=std_ddof):.4f}")
        print(f"F1:  {f1.mean():.4f} +/- {f1.std(ddof=std_ddof):.4f}")
        print(f"ARI: {ari.mean():.4f} +/- {ari.std(ddof=std_ddof):.4f}")
        print(f"Avg Effective Time: {times.mean():.2f}s")
