import torch
import torch.nn as nn
import torch.nn.functional as F

class HGNN_Layer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HGNN_Layer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_ch, out_ch))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, H, dv_inv_sqrt, de_inv):

        dv = dv_inv_sqrt.view(-1, 1) if dv_inv_sqrt.dim() == 1 else dv_inv_sqrt
        de = de_inv.view(-1, 1) if de_inv.dim() == 1 else de_inv


        x = torch.matmul(x, self.weight)


        x = x * dv
        x = torch.sparse.mm(H.t(), x)


        x = x * de


        x = torch.sparse.mm(H, x)


        x = x * dv

        return x


class HGNN(nn.Module):
    def __init__(self, in_ch, n_hid, num_clusters, svd_dim=None, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.svd_dim = svd_dim if svd_dim is not None else num_clusters


        self.hgc1 = HGNN_Layer(in_ch, n_hid)
        self.hgc2 = HGNN_Layer(n_hid, n_hid)

        self.ln1 = nn.LayerNorm(n_hid)
        self.ln2 = nn.LayerNorm(n_hid)

        self.res_proj = nn.Linear(in_ch, n_hid) if in_ch != n_hid else nn.Identity()


        self.cluster_projection_head = nn.Linear(n_hid, self.svd_dim)
        self.cluster_output = nn.Linear(self.svd_dim, num_clusters)


        self.contrast_head = nn.Sequential(
            nn.Linear(n_hid, n_hid),
            nn.ReLU(inplace=True),
            nn.Linear(n_hid, n_hid)
        )


        self.decoder = nn.Linear(n_hid, in_ch)


    def forward(self, x, H, dv_inv_sqrt, de_inv):

        x_residual = self.res_proj(x)


        x1 = self.hgc1(x, H, dv_inv_sqrt, de_inv)
        x1 = self.ln1(x1)
        x1 = F.leaky_relu(x1, 0.2)
        x1 = F.dropout(x1, self.dropout, training=self.training)

        x1 = x1 + x_residual


        x2 = self.hgc2(x1, H, dv_inv_sqrt, de_inv)
        x2 = self.ln2(x2)
        x2 = F.leaky_relu(x2, 0.2)


        x_hidden = x2 + x1 + x_residual


        final_emb = self.cluster_projection_head(x_hidden)
        logits = self.cluster_output(final_emb)

        z_contrast = self.contrast_head(x_hidden)

        x_rec = self.decoder(x_hidden)
        x_rec = F.normalize(x_rec, p=2, dim=1)

        return logits, final_emb, x_rec, z_contrast
