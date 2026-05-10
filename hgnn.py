import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FEGRLLayer(nn.Module):
    def __init__(self, op_in_dim, m_in_dim, edge_dim, hidden_dim):
        super(FEGRLLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim

        self.W_m_1 = nn.Linear(m_in_dim + edge_dim, hidden_dim)
        self.W_m_2 = nn.Linear(m_in_dim, hidden_dim)
        self.W_o_1 = nn.Linear(op_in_dim, hidden_dim)
        self.attn_vec_m = nn.Parameter(torch.randn(1, 2 * hidden_dim))

        self.W_o_2 = nn.Linear(op_in_dim + edge_dim, hidden_dim)
        self.W_o_3 = nn.Linear(op_in_dim, hidden_dim)
        self.W_m_3 = nn.Linear(hidden_dim, hidden_dim)
        self.W_o_prev = nn.Linear(op_in_dim, hidden_dim)
        self.W_o_next = nn.Linear(op_in_dim, hidden_dim)
        self.W_o_5 = nn.Linear(hidden_dim, hidden_dim)
        self.attn_vec_om = nn.Parameter(torch.randn(1, 2 * hidden_dim))
        self.attn_vec_prev = nn.Parameter(torch.randn(1, 2 * hidden_dim))
        self.attn_vec_next = nn.Parameter(torch.randn(1, 2 * hidden_dim))
        self.attn_vec_self = nn.Parameter(torch.randn(1, 2 * hidden_dim))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _score(self, x, attn_vec):
        return torch.exp(
            torch.clamp(
                F.leaky_relu(torch.matmul(x, attn_vec.t()), negative_slope=0.2),
                min=-10,
                max=10
            )
        )

    def forward(self, x_o, x_m, edge_attr, edge_index_om, edge_index_prev, edge_index_next):
        device = x_o.device
        N_m = x_m.size(0)

        feat_o_neighbor = self.W_o_1(x_o)
        feat_m_content = self.W_m_2(x_m)

        v_comb_self = torch.cat([x_m, torch.zeros(N_m, self.edge_dim).to(device)], dim=1)
        feat_m_self_key = self.W_m_1(v_comb_self)
        scores_mm = self._score(torch.cat([feat_m_self_key, feat_m_self_key], dim=1), self.attn_vec_m)

        denom_m = scores_mm.clone()
        agg_total_m = scores_mm * feat_m_content

        if edge_index_om.size(1) > 0:
            src_o = edge_index_om[0]
            tgt_m = edge_index_om[1]
            feat_m_key = self.W_m_1(torch.cat([x_m[tgt_m], edge_attr], dim=1))
            scores_mo = self._score(torch.cat([feat_m_key, feat_o_neighbor[src_o]], dim=1), self.attn_vec_m)
            denom_m.index_add_(0, tgt_m, scores_mo)
            agg_total_m.index_add_(0, tgt_m, scores_mo * feat_o_neighbor[src_o])

        v_prime_m = F.elu(agg_total_m / (denom_m + 1e-9))

        feat_o_self_key = self.W_o_3(x_o)
        feat_o_self_val = self.W_o_3(x_o)
        feat_m_new = self.W_m_3(v_prime_m)
        feat_o_prev = self.W_o_prev(x_o)
        feat_o_next = self.W_o_next(x_o)

        scores_oo = self._score(torch.cat([feat_o_self_key, feat_o_self_key], dim=1), self.attn_vec_self)
        denom_o = scores_oo.clone()
        agg_total_o = scores_oo * feat_o_self_val

        if edge_index_om.size(1) > 0:
            src_o = edge_index_om[0]
            tgt_m = edge_index_om[1]
            feat_mu_key = self.W_o_2(torch.cat([x_o[src_o], edge_attr], dim=1))
            scores_om = self._score(torch.cat([feat_mu_key, feat_m_new[tgt_m]], dim=1), self.attn_vec_om)
            denom_o.index_add_(0, src_o, scores_om)
            agg_total_o.index_add_(0, src_o, scores_om * feat_m_new[tgt_m])

        if edge_index_prev.size(1) > 0:
            tgt = edge_index_prev[0]
            src = edge_index_prev[1]
            scores_prev = self._score(torch.cat([feat_o_self_key[tgt], feat_o_prev[src]], dim=1), self.attn_vec_prev)
            denom_o.index_add_(0, tgt, scores_prev)
            agg_total_o.index_add_(0, tgt, scores_prev * feat_o_prev[src])

        if edge_index_next.size(1) > 0:
            tgt = edge_index_next[0]
            src = edge_index_next[1]
            scores_next = self._score(torch.cat([feat_o_self_key[tgt], feat_o_next[src]], dim=1), self.attn_vec_next)
            denom_o.index_add_(0, tgt, scores_next)
            agg_total_o.index_add_(0, tgt, scores_next * feat_o_next[src])

        mu_prime_r = F.elu(self.W_o_5(agg_total_o / (denom_o + 1e-9)))
        return mu_prime_r, v_prime_m


class FEGRL_Network(nn.Module):
    def __init__(self, num_op_features=8, num_m_features=7, num_edge_features=6, hidden_dim=64, num_layers=2):
        super(FEGRL_Network, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        for l in range(num_layers):
            op_in = num_op_features if l == 0 else hidden_dim
            m_in = num_m_features if l == 0 else hidden_dim
            self.layers.append(
                FEGRLLayer(op_in_dim=op_in, m_in_dim=m_in, edge_dim=num_edge_features, hidden_dim=hidden_dim)
            )

        self.W_g1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K_o = nn.Linear(hidden_dim, hidden_dim)
        self.W_K_m = nn.Linear(hidden_dim, hidden_dim)
        self.W_V_o = nn.Linear(hidden_dim, hidden_dim)
        self.W_V_m = nn.Linear(hidden_dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(4 * hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.critic = nn.Sequential(
            nn.Linear(2 * hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state, device):
        x_o = torch.as_tensor(state['x_o'], dtype=torch.float32, device=device)
        x_m = torch.as_tensor(state['x_m'], dtype=torch.float32, device=device)
        edge_attr = torch.as_tensor(state['edge_attr_om'], dtype=torch.float32, device=device)
        edge_index_om = torch.as_tensor(state['edge_index_om'], dtype=torch.long, device=device)
        edge_index_prev = torch.as_tensor(state['edge_index_prev'], dtype=torch.long, device=device)
        edge_index_next = torch.as_tensor(state['edge_index_next'], dtype=torch.long, device=device)

        mu = x_o
        v = x_m
        for layer in self.layers:
            mu, v = layer(mu, v, edge_attr, edge_index_om, edge_index_prev, edge_index_next)

        h_global_0 = self.W_g1(torch.cat([
            torch.mean(mu, dim=0, keepdim=True),
            torch.mean(v, dim=0, keepdim=True)
        ], dim=1))

        Q_global = self.W_Q(h_global_0)
        e_o = F.softmax(torch.matmul(Q_global, self.W_K_o(mu).t()) / math.sqrt(self.hidden_dim), dim=1)
        e_m = F.softmax(torch.matmul(Q_global, self.W_K_m(v).t()) / math.sqrt(self.hidden_dim), dim=1)
        h_global = torch.cat([
            torch.matmul(e_o, self.W_V_o(mu)),
            torch.matmul(e_m, self.W_V_m(v))
        ], dim=1)

        valid_actions = state['valid_actions']
        value = self.critic(h_global)

        if not valid_actions:
            return None, value, h_global

        logits = []
        for o, m in valid_actions:
            act_feat = torch.cat([h_global.squeeze(0), v[m], mu[o]], dim=0)
            logits.append(self.mlp(act_feat).squeeze(-1))

        logits = torch.stack(logits)
        probs = F.softmax(logits, dim=0)
        return probs, value, h_global