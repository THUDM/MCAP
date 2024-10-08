import random
import sys
import numpy as np
import torch
import torch.nn as nn
from scipy.special import factorial, comb
import scipy.sparse as sp
import torch.nn.functional as F
import datetime


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, n_hops, n_users, interact_mat, args_config,
                 edge_dropout_rate=0.5, mess_dropout_rate=0.1, t_u=1, t_i=2, u2u={}):
        super(GraphConv, self).__init__()

        self.interact_mat = interact_mat
        self.n_users = n_users
        self.n_hops = n_hops
        self.args_config = args_config
        self.edge_dropout_rate = edge_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout
        self.t_u = torch.FloatTensor([t_u])
        self.t_i = torch.FloatTensor([t_i])
        self.embed_dim = args_config.dim
        self.top_u = args_config.top_u
        self.pool = args_config.pool
        if self.pool == "concat":
            self.att1 = torch.nn.Linear(self.embed_dim * 4, self.embed_dim)
        else:
            self.att1 = torch.nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.att2 = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.att3 = torch.nn.Linear(self.embed_dim, 1)
        self.u2u = u2u
        self.device = torch.device("cuda:{}".format(args_config.gpu_id)) if args_config.gpu_id != -1 and torch.cuda.is_available() else torch.device("cpu")

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()
        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()
        i = i[:, dropout_mask]
        v = v[dropout_mask]
        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def get_emb_with_att(self,embs,users):
        # embs: 用户嵌入矩阵
        # u2u: 用户之间的相似度矩阵
        top_u = self.top_u
        batch_users = users
        batch_users_ids = [u.item() for u in batch_users if len(self.u2u[u.item()])>top_u]
        batch_user_sim = [list(self.u2u[u])[:top_u] for u in batch_users_ids]
        user_ids = [[row[0] for row in user_sim] for user_sim in batch_user_sim]
        user_sims = [[row[1] for row in user_sim] for user_sim in batch_user_sim]
        user_sims = torch.tensor(user_sims).to(self.device)
        users_emb = embs[user_ids] * user_sims.unsqueeze(-1)
        u_emb = embs[batch_users_ids].repeat(top_u, 1, 1).transpose(0, 1)
        x = torch.cat((users_emb, u_emb), dim=-1)
        x = F.relu(self.att1(x))
        if self.args_config.att_dropout:
            x = F.dropout(x, training=self.training)
        x = F.relu(self.att2(x))    
        if self.args_config.att_dropout:
            x = F.dropout(x, training=self.training)
        x = self.att3(x)
        att_w = F.softmax(x, dim=1) 
        att_history = torch.matmul(att_w.transpose(-1, -2), users_emb).squeeze(-2)
        embs[batch_users_ids] = att_history
        return embs
    
    def pooling(self, embeddings):
        # [-1, n_hops, channel]
        if self.pool == 'mean':
            return embeddings.mean(dim=1)
        elif self.pool == 'sum':
            return embeddings.sum(dim=1)
        elif self.pool == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:  # final
            return embeddings[:, -1, :]
        
    def forward(self, user_embed, item_embed, batch,
                mess_dropout=True, edge_dropout=True):
        # user_embed: [n_users, channel]
        # item_embed: [n_items, channel]
        # all_embed: [n_users+n_items, channel]
        all_embed = torch.cat([user_embed, item_embed], dim=0)
        agg_embed = all_embed
        embs = [all_embed]
        for hop in range(self.n_hops):
            interact_mat = self._sparse_dropout(self.interact_mat,
                                                self.edge_dropout_rate) if edge_dropout \
                                                                        else self.interact_mat
            agg_embed = torch.sparse.mm(interact_mat, agg_embed)
            if mess_dropout:
                agg_embed = self.dropout(agg_embed)
            embs.append(agg_embed)        
        embs = torch.stack(embs, dim=1)  # [n_entity, n_hops+1, emb_size]
        embs = self.pooling(embs)
        if  batch is not None and self.args_config.with_user:
            users = batch['users']
            embs = self.get_emb_with_att(embs,users)
        return embs[:self.n_users, :], embs[self.n_users:, :]


class MCAP(nn.Module):
    def __init__(self, data_config, args_config, adj_mat, u2u):
        super(MCAP, self).__init__()
        
        self.args_config = args_config
        self.set_seed()
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.adj_mat = adj_mat
        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.edge_dropout = args_config.edge_dropout
        self.edge_dropout_rate = args_config.edge_dropout_rate
        self.n_negs = args_config.n_negs
        self.ns = args_config.ns
        self.K = args_config.K
        self.device = torch.device("cuda:{}".format(args_config.gpu_id)) if args_config.gpu_id != -1 and torch.cuda.is_available() else torch.device("cpu")
        self.t_u = args_config.t_u
        self.t_i = args_config.t_i
        self._init_weight()
        self.user_embed = nn.Parameter(self.user_embed)
        self.item_embed = nn.Parameter(self.item_embed)
        self.gcn = self._init_model(u2u)
        

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.user_embed = initializer(torch.empty(self.n_users, self.emb_size))
        self.item_embed = initializer(torch.empty(self.n_items, self.emb_size))
        # [n_users+n_items, n_users+n_items]
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self, u2u):
        return GraphConv(n_hops=self.context_hops,
                         n_users=self.n_users,
                         interact_mat=self.sparse_norm_adj,
                         args_config=self.args_config,
                         edge_dropout_rate=self.edge_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate,
                         t_u=self.t_u,
                         t_i=self.t_i,
                         u2u = u2u)

    def set_seed(self):
        seed = self.args_config.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def forward(self, batch=None):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']  # [batch_size, n_negs * K]
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed,
                                              self.item_embed,
                                              batch = batch,
                                              edge_dropout=self.edge_dropout,
                                              mess_dropout=self.mess_dropout)
        neg_item = torch.LongTensor([each[0] for each in neg_item]).to(self.device)
        return self.create_bpr_loss(user_gcn_emb[user], item_gcn_emb[pos_item], item_gcn_emb[neg_item],
                                    self.user_embed[user], self.item_embed[pos_item], self.item_embed[neg_item])
        
    def generate(self, split=True):
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed,
                                              self.item_embed,
                                              batch = None,
                                              edge_dropout=False,
                                              mess_dropout=False)
        if split:
            return user_gcn_emb, item_gcn_emb
        else:
            return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, user_gcn_emb, pos_gcn_embs, neg_gcn_embs, users_emb_ego, pos_emb_ego, neg_emb_ego):

        
        batch_size = user_gcn_emb.shape[0]
        pos_scores = torch.sum(torch.mul(user_gcn_emb, pos_gcn_embs), axis=1)
        neg_scores = torch.sum(torch.mul(user_gcn_emb.unsqueeze(dim=1), neg_gcn_embs), axis=-1)
        mf_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores.unsqueeze(dim=1)))
        # cul regularizer
        regularize = (torch.norm(users_emb_ego) ** 2
                       + torch.norm(pos_emb_ego) ** 2
                       + torch.norm(neg_emb_ego) ** 2) / 2 
        emb_loss = self.decay * regularize / batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss

