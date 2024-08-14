# coding: utf-8
#
# user-graph need to be generated by the following script
# tools/generate-u-u-matrix.py
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
import torch_geometric

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_uniform_initialization

class ProjectHead(nn.Module):
    def __init__(self, input_dim=2816, hidden_dim=2048, out_dim=128):
        super(ProjectHead, self).__init__()
        self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim)
            )
        
    def forward(self, feat):
        feat = F.normalize(self.head(feat), dim=1)
        return feat

class CLUBSample(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size, device='cpu'):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())
        self.to(device)

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        sample_size = x_samples.shape[0]
        random_index = torch.randperm(sample_size).long()
        positive = (-(mu - y_samples) ** 2 / logvar.exp() / 2. - logvar / 2.).sum(dim=1)
        negative = (-(mu - y_samples[random_index]) ** 2 / logvar.exp() / 2. - logvar / 2.).sum(dim=1)
        bound = (positive - negative).mean()
        return torch.clamp(bound / 2., min=0.0)

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        llh = (-(mu - y_samples) ** 2 / logvar.exp() / 2. - logvar / 2.).sum(dim=1).mean()
        return llh

    def learning_loss(self, x_samples, y_samples):
        return -self.loglikeli(x_samples, y_samples)
    
class SEA(GeneralRecommender):
    def __init__(self, config, dataset, args):
        super(SEA, self).__init__(config, dataset)

        num_user = self.n_users
        num_item = self.n_items
        batch_size = config['train_batch_size']  # not used
        dim_x = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_mm_layers']
        self.knn_k = config['knn_k']
        self.mm_image_weight = config['mm_image_weight']

        self.alpha_contrast = args.alpha_contrast
        self.temp = args.temp
        self.hidden_dim = args.hidden_dim
        self.out_dim = args.out_dim
        self.ureg = args.ureg
        self.reg_weight = args.reg_weight
        self.t_proj = ProjectHead(input_dim=192, hidden_dim=self.hidden_dim, out_dim=self.out_dim)
        self.v_proj = ProjectHead(input_dim=2048, hidden_dim=self.hidden_dim, out_dim=self.out_dim)
        self.u_embedding_size = self.i_embedding_size = config['embedding_size']
        # self.item_linear = nn.Linear(self.item_raw_features.shape[1], self.i_embedding_size)
        self.t_item_linear = nn.Linear(self.t_feat.shape[1], 384)
        self.v_item_linear = nn.Linear(self.v_feat.shape[1], 4096)
        self.t_trans1 = nn.Linear(192,32)
        self.t_trans2 = nn.Linear(192,32)
        self.v_trans1 = nn.Linear(2048,32)
        self.v_trans2 = nn.Linear(2048,32)

        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.k = 40
        self.aggr_mode = 'add'
        self.dataset = dataset
        self.dropout = config['dropout']
        # self.construction = 'weighted_max'
        self.reg_weight = config['reg_weight']
        self.temp = config['temp']
        self.drop_rate = 0.1
        self.v_rep = None
        self.t_rep = None
        self.v_preference = None
        self.t_preference = None
        self.id_preference = None
        self.dim_latent = 64
        self.dim_feat = 128
        self.mm_adj = None

        self.mlp = nn.Linear(2*dim_x, 2*dim_x)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        self.user_graph_dict = np.load(os.path.join(dataset_path, config['user_graph_dict_file']),
                                       allow_pickle=True).item()

        mm_adj_file = os.path.join(dataset_path, 'mm_adj_{}.pt'.format(self.knn_k))

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)
        else:
            if self.v_feat is not None:
                indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj
                del image_adj
            torch.save(self.mm_adj, mm_adj_file)
        self.mm_adj = self.mm_adj.to(self.device)
        # packing interaction in training into edge_index
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = self.pack_edge_index(train_interactions)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

        # pdb.set_trace()
        self.weight_u = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.num_user, 2, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_u.data = F.softmax(self.weight_u, dim=1)

        self.item_index = torch.zeros([self.num_item], dtype=torch.long)
        index = []
        for i in range(self.num_item):
            self.item_index[i] = i
            index.append(i)
        self.drop_percent = self.drop_rate
        self.single_percent = 1
        self.double_percent = 0

        drop_item = torch.tensor(
            np.random.choice(self.item_index, int(self.num_item * self.drop_percent), replace=False))
        drop_item_single = drop_item[:int(self.single_percent * len(drop_item))]

        self.dropv_node_idx_single = drop_item_single[:int(len(drop_item_single) * 1 / 3)]
        self.dropt_node_idx_single = drop_item_single[int(len(drop_item_single) * 2 / 3):]

        self.dropv_node_idx = self.dropv_node_idx_single
        self.dropt_node_idx = self.dropt_node_idx_single

        mask_cnt = torch.zeros(self.num_item, dtype=int).tolist()
        for edge in edge_index:
            mask_cnt[edge[1] - self.num_user] += 1
        mask_dropv = []
        mask_dropt = []
        for idx, num in enumerate(mask_cnt):
            temp_false = [False] * num
            temp_true = [True] * num
            mask_dropv.extend(temp_false) if idx in self.dropv_node_idx else mask_dropv.extend(temp_true)
            mask_dropt.extend(temp_false) if idx in self.dropt_node_idx else mask_dropt.extend(temp_true)

        edge_index = edge_index[np.lexsort(edge_index.T[1, None])]
        edge_index_dropv = edge_index[mask_dropv]
        edge_index_dropt = edge_index[mask_dropt]

        self.edge_index_dropv = torch.tensor(edge_index_dropv).t().contiguous().to(self.device)
        self.edge_index_dropt = torch.tensor(edge_index_dropt).t().contiguous().to(self.device)

        self.edge_index_dropv = torch.cat((self.edge_index_dropv, self.edge_index_dropv[[1, 0]]), dim=1)
        self.edge_index_dropt = torch.cat((self.edge_index_dropt, self.edge_index_dropt[[1, 0]]), dim=1)

        self.MLP_user = nn.Linear(self.dim_latent * 2, self.dim_latent)

        if self.v_feat is not None:
            self.v_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode, dim_latent=64,
                             device=self.device, features=self.v_feat)
        if self.t_feat is not None:
            self.t_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode, dim_latent=64,
                             device=self.device, features=self.t_feat)


        self.result_embed = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, dim_x)))).to(self.device)

        self.result_embed_guide = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, dim_x)))).to(self.device)
        self.result_embed_v = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, dim_x)))).to(self.device)
        self.result_embed_t = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, dim_x)))).to(self.device)

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def pre_epoch_processing(self):
        self.epoch_user_graph, self.user_weight_matrix = self.topk_sample(self.k)
        self.user_weight_matrix = self.user_weight_matrix.to(self.device)

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        # ndarray([598918, 2]) for ml-imdb
        return np.column_stack((rows, cols))

    def forward(self, interaction):
        user_nodes, pos_item_nodes, neg_item_nodes = interaction[0], interaction[1], interaction[2]
        pos_item_nodes += self.n_users
        neg_item_nodes += self.n_users

        t_item_embeddings = self.t_item_linear(self.t_feat)
        v_item_embeddings = self.v_item_linear(self.v_feat)

        # Prepare for Contrastive learning
        t_dim = int(t_item_embeddings.shape[1] / 2)
        v_dim = int(v_item_embeddings.shape[1] / 2)
        t_emd_proj = self.t_proj(t_item_embeddings[:, :192])
        v_emd_proj = self.v_proj(v_item_embeddings[:, :2048])

        # GCN for id, v, t modalities
        self.v_rep, _ = self.v_gcn(self.edge_index_dropv, self.edge_index, v_item_embeddings)  # 7050, 4096 -> 26495, 64
        self.t_rep, _ = self.t_gcn(self.edge_index_dropt, self.edge_index, t_item_embeddings)  # 7050, 384 -> 26495, 64

        # v, v, id, and vt modalities
        representation = torch.cat((self.v_rep, self.t_rep), dim=1) # -> 26495, 128

        self.v_rep = torch.unsqueeze(self.v_rep, 2) # -> 26495, 64, 1
        self.t_rep = torch.unsqueeze(self.t_rep, 2) # -> 26495, 64, 1

        user_rep = torch.cat((self.v_rep[:self.num_user], self.t_rep[:self.num_user]), dim=2)
        user_rep = self.weight_u.transpose(1, 2) * user_rep
        user_rep = torch.cat((user_rep[:, :, 0], user_rep[:, :, 1]), dim=1)

        item_rep = representation[self.num_user:]

        # build item-item graph
        h = self.buildItemGraph(item_rep)

        user_rep = user_rep
        item_rep = item_rep + h

        # build result embedding
        self.user_rep = user_rep
        self.item_rep = item_rep
        self.result_embed = torch.cat((user_rep, item_rep), dim=0)

        # calculate pos and neg scores
        user_tensor = self.result_embed[user_nodes]
        pos_item_tensor = self.result_embed[pos_item_nodes]
        neg_item_tensor = self.result_embed[neg_item_nodes]
        pos_scores = torch.sum(user_tensor * pos_item_tensor, dim=1)
        neg_scores = torch.sum(user_tensor * neg_item_tensor, dim=1)
        return pos_scores, neg_scores, t_emd_proj, v_emd_proj, t_item_embeddings, v_item_embeddings, t_dim, v_dim

    def buildItemGraph(self, h):
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        return h

    def calculate_loss(self, interaction, club):
        user = interaction[0]
        pos_scores, neg_scores, t_emd_proj, v_emd_proj, t_item_embeddings, v_item_embeddings, t_dim, v_dim = self.forward(interaction)
        loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))

        loss_Solosimloss = self.Solosimloss(t_emd_proj, v_emd_proj, temperature=self.temp)

        t_trans1 = self.t_trans1(t_item_embeddings[:, :192])
        v_trans1 = self.v_trans1(v_item_embeddings[:, :2048])
        t_trans2 = self.t_trans2(t_item_embeddings[:, 192:])
        v_trans2 = self.v_trans2(v_item_embeddings[:, 2048:])

        # mutual information minmization loss in Eq.15
        loss_e1 = self.ureg * club.forward(t_trans1, t_trans2)
        loss_e2 = self.ureg * club.forward(v_trans1, v_trans2)
        
        # negative l_2 lioss in Eq.30
        # loss_e1 = - F.mse_loss(t_trans1, t_trans2)
        # loss_e1 = - F.mse_loss(v_trans1, v_trans2)

        loss = loss_value + (loss_e1 + loss_e2)/2 + self.alpha_contrast * loss_Solosimloss
        return loss, (t_trans1, v_trans1, t_trans2, v_trans2)

    def full_sort_predict(self, interaction):
        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix

    def topk_sample(self, k):
        user_graph_index = []
        count_num = 0
        user_weight_matrix = torch.zeros(len(self.user_graph_dict), k)
        tasike = []
        for i in range(k):
            tasike.append(0)
        for i in range(len(self.user_graph_dict)):
            if len(self.user_graph_dict[i][0]) < k:
                count_num += 1
                if len(self.user_graph_dict[i][0]) == 0:
                    # pdb.set_trace()
                    user_graph_index.append(tasike)
                    continue
                user_graph_sample = self.user_graph_dict[i][0][:k]
                user_graph_weight = self.user_graph_dict[i][1][:k]
                while len(user_graph_sample) < k:
                    rand_index = np.random.randint(0, len(user_graph_sample))
                    user_graph_sample.append(user_graph_sample[rand_index])
                    user_graph_weight.append(user_graph_weight[rand_index])
                user_graph_index.append(user_graph_sample)

                user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
                continue
            user_graph_sample = self.user_graph_dict[i][0][:k]
            user_graph_weight = self.user_graph_dict[i][1][:k]

            user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
            user_graph_index.append(user_graph_sample)

        # pdb.set_trace()
        return user_graph_index, user_weight_matrix

    def print_embd(self):
        return self.result_embed_v, self.result_embed_t
    
    # corresponds to Eq.16
    def Solosimloss(self, view1, view2, temperature: float, b_cos: bool = True):
        if b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

        pos_score = (view1 @ view2.T) / temperature
        score = torch.diag(F.log_softmax(pos_score, dim=1))
        return -score.mean()


class GCN(torch.nn.Module):
    def __init__(self, datasets, batch_size, num_user, num_item, dim_id, aggr_mode,
                 dim_latent=None, device=None, features=None):
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.datasets = datasets
        self.dim_id = dim_id
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode
        self.device = device

        if self.dim_latent:
            self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(num_user, self.dim_latent), dtype=torch.float32, requires_grad=True),
                gain=1).to(self.device))
            self.MLP = nn.Linear(self.dim_feat, 4 * self.dim_latent)
            self.MLP_1 = nn.Linear(4 * self.dim_latent, self.dim_latent)
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

        else:
            self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(num_user, self.dim_feat), dtype=torch.float32, requires_grad=True),
                gain=1).to(self.device))
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

    def forward(self, edge_index_drop, edge_index, features, perturbed=False):
        temp_features = self.MLP_1(F.leaky_relu(self.MLP(features))) if self.dim_latent else features
        x = torch.cat((self.preference, temp_features), dim=0).to(self.device)
        x = F.normalize(x).to(self.device)

        h = self.conv_embed_1(x, edge_index)
        if perturbed:
            random_noise = torch.rand_like(h).to(self.device)
            h += torch.sign(h) * F.normalize(random_noise, dim=-1) * 0.1
        h_1 = self.conv_embed_1(h, edge_index)
        if perturbed:
            random_noise = torch.rand_like(h).to(self.device)
            h_1 += torch.sign(h_1) * F.normalize(random_noise, dim=-1) * 0.1
        # h_2 = self.conv_embed_1(h_1, edge_index)

        x_hat = x + h + h_1
        return x_hat, self.preference


class Base_gcn(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(Base_gcn, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, size=None):
        # pdb.set_trace()
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
            # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        # pdb.set_trace()
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        if self.aggr == 'add':
            # pdb.set_trace()
            row, col = edge_index
            deg = degree(row, size[0], dtype=x_j.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
