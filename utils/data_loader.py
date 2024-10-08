import numpy as np
import scipy.sparse as sp

from collections import defaultdict
import warnings

import torch

warnings.filterwarnings('ignore')

n_users = 0
n_items = 0
dataset = ''
train_user_set = defaultdict(list)
train_item_set = defaultdict(list)
test_user_set = defaultdict(list)
valid_user_set = defaultdict(list)
u2u_co_author = {}
i2i_co_author = {}
u2u_co_venue = {}
i2i_co_venue = {}


def read_cf_dataset(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]
        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])
    return np.array(inter_mat)


def statistics(train_data, valid_data, test_data):
    global n_users, n_items
    n_users = max(max(train_data[:, 0]), max(valid_data[:, 0]), max(test_data[:, 0])) + 1
    n_items = max(max(train_data[:, 1]), max(valid_data[:, 1]), max(test_data[:, 1])) + 1
    
    if dataset not in ['yelp2018', 'gowalla', 'aminer', 'citeulike', 'amazon-book', 'aminer-wechat', 'dblp']:
        n_items -= n_users
        # remap [n_users, n_users+n_items] to [0, n_items]
        train_data[:, 1] -= n_users
        valid_data[:, 1] -= n_users
        test_data[:, 1] -= n_users
        
    cnt_train, cnt_test, cnt_valid = 0, 0, 0
    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
        train_item_set[int(i_id)].append(int(u_id))
        cnt_train += 1
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))
        cnt_test += 1
    for u_id, i_id in valid_data:
        valid_user_set[int(u_id)].append(int(i_id))
        cnt_valid += 1
    print('n_users: ', n_users, '\tn_items: ', n_items)
    print('n_train: ', cnt_train, '\tn_test: ', cnt_test, '\tn_valid: ', cnt_valid)
    print('n_inters: ', cnt_train + cnt_test + cnt_valid)


def build_sparse_graph(adj_list):
    def _bi_norm_lap(adj): 
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()
    
    mat = sum(adj_list)
    return _bi_norm_lap(mat), np.array(mat.sum(1)), np.array(mat.sum(0))

def load_relation_data(file_name):
    u2u = dict() # u2u or i2i
    sim_mat = list()
    print('load file: ', file_name)
    lines = open(file_name, 'r').readlines()
    for line in lines:
        tmps = line.strip().split(' ')
        if len(tmps) == 3:
            p_id1, p_id2, score = int(tmps[0]), int(tmps[1]), int(tmps[2])
            sim_mat.append([p_id1, p_id2, score])
            if p_id1 not in u2u.keys():
                u2u[p_id1] = []
            u2u[p_id1].append(p_id2)
    return np.array(sim_mat), u2u

def load_sim_datas(file_name):
    sim_mat = list()
    print('load file: ', file_name)
    lines = open(file_name, 'r').readlines()
    for line in lines:
        tmps = line.strip().split(' ')
        if len(tmps) == 3:
            p_id1, p_id2, score = int(tmps[0]), int(tmps[1]) - 1, float(tmps[2])
            sim_mat.append([p_id1, p_id2, score])
    return np.array(sim_mat)

def get_adj_features_list(train_cf):
    adj_mat_list = []
    n_all = n_users + n_items
    def _np_mat2sp_adj(np_mat, row_pre, col_pre):
        # 存了两个ui矩阵
        a_rows = np_mat[:, 0] + row_pre 
        a_cols = np_mat[:, 1] + col_pre
        a_vals = [1.] * len(a_rows)

        b_rows = a_cols
        b_cols = a_rows
        b_vals = [1.] * len(b_rows)

        a_adj = sp.coo_matrix((a_vals, (a_rows, a_cols)), shape=(n_all, n_all))
        b_adj = sp.coo_matrix((b_vals, (b_rows, b_cols)), shape=(n_all, n_all))

        return a_adj, b_adj

    R, R_inv = _np_mat2sp_adj(train_cf, row_pre=0, col_pre=n_users)
    adj_mat_list.append(R)
    adj_mat_list.append(R_inv)
    print('\t==========convert user-item into adj mat done.')
    global alpha
    alpha = 1.0
    if args.with_user and args.with_uu_co_author and args.with_uu_co_venue:
        alpha = 0.33
    elif args.with_user and (args.with_uu_co_author or args.with_uu_co_venue):
        alpha = 0.5
    def _np_mat2sym_sp_adj(np_sim_mat, row_pre, col_pre):
        sim_rows = list(np.array(np_sim_mat[:, 0], dtype=int) + row_pre)
        sim_cols = list(np.array(np_sim_mat[:, 1], dtype=int) + col_pre)
        sim_vals = list(np_sim_mat[:, 2] * alpha)

        sim_rows1 = list(np.array(np_sim_mat[:, 1], dtype=int) + row_pre)
        sim_cols1 = list(np.array(np_sim_mat[:, 0], dtype=int) + col_pre)
        sim_vals1 = list(np_sim_mat[:, 2] * alpha)
        return sim_rows, sim_cols, sim_vals, sim_rows1, sim_cols1, sim_vals1
    
    def _np_mat2sym_sp_adj_sim_sort(np_sim_mat, row_pre, col_pre):
        sim_rows = list(np.array(np_sim_mat[:, 0], dtype=int) + row_pre)
        sim_cols = list(np.array(np_sim_mat[:, 1], dtype=int) + col_pre)
        sim_vals = list(np_sim_mat[:, 2] * alpha)
        sim_rows1 = list()
        sim_cols1 = list()
        sim_vals1 = list()
        return sim_rows, sim_cols, sim_vals, sim_rows1, sim_cols1, sim_vals1
    global u2u
    u2u = {}
    if args.with_user:  # 构造uu图，矩阵位置上是 jaccard 系数，(对角矩阵)
        try:
            user_user_mat_np = torch.load(args.data_path + args.dataset + '/' + args.dataset + '_with_user.pt')
            u2u = torch.load(args.data_path + args.dataset + '/' + args.dataset + 'u2u.pt')
        except:
            user_user_mat = list()
            for i in range(0, n_users):
                for j in range(i, n_users):
                    inter = set(train_user_set[i]).intersection(set(train_user_set[j]))
                    union = set(train_user_set[i]).union(set(train_user_set[j]))
                    jaccard = len(inter) * 1.0 / len(union)
                    user_user_mat.append([i, j, jaccard])
                    if jaccard != 0:
                        if i not in u2u.keys(): u2u[i] = set()
                        u2u[i].add((j,jaccard))
                        if j not in u2u.keys(): u2u[j] = set()
                        u2u[j].add((i,jaccard))
            user_user_mat_np = np.array(user_user_mat)
            torch.save(user_user_mat_np, args.data_path + args.dataset + '/' + args.dataset + '_with_user.pt')
            torch.save(u2u, args.data_path + args.dataset + '/' + args.dataset + 'u2u.pt')

        uu_paper_rows, uu_paper_cols, uu_paper_vals, \
        uu_paper_rows1, uu_paper_cols1, uu_paper_vals1 = _np_mat2sym_sp_adj(user_user_mat_np, 0, 0)

    if args.with_uu_co_author:  # uu图，阅读共同author论文的位置为1
        global u2u_co_author
        uu_co_author_file = args.data_path + args.dataset + '/' + args.dataset + '_uu_co_author.txt'
        uu_co_author_data, u2u_co_author = load_relation_data(uu_co_author_file)
        uu_co_author_rows, uu_co_author_cols, uu_co_author_vals, uu_co_author_rows1, \
        uu_co_author_cols1, uu_co_author_vals1 = _np_mat2sym_sp_adj(uu_co_author_data, 0, 0)
        del uu_co_author_data

    if args.with_uu_co_venue:
        global u2u_co_venue
        uu_co_venue_file = args.data_path + args.dataset + '/' + args.dataset + '_uu_co_venue.txt'
        uu_co_venue_data, u2u_co_venue = load_relation_data(uu_co_venue_file)
        uu_co_venue_rows, uu_co_venue_cols, uu_co_venue_vals, uu_co_venue_rows1, \
        uu_co_venue_cols1, uu_co_venue_vals1 = _np_mat2sym_sp_adj(uu_co_venue_data, 0, 0)
        del uu_co_venue_data

    if args.with_user and args.with_uu_co_author == 0 and args.with_uu_co_venue == 0:
        user_adj = sp.coo_matrix((uu_paper_vals + uu_paper_vals1,
                                    (uu_paper_rows + uu_paper_rows1,
                                    uu_paper_cols + uu_paper_cols1)),
                                    shape=(n_all, n_all))
        adj_mat_list.append(user_adj)

    if args.with_user and args.with_uu_co_author and args.with_uu_co_venue:
        user_adj = sp.coo_matrix((uu_paper_vals + uu_paper_vals1 +
                                    uu_co_author_vals + uu_co_author_vals1 +
                                    uu_co_venue_vals + uu_co_venue_vals1,
                                    (uu_paper_rows + uu_paper_rows1 +
                                    uu_co_author_rows + uu_co_author_rows1 +
                                    uu_co_venue_rows + uu_co_venue_rows1,
                                    uu_paper_cols + uu_paper_cols1 +
                                    uu_co_author_cols + uu_co_author_cols1 +
                                    uu_co_venue_cols + uu_co_venue_cols1)),
                                    shape=(n_all, n_all))
        adj_mat_list.append(user_adj)
        
    if args.with_user == 0 and args.with_uu_co_author and args.with_uu_co_venue:
        user_adj = sp.coo_matrix((uu_co_author_vals + uu_co_author_vals1 +
                                    uu_co_venue_vals + uu_co_venue_vals1,
                                    (uu_co_author_rows + uu_co_author_rows1 +
                                    uu_co_venue_rows + uu_co_venue_rows1,
                                    uu_co_author_cols + uu_co_author_cols1 +
                                    uu_co_venue_cols + uu_co_venue_cols1)),
                                    shape=(n_all, n_all))
        adj_mat_list.append(user_adj)
    print('\t==========convert user-user data into adj mat done.')

    # dblp
    if not args.with_sim and args.with_uu_co_author and not args.with_uu_co_venue:
        alpha = 1
        
    if args.with_sim:
        if args.load_ii_sort == 0:
            sim_file = args.data_path + args.dataset + '/' + args.dataset + '_oag_similarity8.txt'
        else:
            sim_file = args.data_path + args.dataset + '/' + args.dataset + f'_oag_similarity8_sort_{args.load_ii_sort}.txt'
        similarity_data = load_sim_datas(sim_file)
        ii_paper_rows, ii_paper_cols, ii_paper_vals, \
        ii_paper_rows1, ii_paper_cols1, ii_paper_vals1 = \
            _np_mat2sym_sp_adj_sim_sort(similarity_data, row_pre=n_users, col_pre=n_users)
        del similarity_data
    if args.with_ii_co_author:
        global i2i_co_author
        ii_co_author_file = args.data_path + args.dataset + '/' + args.dataset + '_ii_co_author.txt'
        ii_co_author_data, i2i_co_author = load_relation_data(ii_co_author_file)
        ii_co_author_rows, ii_co_author_cols, ii_co_author_vals, ii_co_author_rows1, \
        ii_co_author_cols1, ii_co_author_vals1 = \
            _np_mat2sym_sp_adj(ii_co_author_data, row_pre=n_users, col_pre=n_users)
        del ii_co_author_data

    if args.with_ii_co_venue:
        global i2i_co_venue
        ii_co_venue_file = args.data_path + args.dataset + '/' + args.dataset + '_ii_co_venue.txt'
        ii_co_venue_data, i2i_co_venue = load_relation_data(ii_co_venue_file)
        ii_co_venue_rows, ii_co_venue_cols, ii_co_venue_vals, ii_co_venue_rows1, \
        ii_co_venue_cols1, ii_co_venue_vals1 = \
            _np_mat2sym_sp_adj(ii_co_venue_data, row_pre=n_users, col_pre=n_users)
        del ii_co_venue_data

    if args.with_sim and args.with_ii_co_author == 0 and args.with_ii_co_venue == 0:
        item_adj = sp.coo_matrix((ii_paper_vals + ii_paper_vals1,
                                    (ii_paper_rows + ii_paper_rows1,
                                    ii_paper_cols + ii_paper_cols1)),
                                    shape=(n_all, n_all))
        adj_mat_list.append(item_adj)

    if args.with_sim and args.with_ii_co_author and args.with_ii_co_venue:
        item_adj = sp.coo_matrix((ii_paper_vals + ii_paper_vals1 +
                                    ii_co_author_vals + ii_co_author_vals1 +
                                    ii_co_venue_vals + ii_co_venue_vals1,
                                    (ii_paper_rows + ii_paper_rows1 +
                                    ii_co_author_rows + ii_co_author_rows1 +
                                    ii_co_venue_rows + ii_co_venue_rows1,
                                    ii_paper_cols + ii_paper_cols1 +
                                    ii_co_author_cols + ii_co_author_cols1 +
                                    ii_co_venue_cols + ii_co_venue_cols1)),
                                    shape=(n_all, n_all))
        adj_mat_list.append(item_adj)

    if args.with_sim == 0 and args.with_ii_co_author and args.with_ii_co_venue:
        item_adj = sp.coo_matrix((ii_co_author_vals + ii_co_author_vals1 +
                                    ii_co_venue_vals + ii_co_venue_vals1,
                                    (ii_co_author_rows + ii_co_author_rows1 +
                                    ii_co_venue_rows + ii_co_venue_rows1,
                                    ii_co_author_cols + ii_co_author_cols1 +
                                    ii_co_venue_cols + ii_co_venue_cols1)),
                                    shape=(n_all, n_all))
        adj_mat_list.append(item_adj)

    print('\t==========convert item-item data into adj mat done.')
    return adj_mat_list    

    

def load_data(model_args):
    global args, dataset
    args = model_args
    dataset = args.dataset
    directory = args.data_path + dataset + '/'

    print('reading train and test user-item set ...')
    train_cf = read_cf_dataset(directory + 'train.txt')
    test_cf = read_cf_dataset(directory + 'test.txt')
    if args.dataset in ['aminer', 'amazon', 'ml-1m', 'ali', 'citeulike','dblp']:
        valid_cf = read_cf_dataset(directory + 'valid.txt')
    else:
        valid_cf = test_cf
    statistics(train_cf, valid_cf, test_cf)

    print("get_adj_features_list ...")
    adj_bipartite_list = get_adj_features_list(train_cf)
    print('building the adj mat ...')
    norm_mat, indeg, outdeg = build_sparse_graph(adj_bipartite_list)

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
    }
    user_dict = {
        'train_item_set': train_item_set,
        'train_user_set': train_user_set,
        'valid_user_set': valid_user_set if args.dataset not in ['yelp2018', 'gowalla', 'amazon-book', 'aminer-wechat'] else None,
        'test_user_set': test_user_set,
    }
    for k,v in u2u.items():
        u2u[k] = sorted(v, key=lambda x: (-x[1],x[0]))
    print('loading over ...')
    return train_cf, user_dict, n_params, norm_mat, indeg, outdeg, u2u




