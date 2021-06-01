import numpy as np
import torch
import dgl


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, verbose=False, delta=0, save_path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss


class IndexGenerator:
    def __init__(self, batch_size, num_data=None, indices=None, shuffle=True):
        if num_data is not None:
            self.num_data = num_data
            self.indices = np.arange(num_data)
        if indices is not None:
            self.num_data = len(indices)
            self.indices = np.copy(indices)
        self.batch_size = batch_size
        self.iter_counter = 0
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.indices)

    def next(self):
        if self.num_iterations_left() <= 0:
            self.reset()
        self.iter_counter += 1
        return np.copy(self.indices[(self.iter_counter - 1) * self.batch_size:self.iter_counter * self.batch_size])

    def num_iterations(self):
        return int(np.ceil(self.num_data / self.batch_size))

    def num_iterations_left(self):
        return self.num_iterations() - self.iter_counter

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.iter_counter = 0


def parse_adjlist(adjlist, edge_metapath_indices, samples=None, exclude=None,
                  offset1=None, offset2=None, mode=None, use_mask=0):
    edges = []
    nodes = set()
    result_indices = []
    for row, indices in zip(adjlist, edge_metapath_indices):
        row_parsed = list(map(int, row.split(' ')))
        nodes.add(row_parsed[0])
        if len(row_parsed) > 1:
            # sampling neighbors
            if samples is None:
                if exclude is not None:
                    position = [use_mask - 1, use_mask, -use_mask, -use_mask - 1]
                    mask = [False if [ap1 - offset1, f1 - offset2] in exclude or [ap2 - offset1, f2 - offset2]
                            in exclude else True for ap1, f1, ap2, f2 in indices[:, position]]
                    neighbors = np.array(row_parsed[1:])[mask]
                    result_indices.append(indices[mask])
                else:
                    neighbors = row_parsed[1:]
                    result_indices.append(indices)
            else:
                # undersampling frequent neighbors
                unique, counts = np.unique(row_parsed[1:], return_counts=True)
                p = []
                for count in counts:
                    p += [(count ** (3 / 4)) / count] * count
                p = np.array(p)
                p = p / p.sum()
                samples = min(samples, len(row_parsed) - 1)
                sampled_idx = np.sort(np.random.choice(len(row_parsed) - 1, samples, replace=False, p=p))
                if exclude is not None:
                    position = [use_mask - 1, use_mask, -use_mask, -use_mask - 1]
                    mask = [False if [ap1 - offset1, f1 - offset2] in exclude or [ap2 - offset1, f2 - offset2]
                            in exclude else True for ap1, f1, ap2, f2 in indices[sampled_idx][:, position]]
                    neighbors = np.array([row_parsed[i + 1] for i in sampled_idx])[mask]
                    result_indices.append(indices[sampled_idx][mask])
                else:
                    neighbors = [row_parsed[i + 1] for i in sampled_idx]
                    result_indices.append(indices[sampled_idx])
        else:
            neighbors = [row_parsed[0]]
            indices = np.array([[row_parsed[0]] * indices.shape[1]])
            # if mode == 1:
            #     indices += offset
            result_indices.append(indices)
        for dst in neighbors:
            nodes.add(dst)
            edges.append((row_parsed[0], dst))
    mapping = {map_from: map_to for map_to, map_from in enumerate(sorted(nodes))}
    edges = list(map(lambda tup: (mapping[tup[0]], mapping[tup[1]]), edges))
    result_indices = np.vstack(result_indices)
    return edges, result_indices, len(nodes), mapping


def parse_minibatch(adjlists_all, edge_metapath_indices_list, batch_af, batch_pf, device, samples=None, use_masks=None, offset=None):
    """
    We have two batches this time: one for author-field, one for paper-field
    AF: mode 1, mode 2
    PF: mode 0, mode 2

    return: note that the number of A-F-P is bs, 2bs, bs
    """
    g_lists = [[], [], []]
    result_indices_lists = [[], [], []]
    idx_batch_mapped_lists = [[], [], []]

    # AF part, only take 1 and 2
    for mode, (adjlists, edge_metapath_indices_list) in enumerate(zip(adjlists_all, edge_metapath_indices_list)):
        if mode == 0:
            continue
        for adjlist, indices, use_mask in zip(adjlists, edge_metapath_indices_list, use_masks[mode]):
            if use_mask > 0:
                edges, result_indices, num_nodes, mapping = parse_adjlist(
                    [adjlist[row[mode//2]] for row in batch_af], [indices[row[mode//2]] for row in batch_af],
                    samples, batch_af, offset[0], offset[1], mode, use_mask)
            else:
                edges, result_indices, num_nodes, mapping = parse_adjlist(
                    [adjlist[row[mode//2]] for row in batch_af], [indices[row[mode//2]] for row in batch_af],
                    samples, offset1=offset[0], offset2=offset[1], mode=mode)

            g = dgl.DGLGraph(multigraph=True)
            g.add_nodes(num_nodes)
            if len(edges) > 0:
                sorted_index = sorted(range(len(edges)), key=lambda i : edges[i])
                g.add_edges(*list(zip(*[(edges[i][1], edges[i][0]) for i in sorted_index])))
                result_indices = torch.LongTensor(result_indices[sorted_index]).to(device)
            else:
                result_indices = torch.LongTensor(result_indices).to(device)
            g_lists[mode].append(g)
            result_indices_lists[mode].append(result_indices)
            idx_batch_mapped_lists[mode].append(np.array([mapping[row[mode]] for row in batch_af]))

        # PF part, only take 0 and 2
        for mode, (adjlists, edge_metapath_indices_list) in enumerate(zip(adjlists_all, edge_metapath_indices_list)):
            if mode == 1:
                continue
            for adjlist, indices, use_mask in zip(adjlists, edge_metapath_indices_list, use_masks[mode]):
                if use_mask > 0:
                    edges, result_indices, num_nodes, mapping = parse_adjlist(
                        [adjlist[row[mode//2]] for row in batch_pf], [indices[row[mode//2]] for row in batch_pf],
                        samples, batch_pf, offset[0], offset[1], mode, use_mask)
                else:
                    edges, result_indices, num_nodes, mapping = parse_adjlist(
                        [adjlist[row[mode//2]] for row in batch_pf], [indices[row[mode//2]] for row in batch_pf],
                        samples, offset1=offset[0], offset2=offset[1], mode=mode)

                g = dgl.DGLGraph(multigraph=True)
                g.add_nodes(num_nodes)
                if len(edges) > 0:
                    sorted_index = sorted(range(len(edges)), key=lambda i: edges[i])
                    g.add_edges(*list(zip(*[(edges[i][1], edges[i][0]) for i in sorted_index])))
                    result_indices = torch.LongTensor(result_indices[sorted_index]).to(device)
                else:
                    result_indices = torch.LongTensor(result_indices).to(device)
                g_lists[mode].append(g)
                result_indices_lists[mode].append(result_indices)
                idx_batch_mapped_lists[mode].append(np.array([mapping[row[mode]] for row in batch_pf]))

    return g_lists, result_indices_lists, idx_batch_mapped_lists

