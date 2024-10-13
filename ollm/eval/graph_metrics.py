import graph_tool.all as gt
import networkx as nx
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch_geometric.data import Batch
from torch_geometric.nn import SGConv
from torch_geometric.utils import from_networkx

from ollm.utils import (
    Graph,
    batch,
    cosine_sim,
    device,
    embed,
    load_embedding_model,
    textqdm,
)
from ollm.utils.nx_to_gt import nx_to_gt


def embed_graph(
    G: Graph,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 256,
) -> Graph:
    embedder, tokenizer = load_embedding_model(embedding_model)
    nodes_to_embed = [n for n in G.nodes if "embed" not in G.nodes[n]]
    for nodes in batch(textqdm(nodes_to_embed), batch_size=batch_size):
        titles = [G.nodes[n]["title"] for n in nodes]
        embedding = embed(titles, embedder, tokenizer)
        for n, e in zip(nodes, embedding):
            G.nodes[n]["embed"] = e
    return G


def safe_f1(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


@torch.no_grad()
def graph_precision_recall_f1(
    G1: nx.DiGraph,
    G2: nx.DiGraph,
    n_iters: int = 3,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    direction: str = "forward",
) -> tuple[float, float, float] | tuple[None, None, None]:
    if len(G1) == 0 or len(G2) == 0:
        return 0, 0, 0

    # Skip computation if too slow. Time complexity is O(n^2 m)
    n, m = min(len(G1), len(G2)), max(len(G1), len(G2))
    if (n**2 * m) > 20000**3:
        return None, None, None

    G1 = embed_graph(G1, embedding_model=embedding_model)
    G2 = embed_graph(G2, embedding_model=embedding_model)

    if direction == "forward":
        pass
    elif direction == "reverse":
        G1 = G1.reverse(copy=False)
        G2 = G2.reverse(copy=False)
    elif direction == "undirected":
        G1 = G1.to_undirected(as_view=True).to_directed(as_view=True)
        G2 = G2.to_undirected(as_view=True).to_directed(as_view=True)
    else:
        raise ValueError(f"Invalid direction {direction}")

    def nx_to_vec(G: nx.Graph, n_iters) -> torch.Tensor:
        """Compute a graph embedding of shape (n_nodes embed_dim).

        Uses a GCN with identity weights to compute the embedding.
        """

        # Delete all node and edge attributes except for the embedding
        # Otherwise PyG might complain "Not all nodes/edges contain the same attributes"
        G = G.copy()
        for _, _, d in G.edges(data=True):
            d.clear()
        for _, d in G.nodes(data=True):
            for k in list(d.keys()):
                if k != "embed":
                    del d[k]
        pyg_G = from_networkx(G, group_node_attrs=["embed"])

        embed_dim = pyg_G.x.shape[1]
        conv = SGConv(embed_dim, embed_dim, K=n_iters, bias=False).to(device)
        conv.lin.weight.data = torch.eye(embed_dim, device=conv.lin.weight.device)

        pyg_batch = Batch.from_data_list([pyg_G])
        x, edge_index = pyg_batch.x, pyg_batch.edge_index  # type: ignore
        x, edge_index = x.to(device), edge_index.to(device)
        x = conv(x, edge_index)

        return x

    # Compute embeddings
    x1 = nx_to_vec(G1, n_iters)
    x2 = nx_to_vec(G2, n_iters)

    # Cosine similarity matrix
    sim = cosine_sim(x1, x2, dim=-1).cpu().numpy()

    # soft precision, recall, f1
    row_ind, col_ind = linear_sum_assignment(sim, maximize=True)
    score = sim[row_ind, col_ind].sum().item()
    precision = score / len(G1)
    recall = score / len(G2)
    f1 = safe_f1(precision, recall)

    return precision, recall, f1


@torch.no_grad()
def fuzzy_and_continuous_precision_recall_f1(
    G1: nx.DiGraph,
    G2: nx.DiGraph,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 512,
    match_threshold: float = 0.9,
    skip_if_too_slow: bool = True,
) -> (
    tuple[float, float, float, float, float, float]
    | tuple[None, None, None, None, None, None]
):
    # Skip computation if too slow. Time complexity is O(n^2 m)
    s1 = G1.number_of_edges()
    s2 = G2.number_of_edges()
    n = min(s1, s2)
    m = max(s1, s2)
    if n == 0 or m == 0:
        return 0, 0, 0, 0, 0, 0
    if (n**2 * m) > 20000**3 and skip_if_too_slow:
        return None, None, None, None, None, None

    G1 = embed_graph(G1, embedding_model=embedding_model)
    G2 = embed_graph(G2, embedding_model=embedding_model)

    def embed_edges(G, edges):
        u_emb = torch.stack([G.nodes[u]["embed"] for u, _ in edges])
        v_emb = torch.stack([G.nodes[v]["embed"] for _, v in edges])
        return u_emb, v_emb

    def edge_sim(G1, edges1, G2, edges2):
        u1_emb, v1_emb = embed_edges(G1, edges1)
        u2_emb, v2_emb = embed_edges(G2, edges2)
        sim_u = cosine_sim(u1_emb, u2_emb, dim=-1)
        sim_v = cosine_sim(v1_emb, v2_emb, dim=-1)
        return sim_u, sim_v

    sims_u = []
    sims_v = []
    for edge_batch_1 in batch(G1.edges, batch_size):
        sims_u_row = []
        sims_v_row = []
        for edge_batch_2 in batch(G2.edges, batch_size):
            sim_u, sim_v = edge_sim(G1, edge_batch_1, G2, edge_batch_2)
            sims_u_row.append(sim_u)
            sims_v_row.append(sim_v)
        sims_u.append(torch.cat(sims_u_row, dim=-1))
        sims_v.append(torch.cat(sims_v_row, dim=-1))
    sims_u = torch.cat(sims_u, dim=0)
    sims_v = torch.cat(sims_v, dim=0)

    # Continuous precision, recall, f1
    sims = torch.minimum(sims_u, sims_v).cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(sims, maximize=True)
    score = sims[row_ind, col_ind].sum().item()
    precision_continuous = score / s1
    recall_continuous = score / s2
    f1_continuous = safe_f1(precision_continuous, recall_continuous)

    # Fuzzy precision, recall, f1
    hard_sims = (
        ((sims_u >= match_threshold) & (sims_v >= match_threshold)).cpu().numpy()
    )
    precision_fuzzy = hard_sims.any(axis=1).sum().item() / s1
    recall_fuzzy = hard_sims.any(axis=0).sum().item() / s2
    f1_fuzzy = safe_f1(precision_fuzzy, recall_fuzzy)

    return (
        precision_continuous,
        recall_continuous,
        f1_continuous,
        precision_fuzzy,
        recall_fuzzy,
        f1_fuzzy,
    )


def literal_prec_recall_f1(G_pred: nx.Graph, G_true: nx.Graph):
    if len(G_pred) == 0 or len(G_true) == 0:
        return 0, 0, 0

    def title(G, n):
        return G.nodes[n]["title"]

    edges_G = {(title(G_pred, u), title(G_pred, v)) for u, v in G_pred.edges}
    edges_G_true = {(title(G_true, u), title(G_true, v)) for u, v in G_true.edges}
    precision = len(edges_G & edges_G_true) / len(edges_G)
    recall = len(edges_G & edges_G_true) / len(edges_G_true)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1


def motif_distance(G_pred: nx.Graph, G_true: nx.Graph, n: int = 3):
    motifs_pred, counts_pred = gt.motifs(nx_to_gt(G_pred)[0], n)  # type: ignore
    motifs_true, counts_true = gt.motifs(nx_to_gt(G_true)[0], n)  # type: ignore

    all_motifs = motifs_pred[::]
    for motif in motifs_true:
        for existing_motif in all_motifs:
            if gt.isomorphism(motif, existing_motif):
                break
        else:
            all_motifs.append(motif)
    all_counts_pred = np.zeros(len(all_motifs))
    all_counts_true = np.zeros(len(all_motifs))
    for i, motif in enumerate(motifs_pred):
        for j, existing_motif in enumerate(all_motifs):
            if gt.isomorphism(motif, existing_motif):
                all_counts_pred[j] = counts_pred[i]
                break
    for i, motif in enumerate(motifs_true):
        for j, existing_motif in enumerate(all_motifs):
            if gt.isomorphism(motif, existing_motif):
                all_counts_true[j] = counts_true[i]
                break
    all_counts_pred /= all_counts_pred.sum()
    all_counts_true /= all_counts_true.sum()

    wass = np.sum(np.abs(all_counts_true - all_counts_pred)) / 2
    return wass
