import json
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd

from get_S2ORC.utils import ensure_dirs


def build_neighbors(
    edges: list[dict[str, Any]],
    direction: str = "outgoing",
    max_neighbors: int = 50,
) -> dict[str, list[str]]:
    nbrs: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        s, d = e["src_paper_id"], e["dst_paper_id"]
        if direction in ("outgoing", "both"):
            nbrs[s].append(d)
        if direction in ("incoming", "both"):
            nbrs[d].append(s)

    capped: dict[str, list[str]] = {}
    for pid, arr in nbrs.items():
        seen = []
        used = set()
        for x in arr:
            if x in used:
                continue
            used.add(x)
            seen.append(x)
            if len(seen) >= max_neighbors:
                break
        capped[pid] = seen
    return capped


def write_index(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    out_dir: str,
    index_format: str = "csr",
    direction: str = "outgoing",
    max_neighbors: int = 50,
) -> dict[str, Any]:
    ensure_dirs(out_dir)
    node_ids = [n["paper_id"] for n in nodes]
    node_set = set(node_ids)
    neighbors = build_neighbors(edges, direction=direction, max_neighbors=max_neighbors)

    for pid in node_ids:
        neighbors.setdefault(pid, [])

    degrees = [len(neighbors.get(pid, [])) for pid in node_ids]
    stats = {
        "avg_degree": float(np.mean(degrees)) if degrees else 0.0,
        "max_degree": int(np.max(degrees)) if degrees else 0,
        "isolated_nodes": int(sum(1 for d in degrees if d == 0)),
        "direction": direction,
        "max_neighbors": max_neighbors,
    }

    if index_format == "csr":
        try:
            from scipy.sparse import csr_matrix, save_npz
        except Exception:
            index_format = "parquet_neighbors"

    if index_format == "csr":
        id_map = {pid: i for i, pid in enumerate(node_ids)}
        rows = []
        cols = []
        data = []
        for src, dsts in neighbors.items():
            if src not in id_map:
                continue
            sidx = id_map[src]
            for dst in dsts:
                didx = id_map.get(dst)
                if didx is None:
                    continue
                rows.append(sidx)
                cols.append(didx)
                data.append(1)

        mat = csr_matrix((data, (rows, cols)), shape=(len(node_ids), len(node_ids)), dtype=np.int8)
        save_npz(f"{out_dir}/adj_csr.npz", mat)
        with open(f"{out_dir}/id_map.json", "w", encoding="utf-8") as f:
            json.dump(id_map, f, ensure_ascii=False, indent=2)
    elif index_format == "parquet_neighbors":
        recs = [{"paper_id": pid, "neighbors": neighbors.get(pid, [])} for pid in node_ids]
        pd.DataFrame(recs).to_parquet(f"{out_dir}/neighbors.parquet", index=False)
    else:
        raise ValueError("index_format must be csr or parquet_neighbors")

    return stats
