from collections import Counter, defaultdict
from typing import Any


def is_recent_year(year: int | None, min_year: int) -> bool:
    return year is not None and year >= min_year


def node_encodable(node: dict[str, Any], block_count: int) -> bool:
    title_ok = bool((node.get("title") or "").strip())
    abstract_ok = bool((node.get("abstract") or "").strip())
    fulltext_ok = bool(node.get("has_fulltext"))
    # Fallback: allow title-only nodes when abstracts/fulltext are unavailable in sampled shards.
    return title_ok and (abstract_ok or fulltext_ok or block_count > 0 or title_ok)


def filter_and_finalize(
    nodes_by_id: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
    license_allow: set[str],
    keep_unknown_license: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    stats: dict[str, Any] = {}
    stats["before_nodes"] = len(nodes_by_id)
    stats["before_edges"] = len(edges)
    stats["before_blocks"] = len(blocks)

    block_count = Counter([b.get("paper_id") for b in blocks if b.get("paper_id")])

    kept_nodes = {}
    license_counter = Counter()
    for pid, node in nodes_by_id.items():
        lic = (node.get("license") or "unknown").lower().strip() or "unknown"
        license_counter[lic] += 1
        if lic == "unknown" and keep_unknown_license:
            pass
        elif lic not in license_allow:
            continue
        if not node_encodable(node, block_count.get(pid, 0)):
            continue
        kept_nodes[pid] = node

    node_ids = set(kept_nodes.keys())

    kept_edges = []
    out_deg = Counter()
    in_deg = Counter()
    for e in edges:
        src = e.get("src_paper_id")
        dst = e.get("dst_paper_id")
        if not src or not dst:
            continue
        if src not in node_ids or dst not in node_ids:
            continue
        if src == dst:
            continue
        e["is_resolved"] = True
        kept_edges.append(e)
        out_deg[src] += 1
        in_deg[dst] += 1

    connected = set(out_deg.keys()) | set(in_deg.keys())
    kept_nodes = {pid: n for pid, n in kept_nodes.items() if pid in connected}
    node_ids = set(kept_nodes.keys())

    kept_edges = [e for e in kept_edges if e["src_paper_id"] in node_ids and e["dst_paper_id"] in node_ids]
    kept_blocks = [b for b in blocks if b.get("paper_id") in node_ids]

    isolates = len(node_ids - (set([e["src_paper_id"] for e in kept_edges]) | set([e["dst_paper_id"] for e in kept_edges])))

    stats["after_nodes"] = len(kept_nodes)
    stats["after_edges"] = len(kept_edges)
    stats["after_blocks"] = len(kept_blocks)
    stats["isolated_nodes"] = isolates
    stats["license_distribution"] = dict(license_counter)

    return list(kept_nodes.values()), kept_edges, kept_blocks, stats


def split_recent_priority(nodes: list[dict[str, Any]], min_year: int, target_papers: int) -> list[dict[str, Any]]:
    recent = [n for n in nodes if is_recent_year(n.get("year"), min_year)]
    older_or_unknown = [n for n in nodes if n not in recent]
    selected = recent[:target_papers]
    if len(selected) < target_papers:
        selected.extend(older_or_unknown[: target_papers - len(selected)])
    return selected
