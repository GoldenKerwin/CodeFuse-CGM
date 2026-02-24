#!/usr/bin/env python3
import argparse
import os
import random
from collections import defaultdict, deque
from datetime import UTC, datetime

import pandas as pd

from get_S2ORC.datasets_api import (
    S2DatasetsClient,
    extract_dataset_names,
    extract_file_urls,
    normalize_release_ids,
)
from get_S2ORC.downloader import download_dataset_files
from get_S2ORC.filters import filter_and_finalize, split_recent_priority
from get_S2ORC.graph_index import write_index
from get_S2ORC.parser import iter_jsonl_records, parse_record_to_rows
from get_S2ORC.utils import ensure_dirs, estimate_token_len, json_dump, json_load, now_year, setup_logger


def load_local_env(env_path: str = ".env") -> None:
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("\"").strip("'")
            if k and (k not in os.environ):
                os.environ[k] = v


def init_paths():
    ensure_dirs(
        "data/raw",
        "data/processed",
        "data/index",
        "data/meta",
        "logs",
        ".cache/api",
    )


def cmd_list(args):
    logger = setup_logger()
    init_paths()
    client = S2DatasetsClient(logger=logger)

    releases_payload = client.list_releases()
    latest, previous = normalize_release_ids(releases_payload)
    if not latest:
        raise RuntimeError("No releases found from Semantic Scholar datasets API")

    release_payload = client.get_release_datasets(latest)
    datasets = extract_dataset_names(release_payload)

    out_releases = {
        "latest": latest,
        "previous": previous,
        "fetched_at": datetime.now(UTC).isoformat(),
        "raw_count": len(releases_payload),
    }
    json_dump("data/meta/releases.json", out_releases)
    json_dump("data/meta/latest_datasets.json", {"release_id": latest, "datasets": datasets})

    logger.info("latest=%s previous=%s", latest, previous)
    logger.info("datasets(%d): %s", len(datasets), datasets)


def _select_datasets(all_datasets: list[str], explicit: str | None) -> list[str]:
    if explicit:
        return [x.strip() for x in explicit.split(",") if x.strip()]
    keywords = ["papers", "abstracts", "citation", "reference"]
    selected = [d for d in all_datasets if any(k in d.lower() for k in keywords)]
    return selected


def _maybe_get_diff_urls(client, previous: str | None, latest: str, dataset: str, logger):
    if not previous:
        return []
    try:
        payload = client.get_diff_files(previous, latest, dataset)
        urls = extract_file_urls(payload)
        if urls:
            logger.info("Using diff files for dataset=%s (%d files)", dataset, len(urls))
        return urls
    except Exception as e:
        logger.warning("Diff unavailable for dataset=%s: %s; fallback to full", dataset, e)
        return []


def cmd_download(args):
    logger = setup_logger()
    init_paths()
    client = S2DatasetsClient(logger=logger)

    releases_meta = json_load("data/meta/releases.json", default=None)
    if not releases_meta:
        cmd_list(args)
        releases_meta = json_load("data/meta/releases.json", default={})

    latest = args.release_id or releases_meta.get("latest")
    previous = releases_meta.get("previous")
    ds_meta = json_load("data/meta/latest_datasets.json", default={})
    all_datasets = ds_meta.get("datasets") or extract_dataset_names(client.get_release_datasets(latest))

    selected_datasets = _select_datasets(all_datasets, args.datasets)
    if not selected_datasets:
        raise RuntimeError("No datasets selected. Use --datasets explicitly.")

    logger.info("Selected datasets: %s", selected_datasets)

    manifest_path = "data/meta/download_manifest.json"
    for ds in selected_datasets:
        urls = []
        if args.prefer_diff:
            urls = _maybe_get_diff_urls(client, previous, latest, ds, logger)
        if not urls:
            payload = client.get_dataset_files(latest, ds)
            urls = extract_file_urls(payload)
        if not urls:
            logger.warning("No file URLs found for dataset=%s", ds)
            continue
        download_dataset_files(
            dataset_name=ds,
            urls=urls,
            raw_dir="data/raw",
            max_files=args.max_files,
            manifest_path=manifest_path,
            logger=logger,
        )


def _load_subgraph_config(config_path: str | None) -> dict:
    defaults = {
        "num_subgraphs": 100,
        "min_nodes_per_graph": 80,
        "max_nodes_per_graph": 300,
        "min_hops": 1,
        "max_hops": 2,
        "max_attempts_per_graph": 30,
        "candidate_multiplier": 2.0,
        "max_total_attempts_factor": 4.0,
        "enrich_node_cap": 50000,
        "force_fill_to_target": True,
        "topup_rounds": 5,
        "topup_attempts_per_missing": 20,
        "allow_bootstrap_repeats": False,
        "cleanup_scanned_raw": True,
        "round_fetch_until_target": False,
        "round_files_start": 1,
        "round_files_step": 1,
        "enable_graph_api_fallback": False,
        "random_seed": 42,
        "graph_file_prefix": "citation_subgraph",
    }
    if not config_path:
        return defaults
    data = json_load(config_path, default={}) or {}
    cfg = {**defaults, **data}
    cfg["num_subgraphs"] = max(1, int(cfg["num_subgraphs"]))
    cfg["min_nodes_per_graph"] = max(2, int(cfg["min_nodes_per_graph"]))
    cfg["max_nodes_per_graph"] = max(cfg["min_nodes_per_graph"], int(cfg["max_nodes_per_graph"]))
    cfg["min_hops"] = max(1, int(cfg["min_hops"]))
    cfg["max_hops"] = max(cfg["min_hops"], int(cfg["max_hops"]))
    cfg["max_attempts_per_graph"] = max(1, int(cfg["max_attempts_per_graph"]))
    cfg["candidate_multiplier"] = max(1.0, float(cfg.get("candidate_multiplier", 2.0)))
    cfg["max_total_attempts_factor"] = max(1.0, float(cfg.get("max_total_attempts_factor", 4.0)))
    cfg["enrich_node_cap"] = max(1000, int(cfg.get("enrich_node_cap", 50000)))
    cfg["force_fill_to_target"] = bool(cfg.get("force_fill_to_target", True))
    cfg["topup_rounds"] = max(1, int(cfg.get("topup_rounds", 5)))
    cfg["topup_attempts_per_missing"] = max(1, int(cfg.get("topup_attempts_per_missing", 20)))
    cfg["allow_bootstrap_repeats"] = bool(cfg.get("allow_bootstrap_repeats", False))
    cfg["cleanup_scanned_raw"] = bool(cfg.get("cleanup_scanned_raw", True))
    cfg["round_fetch_until_target"] = bool(cfg.get("round_fetch_until_target", False))
    cfg["round_files_start"] = max(1, int(cfg.get("round_files_start", 1)))
    cfg["round_files_step"] = max(1, int(cfg.get("round_files_step", 1)))
    cfg["enable_graph_api_fallback"] = bool(cfg.get("enable_graph_api_fallback", False))
    cfg["random_seed"] = int(cfg["random_seed"])
    cfg["graph_file_prefix"] = str(cfg["graph_file_prefix"])
    return cfg


def _graph_batch_enrich_title_abstract(corpus_ids: list[str], logger) -> dict[str, dict]:
    if not corpus_ids:
        return {}
    client = S2DatasetsClient(logger=logger)
    out = {}
    batch_size = 100
    fields = "corpusId,title,abstract,year,venue,externalIds"
    for i in range(0, len(corpus_ids), batch_size):
        batch = corpus_ids[i : i + batch_size]
        ids = [f"CorpusId:{cid}" for cid in batch]
        try:
            resp = client.graph_paper_batch(ids=ids, fields=fields, use_cache=True)
        except Exception as e:
            logger.warning("Graph API batch enrich failed on batch %d-%d: %s", i, i + len(batch), e)
            continue
        items = resp if isinstance(resp, list) else resp.get("data", []) if isinstance(resp, dict) else []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            cid = item.get("corpusId") or item.get("corpusid")
            if cid is None:
                continue
            title = (item.get("title") or "").strip()
            abstract = (item.get("abstract") or "").strip()
            if not title or not abstract:
                continue
            out[f"corpus:{cid}"] = {
                "paper_id": f"corpus:{cid}",
                "title": title,
                "abstract": abstract,
                "year": item.get("year"),
                "venue": item.get("venue") or "",
                "fields_of_study": [],
                "doi": ((item.get("externalIds") or {}).get("DOI") if isinstance(item.get("externalIds"), dict) else None),
                "arxiv_id": ((item.get("externalIds") or {}).get("ArXiv") if isinstance(item.get("externalIds"), dict) else None),
                "pmid": ((item.get("externalIds") or {}).get("PubMed") if isinstance(item.get("externalIds"), dict) else None),
                "has_fulltext": False,
                "license": "unknown",
            }
    logger.info("Graph API enriched title+abstract nodes: %d / requested %d", len(out), len(corpus_ids))
    return out


def _extract_shard_tag_from_url(url: str) -> str:
    """
    Try extracting shard marker like "__000__" -> "000" from dataset URL basename.
    """
    name = os.path.basename((url or "").split("?", 1)[0])
    parts = name.split("__")
    if len(parts) >= 3 and parts[1].isdigit():
        return parts[1]
    return ""


def _index_urls_by_shard(urls: list[str]) -> dict[str, str]:
    out = {}
    for u in urls:
        tag = _extract_shard_tag_from_url(u)
        if tag and tag not in out:
            out[tag] = u
    return out


def _local_enrich_title_abstract_from_raw(
    target_paper_ids: list[str],
    raw_files: list[str],
    logger,
) -> dict[str, dict]:
    """
    Fill title+abstract directly from local downloaded abstracts/papers shards.
    This avoids Graph API bottleneck and keeps same-release shard consistency.
    """
    if not target_paper_ids:
        return {}
    targets = set(target_paper_ids)
    out: dict[str, dict] = {}

    # Prefer abstracts first (often denser on abstract), then papers for title/year/backfill.
    abstract_files = sorted([p for p in raw_files if "abstracts__" in os.path.basename(p).lower()])
    papers_files = sorted([p for p in raw_files if "papers__" in os.path.basename(p).lower()])
    ordered = abstract_files + papers_files
    if not ordered:
        return {}

    for path in ordered:
        logger.info("Local enrich scan: %s", path)
        for rec in iter_jsonl_records(path, logger=logger):
            node, _, _ = parse_record_to_rows(rec, logger=logger)
            pid = node.get("paper_id")
            if not pid or pid not in targets:
                continue
            old = out.get(pid)
            if old is None:
                old = {
                    "paper_id": pid,
                    "title": "",
                    "abstract": "",
                    "year": node.get("year"),
                    "venue": node.get("venue") or "",
                    "fields_of_study": node.get("fields_of_study") or [],
                    "doi": node.get("doi"),
                    "arxiv_id": node.get("arxiv_id"),
                    "pmid": node.get("pmid"),
                    "has_fulltext": bool(node.get("has_fulltext")),
                    "license": node.get("license") or "unknown",
                }
                out[pid] = old
            title = (node.get("title") or "").strip()
            abstract = (node.get("abstract") or "").strip()
            if (not old.get("title")) and title:
                old["title"] = title
            if (not old.get("abstract")) and abstract:
                old["abstract"] = abstract
            if old.get("year") is None and node.get("year") is not None:
                old["year"] = node.get("year")
            if (not old.get("venue")) and node.get("venue"):
                old["venue"] = node.get("venue")
            if (not old.get("fields_of_study")) and node.get("fields_of_study"):
                old["fields_of_study"] = node.get("fields_of_study") or []
            if (not old.get("doi")) and node.get("doi"):
                old["doi"] = node.get("doi")
            if (not old.get("arxiv_id")) and node.get("arxiv_id"):
                old["arxiv_id"] = node.get("arxiv_id")
            if (not old.get("pmid")) and node.get("pmid"):
                old["pmid"] = node.get("pmid")
            old["has_fulltext"] = bool(old.get("has_fulltext")) or bool(node.get("has_fulltext"))
            if old.get("license") in ("", "unknown") and node.get("license"):
                old["license"] = node.get("license")

        # Early stop when all targets got both title and abstract.
        done = 0
        for pid in targets:
            m = out.get(pid)
            if m and (m.get("title") or "").strip() and (m.get("abstract") or "").strip():
                done += 1
        if done == len(targets):
            break

    final = {
        pid: m
        for pid, m in out.items()
        if (m.get("title") or "").strip() and (m.get("abstract") or "").strip()
    }
    logger.info("Local raw enriched title+abstract nodes: %d / requested %d", len(final), len(target_paper_ids))
    return final


def _extract_citation_graph_from_raw(
    citation_files: list[str],
    logger,
    max_edges_scan: int,
    cleanup_scanned_raw: bool = False,
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]], int]:
    out_adj: dict[str, set[str]] = defaultdict(set)
    in_adj: dict[str, set[str]] = defaultdict(set)
    undirected_adj: dict[str, set[str]] = defaultdict(set)
    scanned_edges = 0
    scanned_rows = 0
    next_progress = 50000
    for path in citation_files:
        logger.info("Scanning citation file for structure: %s", path)
        for rec in iter_jsonl_records(path, logger=logger):
            scanned_rows += 1
            # Fast path for citations dataset rows.
            if isinstance(rec, dict):
                lk = {str(k).lower(): v for k, v in rec.items()}
                citing = (
                    lk.get("citingcorpusid")
                    or lk.get("citingcorpusId")
                    or lk.get("sourcecorpusid")
                    or lk.get("source")
                )
                cited = (
                    lk.get("citedcorpusid")
                    or lk.get("citedcorpusId")
                    or lk.get("targetcorpusid")
                    or lk.get("target")
                )
                if citing is None or cited is None:
                    citing = None
                    cited = None
            else:
                citing = None
                cited = None

            if citing is None or cited is None:
                try:
                    _, row_edges, _ = parse_record_to_rows(rec, logger=None)
                except Exception:
                    row_edges = []
                for e in row_edges:
                    s = e.get("src_paper_id")
                    d = e.get("dst_paper_id")
                    if not s or not d or s == d:
                        continue
                    if not (str(s).startswith("corpus:") and str(d).startswith("corpus:")):
                        continue
                    if d in out_adj[s]:
                        continue
                    out_adj[s].add(d)
                    in_adj[d].add(s)
                    undirected_adj[s].add(d)
                    undirected_adj[d].add(s)
                    scanned_edges += 1
            else:
                s = f"corpus:{citing}"
                d = f"corpus:{cited}"
                if s != d and d not in out_adj[s]:
                    out_adj[s].add(d)
                    in_adj[d].add(s)
                    undirected_adj[s].add(d)
                    undirected_adj[d].add(s)
                    scanned_edges += 1

            if scanned_edges >= next_progress:
                logger.info(
                    "Citation scan progress | rows=%d unique_edges=%d unique_nodes=%d",
                    scanned_rows,
                    scanned_edges,
                    len(set(out_adj.keys()) | set(in_adj.keys())),
                )
                next_progress += 50000
            if scanned_edges >= max_edges_scan:
                logger.info("Reached citation edge scan cap=%d (rows=%d)", max_edges_scan, scanned_rows)
                if cleanup_scanned_raw:
                    try:
                        os.remove(path)
                        logger.info("Removed scanned raw shard: %s", path)
                    except OSError as e:
                        logger.warning("Failed to remove scanned raw shard %s: %s", path, e)
                return out_adj, in_adj, undirected_adj, scanned_edges
        if cleanup_scanned_raw:
            try:
                os.remove(path)
                logger.info("Removed scanned raw shard: %s", path)
            except OSError as e:
                logger.warning("Failed to remove scanned raw shard %s: %s", path, e)
    return out_adj, in_adj, undirected_adj, scanned_edges


def _actual_hop_within_subgraph(seed: str, nodes_set: set[str], undirected_adj: dict[str, set[str]]) -> int:
    if seed not in nodes_set:
        return 0
    q = deque([(seed, 0)])
    seen = {seed}
    max_depth = 0
    while q:
        cur, depth = q.popleft()
        max_depth = max(max_depth, depth)
        for nb in undirected_adj.get(cur, set()):
            if nb not in nodes_set or nb in seen:
                continue
            seen.add(nb)
            q.append((nb, depth + 1))
    return max_depth


def _largest_weakly_connected_nodes(nodes: list[str], undirected_adj: dict[str, set[str]]) -> list[str]:
    node_set = set(nodes)
    seen = set()
    best: list[str] = []
    for n in nodes:
        if n in seen:
            continue
        comp = []
        q = deque([n])
        seen.add(n)
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nb in undirected_adj.get(cur, set()):
                if nb in node_set and nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        if len(comp) > len(best):
            best = comp
    best_set = set(best)
    return [n for n in nodes if n in best_set]


def _build_subgraphs_via_graph_api(args, logger, min_year: int) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    cfg = _load_subgraph_config(args.subgraph_config)
    raw_files = [
        os.path.join(args.raw_dir, n)
        for n in os.listdir(args.raw_dir)
        if n.endswith(".gz") or n.endswith(".jsonl") or n.endswith(".jsonl.gz")
    ]
    citation_files = sorted([p for p in raw_files if "citation" in os.path.basename(p).lower()])
    if not citation_files:
        raise RuntimeError("未找到 citations 原始分片，请先 download --datasets citations")

    explicit_cap = int(getattr(args, "max_citation_edges_scan", 0) or 0)
    if explicit_cap > 0:
        max_edges_scan = explicit_cap
    else:
        auto_cap = max(cfg["num_subgraphs"] * cfg["max_nodes_per_graph"] * 80, 120000)
        max_edges_scan = min(auto_cap, 300000)
    out_adj, in_adj, undirected_adj, scanned_edges = _extract_citation_graph_from_raw(
        citation_files=citation_files,
        logger=logger,
        max_edges_scan=max_edges_scan,
        cleanup_scanned_raw=bool(cfg.get("cleanup_scanned_raw", True)),
    )
    all_nodes = list(set(out_adj.keys()) | set(in_adj.keys()))
    seeds = [n for n in all_nodes if undirected_adj.get(n)]
    if not seeds:
        raise RuntimeError("citations 分片中未构建出可用图结构")
    logger.info(
        "Citation structure built | nodes=%d edges=%d seeds=%d",
        len(all_nodes),
        scanned_edges,
        len(seeds),
    )
    # Limit sampling/enrichment to a high-connectivity node pool so Graph API enrichment stays tractable at 1 rps.
    node_cap = int(cfg.get("enrich_node_cap", 50000))
    if len(all_nodes) > node_cap:
        degree_sorted = sorted(
            all_nodes,
            key=lambda n: len(out_adj.get(n, set())) + len(in_adj.get(n, set())),
            reverse=True,
        )
        node_pool = set(degree_sorted[:node_cap])
        out_adj = {k: {d for d in v if d in node_pool} for k, v in out_adj.items() if k in node_pool}
        in_adj = {k: {s for s in v if s in node_pool} for k, v in in_adj.items() if k in node_pool}
        undirected_adj = {
            k: {x for x in v if x in node_pool}
            for k, v in undirected_adj.items()
            if k in node_pool
        }
        all_nodes = list(node_pool)
        seeds = [n for n in all_nodes if undirected_adj.get(n)]
        logger.info(
            "Applied enrich_node_cap=%d | pooled_nodes=%d pooled_seeds=%d",
            node_cap,
            len(all_nodes),
            len(seeds),
        )
        if not seeds:
            raise RuntimeError("enrich_node_cap 剪裁后无可用 seeds，请增大 enrich_node_cap")

    rng = random.Random(cfg["random_seed"])
    target_candidates = max(int(cfg["num_subgraphs"] * cfg.get("candidate_multiplier", 2.0)), 1000)
    candidate_graphs: list[dict] = []
    seen_signatures = set()
    total_attempts = 0
    max_total_attempts = int(cfg["num_subgraphs"] * cfg["max_attempts_per_graph"] * cfg.get("max_total_attempts_factor", 4.0))

    while len(candidate_graphs) < target_candidates and total_attempts < max_total_attempts:
        total_attempts += 1
        seed = rng.choice(seeds)
        expanded, requested_hop = _expand_subgraph_nodes(
            seed=seed,
            undirected_adj=undirected_adj,
            min_hops=cfg["min_hops"],
            max_hops=cfg["max_hops"],
            rng=rng,
        )
        if len(expanded) < cfg["min_nodes_per_graph"]:
            continue
        if len(expanded) > cfg["max_nodes_per_graph"]:
            expanded = expanded[: cfg["max_nodes_per_graph"]]
        expanded = _largest_weakly_connected_nodes(expanded, undirected_adj)
        if len(expanded) < cfg["min_nodes_per_graph"]:
            continue
        expanded_set = set(expanded)
        induced_edges = []
        for s in expanded:
            for d in out_adj.get(s, set()):
                if d in expanded_set and s != d:
                    induced_edges.append((s, d))
        if not induced_edges:
            continue
        sig = tuple(sorted(expanded_set))
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)
        candidate_graphs.append(
            {
                "seed": seed,
                "requested_hop": int(requested_hop),
                "node_ids": expanded,
                "edges": induced_edges,
            }
        )
        if total_attempts % 50000 == 0:
            logger.info(
                "Candidate sampling progress | attempts=%d/%d candidates=%d/%d",
                total_attempts,
                max_total_attempts,
                len(candidate_graphs),
                target_candidates,
            )

    logger.info(
        "Candidate subgraphs sampled | candidates=%d attempts=%d target=%d",
        len(candidate_graphs),
        total_attempts,
        target_candidates,
    )
    if not candidate_graphs:
        raise RuntimeError("未采样到候选子图，请增加 citations 分片数或放宽子图参数")

    unique_corpus_ids = []
    seen_cids = set()
    for g in candidate_graphs:
        for pid in g["node_ids"]:
            cid = pid.split(":", 1)[1] if ":" in pid else pid
            if cid in seen_cids:
                continue
            seen_cids.add(cid)
            unique_corpus_ids.append(cid)

    # Prefer local raw enrichment (abstracts/papers shards) to avoid Graph API bottleneck.
    raw_files_for_enrich = [
        os.path.join(args.raw_dir, n)
        for n in os.listdir(args.raw_dir)
        if n.endswith(".gz") or n.endswith(".jsonl") or n.endswith(".jsonl.gz")
    ]
    target_pids = [f"corpus:{cid}" for cid in unique_corpus_ids]
    enriched = _local_enrich_title_abstract_from_raw(target_pids, raw_files_for_enrich, logger=logger)
    if not enriched and cfg.get("enable_graph_api_fallback", False):
        logger.warning("Local raw enrichment got 0 nodes; fallback to Graph API batch enrich.")
        enriched = _graph_batch_enrich_title_abstract(unique_corpus_ids, logger=logger)
    if not enriched:
        raise RuntimeError(
            "未从本地 raw 分片补齐到任何 title+abstract 节点。请下载对齐的 abstracts/papers 分片，"
            "或在配置启用 enable_graph_api_fallback。"
        )

    selected_graphs: list[dict] = []
    used_graph_node_ids: set[str] = set()
    subgraph_stats: list[dict] = []
    for idx, cand in enumerate(candidate_graphs):
        if len(selected_graphs) >= cfg["num_subgraphs"]:
            break
        filtered_nodes = [nid for nid in cand["node_ids"] if nid in enriched]
        if len(filtered_nodes) > cfg["max_nodes_per_graph"]:
            filtered_nodes = filtered_nodes[: cfg["max_nodes_per_graph"]]
        filtered_nodes = _largest_weakly_connected_nodes(filtered_nodes, undirected_adj) if filtered_nodes else []
        if len(filtered_nodes) < cfg["min_nodes_per_graph"]:
            continue
        filtered_set = set(filtered_nodes)
        induced_edges = [(s, d) for (s, d) in cand["edges"] if s in filtered_set and d in filtered_set and s != d]
        if not induced_edges:
            continue
        actual_hop = _actual_hop_within_subgraph(cand["seed"], filtered_set, undirected_adj)
        if not (cfg["min_hops"] <= actual_hop <= cfg["max_hops"]):
            continue

        node_token_lens = []
        graph_nodes = []
        for nid in filtered_nodes:
            meta = enriched[nid]
            text = f"title: {meta['title']}\nabstract: {meta['abstract']}"
            tlen = estimate_token_len(text)
            node_token_lens.append(tlen)
            graph_nodes.append({"id": nid, "nodeType": "TextFile", "name": "paper", "text": text})
            used_graph_node_ids.add(nid)

        graph_edges = [{"source": s, "target": d} for (s, d) in induced_edges]
        gname = f"{cfg['graph_file_prefix']}_{len(selected_graphs):03d}.json"
        seed_title = enriched.get(cand["seed"], {}).get("title") or enriched[filtered_nodes[0]]["title"]
        selected_graphs.append(
            {
                "graph_file": gname,
                "seed_node": cand["seed"],
                "used_hop": int(actual_hop),
                "requested_hop": int(cand["requested_hop"]),
                "nodes": graph_nodes,
                "edges": graph_edges,
                "seed_title": seed_title,
                "graph_text_token_len": int(sum(node_token_lens)),
                "node_token_lens": node_token_lens,
            }
        )
        subgraph_stats.append(
            {
                "graph_file": gname,
                "seed_node": cand["seed"],
                "requested_hop": int(cand["requested_hop"]),
                "used_hop": int(actual_hop),
                "node_count": len(graph_nodes),
                "edge_count": len(graph_edges),
                "graph_text_token_len": int(sum(node_token_lens)),
                "node_token_mean": float(sum(node_token_lens) / len(node_token_lens)) if node_token_lens else 0.0,
                "node_token_min": int(min(node_token_lens)) if node_token_lens else 0,
                "node_token_max": int(max(node_token_lens)) if node_token_lens else 0,
            }
        )

    target_subgraphs = int(cfg["num_subgraphs"])
    if cfg.get("force_fill_to_target", True) and len(selected_graphs) < target_subgraphs:
        logger.warning(
            "Selected subgraphs < target (%d/%d). Start top-up rounds.",
            len(selected_graphs),
            target_subgraphs,
        )
        topup_seeds = [nid for nid in enriched.keys() if undirected_adj.get(nid)]
        for ridx in range(int(cfg.get("topup_rounds", 5))):
            if len(selected_graphs) >= target_subgraphs:
                break
            missing = target_subgraphs - len(selected_graphs)
            attempts = max(10000, missing * int(cfg.get("topup_attempts_per_missing", 20)))
            added = 0
            for _ in range(attempts):
                if len(selected_graphs) >= target_subgraphs:
                    break
                if not topup_seeds:
                    break
                seed = rng.choice(topup_seeds)
                expanded, requested_hop = _expand_subgraph_nodes(
                    seed=seed,
                    undirected_adj=undirected_adj,
                    min_hops=cfg["min_hops"],
                    max_hops=cfg["max_hops"],
                    rng=rng,
                )
                filtered_nodes = [nid for nid in expanded if nid in enriched]
                if len(filtered_nodes) > cfg["max_nodes_per_graph"]:
                    filtered_nodes = filtered_nodes[: cfg["max_nodes_per_graph"]]
                filtered_nodes = _largest_weakly_connected_nodes(filtered_nodes, undirected_adj) if filtered_nodes else []
                if len(filtered_nodes) < cfg["min_nodes_per_graph"]:
                    continue
                filtered_set = set(filtered_nodes)
                sig = tuple(sorted(filtered_set))
                if sig in seen_signatures:
                    continue
                induced_edges = []
                for s in filtered_nodes:
                    for d in out_adj.get(s, set()):
                        if d in filtered_set and s != d:
                            induced_edges.append((s, d))
                if not induced_edges:
                    continue
                actual_hop = _actual_hop_within_subgraph(seed, filtered_set, undirected_adj)
                if not (cfg["min_hops"] <= actual_hop <= cfg["max_hops"]):
                    continue
                seen_signatures.add(sig)

                node_token_lens = []
                graph_nodes = []
                for nid in filtered_nodes:
                    meta = enriched[nid]
                    text = f"title: {meta['title']}\nabstract: {meta['abstract']}"
                    tlen = estimate_token_len(text)
                    node_token_lens.append(tlen)
                    graph_nodes.append({"id": nid, "nodeType": "TextFile", "name": "paper", "text": text})
                    used_graph_node_ids.add(nid)

                graph_edges = [{"source": s, "target": d} for (s, d) in induced_edges]
                gname = f"{cfg['graph_file_prefix']}_{len(selected_graphs):03d}.json"
                seed_title = enriched.get(seed, {}).get("title") or enriched[filtered_nodes[0]]["title"]
                selected_graphs.append(
                    {
                        "graph_file": gname,
                        "seed_node": seed,
                        "used_hop": int(actual_hop),
                        "requested_hop": int(requested_hop),
                        "nodes": graph_nodes,
                        "edges": graph_edges,
                        "seed_title": seed_title,
                        "graph_text_token_len": int(sum(node_token_lens)),
                        "node_token_lens": node_token_lens,
                    }
                )
                subgraph_stats.append(
                    {
                        "graph_file": gname,
                        "seed_node": seed,
                        "requested_hop": int(requested_hop),
                        "used_hop": int(actual_hop),
                        "node_count": len(graph_nodes),
                        "edge_count": len(graph_edges),
                        "graph_text_token_len": int(sum(node_token_lens)),
                        "node_token_mean": float(sum(node_token_lens) / len(node_token_lens)) if node_token_lens else 0.0,
                        "node_token_min": int(min(node_token_lens)) if node_token_lens else 0,
                        "node_token_max": int(max(node_token_lens)) if node_token_lens else 0,
                    }
                )
                added += 1
            logger.info(
                "Top-up round %d finished | added=%d total=%d target=%d attempts=%d",
                ridx + 1,
                added,
                len(selected_graphs),
                target_subgraphs,
                attempts,
            )
            if added == 0:
                logger.info("Top-up round %d had no gain; stop top-up early.", ridx + 1)
                break

        if len(selected_graphs) < target_subgraphs and cfg.get("allow_bootstrap_repeats", True) and selected_graphs:
            logger.warning(
                "Top-up rounds still insufficient (%d/%d). Bootstrap repeating existing valid subgraphs.",
                len(selected_graphs),
                target_subgraphs,
            )
            base_graphs = list(selected_graphs)
            bidx = 0
            while len(selected_graphs) < target_subgraphs:
                base = base_graphs[bidx % len(base_graphs)]
                gname = f"{cfg['graph_file_prefix']}_{len(selected_graphs):03d}.json"
                selected_graphs.append(
                    {
                        "graph_file": gname,
                        "seed_node": base.get("seed_node"),
                        "used_hop": int(base.get("used_hop", 0)),
                        "requested_hop": int(base.get("requested_hop", 0)),
                        "nodes": base["nodes"],
                        "edges": base["edges"],
                        "seed_title": base.get("seed_title", ""),
                        "graph_text_token_len": int(base.get("graph_text_token_len", 0)),
                        "node_token_lens": list(base.get("node_token_lens", [])),
                    }
                )
                lns = list(base.get("node_token_lens", []))
                subgraph_stats.append(
                    {
                        "graph_file": gname,
                        "seed_node": base.get("seed_node"),
                        "requested_hop": int(base.get("requested_hop", 0)),
                        "used_hop": int(base.get("used_hop", 0)),
                        "node_count": len(base["nodes"]),
                        "edge_count": len(base["edges"]),
                        "graph_text_token_len": int(base.get("graph_text_token_len", 0)),
                        "node_token_mean": float(sum(lns) / len(lns)) if lns else 0.0,
                        "node_token_min": int(min(lns)) if lns else 0,
                        "node_token_max": int(max(lns)) if lns else 0,
                    }
                )
                bidx += 1
            logger.info("Bootstrap fill complete | total=%d target=%d", len(selected_graphs), target_subgraphs)

    if not selected_graphs:
        raise RuntimeError(
            "候选子图在 title+abstract 严格过滤后全部失效。请增加 citations 覆盖或降低 hop/节点范围。"
        )

    final_node_ids = set()
    final_edge_keys = set()
    final_edges: list[dict] = []
    for g in selected_graphs:
        for n in g["nodes"]:
            final_node_ids.add(n["id"])
        for e in g["edges"]:
            ek = (e["source"], e["target"])
            if ek in final_edge_keys:
                continue
            final_edge_keys.add(ek)
            final_edges.append(
                {
                    "src_paper_id": e["source"],
                    "dst_paper_id": e["target"],
                    "is_resolved": True,
                    "context": "",
                }
            )

    final_nodes = []
    final_blocks = []
    for pid in sorted(final_node_ids):
        meta = enriched[pid]
        if not (meta.get("title") and meta.get("abstract")):
            continue
        final_nodes.append(meta)
        final_blocks.append(
            {
                "paper_id": pid,
                "view_type": "abstract",
                "text": meta["abstract"],
                "token_len": estimate_token_len(meta["abstract"]),
                "section_path": "abstract",
            }
        )

    stats = {
        "construction_mode": "graph_api_strict",
        "scanned_citation_edges": scanned_edges,
        "citation_nodes": len(all_nodes),
        "candidate_subgraphs": len(candidate_graphs),
        "selected_subgraphs": len(selected_graphs),
        "final_nodes": len(final_nodes),
        "final_edges": len(final_edges),
        "final_blocks": len(final_blocks),
        "min_year_requested": min_year,
        "note": "节点严格要求 title+abstract；结构来自 citations，文本通过 Graph API batch 补齐",
    }
    meta = {"subgraph_config": cfg, "subgraph_stats_detail": subgraph_stats}
    return final_nodes, final_edges, final_blocks, selected_graphs, {"stats": stats, "meta": meta}


def _expand_subgraph_nodes(
    seed: str,
    undirected_adj: dict[str, set[str]],
    min_hops: int,
    max_hops: int,
    rng: random.Random,
) -> tuple[list[str], int]:
    hop = rng.randint(min_hops, max_hops)
    visited = {seed}
    order = [seed]
    q = deque([(seed, 0)])
    while q:
        node, depth = q.popleft()
        if depth >= hop:
            continue
        nbrs = list(undirected_adj.get(node, set()))
        rng.shuffle(nbrs)
        for nb in nbrs:
            if nb in visited:
                continue
            visited.add(nb)
            order.append(nb)
            q.append((nb, depth + 1))
    return order, hop


def _export_cgm(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
    valid_ratio: float = 0.05,
    graph_filename: str = "citation_graph.json",
    subgraph_config_path: str | None = None,
):
    import json

    graph_dir = "data/processed/cgm_graphs"
    split_dir = "data/processed/cgm_splits"
    ensure_dirs(graph_dir, split_dir)

    if nodes_df.empty or "paper_id" not in nodes_df.columns:
        return

    node_meta = {}
    for row in nodes_df.itertuples(index=False):
        title = (row.title or "").strip()
        abstract = (row.abstract or "").strip()
        if not title or not abstract:
            continue
        node_meta[row.paper_id] = {"title": title, "abstract": abstract}
    all_node_ids = set(node_meta.keys())

    out_adj: dict[str, set[str]] = defaultdict(set)
    in_adj: dict[str, set[str]] = defaultdict(set)
    directed_edges: list[tuple[str, str]] = []
    if (not edges_df.empty) and ("src_paper_id" in edges_df.columns):
        for e in edges_df.itertuples(index=False):
            src = e.src_paper_id
            dst = e.dst_paper_id
            if src in all_node_ids and dst in all_node_ids and src != dst:
                out_adj[src].add(dst)
                in_adj[dst].add(src)
                directed_edges.append((src, dst))

    # keep natural directed edges; undirected adjacency is only for neighborhood expansion.
    undirected_adj: dict[str, set[str]] = defaultdict(set)
    for s, d in directed_edges:
        undirected_adj[s].add(d)
        undirected_adj[d].add(s)

    cfg = _load_subgraph_config(subgraph_config_path)
    rng = random.Random(cfg["random_seed"])
    seeds = [nid for nid in all_node_ids if undirected_adj.get(nid)]
    if not seeds:
        seeds = list(all_node_ids)

    samples = []
    generated = 0
    subgraph_stats = []
    for i in range(cfg["num_subgraphs"]):
        chosen_nodes = None
        used_hop = None
        for _ in range(cfg["max_attempts_per_graph"]):
            seed = rng.choice(seeds)
            expanded, hop = _expand_subgraph_nodes(
                seed=seed,
                undirected_adj=undirected_adj,
                min_hops=cfg["min_hops"],
                max_hops=cfg["max_hops"],
                rng=rng,
            )
            if len(expanded) < cfg["min_nodes_per_graph"]:
                continue
            if len(expanded) > cfg["max_nodes_per_graph"]:
                expanded = expanded[: cfg["max_nodes_per_graph"]]
            chosen_nodes = expanded
            used_hop = hop
            break

        if not chosen_nodes:
            continue

        chosen_set = set(chosen_nodes)
        graph_nodes = []
        node_token_lens = []
        for nid in chosen_nodes:
            meta = node_meta[nid]
            txt = f"title: {meta['title']}\nabstract: {meta['abstract']}"
            tlen = estimate_token_len(txt)
            node_token_lens.append(tlen)
            graph_nodes.append({"id": nid, "nodeType": "TextFile", "name": "paper", "text": txt})

        graph_edges = [{"source": s, "target": d} for s, d in directed_edges if s in chosen_set and d in chosen_set]
        if not graph_edges:
            continue

        graph_text_token_len = int(sum(node_token_lens))

        gname = f"{cfg['graph_file_prefix']}_{i:03d}.json"
        gpath = os.path.abspath(os.path.join(graph_dir, gname))
        with open(gpath, "w", encoding="utf-8") as f:
            json.dump({"nodes": graph_nodes, "edges": graph_edges}, f, ensure_ascii=False)

        seed_title = node_meta[chosen_nodes[0]]["title"]
        prompt = f"请基于该引文子图生成结构化综述，重点论文：{seed_title or chosen_nodes[0]}。"
        answer = "该样本为预训练占位答案，请在后训练阶段替换为高质量监督综述。"
        samples.append(
            {
                "graph": gpath,
                "prompt": prompt,
                "answer": answer,
            }
        )
        subgraph_stats.append(
            {
                "graph_file": gname,
                "seed_node": chosen_nodes[0],
                "used_hop": int(used_hop if used_hop is not None else 0),
                "node_count": len(graph_nodes),
                "edge_count": len(graph_edges),
                "graph_text_token_len": graph_text_token_len,
                "node_token_mean": float(sum(node_token_lens) / len(node_token_lens)) if node_token_lens else 0.0,
                "node_token_min": int(min(node_token_lens)) if node_token_lens else 0,
                "node_token_max": int(max(node_token_lens)) if node_token_lens else 0,
            }
        )
        generated += 1

    random.shuffle(samples)
    n_valid = max(1, int(len(samples) * valid_ratio)) if samples else 0
    valid = samples[:n_valid]
    train = samples[n_valid:]

    def write_jsonl(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(os.path.join(split_dir, "train.jsonl"), train)
    write_jsonl(os.path.join(split_dir, "valid.jsonl"), valid)
    def _summary(vals: list[int]) -> dict:
        if not vals:
            return {"mean": 0.0, "min": 0, "max": 0}
        return {"mean": float(sum(vals) / len(vals)), "min": int(min(vals)), "max": int(max(vals))}

    node_vals = [x["node_count"] for x in subgraph_stats]
    edge_vals = [x["edge_count"] for x in subgraph_stats]
    hop_vals = [x["used_hop"] for x in subgraph_stats]
    graph_text_token_vals = [x["graph_text_token_len"] for x in subgraph_stats]
    per_graph_node_token_mean_vals = [x["node_token_mean"] for x in subgraph_stats]
    node_token_min_vals = [x["node_token_min"] for x in subgraph_stats]
    node_token_max_vals = [x["node_token_max"] for x in subgraph_stats]

    def _summary_float(vals: list[float]) -> dict:
        if not vals:
            return {"mean": 0.0, "min": 0.0, "max": 0.0}
        return {"mean": float(sum(vals) / len(vals)), "min": float(min(vals)), "max": float(max(vals))}

    summary = {
        "subgraph_count": generated,
        "node_count": _summary(node_vals),
        "edge_count": _summary(edge_vals),
        "hop_count": _summary(hop_vals),
        "graph_text_token_len": _summary(graph_text_token_vals),
        "per_graph_node_token_mean": _summary_float(per_graph_node_token_mean_vals),
        "per_graph_node_token_min": _summary(node_token_min_vals),
        "per_graph_node_token_max": _summary(node_token_max_vals),
    }

    meta = {
        "generated_subgraphs": generated,
        "train_samples": len(train),
        "valid_samples": len(valid),
        "subgraph_config": cfg,
        "subgraph_stats_summary": summary,
        "subgraph_stats_detail": subgraph_stats,
    }
    json_dump(os.path.join(split_dir, "subgraph_export_meta.json"), meta)
    return meta


def _export_cgm_prebuilt(
    subgraphs: list[dict],
    valid_ratio: float = 0.05,
    encoder_chunk_tokenizer_path: str = "/root/autodl-tmp/CodeFuse-CGM/encoder/specter2_base",
    append_mode: bool = False,
):
    import json

    graph_dir = "data/processed/cgm_graphs"
    split_dir = "data/processed/cgm_splits"
    ensure_dirs(graph_dir, split_dir)
    if not append_mode:
        for name in os.listdir(graph_dir):
            if name.endswith(".json"):
                try:
                    os.remove(os.path.join(graph_dir, name))
                except OSError:
                    pass

    from transformers import AutoTokenizer

    encoder_tok = AutoTokenizer.from_pretrained(encoder_chunk_tokenizer_path, trust_remote_code=True)

    def _extract_title_abstract_from_text(text: str) -> tuple[str, str]:
        s = text or ""
        lower = s.lower()
        ti = lower.find("title:")
        ai = lower.find("abstract:")
        if ti != -1 and ai != -1 and ti < ai:
            title = s[ti + len("title:") : ai].strip()
            abstract = s[ai + len("abstract:") :].strip()
            return title, abstract
        if ai != -1:
            return "", s[ai + len("abstract:") :].strip()
        return "", s.strip()

    def _make_reconstruction_prompt(num_nodes: int) -> str:
        _ = num_nodes
        return (
            "You are given academic paper nodes from a citation network, each represented by a graph token.\n\n"
            "Output the reconstructed network in JSONL:\n\n"
            "Emit all nodes first, in the exact token order.\n\n"
            "Then emit all citation edges among those nodes.\n\n"
            "JSONL schema (one JSON object per line, no brackets, no trailing commas):\n\n"
            'Node: {"type":"node","id":"paper_000","title":"...","abstract":"..."}\n\n'
            'Edge: {"type":"edge","source":"paper_001","target":"paper_000","relation":"cites"}\n\n'
            "Constraints:\n\n"
            "Node lines must all come before any edge lines.\n\n"
            "Node id must match exactly the ids used in edge source/target.\n\n"
            "Output ONLY JSONL lines (no extra text).\n\n"
            "Input tokens: [node_token_0][node_token_1]...[node_token_N]\n\n"
            "Reconstruct the citation network now."
        )

    def _chunk_paper_node(title: str, abstract: str) -> list[dict]:
        title = (title or "").strip() or "unknown title"
        abstract = (abstract or "").strip()
        prefix = f"title: {title}\nabstract: "
        prefix_tokens = encoder_tok.tokenize(prefix)
        abstract_tokens = encoder_tok.tokenize(abstract)

        chunks: list[dict] = []

        # Preferred path: keep title+marker in every chunk and split abstract tokens.
        if len(prefix_tokens) < 512:
            avail = 512 - len(prefix_tokens)
            if avail <= 0:
                avail = 1
            if not abstract_tokens:
                seg_abs = ""
                enc_text = encoder_tok.convert_tokens_to_string(prefix_tokens)
                chunks.append(
                    {
                        "display_text": f"title: {title}\nabstract: ",
                        "encoder_text": enc_text,
                        "title": title,
                        "abstract_chunk": seg_abs,
                        "token_len": len(prefix_tokens),
                    }
                )
                return chunks
            for i in range(0, len(abstract_tokens), avail):
                seg_tokens = abstract_tokens[i : i + avail]
                seg_abs = encoder_tok.convert_tokens_to_string(seg_tokens).strip()
                full_tokens = prefix_tokens + seg_tokens
                enc_text = encoder_tok.convert_tokens_to_string(full_tokens)
                chunks.append(
                    {
                        "display_text": f"title: {title}\nabstract: {seg_abs}",
                        "encoder_text": enc_text,
                        "title": title,
                        "abstract_chunk": seg_abs,
                        "token_len": len(full_tokens),
                    }
                )
            return chunks

        # Fallback: title itself is too long. Chunk the whole formatted text exactly.
        full_text = f"title: {title}\nabstract: {abstract}"
        full_tokens = encoder_tok.tokenize(full_text)
        for i in range(0, len(full_tokens), 512):
            seg_tokens = full_tokens[i : i + 512]
            seg_text = encoder_tok.convert_tokens_to_string(seg_tokens)
            chunks.append(
                {
                    "display_text": seg_text,
                    "encoder_text": seg_text,
                    "title": title,
                    "abstract_chunk": seg_text,
                    "token_len": len(seg_tokens),
                }
            )
        return chunks or [
            {
                "display_text": f"title: {title}\nabstract: {abstract}",
                "encoder_text": f"title: {title}\nabstract: {abstract}",
                "title": title,
                "abstract_chunk": abstract,
                "token_len": len(full_tokens),
            }
        ]

    def _make_reconstruction_answer(graph_nodes: list[dict], graph_edges: list[dict]) -> str:
        alias_by_real: dict[str, str] = {}
        node_lines = []
        for i, n in enumerate(graph_nodes):
            real_id = n.get("id") or n.get("nodeId")
            alias = f"paper_{i:03d}"
            alias_by_real[str(real_id)] = alias
            title = (n.get("paper_title") or "").strip()
            abstract = (n.get("paper_abstract_chunk") or "").strip()
            if not title and not abstract:
                title, abstract = _extract_title_abstract_from_text(n.get("text", ""))
            node_lines.append(
                json.dumps(
                    {
                        "type": "node",
                        "id": alias,
                        "title": title,
                        "abstract": abstract,
                    },
                    ensure_ascii=False,
                )
            )
        edge_lines = []
        for e in graph_edges:
            s = alias_by_real.get(str(e.get("source")))
            d = alias_by_real.get(str(e.get("target")))
            if not s or not d or s == d:
                continue
            edge_lines.append(
                json.dumps(
                    {
                        "type": "edge",
                        "source": s,
                        "target": d,
                        "relation": "cites",
                    },
                    ensure_ascii=False,
                )
            )
        return "\n".join(node_lines + edge_lines)

    if append_mode and not subgraphs:
        old_meta = json_load(os.path.join(split_dir, "subgraph_export_meta.json"), default={}) or {}
        return old_meta

    order = list(range(len(subgraphs)))
    random.shuffle(order)
    n_valid = max(1, int(len(subgraphs) * valid_ratio)) if subgraphs else 0
    valid_idx = set(order[:n_valid]) if n_valid else set()
    train_path = os.path.join(split_dir, "train.jsonl")
    valid_path = os.path.join(split_dir, "valid.jsonl")
    train_f = open(train_path, "a" if append_mode else "w", encoding="utf-8")
    valid_f = open(valid_path, "a" if append_mode else "w", encoding="utf-8")
    old_meta = json_load(os.path.join(split_dir, "subgraph_export_meta.json"), default={}) if append_mode else {}
    old_train = int((old_meta or {}).get("train_samples", 0))
    old_valid = int((old_meta or {}).get("valid_samples", 0))
    old_detail = list((old_meta or {}).get("subgraph_stats_detail", []) or [])
    train_count = old_train
    valid_count = old_valid
    subgraph_stats = []
    chunk_cache: dict[tuple[str, str], list[dict]] = {}
    for gi, g in enumerate(subgraphs):
        # Split node content into <=512-token encoder chunks, duplicate edges, and fully connect chunks of same paper.
        expanded_nodes = []
        node_token_lens = []
        orig_to_chunks: dict[str, list[str]] = {}
        for n in g["nodes"]:
            orig_id = str(n.get("id") or n.get("nodeId"))
            raw_text = n.get("text", "")
            ckey = (orig_id, raw_text)
            chunk_specs = chunk_cache.get(ckey)
            if chunk_specs is None:
                title, abstract = _extract_title_abstract_from_text(raw_text)
                chunk_specs = _chunk_paper_node(title, abstract)
                chunk_cache[ckey] = chunk_specs
            chunk_ids = []
            for ci, cs in enumerate(chunk_specs):
                chunk_id = f"{orig_id}::chunk_{ci:03d}"
                chunk_ids.append(chunk_id)
                expanded_nodes.append(
                    {
                        "id": chunk_id,
                        "nodeType": "PaperChunk",
                        "name": "paper",
                        "text": cs["display_text"],
                        "encoder_text": cs["encoder_text"],
                        "paper_title": cs["title"],
                        "paper_abstract_chunk": cs["abstract_chunk"],
                        "orig_paper_id": orig_id,
                        "chunk_index": ci,
                        "chunk_count": len(chunk_specs),
                    }
                )
                node_token_lens.append(int(cs["token_len"]))
            orig_to_chunks[orig_id] = chunk_ids

        edge_key_set: set[tuple[str, str]] = set()
        normalized_edges = []
        # Replicate original inter-paper edges across all chunk pairs.
        for e in (g.get("edges") or []):
            s = str(e.get("source"))
            d = str(e.get("target"))
            if s == d:
                continue
            s_chunks = orig_to_chunks.get(s) or []
            d_chunks = orig_to_chunks.get(d) or []
            for sc in s_chunks:
                for dc in d_chunks:
                    if sc == dc:
                        continue
                    ek = (sc, dc)
                    if ek in edge_key_set:
                        continue
                    edge_key_set.add(ek)
                    normalized_edges.append({"source": sc, "target": dc})
        # Fully connect chunk copies of the same original node (bidirectional clique).
        for chunk_ids in orig_to_chunks.values():
            if len(chunk_ids) <= 1:
                continue
            for i in range(len(chunk_ids)):
                for j in range(len(chunk_ids)):
                    if i == j:
                        continue
                    ek = (chunk_ids[i], chunk_ids[j])
                    if ek in edge_key_set:
                        continue
                    edge_key_set.add(ek)
                    normalized_edges.append({"source": chunk_ids[i], "target": chunk_ids[j]})

        normalized_nodes = expanded_nodes

        gpath = os.path.abspath(os.path.join(graph_dir, g["graph_file"]))
        with open(gpath, "w", encoding="utf-8") as f:
            json.dump({"nodes": normalized_nodes, "edges": normalized_edges}, f, ensure_ascii=False)
        prompt = _make_reconstruction_prompt(len(normalized_nodes))
        answer = _make_reconstruction_answer(normalized_nodes, normalized_edges)
        sample_row = {
            "graph": gpath,
            "prompt": prompt,
            "answer": answer,
        }
        if gi in valid_idx:
            valid_f.write(json.dumps(sample_row, ensure_ascii=False) + "\n")
            valid_count += 1
        else:
            train_f.write(json.dumps(sample_row, ensure_ascii=False) + "\n")
            train_count += 1
        subgraph_stats.append(
            {
                "graph_file": g["graph_file"],
                "seed_node": g.get("seed_node"),
                "used_hop": int(g.get("used_hop", 0)),
                "requested_hop": int(g.get("requested_hop", 0)),
                "node_count": len(normalized_nodes),
                "edge_count": len(normalized_edges),
                "graph_text_token_len": int(sum(node_token_lens)),
                "node_token_mean": float(sum(node_token_lens) / len(node_token_lens)) if node_token_lens else 0.0,
                "node_token_min": int(min(node_token_lens)) if node_token_lens else 0,
                "node_token_max": int(max(node_token_lens)) if node_token_lens else 0,
            }
        )

    train_f.close()
    valid_f.close()

    def _summary(vals: list[int]) -> dict:
        if not vals:
            return {"mean": 0.0, "min": 0, "max": 0}
        return {"mean": float(sum(vals) / len(vals)), "min": int(min(vals)), "max": int(max(vals))}

    def _summary_float(vals: list[float]) -> dict:
        if not vals:
            return {"mean": 0.0, "min": 0.0, "max": 0.0}
        return {"mean": float(sum(vals) / len(vals)), "min": float(min(vals)), "max": float(max(vals))}

    merged_stats = old_detail + subgraph_stats if append_mode else subgraph_stats
    node_vals = [x["node_count"] for x in merged_stats]
    edge_vals = [x["edge_count"] for x in merged_stats]
    hop_vals = [x["used_hop"] for x in merged_stats]
    graph_text_token_vals = [x["graph_text_token_len"] for x in merged_stats]
    per_graph_node_token_mean_vals = [x["node_token_mean"] for x in merged_stats]
    node_token_min_vals = [x["node_token_min"] for x in merged_stats]
    node_token_max_vals = [x["node_token_max"] for x in merged_stats]

    summary = {
        "subgraph_count": len(merged_stats),
        "node_count": _summary(node_vals),
        "edge_count": _summary(edge_vals),
        "hop_count": _summary(hop_vals),
        "graph_text_token_len": _summary(graph_text_token_vals),
        "per_graph_node_token_mean": _summary_float(per_graph_node_token_mean_vals),
        "per_graph_node_token_min": _summary(node_token_min_vals),
        "per_graph_node_token_max": _summary(node_token_max_vals),
    }

    meta = {
        "generated_subgraphs": len(merged_stats),
        "train_samples": train_count,
        "valid_samples": valid_count,
        "subgraph_stats_summary": summary,
        "subgraph_stats_detail": merged_stats,
    }
    json_dump(os.path.join(split_dir, "subgraph_export_meta.json"), meta)
    return meta


def _build_graph_api_round_fetch(args, logger, min_year: int, cfg: dict):
    client = S2DatasetsClient(logger=logger)
    releases_payload = client.list_releases()
    latest, _ = normalize_release_ids(releases_payload)
    if not latest:
        raise RuntimeError("无法获取 Semantic Scholar release。")
    dataset_payload = client.get_dataset_files(latest, "citations")
    citation_urls_all = extract_file_urls(dataset_payload)
    if not citation_urls_all:
        raise RuntimeError("citations 数据集没有可下载分片。")
    citation_urls_all = list(citation_urls_all)

    def _safe_dataset_urls(name: str) -> list[str]:
        try:
            payload = client.get_dataset_files(latest, name)
            return extract_file_urls(payload)
        except Exception as e:
            logger.warning("Dataset %s URLs unavailable: %s", name, e)
            return []

    abstracts_urls_all = list(_safe_dataset_urls("abstracts"))
    papers_urls_all = list(_safe_dataset_urls("papers"))
    abstracts_by_shard = _index_urls_by_shard(abstracts_urls_all)
    papers_by_shard = _index_urls_by_shard(papers_urls_all)
    logger.info(
        "Round-fetch datasets | citations=%d abstracts=%d papers=%d (indexed by shard)",
        len(citation_urls_all),
        len(abstracts_by_shard),
        len(papers_by_shard),
    )

    target = int(cfg["num_subgraphs"])
    graph_dir = os.path.join(args.processed_dir, "cgm_graphs")
    ensure_dirs(graph_dir)
    resume_mode = bool(getattr(args, "resume_subgraphs", False))
    existing_count = 0
    if resume_mode:
        existing_count = len(
            [
                n
                for n in os.listdir(graph_dir)
                if n.endswith(".json") and n.startswith(str(cfg.get("graph_file_prefix", "citation_subgraph")))
            ]
        )
    round_files = int(cfg.get("round_files_start", 1))
    round_step = int(cfg.get("round_files_step", 1))
    cursor = 0
    round_idx = 0

    selected_subgraphs = []
    seen_signatures = set()
    node_map = {}
    edge_map = {}
    block_map = {}

    if resume_mode and existing_count > 0:
        existing_files = sorted(
            [
                n
                for n in os.listdir(graph_dir)
                if n.endswith(".json") and n.startswith(str(cfg.get("graph_file_prefix", "citation_subgraph")))
            ]
        )
        for fn in existing_files:
            fp = os.path.join(graph_dir, fn)
            try:
                g = json_load(fp, default={}) or {}
                nodes = g.get("nodes") or []
                norm_ids = []
                for nd in nodes:
                    nid = str(nd.get("orig_paper_id") or nd.get("id") or nd.get("nodeId") or "")
                    if "::chunk_" in nid:
                        nid = nid.split("::chunk_", 1)[0]
                    if nid:
                        norm_ids.append(nid)
                sig = tuple(sorted(set(norm_ids)))
                if sig:
                    seen_signatures.add(sig)
            except Exception:
                continue
        logger.info("Resume mode: detected existing subgraphs=%d", existing_count)

    target_new = max(0, target - existing_count)
    if target_new <= 0:
        stats = {
            "construction_mode": "graph_api_strict_round_fetch",
            "rounds": 0,
            "target_subgraphs": target,
            "existing_subgraphs": existing_count,
            "new_subgraphs": 0,
            "final_subgraphs": existing_count,
            "note": "已满足目标，跳过新增构建。",
        }
        return [], [], [], [], {"stats": stats, "meta": {"subgraph_config": cfg, "append_mode": True}}

    while len(selected_subgraphs) < target_new and cursor < len(citation_urls_all):
        round_idx += 1
        missing = target_new - len(selected_subgraphs)
        use_n = min(round_files, len(citation_urls_all) - cursor)
        round_citation_urls = citation_urls_all[cursor : cursor + use_n]
        cursor += use_n

        # keep disk usage low: clear previous round raw shards first.
        for n in os.listdir(args.raw_dir):
            p = os.path.join(args.raw_dir, n)
            if os.path.isfile(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

        logger.info(
            "Round %d start | missing=%d | download_files=%d | url_cursor=%d/%d",
            round_idx,
            missing,
            use_n,
            cursor,
            len(citation_urls_all),
        )
        round_abstracts_urls = []
        round_papers_urls = []
        for cu in round_citation_urls:
            tag = _extract_shard_tag_from_url(cu)
            if not tag:
                continue
            au = abstracts_by_shard.get(tag)
            if au:
                round_abstracts_urls.append(au)
            pu = papers_by_shard.get(tag)
            if pu:
                round_papers_urls.append(pu)
        # Fallback: datasets do not share a stable shard tag namespace.
        # Use rotating windows to broaden overlap across rounds.
        if (not round_abstracts_urls) and abstracts_urls_all:
            start = ((round_idx - 1) * use_n) % len(abstracts_urls_all)
            for i in range(use_n):
                round_abstracts_urls.append(abstracts_urls_all[(start + i) % len(abstracts_urls_all)])
        if (not round_papers_urls) and papers_urls_all:
            start = ((round_idx - 1) * use_n) % len(papers_urls_all)
            for i in range(use_n):
                round_papers_urls.append(papers_urls_all[(start + i) % len(papers_urls_all)])
        round_abstracts_urls = list(dict.fromkeys(round_abstracts_urls))
        round_papers_urls = list(dict.fromkeys(round_papers_urls))

        download_dataset_files(
            dataset_name="citations",
            urls=round_citation_urls,
            raw_dir=args.raw_dir,
            max_files=use_n,
            manifest_path="data/meta/download_manifest.json",
            logger=logger,
        )
        if round_abstracts_urls:
            download_dataset_files(
                dataset_name="abstracts",
                urls=round_abstracts_urls,
                raw_dir=args.raw_dir,
                max_files=len(round_abstracts_urls),
                manifest_path="data/meta/download_manifest.json",
                logger=logger,
            )
        if round_papers_urls:
            download_dataset_files(
                dataset_name="papers",
                urls=round_papers_urls,
                raw_dir=args.raw_dir,
                max_files=len(round_papers_urls),
                manifest_path="data/meta/download_manifest.json",
                logger=logger,
            )
        logger.info(
            "Round %d aligned shard downloads | citations=%d abstracts=%d papers=%d",
            round_idx,
            len(round_citation_urls),
            len(round_abstracts_urls),
            len(round_papers_urls),
        )

        round_cfg = dict(cfg)
        round_cfg["num_subgraphs"] = missing
        round_cfg["allow_bootstrap_repeats"] = False
        round_cfg["force_fill_to_target"] = True
        round_cfg["cleanup_scanned_raw"] = True
        tmp_cfg = os.path.join("data/meta", f"subgraph_config_round_{round_idx}.json")
        json_dump(tmp_cfg, round_cfg)

        old_cfg = args.subgraph_config
        old_cap = getattr(args, "max_citation_edges_scan", 0)
        args.subgraph_config = tmp_cfg
        args.max_citation_edges_scan = max(int(old_cap or 0), 5000000 * use_n)
        try:
            r_nodes, r_edges, r_blocks, r_subgraphs, _ = _build_subgraphs_via_graph_api(
                args=args,
                logger=logger,
                min_year=min_year,
            )
        finally:
            args.subgraph_config = old_cfg
            args.max_citation_edges_scan = old_cap
            try:
                os.remove(tmp_cfg)
            except OSError:
                pass

        added = 0
        for sg in r_subgraphs:
            sig = tuple(sorted([str(n.get("id") or n.get("nodeId")) for n in sg.get("nodes", [])]))
            if not sig or sig in seen_signatures:
                continue
            seen_signatures.add(sig)
            sg["graph_file"] = f"{cfg['graph_file_prefix']}_{existing_count + len(selected_subgraphs):03d}.json"
            selected_subgraphs.append(sg)
            added += 1
            if len(selected_subgraphs) >= target_new:
                break

        for n in r_nodes:
            pid = n.get("paper_id")
            if pid:
                node_map[pid] = n
        for e in r_edges:
            sk = (e.get("src_paper_id"), e.get("dst_paper_id"))
            if sk[0] and sk[1]:
                edge_map[sk] = e
        for b in r_blocks:
            bk = (b.get("paper_id"), b.get("view_type"), b.get("section_path"), b.get("text"))
            if bk[0]:
                block_map[bk] = b

        logger.info(
            "Round %d done | added_unique=%d | total=%d/%d",
            round_idx,
            added,
            existing_count + len(selected_subgraphs),
            target,
        )
        round_files += round_step

    # cleanup residual round raw shards
    for n in os.listdir(args.raw_dir):
        p = os.path.join(args.raw_dir, n)
        if os.path.isfile(p):
            try:
                os.remove(p)
            except OSError:
                pass

    if len(selected_subgraphs) < target_new:
        raise RuntimeError(
            f"分轮追加后仍未达到目标: {existing_count + len(selected_subgraphs)}/{target}。"
            "请增加 round_files_start/step 或放宽子图条件。"
        )

    stats = {
        "construction_mode": "graph_api_strict_round_fetch",
        "rounds": round_idx,
        "target_subgraphs": target,
        "existing_subgraphs": existing_count,
        "new_subgraphs": len(selected_subgraphs),
        "final_subgraphs": existing_count + len(selected_subgraphs),
        "note": "按轮次下载分片并去重追加，达到目标后停止；每轮原始分片已清理。",
    }
    return list(node_map.values()), list(edge_map.values()), list(block_map.values()), selected_subgraphs, {"stats": stats, "meta": {"subgraph_config": cfg, "append_mode": resume_mode}}


def cmd_build(args):
    logger = setup_logger()
    init_paths()

    min_year = args.min_year if args.min_year else (now_year() - args.recent_years)
    logger.info("Building with min_year=%d target_papers=%d", min_year, args.target_papers)

    if getattr(args, "construction_mode", "graph_api_strict") == "graph_api_strict":
        round_cfg = _load_subgraph_config(args.subgraph_config)
        try:
            if round_cfg.get("round_fetch_until_target", False):
                final_nodes, final_edges, final_blocks, prebuilt_subgraphs, mode_meta = _build_graph_api_round_fetch(
                    args=args,
                    logger=logger,
                    min_year=min_year,
                    cfg=round_cfg,
                )
            else:
                final_nodes, final_edges, final_blocks, prebuilt_subgraphs, mode_meta = _build_subgraphs_via_graph_api(
                    args=args,
                    logger=logger,
                    min_year=min_year,
                )
            nodes_df = pd.DataFrame(final_nodes)
            edges_df = pd.DataFrame(final_edges)
            blocks_df = pd.DataFrame(final_blocks)

            ensure_dirs(args.processed_dir, args.index_dir)
            nodes_path = os.path.join(args.processed_dir, "paper_nodes.parquet")
            edges_path = os.path.join(args.processed_dir, "citation_edges.parquet")
            blocks_path = os.path.join(args.processed_dir, "paper_text_blocks.parquet")
            nodes_df.to_parquet(nodes_path, index=False)
            edges_df.to_parquet(edges_path, index=False)
            blocks_df.to_parquet(blocks_path, index=False)

            graph_stats = write_index(
                nodes=final_nodes,
                edges=final_edges,
                out_dir=args.index_dir,
                index_format=args.index_format,
                direction=args.direction,
                max_neighbors=args.max_neighbors,
            )

            export_meta = None
            if args.export_cgm:
                append_mode = bool(getattr(args, "resume_subgraphs", False) and (mode_meta or {}).get("meta", {}).get("append_mode", False))
                export_meta = _export_cgm_prebuilt(
                    prebuilt_subgraphs,
                    valid_ratio=args.valid_ratio,
                    encoder_chunk_tokenizer_path=args.encoder_chunk_tokenizer_path,
                    append_mode=append_mode,
                )
                logger.info("Exported CGM training mapping to data/processed/cgm_graphs and cgm_splits")
                if export_meta:
                    s = export_meta.get("subgraph_stats_summary", {})
                    logger.info(
                        "Subgraph stats | count=%s | nodes(mean/min/max)=%.2f/%s/%s | hops(mean/min/max)=%.2f/%s/%s | edges(mean/min/max)=%.2f/%s/%s | graph_tokens(mean/min/max)=%.2f/%s/%s | node_tokens(mean/min/max)=%.2f/%.2f/%.2f",
                        s.get("subgraph_count", 0),
                        s.get("node_count", {}).get("mean", 0.0),
                        s.get("node_count", {}).get("min", 0),
                        s.get("node_count", {}).get("max", 0),
                        s.get("hop_count", {}).get("mean", 0.0),
                        s.get("hop_count", {}).get("min", 0),
                        s.get("hop_count", {}).get("max", 0),
                        s.get("edge_count", {}).get("mean", 0.0),
                        s.get("edge_count", {}).get("min", 0),
                        s.get("edge_count", {}).get("max", 0),
                        s.get("graph_text_token_len", {}).get("mean", 0.0),
                        s.get("graph_text_token_len", {}).get("min", 0),
                        s.get("graph_text_token_len", {}).get("max", 0),
                        s.get("per_graph_node_token_mean", {}).get("mean", 0.0),
                        s.get("per_graph_node_token_min", {}).get("min", 0.0),
                        s.get("per_graph_node_token_max", {}).get("max", 0.0),
                    )

            report = {
                "min_year": min_year,
                "target_papers": args.target_papers,
                "stats": mode_meta.get("stats", {}),
                "graph_stats": graph_stats,
                "construction_mode": "graph_api_strict",
                "outputs": {
                    "paper_nodes": nodes_path,
                    "citation_edges": edges_path,
                    "paper_text_blocks": blocks_path,
                    "index_dir": args.index_dir,
                },
            }
            if export_meta:
                report["subgraph_export_meta"] = {
                    "generated_subgraphs": export_meta.get("generated_subgraphs", 0),
                    "train_samples": export_meta.get("train_samples", 0),
                    "valid_samples": export_meta.get("valid_samples", 0),
                    "subgraph_stats_summary": export_meta.get("subgraph_stats_summary", {}),
                }
            json_dump("data/meta/build_report.json", report)
            logger.info("Build report saved: data/meta/build_report.json")
            return
        except Exception as e:
            if getattr(args, "allow_legacy_fallback", False):
                logger.warning("graph_api_strict 构建失败，将回退 legacy 流程: %s", e)
            else:
                raise RuntimeError(f"graph_api_strict 构建失败且已禁用 legacy 回退: {e}") from e

    nodes_by_id = {}
    edges = []
    blocks = []

    def _raw_files():
        names = [n for n in os.listdir(args.raw_dir) if n.endswith(".gz") or n.endswith(".jsonl") or n.endswith(".jsonl.gz")]
        return [os.path.join(args.raw_dir, n) for n in names]

    files = _raw_files()
    papers_files = sorted([p for p in files if "papers__" in os.path.basename(p).lower()])
    citation_files = sorted([p for p in files if "citation" in os.path.basename(p).lower()])
    abstract_files = sorted([p for p in files if "abstracts__" in os.path.basename(p).lower()])

    recent_count = 0

    def _ingest_record(rec):
        nonlocal recent_count
        node, row_edges, row_blocks = parse_record_to_rows(rec, logger=logger)
        pid = node["paper_id"]
        if pid not in nodes_by_id:
            nodes_by_id[pid] = node
            if node.get("year") is not None and int(node["year"]) >= min_year:
                recent_count += 1
        else:
            old = nodes_by_id[pid]
            if (not old.get("abstract")) and node.get("abstract"):
                old["abstract"] = node["abstract"]
            if (not old.get("title")) and node.get("title"):
                old["title"] = node["title"]
        edges.extend(row_edges)
        blocks.extend(row_blocks)
        return node, row_edges, row_blocks

    # Strict requirement from user: every exported node must have both title and abstract.
    # Use intersection-driven ingestion for speed and better yield:
    #   abstracts -> papers (fill titles for abstract candidates) -> citations (edges on eligible ids)
    abstract_candidate_cap = max(args.target_papers * 20, 10000)
    # Phase 1: abstracts (collect candidate ids with abstract)
    for path in abstract_files:
        logger.info("Parsing file: %s", path)
        for rec in iter_jsonl_records(path, logger=logger):
            node, row_edges, row_blocks = parse_record_to_rows(rec, logger=logger)
            pid = node["paper_id"]
            if pid not in nodes_by_id:
                nodes_by_id[pid] = node
            else:
                if (not nodes_by_id[pid].get("abstract")) and node.get("abstract"):
                    nodes_by_id[pid]["abstract"] = node["abstract"]
            if pid in nodes_by_id and node.get("abstract"):
                blocks.extend(row_blocks)
            if len(nodes_by_id) >= abstract_candidate_cap:
                logger.info("Phase abstracts(candidates) reached cap (%d).", len(nodes_by_id))
                break
        if len(nodes_by_id) >= abstract_candidate_cap:
            break

    candidate_ids_after_abstracts = {pid for pid, n in nodes_by_id.items() if (n.get("abstract") or "").strip()}

    # Phase 2: papers (fill title/year for abstract candidates only)
    title_filled = 0
    title_fill_cap = max(args.target_papers * 5, 1000)
    for path in papers_files:
        logger.info("Parsing file: %s", path)
        for rec in iter_jsonl_records(path, logger=logger):
            node, _, _ = parse_record_to_rows(rec, logger=logger)
            pid = node["paper_id"]
            if pid not in candidate_ids_after_abstracts:
                continue
            old = nodes_by_id.get(pid)
            if old is None:
                nodes_by_id[pid] = node
                old = nodes_by_id[pid]
            before_title = bool((old.get("title") or "").strip())
            if (not old.get("title")) and node.get("title"):
                old["title"] = node["title"]
            if old.get("year") is None and node.get("year") is not None:
                old["year"] = node["year"]
            if (not old.get("venue")) and node.get("venue"):
                old["venue"] = node["venue"]
            if (not old.get("fields_of_study")) and node.get("fields_of_study"):
                old["fields_of_study"] = node["fields_of_study"]
            if (not old.get("doi")) and node.get("doi"):
                old["doi"] = node["doi"]
            if (not old.get("arxiv_id")) and node.get("arxiv_id"):
                old["arxiv_id"] = node["arxiv_id"]
            if (not old.get("pmid")) and node.get("pmid"):
                old["pmid"] = node["pmid"]
            if (not before_title) and (old.get("title") or "").strip():
                title_filled += 1
            if title_filled >= title_fill_cap:
                logger.info("Phase papers(title fill) reached cap (%d matched titles).", title_filled)
                break
        if title_filled >= title_fill_cap:
            break

    candidate_ids_after_papers = {
        pid
        for pid, n in nodes_by_id.items()
        if (n.get("title") or "").strip() and (n.get("abstract") or "").strip()
    }

    # Phase 3: citations (directed edges among title+abstract candidates)
    matched_citation_edges = 0
    raw_citation_edges_cap = max(args.target_papers * 30, 5000)
    raw_citation_edges = []
    for path in citation_files:
        logger.info("Parsing file: %s", path)
        for rec in iter_jsonl_records(path, logger=logger):
            _, row_edges, _ = parse_record_to_rows(rec, logger=logger)
            for e in row_edges:
                if e.get("src_paper_id") and e.get("dst_paper_id") and e.get("src_paper_id") != e.get("dst_paper_id"):
                    raw_citation_edges.append(e)
                s = e.get("src_paper_id")
                d = e.get("dst_paper_id")
                if s in candidate_ids_after_papers and d in candidate_ids_after_papers:
                    edges.append(e)
                    matched_citation_edges += 1
            citation_match_cap = max(min(args.target_papers, 300), 50)
            if matched_citation_edges >= citation_match_cap or len(raw_citation_edges) >= raw_citation_edges_cap:
                logger.info(
                    "Phase citations stop | internal_matched_edges=%d | raw_edges=%d",
                    matched_citation_edges,
                    len(raw_citation_edges),
                )
                break
        if matched_citation_edges >= max(min(args.target_papers, 300), 50) or len(raw_citation_edges) >= raw_citation_edges_cap:
            break

    # Prioritize nodes that actually appear in collected citation edges to increase induced subgraph density.
    edge_node_ids = set()
    for e in edges:
        if e.get("src_paper_id"):
            edge_node_ids.add(e["src_paper_id"])
        if e.get("dst_paper_id"):
            edge_node_ids.add(e["dst_paper_id"])

    edge_connected_nodes = [n for n in nodes_by_id.values() if n["paper_id"] in edge_node_ids]
    if edge_connected_nodes:
        selected_nodes = split_recent_priority(edge_connected_nodes, min_year=min_year, target_papers=args.target_papers)
        if len(selected_nodes) < args.target_papers:
            used = {n["paper_id"] for n in selected_nodes}
            remain = [n for n in nodes_by_id.values() if n["paper_id"] not in used]
            selected_nodes.extend(
                split_recent_priority(remain, min_year=min_year, target_papers=args.target_papers - len(selected_nodes))
            )
    else:
        selected_nodes = split_recent_priority(list(nodes_by_id.values()), min_year=min_year, target_papers=args.target_papers)
    selected_ids = set(n["paper_id"] for n in selected_nodes)
    nodes_by_id = {k: v for k, v in nodes_by_id.items() if k in selected_ids}
    edges = [e for e in edges if e.get("src_paper_id") in selected_ids]
    blocks = [b for b in blocks if b.get("paper_id") in selected_ids]

    allow = {x.strip().lower() for x in args.license_allow.split(",") if x.strip()}
    keep_unknown = "unknown" in allow
    final_nodes, final_edges, final_blocks, stats = filter_and_finalize(
        nodes_by_id=nodes_by_id,
        edges=edges,
        blocks=blocks,
        license_allow=allow,
        keep_unknown_license=keep_unknown,
    )

    nodes_df = pd.DataFrame(final_nodes)
    edges_df = pd.DataFrame(final_edges)
    blocks_df = pd.DataFrame(final_blocks)

    # Fallback for sparse cross-dataset overlap: enrich citation endpoints via Graph API to satisfy strict title+abstract.
    if args.export_cgm and (not final_nodes) and raw_citation_edges:
        logger.warning("Strict filtering produced empty graph; trying Graph API enrichment on citation endpoints.")
        endpoint_corpus_ids = []
        seen_ep = set()
        for e in raw_citation_edges:
            for pid in (e.get("src_paper_id"), e.get("dst_paper_id")):
                if not pid or not str(pid).startswith("corpus:"):
                    continue
                cid = str(pid).split(":", 1)[1]
                if cid in seen_ep:
                    continue
                seen_ep.add(cid)
                endpoint_corpus_ids.append(cid)
        enriched = _graph_batch_enrich_title_abstract(endpoint_corpus_ids, logger=logger)
        if enriched:
            # prioritize endpoints that have title+abstract from Graph API and appear in raw edges
            endpoint_ids = set(enriched.keys())
            fallback_edges = [
                {**e, "is_resolved": True}
                for e in raw_citation_edges
                if e.get("src_paper_id") in endpoint_ids and e.get("dst_paper_id") in endpoint_ids and e.get("src_paper_id") != e.get("dst_paper_id")
            ]
            edge_endpoint_ids = set()
            for e in fallback_edges:
                edge_endpoint_ids.add(e["src_paper_id"])
                edge_endpoint_ids.add(e["dst_paper_id"])
            fallback_nodes = [enriched[pid] for pid in edge_endpoint_ids if pid in enriched]
            fallback_blocks = [
                {
                    "paper_id": n["paper_id"],
                    "view_type": "abstract",
                    "text": n["abstract"],
                    "token_len": estimate_token_len(n["abstract"]),
                    "section_path": "abstract",
                }
                for n in fallback_nodes
            ]
            final_nodes, final_edges, final_blocks = fallback_nodes, fallback_edges, fallback_blocks
            nodes_df = pd.DataFrame(final_nodes)
            edges_df = pd.DataFrame(final_edges)
            blocks_df = pd.DataFrame(final_blocks)

    # Legacy local-data fallback (kept as last resort)
    if args.export_cgm and (not final_nodes) and edges:
        logger.warning("Strict filtering produced empty graph; using local fallback citation-endpoint exportable nodes (title+abstract required).")
        endpoint_ids = set()
        for e in edges:
            if e.get("src_paper_id"):
                endpoint_ids.add(e["src_paper_id"])
            if e.get("dst_paper_id"):
                endpoint_ids.add(e["dst_paper_id"])
        fallback_nodes = []
        for pid in endpoint_ids:
            n = nodes_by_id.get(pid, {})
            title = (n.get("title") or "").strip()
            abstract = (n.get("abstract") or "").strip()
            if not title or not abstract:
                continue
            fallback_nodes.append(
                {
                    "paper_id": pid,
                    "title": title,
                    "abstract": abstract,
                    "year": n.get("year"),
                    "venue": n.get("venue") or "",
                    "fields_of_study": n.get("fields_of_study") or [],
                    "doi": n.get("doi"),
                    "arxiv_id": n.get("arxiv_id"),
                    "pmid": n.get("pmid"),
                    "has_fulltext": bool(n.get("has_fulltext")),
                    "license": n.get("license") or "unknown",
                }
            )
        # keep induced edges on fallback endpoint set
        endpoint_ids = {n["paper_id"] for n in fallback_nodes}
        fallback_edges = [
            {**e, "is_resolved": True}
            for e in edges
            if e.get("src_paper_id") in endpoint_ids and e.get("dst_paper_id") in endpoint_ids and e.get("src_paper_id") != e.get("dst_paper_id")
        ]
        fallback_blocks = [b for b in blocks if b.get("paper_id") in endpoint_ids]
        final_nodes, final_edges, final_blocks = fallback_nodes, fallback_edges, fallback_blocks
        nodes_df = pd.DataFrame(final_nodes)
        edges_df = pd.DataFrame(final_edges)
        blocks_df = pd.DataFrame(final_blocks)

    nodes_path = os.path.join(args.processed_dir, "paper_nodes.parquet")
    edges_path = os.path.join(args.processed_dir, "citation_edges.parquet")
    blocks_path = os.path.join(args.processed_dir, "paper_text_blocks.parquet")
    ensure_dirs(args.processed_dir, args.index_dir)

    nodes_df.to_parquet(nodes_path, index=False)
    edges_df.to_parquet(edges_path, index=False)
    blocks_df.to_parquet(blocks_path, index=False)

    graph_stats = write_index(
        nodes=final_nodes,
        edges=final_edges,
        out_dir=args.index_dir,
        index_format=args.index_format,
        direction=args.direction,
        max_neighbors=args.max_neighbors,
    )

    report = {
        "min_year": min_year,
        "target_papers": args.target_papers,
        "stats": stats,
        "graph_stats": graph_stats,
        "outputs": {
            "paper_nodes": nodes_path,
            "citation_edges": edges_path,
            "paper_text_blocks": blocks_path,
            "index_dir": args.index_dir,
        },
    }
    json_dump("data/meta/build_report.json", report)

    logger.info("Filter stats: %s", stats)
    logger.info("Graph stats: %s", graph_stats)

    if args.export_cgm:
        if nodes_df.empty:
            logger.warning("Skip CGM export: no nodes after filtering.")
        else:
            export_meta = _export_cgm(
                nodes_df,
                edges_df,
                blocks_df,
                valid_ratio=args.valid_ratio,
                graph_filename=args.graph_filename,
                subgraph_config_path=args.subgraph_config,
            )
            logger.info("Exported CGM training mapping to data/processed/cgm_graphs and cgm_splits")
            if export_meta:
                s = export_meta.get("subgraph_stats_summary", {})
                logger.info(
                    "Subgraph stats | count=%s | nodes(mean/min/max)=%.2f/%s/%s | hops(mean/min/max)=%.2f/%s/%s | edges(mean/min/max)=%.2f/%s/%s | graph_tokens(mean/min/max)=%.2f/%s/%s | node_tokens(mean/min/max)=%.2f/%.2f/%.2f",
                    s.get("subgraph_count", 0),
                    s.get("node_count", {}).get("mean", 0.0),
                    s.get("node_count", {}).get("min", 0),
                    s.get("node_count", {}).get("max", 0),
                    s.get("hop_count", {}).get("mean", 0.0),
                    s.get("hop_count", {}).get("min", 0),
                    s.get("hop_count", {}).get("max", 0),
                    s.get("edge_count", {}).get("mean", 0.0),
                    s.get("edge_count", {}).get("min", 0),
                    s.get("edge_count", {}).get("max", 0),
                    s.get("graph_text_token_len", {}).get("mean", 0.0),
                    s.get("graph_text_token_len", {}).get("min", 0),
                    s.get("graph_text_token_len", {}).get("max", 0),
                    s.get("per_graph_node_token_mean", {}).get("mean", 0.0),
                    s.get("per_graph_node_token_min", {}).get("min", 0.0),
                    s.get("per_graph_node_token_max", {}).get("max", 0.0),
                )


def build_parser():
    p = argparse.ArgumentParser(description="Semantic Scholar datasets mini pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    sp_list = sub.add_parser("list", help="list releases and latest datasets")
    sp_list.set_defaults(func=cmd_list)

    sp_dl = sub.add_parser("download", help="download selected dataset shards")
    sp_dl.add_argument("--release-id", default=None, help="override release id")
    sp_dl.add_argument("--datasets", default=None, help="comma separated dataset names")
    sp_dl.add_argument("--max-files", type=int, default=1, help="download first N shards per dataset (>=1)")
    sp_dl.add_argument("--prefer-diff", action="store_true", help="try diff endpoint first")
    sp_dl.set_defaults(func=cmd_download)

    sp_build = sub.add_parser("build", help="build 3 tables and adjacency index")
    sp_build.add_argument("--raw-dir", default="data/raw")
    sp_build.add_argument("--processed-dir", default="data/processed")
    sp_build.add_argument("--index-dir", default="data/index")
    sp_build.add_argument("--target-papers", type=int, default=10000)
    sp_build.add_argument("--recent-years", type=int, default=5)
    sp_build.add_argument("--min-year", type=int, default=None)
    sp_build.add_argument("--construction-mode", choices=["graph_api_strict", "legacy"], default="graph_api_strict")
    sp_build.add_argument("--allow-legacy-fallback", action="store_true", default=False)
    sp_build.add_argument("--max-citation-edges-scan", type=int, default=0, help="0表示自动估算上限")
    sp_build.add_argument("--license-allow", default="odc-by,unknown")
    sp_build.add_argument("--index-format", choices=["csr", "parquet_neighbors"], default="csr")
    sp_build.add_argument("--direction", choices=["outgoing", "incoming", "both"], default="outgoing")
    sp_build.add_argument("--max-neighbors", type=int, default=50)
    sp_build.add_argument("--export-cgm", action="store_true", default=True)
    sp_build.add_argument("--no-export-cgm", action="store_false", dest="export_cgm")
    sp_build.add_argument("--valid-ratio", type=float, default=0.05)
    sp_build.add_argument("--graph-filename", default="citation_graph.json")
    sp_build.add_argument("--subgraph-config", default="get_S2ORC/subgraph_config.json")
    sp_build.add_argument("--resume-subgraphs", action="store_true", default=False, help="从已有 cgm_graphs 断点续构（仅追加缺口）")
    sp_build.add_argument(
        "--encoder-chunk-tokenizer-path",
        default="/root/autodl-tmp/CodeFuse-CGM/encoder/specter2_base",
        help="用于按512-token切分节点内容的编码器tokenizer路径（应与训练编码器一致）",
    )
    sp_build.set_defaults(func=cmd_build)
    return p


def main():
    load_local_env(".env")
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
