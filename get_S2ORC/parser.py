import gzip
import json
import os
import random
from collections.abc import Iterator
from typing import Any

from get_S2ORC.utils import clean_text, clip_text, estimate_token_len, sha1_text


UNKNOWN_FIELD_LOG_LIMIT = 20
_unknown_field_log_count = 0


def iter_jsonl_records(file_path: str, logger=None) -> Iterator[dict[str, Any]]:
    opener = gzip.open if file_path.endswith(".gz") else open
    mode = "rt"
    with opener(file_path, mode, encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except Exception as e:
                if logger:
                    logger.warning("JSON decode error in %s:%d: %s", file_path, i, e)


def _pick(d: dict[str, Any], keys: list[str]) -> Any:
    def _ci_get(obj: dict[str, Any], key: str):
        if key in obj:
            return obj[key]
        lk = key.lower()
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() == lk:
                return v
        return None

    def _path_get(obj: dict[str, Any], path: str):
        cur: Any = obj
        for part in path.split("."):
            if not isinstance(cur, dict):
                return None
            cur = _ci_get(cur, part)
            if cur is None:
                return None
        return cur

    for k in keys:
        v = _path_get(d, k)
        if v not in (None, ""):
            return v
    return None


def build_paper_id(record: dict[str, Any]) -> str:
    pid = _pick(record, ["paperId", "paper_id", "id"]) 
    if pid:
        return f"paper:{pid}"
    corpus = _pick(record, ["corpusId", "corpus_id"])
    if corpus:
        return f"corpus:{corpus}"
    doi = _pick(record, ["doi", "externalIds.DOI"])
    if doi:
        return f"doi:{str(doi).lower()}"
    arxiv_id = _extract_external(record, ["ArXiv", "arxiv", "arXiv"])
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    pmid = _extract_external(record, ["PubMed", "PMID", "pmid"])
    if pmid:
        return f"pmid:{pmid}"
    title = clean_text(_pick(record, ["title", "paper_title"]) or "")
    year = _pick(record, ["year", "publicationYear"]) or ""
    return f"sha1:{sha1_text(f'{title}|{year}')[:20]}"


def _extract_external(record: dict[str, Any], keys: list[str]) -> str | None:
    ext = _pick(record, ["externalIds", "externalids"]) or {}
    if isinstance(ext, dict):
        for k in keys:
            v = _pick(ext, [k])
            if v:
                return str(v)
    return None


def _normalize_year(y: Any) -> int | None:
    if y is None:
        return None
    try:
        yi = int(y)
        if 1800 <= yi <= 2100:
            return yi
    except Exception:
        return None
    return None


def _extract_license(record: dict[str, Any]) -> str:
    lic = _pick(record, ["license", "openAccessPdf.license", "content.license"])
    if isinstance(lic, dict):
        lic = lic.get("name")
    if not lic:
        return "unknown"
    return str(lic).strip().lower()


def _flatten_fields(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, list):
        for x in value:
            if isinstance(x, str):
                out.append(x)
            elif isinstance(x, dict):
                name = x.get("name") or x.get("category")
                if name:
                    out.append(str(name))
    elif isinstance(value, str):
        out = [value]
    return list(dict.fromkeys([clean_text(x) for x in out if clean_text(x)]))


def _extract_sections(record: dict[str, Any]) -> list[dict[str, str]]:
    candidates = []
    for k in ["sections", "fullText", "full_text", "body_text", "content"]:
        v = _pick(record, [k])
        if v is not None:
            candidates.append((k, v))
    sections: list[dict[str, str]] = []
    for k, v in candidates:
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    heading = clean_text(item.get("heading") or item.get("section") or item.get("title") or "")
                    text = clean_text(item.get("text") or item.get("content") or "")
                    if text:
                        sections.append({"heading": heading, "text": text, "section_path": heading or k})
        elif isinstance(v, dict):
            for sk, sv in v.items():
                if isinstance(sv, str):
                    text = clean_text(sv)
                    if text:
                        sections.append({"heading": clean_text(sk), "text": text, "section_path": clean_text(sk)})
                elif isinstance(sv, dict):
                    text = clean_text(sv.get("text") or sv.get("content") or "")
                    if text:
                        hd = clean_text(sv.get("heading") or sk)
                        sections.append({"heading": hd, "text": text, "section_path": hd})
    return sections


def _extract_from_s2orc_content(record: dict[str, Any]) -> dict[str, Any]:
    content = _pick(record, ["content"]) or {}
    if not isinstance(content, dict):
        return {}
    text = content.get("text") or ""
    ann = content.get("annotations") or {}
    out = {}

    # Many S2ORC shards encode title/abstract in annotations as plain strings.
    if isinstance(ann, dict):
        title_ann = ann.get("title")
        abs_ann = ann.get("abstract")
        if isinstance(title_ann, str) and title_ann.strip():
            out["title"] = clean_text(title_ann)
        if isinstance(abs_ann, str) and abs_ann.strip():
            out["abstract"] = clean_text(abs_ann)

    if text and not out.get("abstract"):
        # Fallback: use the first part of content text as a weak abstract-like snippet.
        out["abstract"] = clip_text(clean_text(text), 4000)
    return out


def _view_type_from_heading(heading: str) -> str:
    h = heading.lower()
    if any(x in h for x in ["intro", "background"]):
        return "intro"
    if any(x in h for x in ["method", "approach", "experiment", "setup"]):
        return "method"
    if any(x in h for x in ["conclusion", "discussion", "future"]):
        return "conclusion"
    return "random_span"


def _extract_refs(record: dict[str, Any]) -> list[dict[str, Any]]:
    refs = []
    # citation dataset often stores one directed edge per row
    citing = _pick(record, ["citingcorpusid", "citingCorpusId", "sourcecorpusid", "source"])
    cited = _pick(record, ["citedcorpusid", "citedCorpusId", "targetcorpusid", "target"])
    if citing is not None and cited is not None:
        refs.append({"corpusId": cited, "_src_corpus": citing})

    for k in ["references", "citations", "outboundCitations", "citation", "reference"]:
        v = _pick(record, [k])
        if isinstance(v, list):
            refs.extend(v)
    return refs


def _extract_ref_dst(ref: Any) -> str | None:
    if isinstance(ref, str):
        return f"paper:{ref}"
    if isinstance(ref, dict):
        rid = _pick(ref, ["paperId", "paper_id", "id"])
        if rid:
            return f"paper:{rid}"
        cid = _pick(ref, ["corpusId", "corpus_id"])
        if cid:
            return f"corpus:{cid}"
        doi = _pick(ref, ["doi"])
        if doi:
            return f"doi:{str(doi).lower()}"
    return None


def parse_record_to_rows(record: dict[str, Any], logger=None) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    s2orc_fields = _extract_from_s2orc_content(record)
    paper_id = build_paper_id(record)
    title = clean_text(_pick(record, ["title", "paper_title", "name"]) or s2orc_fields.get("title") or "")
    abstract = clean_text(_pick(record, ["abstract", "summary"]) or s2orc_fields.get("abstract") or "")
    year = _normalize_year(_pick(record, ["year", "publicationYear"]))
    venue = clean_text(_pick(record, ["venue", "journal", "publicationVenue", "booktitle"]) or "")
    fields = _flatten_fields(_pick(record, ["fieldsOfStudy", "s2FieldsOfStudy", "magFieldsOfStudy"]))

    doi = _pick(record, ["doi"]) or _extract_external(record, ["DOI", "doi"])
    arxiv_id = _extract_external(record, ["ArXiv", "arXiv", "arxiv"])
    pmid = _extract_external(record, ["PubMed", "PMID", "pmid"])
    license_name = _extract_license(record)

    sections = _extract_sections(record)
    has_fulltext = len(sections) > 0

    node = {
        "paper_id": paper_id,
        "title": title,
        "abstract": abstract,
        "year": year,
        "venue": venue,
        "fields_of_study": fields,
        "doi": clean_text(doi) if doi else None,
        "arxiv_id": clean_text(arxiv_id) if arxiv_id else None,
        "pmid": clean_text(pmid) if pmid else None,
        "has_fulltext": has_fulltext,
        "license": license_name if license_name else "unknown",
    }

    blocks: list[dict[str, Any]] = []
    if abstract:
        blocks.append(
            {
                "paper_id": paper_id,
                "view_type": "abstract",
                "text": abstract,
                "token_len": estimate_token_len(abstract),
                "section_path": "abstract",
            }
        )

    for sec in sections:
        txt = clip_text(sec["text"], 4000)
        if not txt:
            continue
        view_type = _view_type_from_heading(sec.get("heading", ""))
        blocks.append(
            {
                "paper_id": paper_id,
                "view_type": view_type,
                "text": txt,
                "token_len": estimate_token_len(txt),
                "section_path": sec.get("section_path") or None,
            }
        )

    refs = _extract_refs(record)
    edges: list[dict[str, Any]] = []
    record_src = _pick(record, ["citingcorpusid", "citingCorpusId", "sourcecorpusid", "source"])
    for ref in refs:
        dst = _extract_ref_dst(ref)
        src = paper_id
        if isinstance(ref, dict) and ref.get("_src_corpus") is not None:
            src = f"corpus:{ref.get('_src_corpus')}"
        elif record_src is not None:
            src = f"corpus:{record_src}"
        context = ""
        if isinstance(ref, dict):
            context = clean_text(ref.get("context") or ref.get("mention") or ref.get("snippet") or "")
        edges.append(
            {
                "src_paper_id": src,
                "dst_paper_id": dst,
                "is_resolved": bool(dst),
                "context": clip_text(context, 300) if context else "",
            }
        )

    global _unknown_field_log_count
    if logger and _unknown_field_log_count < UNKNOWN_FIELD_LOG_LIMIT and random.random() < 0.0005:
        known = {
            "paperId", "paper_id", "id", "corpusId", "title", "abstract", "year", "venue", "journal",
            "fieldsOfStudy", "s2FieldsOfStudy", "magFieldsOfStudy", "externalIds", "doi", "references",
            "citations", "outboundCitations", "sections", "fullText", "body_text", "content",
            "openAccessPdf", "publicationYear", "publicationVenue", "booktitle", "summary",
        }
        unknown = [k for k in record.keys() if k not in known]
        if unknown:
            logger.info("Unknown fields sample: %s", unknown[:15])
            _unknown_field_log_count += 1

    return node, edges, blocks


def iter_raw_records(raw_dir: str, logger=None):
    def _prio(name: str) -> tuple[int, str]:
        n = name.lower()
        if "papers__" in n:
            return (0, name)
        if "citation" in n:
            return (1, name)
        if "abstracts__" in n:
            return (2, name)
        return (3, name)

    for name in sorted(os.listdir(raw_dir), key=_prio):
        if not (name.endswith(".jsonl") or name.endswith(".jsonl.gz") or name.endswith(".gz")):
            continue
        path = os.path.join(raw_dir, name)
        if logger:
            logger.info("Parsing file: %s", path)
        for rec in iter_jsonl_records(path, logger=logger):
            yield rec


def iter_raw_records_with_source(raw_dir: str, logger=None):
    def _prio(name: str) -> tuple[int, str]:
        n = name.lower()
        if "papers__" in n:
            return (0, name)
        if "citation" in n:
            return (1, name)
        if "abstracts__" in n:
            return (2, name)
        return (3, name)

    for name in sorted(os.listdir(raw_dir), key=_prio):
        if not (name.endswith(".jsonl") or name.endswith(".jsonl.gz") or name.endswith(".gz")):
            continue
        path = os.path.join(raw_dir, name)
        if logger:
            logger.info("Parsing file: %s", path)
        for rec in iter_jsonl_records(path, logger=logger):
            yield name, rec
