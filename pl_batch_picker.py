import csv
import json
import math
import os
import re
import threading
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pymssql
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import numpy as np
except Exception:  # numpy is optional
    np = None


QUEUE_FIELDS = [
    "VehicleId_Motor",
    "Category",
    "SubCategory",
    "row_cnt",
    "status",
    "processed_at",
    "notes",
]


@dataclass
class RowScore:
    row: Dict[str, Any]
    like: int
    score_desc: int
    score_term: int
    score_value: int
    category_state: str
    domain_state: str
    has_generic: bool
    has_object: bool


def _to_text(v: Any) -> str:
    return "" if v is None else str(v)


def _norm_text(v: Any) -> str:
    return _to_text(v).strip().lower()


def _tokens(text: str) -> List[str]:
    t = _to_text(text).lower()
    if not t:
        return []
    # Fix: avoid splitting "A/C" into meaningless single-letter tokens "a","c"
    t = re.sub(r"\b(a)\s*/\s*(c)\b", "ac", t)
    toks = re.findall(r"[a-z0-9_\-]+", t)
    # Drop single-letter alphabetic noise tokens (a,c,l,r, etc.), keep single-digit numbers
    return [x for x in toks if not (len(x) == 1 and x.isalpha())]


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _locate_kb_root(root: Path) -> Optional[Path]:
    """定位真正 KB 根目录（含 kb_manifest.json）。"""
    if (root / "kb_manifest.json").exists():
        return root
    matches = [m for m in root.rglob("kb_manifest.json") if m.is_file()]
    if not matches:
        return None
    # 命中多份时：选最浅层，若并列按路径字典序
    matches_sorted = sorted(matches, key=lambda m: (len(m.relative_to(root).parts), str(m)))
    return matches_sorted[0].parent


def load_kb(path: str = "kb") -> Dict[str, Any]:
    default_scoring = {
        "score_desc_weight": 0.60,
        "score_term_weight": 0.30,
        "score_value_weight": 0.10,
        "gpg_delta_min": -4,
        "gpg_delta_max": 10,
    }
    kb: Dict[str, Any] = {
        "kb_name": "fallback-kb",
        "version": "0",
        "description": "manifest/CSV KB missing, fallback defaults",
        "scoring_defaults": default_scoring,
        "norm_map": {},
        "concept_map": {},
        "weight_map": {},
        "norm_to_concepts": {},
        "adjacency": {},
        "guard": {},
        "stopwords": set(),
        "pl_norm_map": {},
        "pl_concept_map": {},
        "pl_adjacency": {},
        "pl_light_map": {},
        "pl_enabled": False,
        "category_alias": {},
        "weak_token_weight": {},
        "strong_token_weight": {},
    }

    root = Path(path)
    tmp_dir = None
    if root.is_file() and root.suffix.lower() == ".zip":
        tmp_dir = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(root, "r") as zf:
            zf.extractall(tmp_dir.name)
        root = Path(tmp_dir.name)
    elif not root.exists() and Path("kb.zip").exists():
        tmp_dir = tempfile.TemporaryDirectory()
        with zipfile.ZipFile("kb.zip", "r") as zf:
            zf.extractall(tmp_dir.name)
        root = Path(tmp_dir.name)

    # 关键修复：自动定位 kb_manifest.json 所在目录（兼容多一层目录/zip 顶层目录）
    kb_root = _locate_kb_root(root)
    if kb_root is None:
        if tmp_dir:
            tmp_dir.cleanup()
        return kb

    matches = [m for m in root.rglob("kb_manifest.json") if m.is_file()]
    if len(matches) > 1:
        kb["description"] = f"{kb['description']} | multiple manifests found, using shallowest: {kb_root}"

    manifest_path = kb_root / "kb_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        if tmp_dir:
            tmp_dir.cleanup()
        return kb

    kb["kb_name"] = manifest.get("kb_name", kb["kb_name"])
    kb["version"] = manifest.get("version", kb["version"])
    kb["description"] = manifest.get("description", kb["description"])
    kb["scoring_defaults"] = {**default_scoring, **(manifest.get("scoring_defaults") or {})}

    files = manifest.get("files") or {}
    extras = manifest.get("extras") or {}
    syn_path = kb_root / files.get("lexicon_synonyms", "lexicon/lexicon_synonyms.csv")
    link_path = kb_root / files.get("concept_links", "lexicon/concept_links.csv")
    guard_path = kb_root / files.get("domain_guard", "rules/domain_guard.csv")
    stop_path = kb_root / files.get("url_stopwords", "rules/url_stopwords.csv")
    pl_syn_path = kb_root / extras.get("pl_synonyms", "lexicon/pl_synonyms.csv")
    pl_link_path = kb_root / extras.get("pl_concept_links", "lexicon/pl_concept_links.csv")
    pl_map_path = kb_root / extras.get("pl_light_match_map", "rules/pl_light_match_map.csv")
    alias_path = kb_root / "rules/category_aliases.csv"
    weak_tokens_path = kb_root / "rules/weak_tokens.csv"
    strong_tokens_path = kb_root / "rules/strong_tokens.csv"

    norm_map: Dict[str, str] = {}
    concept_map: Dict[str, str] = {}
    weight_map: Dict[str, float] = {}
    norm_to_concepts: Dict[str, Set[str]] = {}
    for r in _read_csv_dicts(syn_path):
        term = _norm_text(r.get("term"))
        if not term:
            continue
        norm = _norm_text(r.get("norm")) or term
        concept = _norm_text(r.get("concept"))
        weight = _safe_float(r.get("weight"), 1.0)
        norm_map[term] = norm
        concept_map[term] = concept
        weight_map[term] = weight
        norm_to_concepts.setdefault(norm, set())
        if concept:
            norm_to_concepts[norm].add(concept)

    adjacency: Dict[str, List[Tuple[str, float]]] = {}
    for r in _read_csv_dicts(link_path):
        a = _norm_text(r.get("concept_a"))
        b = _norm_text(r.get("concept_b"))
        s = _safe_float(r.get("score"), 0.0)
        if not a or not b:
            continue
        adjacency.setdefault(a, []).append((b, s))
        adjacency.setdefault(b, []).append((a, s))

    guard: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in _read_csv_dicts(guard_path):
        cat = _norm_text(r.get("new_category"))
        dom = _norm_text(r.get("old_url_domain"))
        if not cat or not dom:
            continue
        p = _safe_float(r.get("penalty"), 0.0)
        guard[(cat, dom)] = {
            "mode": _norm_text(r.get("mode")) or "allow",
            # 关键修复：兼容 KB 里负数惩罚，统一转为正惩罚幅度
            "penalty": (-p if p < 0 else p),
            "note": _to_text(r.get("note")),
        }

    stopwords: Set[str] = set()
    for r in _read_csv_dicts(stop_path):
        t = _norm_text(r.get("term"))
        if t:
            stopwords.add(t)

    pl_norm_map: Dict[str, str] = {}
    pl_concept_map: Dict[str, str] = {}
    for r in _read_csv_dicts(pl_syn_path):
        term = _norm_text(r.get("term"))
        if not term:
            continue
        pl_norm_map[term] = _norm_text(r.get("norm")) or term
        pl_concept_map[term] = _norm_text(r.get("pl_concept"))

    pl_adjacency: Dict[str, List[Tuple[str, float]]] = {}
    for r in _read_csv_dicts(pl_link_path):
        a = _norm_text(r.get("pl_concept_a"))
        b = _norm_text(r.get("pl_concept_b"))
        s = _safe_float(r.get("score"), 0.0)
        if not a or not b:
            continue
        pl_adjacency.setdefault(a, []).append((b, s))
        pl_adjacency.setdefault(b, []).append((a, s))

    pl_light_map: Dict[str, List[Tuple[str, str, float]]] = {}
    for r in _read_csv_dicts(pl_map_path):
        tok = _norm_text(r.get("gpg_token"))
        if not tok:
            continue
        pl_light_map.setdefault(tok, []).append(
            (
                _norm_text(r.get("motor_category")),
                _norm_text(r.get("motor_subcategory")),
                _safe_float(r.get("score"), 0.0),
            )
        )

    category_alias: Dict[str, str] = {}
    for r in _read_csv_dicts(alias_path):
        raw = _norm_text(r.get("raw"))
        canon = _norm_text(r.get("canon"))
        if raw and canon:
            category_alias[raw] = canon

    weak_token_weight: Dict[str, float] = {}
    for r in _read_csv_dicts(weak_tokens_path):
        term = _norm_text(r.get("term"))
        if term:
            weak_token_weight[term] = _safe_float(r.get("weight"), 0.2)

    strong_token_weight: Dict[str, float] = {}
    for r in _read_csv_dicts(strong_tokens_path):
        term = _norm_text(r.get("term"))
        if term:
            strong_token_weight[term] = _safe_float(r.get("weight"), 2.0)

    kb.update(
        {
            "norm_map": norm_map,
            "concept_map": concept_map,
            "weight_map": weight_map,
            "norm_to_concepts": norm_to_concepts,
            "adjacency": adjacency,
            "guard": guard,
            "stopwords": stopwords,
            "pl_norm_map": pl_norm_map,
            "pl_concept_map": pl_concept_map,
            "pl_adjacency": pl_adjacency,
            "pl_light_map": pl_light_map,
            "pl_enabled": bool(pl_norm_map or pl_adjacency or pl_light_map),
            "category_alias": category_alias,
            "weak_token_weight": weak_token_weight,
            "strong_token_weight": strong_token_weight,
        }
    )
    if tmp_dir:
        tmp_dir.cleanup()
    return kb


def _normalize_tokens(text: str, kb: Dict[str, Any]) -> List[str]:
    norm_map: Dict[str, str] = kb.get("norm_map", {})
    return [norm_map.get(t, t) for t in _tokens(text)]


def _concepts_from_tokens(tokens: Sequence[str], kb: Dict[str, Any]) -> Set[str]:
    concept_map: Dict[str, str] = kb.get("concept_map", {})
    norm_to_concepts: Dict[str, Set[str]] = kb.get("norm_to_concepts", {})
    concepts: Set[str] = set()
    for t in tokens:
        c = _norm_text(concept_map.get(t))
        if c:
            concepts.add(c)
        for c2 in norm_to_concepts.get(t, set()):
            if c2:
                concepts.add(_norm_text(c2))
    return concepts


def _clean_gpg_tokens(gpg: str, kb: Dict[str, Any]) -> List[str]:
    stopwords: Set[str] = kb.get("stopwords", set())
    raw = _normalize_tokens(gpg, kb)
    return [t for t in raw if t and t not in stopwords]


def _infer_old_domain(gpg_tokens: Sequence[str]) -> str:
    txt = " ".join(gpg_tokens)
    if any(k in txt for k in ["engine", "fuel", "tool"]):
        return "engine_fuel_tool"
    if any(k in txt for k in ["chassis", "train", "transmission", "power"]):
        return "power_train_chassis"
    if "electrical" in txt or "wire" in txt:
        return "electrical"
    if "body" in txt or "door" in txt:
        return "body"
    if any(k in txt for k in ["fastener", "bolt", "nut", "screw", "washer"]):
        return "fasteners"
    return "unknown"


def _category_state_from_guard(category: str, old_domain: str, kb: Dict[str, Any]) -> Tuple[str, float]:
    rec = kb.get("guard", {}).get((_norm_text(category), _norm_text(old_domain)))
    if not rec:
        return "PASS", 0.0
    mode = _norm_text(rec.get("mode"))
    penalty = _safe_float(rec.get("penalty"), 0.0)
    if mode == "deny":
        return "HARD-FAIL", max(0.0, penalty)
    if mode == "penalize":
        return "UNCERTAIN", max(0.0, penalty)
    return "PASS", 0.0


def _concept_link_score(a_concepts: Set[str], b_concepts: Set[str], kb: Dict[str, Any]) -> float:
    if not a_concepts or not b_concepts:
        return 0.0
    adjacency = kb.get("adjacency", {})
    score = 0.0
    for c in a_concepts:
        for other, s in adjacency.get(c, []):
            if other in b_concepts:
                score += _safe_float(s, 0.0)
    return score


def _weighted_overlap(desc_tokens: Sequence[str], sub_tokens: Sequence[str], kb: Dict[str, Any]) -> float:
    overlap = set(desc_tokens) & set(sub_tokens)
    if not overlap:
        return 0.0
    wmap = kb.get("weight_map", {})
    weak = kb.get("weak_token_weight", {})
    strong = kb.get("strong_token_weight", {})
    total = 0.0
    for tok in overlap:
        w = _safe_float(wmap.get(tok, 1.0), 1.0)
        if tok in weak:
            w *= _safe_float(weak.get(tok, 0.2), 0.2)
        if tok in strong:
            w *= _safe_float(strong.get(tok, 2.0), 2.0)
        total += w
    return total


def safe_percentiles(values: List[float]) -> List[int]:
    if not values:
        return []
    if np is not None:
        arr = np.array(values, dtype=float)
        order = arr.argsort().argsort()
        denom = max(1, len(values) - 1)
        return [int(round(100 * (x / denom))) for x in order]
    pairs = sorted((v, i) for i, v in enumerate(values))
    out = [0] * len(values)
    denom = max(1, len(values) - 1)
    for rank, (_, idx) in enumerate(pairs):
        out[idx] = int(round(100 * (rank / denom)))
    return out


def _extract_object_phrase(text: str) -> str:
    relation_terms = ["for", "with", "used", "used for", "w", "of"]
    rels = sorted(relation_terms, key=len, reverse=True)
    for r in rels:
        m = re.search(rf"\b{re.escape(r)}\b\s+([a-z0-9\-\s]{{2,60}})", text)
        if m:
            return m.group(1).strip()
    return ""


def score_pl_rows(rows: List[Dict[str, Any]], kb: Dict[str, Any]) -> Tuple[List[RowScore], int]:
    value_scores = safe_percentiles([_safe_float(r.get("value1"), 0.0) for r in rows])
    row_scores: List[RowScore] = []

    for i, row in enumerate(rows):
        desc_all = f"{_to_text(row.get('PNCDesc'))} {_to_text(row.get('PartDescription'))}".strip().lower()
        gpg = _to_text(row.get("SubCategory_GPG")).lower()
        subcategory = _to_text(row.get("SubCategory"))
        category_raw = _to_text(row.get("Category"))
        category_alias = kb.get("category_alias", {})
        category = category_alias.get(_norm_text(category_raw), _norm_text(category_raw))

        desc_tokens = _normalize_tokens(desc_all, kb)
        desc_concepts = _concepts_from_tokens(desc_tokens, kb)
        gpg_tokens = _clean_gpg_tokens(gpg, kb)
        gpg_concepts = _concepts_from_tokens(gpg_tokens, kb)
        sub_tokens = _normalize_tokens(subcategory.lower(), kb)
        sub_concepts = _concepts_from_tokens(sub_tokens, kb)

        obj = _extract_object_phrase(desc_all)
        obj_tokens = _normalize_tokens(obj, kb)
        obj_concepts = _concepts_from_tokens(obj_tokens, kb)
        has_object = bool(obj_tokens)

        old_domain = _infer_old_domain(gpg_tokens)
        c_state, guard_penalty = _category_state_from_guard(category, old_domain, kb)

        w_overlap = _weighted_overlap(desc_tokens, sub_tokens, kb)
        c_overlap = len(desc_concepts & sub_concepts)
        near = _concept_link_score(desc_concepts, sub_concepts, kb)

        if c_state == "HARD-FAIL":
            d_state = "OUT"
        elif (w_overlap >= 1.2) or (c_overlap >= 1):
            d_state = "HIT"
        elif near >= 1.0:
            d_state = "NEAR"
        else:
            d_state = "OUT"

        generic_fastener = "generic_fastener" in desc_concepts or any(t in desc_tokens for t in ["bolt", "nut", "screw", "washer", "fastener"])
        sub_fastener = "fastener" in " ".join(sub_tokens)

        score_desc = 35
        if c_state == "PASS":
            score_desc += 15
        elif c_state == "UNCERTAIN":
            score_desc += 6
        if d_state == "HIT":
            score_desc += 30
        elif d_state == "NEAR":
            score_desc += 15
        if has_object:
            score_desc += 6
            if len(obj_tokens) >= 2:
                score_desc += 4
        else:
            score_desc -= 4
        score_desc -= int(round(guard_penalty))
        score_desc = max(0, min(100, score_desc))

        term_base = 25 + min(40, int(round(w_overlap * 10 + c_overlap * 15 + near * 6)))
        gpg_overlap = len(gpg_concepts & sub_concepts)
        gpg_near = _concept_link_score(gpg_concepts, sub_concepts, kb)
        if gpg_overlap > 0:
            gpg_delta = 8 + min(2, gpg_overlap)
        elif gpg_near >= 1.2:
            gpg_delta = min(6, int(round(gpg_near * 2)))
        else:
            gpg_delta = -3 if gpg_tokens else 0
        gpg_delta = max(int(kb["scoring_defaults"].get("gpg_delta_min", -4)), min(int(kb["scoring_defaults"].get("gpg_delta_max", 10)), gpg_delta))

        pl_delta = 0
        if kb.get("pl_enabled"):
            pl_max_bonus = int(kb["scoring_defaults"].get("pl_light_max_bonus", 6))
            pl_max_penalty = int(kb["scoring_defaults"].get("pl_light_max_penalty", 3))
            pl_map = kb.get("pl_light_map", {})
            cat_n = _norm_text(category)
            sub_n = _norm_text(subcategory)
            best_bonus = 0
            best_penalty = 0
            for tok in gpg_tokens:
                for m_cat, m_sub, m_score in pl_map.get(tok, []):
                    if m_score > 1.0:
                        # 关键修复：兼容 1~6 刻度分值，先按 bonus 上限归一化
                        s = max(0.0, min(1.0, m_score / float(max(1, pl_max_bonus))))
                    else:
                        s = max(0.0, min(1.0, m_score))
                    if m_cat == cat_n and m_sub == sub_n:
                        best_bonus = max(best_bonus, int(round(pl_max_bonus * s)))
                    else:
                        best_penalty = max(best_penalty, int(round(pl_max_penalty * s)))
            pl_delta = best_bonus if best_bonus > 0 else -best_penalty

        score_term = max(0, min(100, term_base + gpg_delta + pl_delta - int(round(guard_penalty / 2))))
        score_value = value_scores[i]

        w_desc = float(kb["scoring_defaults"].get("score_desc_weight", 0.60))
        w_term = float(kb["scoring_defaults"].get("score_term_weight", 0.30))
        w_value = float(kb["scoring_defaults"].get("score_value_weight", 0.10))
        like_raw = int(round(w_desc * score_desc + w_term * score_term + w_value * score_value))

        old_strength = (0.80 * gpg_overlap + 0.50 * gpg_near)
        strength = (1.00 * w_overlap + 1.20 * c_overlap + 0.60 * near) + (0.25 * old_strength) + (0.30 if has_object else 0)
        k = 0.55
        sat = 1 - math.exp(-k * max(0, strength))
        if d_state == "HIT":
            conf = 0.55 + 0.45 * sat
        elif d_state == "NEAR":
            conf = 0.25 + 0.35 * sat
        else:
            conf = 0.05 + 0.20 * sat

        if c_state == "UNCERTAIN":
            conf *= 0.85
        elif c_state == "HARD-FAIL":
            conf *= 0.20
        conf *= max(0.0, 1.0 - guard_penalty / 18)
        if generic_fastener and not has_object:
            conf *= 0.60
        if generic_fastener and (not sub_fastener):
            conf *= 0.80
        conf = max(0.0, min(1.0, conf))

        like = round(1 + conf * (like_raw - 1))
        like = max(1, min(100, like))
        if c_state == "HARD-FAIL":
            like = min(like, 15)

        row_scores.append(
            RowScore(
                row=row,
                like=like,
                score_desc=score_desc,
                score_term=score_term,
                score_value=score_value,
                category_state=c_state,
                domain_state=d_state,
                has_generic=generic_fastener,
                has_object=has_object,
            )
        )

    removed_dup = dedupe_by_spn(row_scores)
    return row_scores, removed_dup


def dedupe_by_spn(rows: List[RowScore]) -> int:
    by_spn: Dict[str, RowScore] = {}
    removed = 0
    for rs in rows:
        spn = _norm_text(rs.row.get("SearchPartNumber"))
        if not spn:
            continue
        old = by_spn.get(spn)
        if old is None:
            by_spn[spn] = rs
            continue
        keep_new = (rs.like > old.like) or (rs.like == old.like and rs.score_value > old.score_value)
        if keep_new:
            old.row["_dedup_drop"] = True
            by_spn[spn] = rs
        else:
            rs.row["_dedup_drop"] = True
        removed += 1
    return removed


def pick_ispick(rows: List[RowScore], global_picked_spn: Set[str]) -> None:
    for r in rows:
        r.row["_ispick"] = None
    tier1 = [r for r in rows if r.like >= 15 and not r.row.get("_dedup_drop") and r.domain_state == "HIT"]
    tier2 = [r for r in rows if r.like >= 15 and not r.row.get("_dedup_drop") and r.domain_state == "NEAR"]
    tier3 = [r for r in rows if r.like >= 15 and not r.row.get("_dedup_drop") and r.has_generic and not r.has_object and r.like <= 35]

    picked: List[RowScore] = []

    def fill(pool: List[RowScore], allow_t3: bool = False):
        nonlocal picked
        pool = sorted(pool, key=lambda x: (x.like, x.score_desc, x.score_term, x.score_value), reverse=True)
        while pool and len(picked) < 10:
            best = pool[0]
            best_like = best.like
            choose = best

            # 覆盖优先：在不降质（最多让步3分）前提下优先选未覆盖SPN
            for cand in pool:
                cand_spn = _norm_text(cand.row.get("SearchPartNumber"))
                if any(_norm_text(x.row.get("SearchPartNumber")) == cand_spn for x in picked):
                    continue
                if cand_spn and cand_spn not in global_picked_spn and cand.like >= best_like - 3:
                    choose = cand
                    break

            pool.remove(choose)
            choose_spn = _norm_text(choose.row.get("SearchPartNumber"))
            if any(_norm_text(x.row.get("SearchPartNumber")) == choose_spn for x in picked):
                continue
            if not allow_t3 and choose in tier3:
                continue
            picked.append(choose)
            if choose_spn:
                global_picked_spn.add(choose_spn)

    fill(tier1)
    fill(tier2)
    if len(picked) < 10:
        fill(tier3, allow_t3=True)

    for p in picked[:10]:
        p.row["_ispick"] = 1


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("PL Batch Like/ispick Processor")
        self.kb_path = tk.StringVar(value="kb")
        self.kb = load_kb(self.kb_path.get())
        self.log_var = tk.StringVar(value="Ready")
        self.progress_var = tk.StringVar(value="Stage: idle")
        self.pl_var = tk.StringVar(value="PL: 0/0")
        self.running = False
        self.pause_event = threading.Event()
        self.stop_requested = False
        self.worker_thread = None

        frm = ttk.Frame(root, padding=8)
        frm.grid(sticky="nsew")
        for i, label in enumerate(["Host", "User", "Password", "Database", "Schema", "Table", "KB Path"]):
            ttk.Label(frm, text=label).grid(row=i, column=0, sticky="w")
        self.host = tk.StringVar()
        self.user = tk.StringVar()
        self.pwd = tk.StringVar()
        self.db = tk.StringVar()
        self.schema = tk.StringVar(value="dbo")
        self.table = tk.StringVar()

        ttk.Entry(frm, textvariable=self.host, width=40).grid(row=0, column=1, sticky="ew")
        ttk.Entry(frm, textvariable=self.user, width=40).grid(row=1, column=1, sticky="ew")
        ttk.Entry(frm, textvariable=self.pwd, width=40, show="*").grid(row=2, column=1, sticky="ew")

        self.db_combo = ttk.Combobox(frm, textvariable=self.db, width=38, state="readonly")
        self.db_combo.grid(row=3, column=1, sticky="ew")
        self.db_combo.bind("<<ComboboxSelected>>", self.on_database_selected)

        self.schema_combo = ttk.Combobox(frm, textvariable=self.schema, width=38, state="readonly")
        self.schema_combo.grid(row=4, column=1, sticky="ew")

        self.table_combo = ttk.Combobox(frm, textvariable=self.table, width=38, state="readonly")
        self.table_combo.grid(row=5, column=1, sticky="ew")

        kb_row = ttk.Frame(frm)
        kb_row.grid(row=6, column=1, sticky="ew")
        kb_row.columnconfigure(0, weight=1)
        ttk.Entry(kb_row, textvariable=self.kb_path).grid(row=0, column=0, sticky="ew")
        ttk.Button(kb_row, text="Browse KB", command=self.browse_kb).grid(row=0, column=1, padx=(6, 0))
        ttk.Button(kb_row, text="Load KB", command=self.reload_kb).grid(row=0, column=2, padx=(6, 0))

        ttk.Button(frm, text="Load Databases", command=self.load_databases).grid(row=7, column=0, pady=4, sticky="w")
        ctl_row = ttk.Frame(frm)
        ctl_row.grid(row=7, column=1, pady=4, sticky="w")
        ttk.Button(ctl_row, text="Load Tables", command=self.load_tables).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(ctl_row, text="Start", command=self.start).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(ctl_row, text="Pause", command=self.pause_run).grid(row=0, column=2, padx=(0, 6))
        ttk.Button(ctl_row, text="Resume", command=self.resume_run).grid(row=0, column=3, padx=(0, 6))
        ttk.Button(ctl_row, text="Stop", command=self.stop_run).grid(row=0, column=4)
        ttk.Label(frm, textvariable=self.progress_var).grid(row=8, column=0, columnspan=2, sticky="w")
        ttk.Label(frm, textvariable=self.pl_var).grid(row=9, column=0, columnspan=2, sticky="w")
        ttk.Label(frm, textvariable=self.log_var).grid(row=10, column=0, columnspan=2, sticky="w")

        self.progress = ttk.Progressbar(frm, orient="horizontal", mode="determinate", maximum=100)
        self.progress.grid(row=11, column=0, columnspan=2, sticky="ew", pady=(2, 4))

        self.text = tk.Text(frm, width=100, height=18)
        self.text.grid(row=12, column=0, columnspan=2, sticky="nsew")
        frm.columnconfigure(1, weight=1)
        self.reload_kb()

    def ui(self, fn):
        self.root.after(0, fn)

    def log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        self.ui(lambda: (self.text.insert("end", line), self.text.see("end"), self.log_var.set(msg)))

    def set_progress_indeterminate(self, active: bool):
        def _u():
            if active:
                self.progress.configure(mode="indeterminate")
                self.progress.start(10)
            else:
                self.progress.stop()
                self.progress.configure(mode="determinate", value=0, maximum=100)
        self.ui(_u)

    def set_progress_value(self, current: int, total: int):
        def _u():
            self.progress.stop()
            self.progress.configure(mode="determinate", maximum=max(1, total), value=max(0, min(current, total)))
        self.ui(_u)

    def browse_kb(self):
        chosen_dir = filedialog.askdirectory(title="Select KB folder (contains kb_manifest.json)")
        if chosen_dir:
            self.kb_path.set(chosen_dir)
            return
        chosen_zip = filedialog.askopenfilename(
            title="Or select KB zip",
            filetypes=[("Zip files", "*.zip"), ("All files", "*.*")],
        )
        if chosen_zip:
            self.kb_path.set(chosen_zip)

    def reload_kb(self):
        self.kb = load_kb(self.kb_path.get().strip() or "kb")
        self.log(f"KB loaded: {self.kb.get('kb_name')} v{self.kb.get('version')} - {self.kb.get('description')}")

    def connect(self, database: Optional[str] = None):
        db_name = (database or self.db.get()).strip()
        conn_kwargs = dict(
            host=self.host.get().strip(),
            user=self.user.get().strip(),
            password=self.pwd.get(),
            tds_version="7.0",
        )
        if db_name:
            conn_kwargs["database"] = db_name
        return pymssql.connect(**conn_kwargs)

    def load_databases(self):
        try:
            with self.connect(database="master") as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT name FROM sys.databases WHERE state = 0 ORDER BY name")
                    dbs = [r[0] for r in cur.fetchall()]
            self.db_combo["values"] = dbs
            if dbs:
                self.db.set(dbs[0])
                self.on_database_selected(None)
            self.log(f"Loaded {len(dbs)} databases")
        except Exception as e:
            messagebox.showerror("Load databases failed", str(e))

    def on_database_selected(self, _event):
        self.table.set("")
        self.table_combo["values"] = []
        self.load_tables()

    def load_tables(self):
        if not self.db.get().strip():
            messagebox.showwarning("Select database", "请先加载并选择数据库")
            return
        try:
            with self.connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT TABLE_SCHEMA, TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
                        "WHERE TABLE_TYPE='BASE TABLE' ORDER BY TABLE_SCHEMA, TABLE_NAME"
                    )
                    table_pairs = cur.fetchall()
            schemas = sorted({r[0] for r in table_pairs})
            if schemas:
                self.schema_combo["values"] = schemas
                if self.schema.get() not in schemas:
                    self.schema.set(schemas[0])

            selected_schema = self.schema.get().strip()
            tables = [name for sch, name in table_pairs if sch == selected_schema]
            if not tables and table_pairs:
                selected_schema = table_pairs[0][0]
                self.schema.set(selected_schema)
                tables = [name for sch, name in table_pairs if sch == selected_schema]

            self.table_combo["values"] = tables
            if tables:
                self.table.set(tables[0])
            self.log(f"Loaded {len(table_pairs)} tables, schema={selected_schema}, options={len(tables)}")
        except Exception as e:
            messagebox.showerror("Load tables failed", str(e))

    def pause_run(self):
        if self.running:
            self.pause_event.set()
            self.log("Paused")

    def resume_run(self):
        if self.running:
            self.pause_event.clear()
            self.log("Resumed")

    def stop_run(self):
        if self.running:
            self.stop_requested = True
            self.pause_event.clear()
            self.ui(lambda: self.progress_var.set("Stage: stopping"))
            self.log("Stop requested")

    def _wait_if_paused_or_stop(self) -> bool:
        while self.pause_event.is_set() and not self.stop_requested:
            self.ui(lambda: self.progress_var.set("Stage: paused"))
            self.pause_event.wait(0.2)
        return not self.stop_requested

    def start(self):
        if self.running:
            return
        if not self.db.get().strip() or not self.table.get().strip() or not self.schema.get().strip():
            messagebox.showwarning("Missing selection", "请先选择数据库、Schema、表")
            return
        self.stop_requested = False
        self.pause_event.clear()
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def _worker(self):
        try:
            self.run_main()
        except Exception as e:
            self.log(f"ERROR: {e}")
            self.ui(lambda: messagebox.showerror("Run failed", str(e)))
        finally:
            self.pause_event.clear()
            self.running = False
            self.worker_thread = None

    def _validate_required_columns(self, conn, schema: str, table_name: str) -> Dict[str, str]:
        required = [
            "VehicleId_Motor", "Category", "SubCategory", "SearchPartNumber", "PNCDesc",
            "PartDescription", "value1", "PartNumber", "SubCategory_GPG", "like", "ispick"
        ]
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s",
                (schema, table_name),
            )
            raw_cols = [r[0] for r in cur.fetchall()]
        by_lower = {str(c).lower(): str(c) for c in raw_cols}
        miss = [c for c in required if c.lower() not in by_lower]
        if miss:
            raise RuntimeError(f"Missing required columns in {schema}.{table_name}: {', '.join(miss)}")
        return {c: by_lower[c.lower()] for c in required}

    def run_main(self):
        self.set_progress_indeterminate(True)
        schema = self.schema.get().strip()
        table = self.table.get().strip()
        mod_table = f"{table}_Mod"
        queue_file = Path(f"pl_queue_{schema}_{table}.csv")

        def _generate_queue(conn):
            self.ui(lambda: self.progress_var.set("Stage: generate PL queue"))
            with conn.cursor(as_dict=True) as cur:
                cur.execute(
                    f"SELECT VehicleId_Motor,Category,SubCategory,COUNT(*) AS row_cnt "
                    f"FROM [{schema}].[{mod_table}] GROUP BY VehicleId_Motor,Category,SubCategory"
                )
                queues = cur.fetchall()
            with queue_file.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=QUEUE_FIELDS)
                w.writeheader()
                for q in queues:
                    w.writerow({**q, "status": "PENDING", "processed_at": "", "notes": ""})
            self.log(f"Generated queue: {len(queues)} PL")

        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 AS x FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s", (schema, mod_table))
                mod_exists = cur.fetchone() is not None

            queue_exists = queue_file.exists() and os.access(queue_file, os.R_OK)
            has_trace = False
            if mod_exists:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT CASE WHEN EXISTS(SELECT 1 FROM [{schema}].[{mod_table}] WHERE ispick=1 OR [like] IS NOT NULL) THEN 1 ELSE 0 END AS has_trace"
                    )
                    has_trace = bool(cur.fetchone()[0])

            resume_possible = queue_exists or mod_exists or has_trace
            if resume_possible:
                details = ["Detected previous progress:"]
                if queue_exists:
                    details.append(f"- {queue_file.name} exists")
                if mod_exists:
                    details.append(f"- table {schema}.{mod_table} exists")
                if has_trace:
                    details.append("- _Mod has processed traces (ispick=1 or like not null)")
                details.append("Continue from last progress?")
                resume_choice = messagebox.askyesnocancel("Resume?", "\n".join(details))
                if resume_choice is None:
                    self.log("User cancelled; exiting without changes")
                    self.ui(self.root.destroy)
                    return
                resume_mode = bool(resume_choice)
            else:
                resume_mode = False

            if not resume_mode:
                self.ui(lambda: self.progress_var.set("Stage: copy _Mod"))
                if mod_exists:
                    drop = messagebox.askyesno("_Mod exists", f"{schema}.{mod_table} already exists. Drop and recreate?")
                    if drop:
                        with conn.cursor() as cur:
                            cur.execute(f"DROP TABLE [{schema}].[{mod_table}]")
                        conn.commit()
                        mod_exists = False
                    else:
                        self.log("User canceled rebuild because _Mod exists")
                        self.set_progress_indeterminate(False)
                        return
                if not mod_exists:
                    with conn.cursor() as cur:
                        cur.execute(f"SELECT * INTO [{schema}].[{mod_table}] FROM [{schema}].[{table}]")
                    conn.commit()
                    self.log(f"Created {schema}.{mod_table}")
                colmap = self._validate_required_columns(conn, schema, mod_table)
                _generate_queue(conn)
                global_picked: Set[str] = set()
            else:
                self.log("Resume mode: keep existing _Mod and queue")
                colmap = self._validate_required_columns(conn, schema, mod_table)
                if not queue_exists:
                    _generate_queue(conn)
                with conn.cursor(as_dict=True) as cur:
                    cur.execute(f"SELECT DISTINCT [{colmap['SearchPartNumber']}] AS SearchPartNumber FROM [{schema}].[{mod_table}] WHERE [{colmap['ispick']}]=1")
                    global_picked = {_norm_text(r["SearchPartNumber"]) for r in cur.fetchall() if r.get("SearchPartNumber")}

            self.ui(lambda: self.progress_var.set("Stage: process PL loop"))
            queue_rows = list(csv.DictReader(queue_file.open("r", encoding="utf-8")))
            total = len(queue_rows)
            processed = 0
            duplicates_removed = 0
            dup_pl_count = 0
            like_counter = Counter()
            selected_count = 0
            full_10_count = 0
            empty_pl_count = 0

            for i, q in enumerate(queue_rows, start=1):
                if not self._wait_if_paused_or_stop():
                    self.log("Stopped by user")
                    break
                stop_pl = False
                if q["status"] == "DONE":
                    processed += 1
                    continue
                key = (q["VehicleId_Motor"], q["Category"], q["SubCategory"])
                self.ui(lambda i=i, total=total, key=key, rc=q['row_cnt']: self.pl_var.set(f"PL: {i}/{total} {key} rows={rc}"))
                try:
                    with conn.cursor(as_dict=True) as cur:
                        cur.execute(
                            f"SELECT [{colmap['VehicleId_Motor']}] AS VehicleId_Motor, [{colmap['Category']}] AS Category, [{colmap['SubCategory']}] AS SubCategory, "
                            f"[{colmap['PNCDesc']}] AS PNCDesc, [{colmap['PartNumber']}] AS PartNumber, [{colmap['SubCategory_GPG']}] AS SubCategory_GPG, "
                            f"[{colmap['SearchPartNumber']}] AS SearchPartNumber, [{colmap['PartDescription']}] AS PartDescription, [{colmap['value1']}] AS value1 "
                            f"FROM [{schema}].[{mod_table}] WHERE [{colmap['VehicleId_Motor']}]=%s AND [{colmap['Category']}]=%s AND [{colmap['SubCategory']}]=%s",
                            key,
                        )
                        rows = cur.fetchall()
                    if self.stop_requested:
                        self.log("Stopped by user before scoring")
                        break
                    scored, dedup_removed = score_pl_rows(rows, self.kb)
                    duplicates_removed += dedup_removed
                    if dedup_removed > 0:
                        dup_pl_count += 1
                    pick_ispick(scored, global_picked)
                    picked_now = sum(1 for r in scored if r.row.get("_ispick") == 1)
                    selected_count += picked_now
                    if picked_now == 10:
                        full_10_count += 1
                    if picked_now == 0:
                        empty_pl_count += 1

                    with conn.cursor() as cur:
                        for rs in scored:
                            if self.stop_requested:
                                conn.rollback()
                                q["status"] = "PENDING"
                                q["notes"] = "stopped_before_commit"
                                stop_pl = True
                                break
                            like = rs.like
                            ispick = rs.row.get("_ispick")
                            params = (
                                like,
                                ispick,
                                rs.row.get("VehicleId_Motor"),
                                rs.row.get("Category"),
                                rs.row.get("SubCategory"),
                                rs.row.get("PNCDesc"),
                                rs.row.get("PNCDesc"),
                                rs.row.get("PartNumber"),
                                rs.row.get("PartNumber"),
                                rs.row.get("SubCategory_GPG"),
                                rs.row.get("SubCategory_GPG"),
                                rs.row.get("SearchPartNumber"),
                                rs.row.get("SearchPartNumber"),
                            )
                            sql = (
                                f"UPDATE [{schema}].[{mod_table}] SET [{colmap['like']}]=%s, [{colmap['ispick']}]=%s "
                                f"WHERE [{colmap['VehicleId_Motor']}]=%s AND [{colmap['Category']}]=%s AND [{colmap['SubCategory']}]=%s "
                                f"AND (([{colmap['PNCDesc']}]=%s) OR ([{colmap['PNCDesc']}] IS NULL AND %s IS NULL)) "
                                f"AND (([{colmap['PartNumber']}]=%s) OR ([{colmap['PartNumber']}] IS NULL AND %s IS NULL)) "
                                f"AND (([{colmap['SubCategory_GPG']}]=%s) OR ([{colmap['SubCategory_GPG']}] IS NULL AND %s IS NULL)) "
                                f"AND (([{colmap['SearchPartNumber']}]=%s) OR ([{colmap['SearchPartNumber']}] IS NULL AND %s IS NULL))"
                            )
                            cur.execute(sql, params)
                            if cur.rowcount == 0:
                                note = (q.get("notes") or "") + "|miss_update"
                                q["notes"] = note[:240]
                            elif cur.rowcount > 1:
                                raise RuntimeError("Unsafe UPDATE matched >1 row; stop current PL")
                            like_counter[self._bucket(like)] += 1
                        if stop_pl:
                            break
                        conn.commit()

                    if stop_pl:
                        self.log("Stopped by user during writeback")
                        break
                    q["status"] = "DONE"
                    q["processed_at"] = datetime.now().isoformat(timespec="seconds")
                    q["notes"] = ""
                    processed += 1
                except Exception as e:
                    q["status"] = "FAIL"
                    q["notes"] = str(e)[:240]
                    conn.rollback()
                self.set_progress_value(i, total)
                self._save_queue(queue_file, queue_rows)

            self.set_progress_indeterminate(False)
            self._save_queue(queue_file, queue_rows)
            avg_pick = round(selected_count / total, 2) if total else 0
            summary = (
                f"总PL={total}, DONE={processed}, ispick=1总行数={selected_count}, 未选中PL数={empty_pl_count}, "
                f"选满10个PL数={full_10_count}, 平均每PL选中数={avg_pick}\n"
                f"PL内SPN重复剔除行数={duplicates_removed}, 涉及PL数={dup_pl_count}\n"
                f"like分布: <=15={like_counter['<=15']} 16-20={like_counter['16-20']} 21-35={like_counter['21-35']} "
                f"36-60={like_counter['36-60']} 61-80={like_counter['61-80']} 81-100={like_counter['81-100']}"
            )
            self.log(summary)

    @staticmethod
    def _save_queue(path: Path, rows: List[Dict[str, str]]):
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=QUEUE_FIELDS)
            w.writeheader()
            w.writerows(rows)

    @staticmethod
    def _bucket(v: int) -> str:
        if v <= 15:
            return "<=15"
        if v <= 20:
            return "16-20"
        if v <= 35:
            return "21-35"
        if v <= 60:
            return "36-60"
        if v <= 80:
            return "61-80"
        return "81-100"


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
