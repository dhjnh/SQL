import os
import re
import csv
import json
import time
import collections
import queue
import threading
import warnings
import zipfile
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set, Optional

import pandas as pd
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from tkinter import filedialog
import pymssql

# ============================
# SQL连接方式：对齐你的一体化V2.py
# pymssql.connect(host=..., user=..., password=..., tds_version="7.0", database=可选)
# ============================
TDS_VERSION = "7.0"
DONE_SUFFIX = "_Done"
CHUNK_VALUES = 500  # UPDATE VALUES 每批行数
DEFAULT_KB_DIR = "./kb_v4.3_local_flex"

PUNCT_RE = re.compile(r"[-_/()\[\],\.]")
SPACE_RE = re.compile(r"\s+")


def _as_lower_set(items: List[str]) -> Set[str]:
    return {str(x).strip().lower() for x in items if str(x).strip()}


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"JSON parse failed: {path} ({e})") from e


def _read_txt_set(path: Path) -> Set[str]:
    out = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.add(s.lower())
    return out


def _read_alias_csv(path: Path) -> Dict[str, str]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            rows.append([c.strip() for c in row])
    if not rows:
        return {}

    start = 0
    first = [c.lower() for c in rows[0]]
    if len(first) >= 2 and first[0] == "from" and first[1] == "to":
        start = 1

    alias = {}
    for row in rows[start:]:
        if len(row) < 2:
            continue
        src = row[0].strip().lower()
        dst = row[1].strip().lower()
        if src and dst:
            alias[src] = dst
    return alias


def _validate_required(mapping: Dict[str, Any], keys: List[str], ctx: str):
    for k in keys:
        if k not in mapping:
            raise RuntimeError(f"Missing required field '{k}' in {ctx}")


def load_kb(kb_dir: str) -> Dict[str, Any]:
    base = Path(kb_dir)
    manifest_path = base / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"KB manifest not found: {manifest_path}")

    manifest = _read_json(manifest_path)
    _validate_required(manifest, ["version", "pl_cap", "files"], str(manifest_path))
    files = manifest["files"]
    if not isinstance(files, dict):
        raise RuntimeError("manifest.files must be an object")

    _validate_required(
        files,
        [
            "domains",
            "domain_boost_rules",
            "domain_relations",
            "markers",
            "ngram_rules",
            "token_alias",
            "lists",
        ],
        "manifest.files",
    )

    lists_mapping = files["lists"]
    if not isinstance(lists_mapping, dict):
        raise RuntimeError("manifest.files.lists must be an object")

    required_list_keys = [
        "noise",
        "loc_tokens",
        "form_tokens",
        "fastener_tokens",
        "standard_words",
        "plural_except",
        "key_whitelist",
        "weak_overlap",
        "legacy_high_extra",
        "hvac_interior_cues",
    ]
    _validate_required(lists_mapping, required_list_keys, "manifest.files.lists")

    if ("compat_ignore" not in lists_mapping) and ("compat_ignore_extra" not in lists_mapping):
        raise RuntimeError("manifest.files.lists must contain compat_ignore or compat_ignore_extra")

    def fp(rel: str) -> Path:
        p2 = base / rel
        if not p2.exists():
            raise RuntimeError(f"KB file not found: {p2}")
        return p2

    domains = _read_json(fp(files["domains"]))
    domain_boost_rules_raw = _read_json(fp(files["domain_boost_rules"]))
    domain_relations = _read_json(fp(files["domain_relations"]))
    markers = _read_json(fp(files["markers"]))
    ngram_rules = _read_json(fp(files["ngram_rules"]))
    token_alias = _read_alias_csv(fp(files["token_alias"]))

    list_sets = {k: _read_txt_set(fp(v)) for k, v in lists_mapping.items()}

    if not isinstance(domains, dict) or not domains:
        raise RuntimeError("domains.json must be a non-empty object")
    for dom, toks in domains.items():
        if not isinstance(toks, list):
            raise RuntimeError(f"domains[{dom}] must be a list")

    canonical_domains = {str(k).lower(): str(k) for k in domains.keys()}

    if isinstance(domain_boost_rules_raw, list):
        domain_boost_rules = collections.defaultdict(list)
        for i, r in enumerate(domain_boost_rules_raw):
            if not isinstance(r, dict):
                raise RuntimeError(f"domain_boost_rules[{i}] must be object")
            dom = r.get("domain")
            if not dom:
                raise RuntimeError(f"domain_boost_rules[{i}] missing domain")
            dkey = str(dom).lower()
            if dkey not in canonical_domains:
                raise RuntimeError(f"domain_boost_rules[{i}] domain not in domains: {dom}")
            norm_dom = canonical_domains[dkey]
            r2 = dict(r)
            r2["domain"] = norm_dom
            domain_boost_rules[norm_dom].append(r2)
        domain_boost_rules = dict(domain_boost_rules)
    elif isinstance(domain_boost_rules_raw, dict):
        domain_boost_rules = {}
        for dom, rules in domain_boost_rules_raw.items():
            if not isinstance(rules, list):
                raise RuntimeError(f"domain_boost_rules[{dom}] must be a list")
            dkey = str(dom).lower()
            if dkey not in canonical_domains:
                raise RuntimeError(f"domain_boost_rules domain not in domains: {dom}")
            norm_dom = canonical_domains[dkey]
            out_rules = []
            for i, rule in enumerate(rules):
                if not isinstance(rule, dict):
                    raise RuntimeError(f"domain_boost_rules[{dom}][{i}] must be object")
                if "domain" in rule and str(rule["domain"]).lower() != dkey:
                    raise RuntimeError(
                        f"domain_boost_rules[{dom}][{i}].domain mismatch: {rule['domain']}"
                    )
                r2 = dict(rule)
                r2["domain"] = norm_dom
                out_rules.append(r2)
            domain_boost_rules[norm_dom] = out_rules
    else:
        raise RuntimeError("domain_boost_rules.json must be an object or list")

    for dom, rules in domain_boost_rules.items():
        for i, rule in enumerate(rules):
            if not isinstance(rule, dict):
                raise RuntimeError(f"domain_boost_rules[{dom}][{i}] must be object")
            terms = rule.get("tokens") or rule.get("terms") or []
            if not isinstance(terms, (list, tuple, set)):
                raise RuntimeError(f"domain_boost_rules[{dom}][{i}] tokens/terms must be list-like")
            rule["_terms_set"] = {str(x).lower() for x in terms if str(x).strip()}

    _validate_required(
        domain_relations,
        [
            "strong_domains",
            "neighbor_domains",
            "candidate_fallback_neighbors",
            "special_domain_names",
        ],
        "domain_relations.json",
    )
    _validate_required(domain_relations["special_domain_names"], ["fastener", "unknown"], "domain_relations.special_domain_names")

    if not isinstance(markers, dict):
        raise RuntimeError("markers.json must be an object")
    if not isinstance(ngram_rules, list):
        raise RuntimeError("ngram_rules.json must be a list")

    normalized_ngrams = []
    for i, rule in enumerate(ngram_rules):
        if not isinstance(rule, dict):
            raise RuntimeError(f"ngram_rules[{i}] must be an object")
        _validate_required(rule, ["add"], f"ngram_rules[{i}]")
        all_terms = rule.get("all", [])
        any_groups = rule.get("any", [])
        if "all" in rule and not isinstance(all_terms, list):
            raise RuntimeError(f"ngram_rules[{i}].all must be list")
        if "any" in rule:
            if not isinstance(any_groups, list):
                raise RuntimeError(f"ngram_rules[{i}].any must be list[list/tuple]")
            for j, grp in enumerate(any_groups):
                if not isinstance(grp, (list, tuple)):
                    raise RuntimeError(f"ngram_rules[{i}].any[{j}] must be list/tuple")
        if (not all_terms) and (not any_groups):
            raise RuntimeError(f"ngram_rules[{i}] cannot have both all/any empty")

        nr = {
            "add": str(rule["add"]).lower(),
            "all": [str(x).lower() for x in all_terms],
            "any": [[str(x).lower() for x in grp] for grp in any_groups],
            "any_sets": [{str(x).lower() for x in grp} for grp in any_groups],
        }
        normalized_ngrams.append(nr)

    neighbors_raw = domain_relations["neighbor_domains"]
    if not isinstance(neighbors_raw, list):
        raise RuntimeError("domain_relations.neighbor_domains must be a list")
    neighbor_domains = set()
    for i, pair in enumerate(neighbors_raw):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise RuntimeError(f"domain_relations.neighbor_domains[{i}] must be length-2 list")
        a = str(pair[0]).lower()
        b = str(pair[1]).lower()
        neighbor_domains.add((a, b))
        neighbor_domains.add((b, a))

    cue_raw = domain_relations.get("cue_required_neighbor_domains", [])
    if "cue_required_neighbor_domains" in domain_relations and not isinstance(cue_raw, list):
        raise RuntimeError("domain_relations.cue_required_neighbor_domains must be a list")
    cue_required_neighbor_domains = set()
    for i, pair in enumerate(cue_raw):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise RuntimeError(f"domain_relations.cue_required_neighbor_domains[{i}] must be length-2 list")
        a = str(pair[0]).lower()
        b = str(pair[1]).lower()
        cue_required_neighbor_domains.add((a, b))
        cue_required_neighbor_domains.add((b, a))

    raw_fallback_neighbors = domain_relations["candidate_fallback_neighbors"]
    if not isinstance(raw_fallback_neighbors, dict):
        raise RuntimeError("domain_relations.candidate_fallback_neighbors must be an object")
    candidate_fallback_neighbors = {}
    for raw_k, raw_vals in raw_fallback_neighbors.items():
        k_lower = str(raw_k).lower()
        if k_lower not in canonical_domains:
            raise RuntimeError(f"domain_relations.candidate_fallback_neighbors key not in domains: {raw_k}")
        if not isinstance(raw_vals, list):
            raise RuntimeError(f"domain_relations.candidate_fallback_neighbors[{raw_k}] must be a list")
        key_canon = canonical_domains[k_lower].lower()
        vals = []
        for j, raw_v in enumerate(raw_vals):
            v_lower = str(raw_v).lower()
            if v_lower not in canonical_domains:
                raise RuntimeError(
                    f"domain_relations.candidate_fallback_neighbors[{raw_k}][{j}] not in domains: {raw_v}"
                )
            vals.append(canonical_domains[v_lower].lower())
        candidate_fallback_neighbors[key_canon] = vals

    noise = list_sets["noise"]
    loc_tokens = list_sets["loc_tokens"]
    form_tokens = list_sets["form_tokens"]
    fastener_tokens = list_sets["fastener_tokens"]
    compat_ignore = set()
    compat_ignore |= noise | loc_tokens | form_tokens | fastener_tokens
    compat_ignore |= list_sets.get("compat_ignore", set())
    compat_ignore |= list_sets.get("compat_ignore_extra", set())
    list_sets["compat_ignore"] = compat_ignore

    kb = {
        "version": str(manifest["version"]),
        "pl_cap": int(manifest["pl_cap"]),
        "domains": {k: _as_lower_set(v) for k, v in domains.items()},
        "domain_boost_rules": domain_boost_rules,
        "domain_relations": {
            "strong_domains": _as_lower_set(domain_relations["strong_domains"]),
            "neighbor_domains": neighbor_domains,
            "candidate_fallback_neighbors": candidate_fallback_neighbors,
            "special_domain_names": {
                "fastener": str(domain_relations["special_domain_names"]["fastener"]),
                "unknown": str(domain_relations["special_domain_names"]["unknown"]),
                "conflict": str(domain_relations["special_domain_names"].get("conflict", "Conflict")),
            },
            "cue_required_neighbor_domains": cue_required_neighbor_domains,
            "canonical_map": canonical_domains,
        },
        "markers": {str(k).lower(): _as_lower_set(v) for k, v in markers.items()},
        "ngram_rules": normalized_ngrams,
        "token_alias": {k.lower(): v.lower() for k, v in token_alias.items()},
        "lists": list_sets,
    }
    return kb



KB = None
NOISE = set()
LOC_TOKENS = set()
FORM_TOKENS = set()
FASTENER_TOKENS = set()
STANDARD_WORDS = set()
COMPAT_IGNORE = set()
PLURAL_EXCEPT = set()
TOKEN_ALIAS = {}
KEY_WHITELIST = set()
WEAK_OVERLAP = set()
PL_CAP = 10
UNKNOWN_NAME = "Unknown"
FASTENER_NAME = "Fastener/Standard"
CONFLICT_NAME = "Conflict"


def apply_kb(kb: Dict[str, Any]):
    global KB, NOISE, LOC_TOKENS, FORM_TOKENS, FASTENER_TOKENS, STANDARD_WORDS
    global COMPAT_IGNORE, PLURAL_EXCEPT, TOKEN_ALIAS, KEY_WHITELIST, WEAK_OVERLAP, PL_CAP
    global UNKNOWN_NAME, FASTENER_NAME, CONFLICT_NAME
    KB = kb
    NOISE = KB["lists"]["noise"]
    LOC_TOKENS = KB["lists"]["loc_tokens"]
    FORM_TOKENS = KB["lists"]["form_tokens"]
    FASTENER_TOKENS = KB["lists"]["fastener_tokens"]
    STANDARD_WORDS = KB["lists"]["standard_words"]
    COMPAT_IGNORE = KB["lists"]["compat_ignore"]
    PLURAL_EXCEPT = KB["lists"]["plural_except"]
    TOKEN_ALIAS = KB["token_alias"]
    KEY_WHITELIST = KB["lists"]["key_whitelist"]
    WEAK_OVERLAP = KB["lists"]["weak_overlap"]
    PL_CAP = KB["pl_cap"]
    UNKNOWN_NAME = KB["domain_relations"]["special_domain_names"]["unknown"]
    FASTENER_NAME = KB["domain_relations"]["special_domain_names"]["fastener"]
    CONFLICT_NAME = KB["domain_relations"]["special_domain_names"].get("conflict", "Conflict")
    _desc_cache.clear()
    _legacy_cache.clear()


def ensure_kb_loaded():
    if KB is None:
        raise RuntimeError("KB not loaded. Please load KB first.")


def _stable_text(v: Optional[str]) -> str:
    try:
        if v is None or pd.isna(v):
            return ""
    except Exception:
        if v is None:
            return ""
    return str(v)


def norm_text(s: Optional[str]) -> str:
    if s is None or pd.isna(s):
        return ""
    s = str(s).lower()
    s = s.replace("a/c", "ac").replace("a-c", "ac").replace("w/", "with ").replace("wo/", "without ")
    s = s.replace("p/s", "ps").replace("p-s", "ps")
    s = PUNCT_RE.sub(" ", s)
    return SPACE_RE.sub(" ", s).strip()


def norm_token(t: str) -> str:
    if not t:
        return t
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("s") and len(t) > 3 and (t not in PLURAL_EXCEPT) and (not t.endswith("ss")):
        return t[:-1]
    return t


def tokenize(s: Optional[str]) -> List[str]:
    ensure_kb_loaded()
    s = norm_text(s)
    if not s:
        return []
    toks = [TOKEN_ALIAS.get(norm_token(x), norm_token(x)) for x in s.split() if x]
    st = set(toks)

    for rule in KB["ngram_rules"]:
        all_terms = rule.get("all", [])
        any_groups = rule.get("any_sets", [])
        add = rule["add"]
        if all_terms and not all(t in st for t in all_terms):
            continue
        ok_any = True
        if any_groups:
            for gset in any_groups:
                if not (st & gset):
                    ok_any = False
                    break
        if ok_any:
            toks.append(add)
            st.add(add)
    return toks


def domain_tag(tokens: List[str]) -> str:
    ensure_kb_loaded()
    toks = set(tokens)
    hits = {}

    for dom, dset in KB["domains"].items():
        score = len(toks & dset)
        for r in KB["domain_boost_rules"].get(dom, []):
            mode = str(r.get("mode", "")).lower()
            add = int(r.get("add", 0))
            terms = r.get("_terms_set", set())
            if mode == "any" and (toks & terms):
                score += add
            elif mode == "all" and terms and terms.issubset(toks):
                score += add
        if score:
            hits[dom] = score

    if hits:
        top = sorted(hits.items(), key=lambda x: (x[1], x[0]), reverse=True)
        if len(top) > 1 and top[1][1] == top[0][1] and top[1][0] != top[0][0]:
            return CONFLICT_NAME
        return top[0][0]

    if toks & FASTENER_TOKENS:
        return FASTENER_NAME
    return UNKNOWN_NAME




def _dom_key(dom: str) -> str:
    return str(dom or "").lower()


def _canonical_dom_name(dom: str) -> str:
    ensure_kb_loaded()
    return KB["domain_relations"]["canonical_map"].get(_dom_key(dom), str(dom))

def key_tokens(tok_set: Set[str]) -> Set[str]:
    out = set()
    for t in tok_set:
        if (not t) or (t in COMPAT_IGNORE) or t.isdigit():
            continue
        if len(t) >= 3 or t in KEY_WHITELIST:
            out.add(t)
    return out


def _marker(name: str) -> Set[str]:
    ensure_kb_loaded()
    return KB["markers"].get(str(name).lower(), set())


def neighbor_allowed(dom_a: str, dom_b: str, desc_set: Set[str], pl_set: Set[str]) -> bool:
    a = _dom_key(dom_a)
    b = _dom_key(dom_b)
    if (a, b) not in KB["domain_relations"]["neighbor_domains"]:
        return False

    cue_pairs = KB["domain_relations"].get("cue_required_neighbor_domains", set())
    if (a, b) in cue_pairs:
        cues = KB["lists"]["hvac_interior_cues"]
        return bool((desc_set | pl_set) & cues)
    return True



@dataclass
class PLRec:
    pid: int
    vid: str
    cat: str
    sub: str
    remain: int
    dom: str
    toks: Set[str]
    subtoks: Set[str]
    key_toks: Set[str]
    key_subtoks: Set[str]
    is_sink: int


def pl_guard_reject(desc_set: Set[str], pl: "PLRec") -> bool:
    ps = set(pl.subtoks)
    airbag = _marker("airbag_markers")
    exhaust = _marker("exhaust_markers")
    trans = _marker("trans_markers")
    airinj = _marker("airinj_markers")
    intake = _marker("intake_markers")
    valvetrain = _marker("valvetrain_markers")
    eng_int = _marker("engine_internals")

    if ps & airbag and not (desc_set & airbag):
        return True
    if (desc_set & exhaust) and (_dom_key(pl.dom) == _dom_key(_canonical_dom_name("Engine/Powertrain"))) and (not (ps & exhaust)) and (not (ps & trans)):
        return True
    if desc_set & airinj:
        if _dom_key(pl.dom) in {_dom_key(_canonical_dom_name("Interior/Trim")), _dom_key(_canonical_dom_name("Body/Exterior"))}:
            return True
        if (ps & intake) and not (ps & airinj):
            return True
    if (desc_set & valvetrain) and (not (desc_set & intake)) and (ps & intake):
        return True
    if (desc_set & trans) and (ps & eng_int) and not (ps & trans):
        return True
    if (ps & eng_int) and not (desc_set & eng_int):
        return True
    if (ps & {"luggage", "carrier", "roofrack", "rack", "crossbar"}) and not (
        desc_set & {"luggage", "carrier", "roofrack", "rack", "crossbar", "roof"}
    ):
        return True
    if (ps & {"seattrack", "track", "rail", "slider", "adjuster", "seatrail"}) and not (
        desc_set & {"seattrack", "track", "rail", "slider", "adjuster", "seatrail", "seat"}
    ):
        return True
    if (ps & {"dash", "dashboard", "instrument", "cluster", "panel"}) and (
        desc_set & {"seattrack", "track", "rail", "slider", "adjuster", "seatrail"}
    ):
        return True
    if (desc_set & {"valvebody", "oilstrainer", "atf", "accumulator", "torqueconverter", "transmission", "automatic"}) and (
        _dom_key(pl.dom) in {_dom_key(_canonical_dom_name("Body/Exterior")), _dom_key(_canonical_dom_name("Interior/Trim"))}
    ):
        return True
    return False


_desc_cache = {}
_legacy_cache = {}
DESC_CACHE_MAX = 200000
LEGACY_CACHE_MAX = 200000


def desc_features(pnc: Optional[str], desc: Optional[str]):
    pnc_s = _stable_text(pnc)
    desc_s = _stable_text(desc)
    key = (pnc_s, desc_s)
    v = _desc_cache.get(key)
    if v is not None:
        return v
    toks = tokenize(pnc_s + " " + desc_s)
    dset = set(toks)
    has_loc = any(t in LOC_TOKENS for t in toks)
    fast_present = any(t in FASTENER_TOKENS for t in toks)
    dom = domain_tag(toks)
    generic = {
        "part",
        "component",
        "repair",
        "kit",
        "set",
        "assembly",
        "assy",
        "subassembly",
        "sub",
        "adjusting",
        "protector",
        "system",
        "control",
        "components",
        "support",
        "holder",
        "mount",
    }
    obj = [
        t
        for t in toks
        if t not in NOISE and t not in LOC_TOKENS and t not in FORM_TOKENS and len(t) > 2 and (not t.isdigit()) and t not in generic
    ]
    fast_no_loc = bool(fast_present and (not has_loc) and _dom_key(dom) == _dom_key(FASTENER_NAME))
    v = (toks, obj, dom, has_loc, fast_present, fast_no_loc, dset)
    if len(_desc_cache) > DESC_CACHE_MAX:
        _desc_cache.clear()
    _desc_cache[key] = v
    return v


def legacy_features(subcat_gpg: Optional[str]):
    key = _stable_text(subcat_gpg)
    v = _legacy_cache.get(key)
    if v is not None:
        return v
    s = key
    ps = s.split("/")
    segs = []
    if ps and ps[-1]:
        segs.append(ps[-1])
    if len(ps) >= 2 and ps[-2]:
        segs.append(ps[-2])
    if len(ps) >= 3 and ps[-3]:
        segs.append(ps[-3])
    toks = []
    for seg in segs:
        toks.extend(tokenize(seg))
    tset = set(toks)
    dom = domain_tag(toks)

    marker_names = [
        "cooling_markers",
        "steering_markers",
        "trans_markers",
        "engine_internals",
        "valvetrain_markers",
        "intake_markers",
        "airbag_markers",
        "exhaust_markers",
        "airinj_markers",
    ]
    high_mark = set()
    for n in marker_names:
        high_mark |= _marker(n)
    high_mark |= KB["lists"]["legacy_high_extra"]

    if tset & high_mark:
        conf = 2
    elif _dom_key(dom) in {_dom_key(_canonical_dom_name("Body/Exterior")), _dom_key(_canonical_dom_name("Interior/Trim")), _dom_key(FASTENER_NAME)}:
        conf = 1
    elif dom.lower() in KB["domain_relations"]["strong_domains"]:
        conf = 2
    else:
        conf = 0
    v = (tset, dom, conf)
    if len(_legacy_cache) > LEGACY_CACHE_MAX:
        _legacy_cache.clear()
    _legacy_cache[key] = v
    return v


def build_pl_index(pl_df: pd.DataFrame):
    pl_df = pl_df.copy()
    sink_tokens = {"hydraulic", "pulleys", "pulley", "moldings", "molding", "system", "components"}
    if "allready_Pick" not in pl_df.columns:
        pl_df["allready_Pick"] = 0
    pl_df["allready_Pick"] = pd.to_numeric(pl_df["allready_Pick"], errors="coerce").fillna(0).astype(int)
    pl_df["RemainSlots"] = (PL_CAP - pl_df["allready_Pick"]).clip(lower=0).astype(int)
    pl_recs = {}
    pl_by_vehicle = collections.defaultdict(list)
    token_index = collections.defaultdict(lambda: collections.defaultdict(list))
    domain_sorted = collections.defaultdict(lambda: collections.defaultdict(list))
    std_pls = collections.defaultdict(set)
    for pid, r in enumerate(pl_df.itertuples(index=False), start=0):
        rem = int(getattr(r, "RemainSlots", 0))
        if rem <= 0:
            continue
        vid = _stable_text(getattr(r, "VehicleId_Motor", "")).strip()
        cat = _stable_text(getattr(r, "Category", "")).strip()
        sub = _stable_text(getattr(r, "SubCategory", "")).strip()
        toks_all = set(tokenize(cat + " " + sub))
        subtoks = set(tokenize(sub))
        dom = domain_tag(list(toks_all))
        kt = key_tokens(toks_all)
        ksub = key_tokens(subtoks)
        is_sink = 1 if (kt & sink_tokens) else 0
        rec = PLRec(
            pid=pid,
            vid=vid,
            cat=cat,
            sub=sub,
            remain=rem,
            dom=dom,
            toks=toks_all,
            subtoks=subtoks,
            key_toks=kt,
            key_subtoks=ksub,
            is_sink=is_sink,
        )
        pl_recs[pid] = rec
        pl_by_vehicle[vid].append(pid)
        for t in ksub:
            token_index[vid][t].append(pid)
        extra = [t for t in kt if (t not in ksub) and (t not in WEAK_OVERLAP) and (t not in COMPAT_IGNORE)]
        extra = sorted(extra, key=lambda x: (-len(x), x))[:6]
        for t in extra:
            token_index[vid][t].append(pid)
        if _dom_key(dom) == _dom_key(FASTENER_NAME) or (toks_all & STANDARD_WORDS):
            std_pls[vid].add(pid)
    for vid, pids in pl_by_vehicle.items():
        dm = collections.defaultdict(list)
        for pid in pids:
            dm[pl_recs[pid].dom].append(pid)
        for dom, pids2 in dm.items():
            domain_sorted[vid][dom] = sorted(pids2, key=lambda x: (pl_recs[x].remain, pl_recs[x].sub), reverse=True)
    return pl_recs, pl_by_vehicle, token_index, domain_sorted, std_pls


def candidate_pls(vid: str, obj_key: Set[str], dom_hint: str, fast_no_loc: bool, pl_by_vehicle, token_index, domain_sorted, std_pls, max_fallback=30, max_total=60):
    if fast_no_loc:
        cand = list(std_pls.get(vid, set())) or pl_by_vehicle.get(vid, [])
        return list(dict.fromkeys(cand))[:max_total]
    cand = []
    seen = set()
    for t in sorted(obj_key, key=lambda x: (-len(x), x))[:12]:
        for pid in token_index[vid].get(t, []):
            if pid in seen:
                continue
            seen.add(pid); cand.append(pid)
            if len(cand) >= max_total:
                break
        if len(cand) >= max_total:
            break
    if len(cand) < 12:
        for pid in domain_sorted[vid].get(dom_hint, [])[:max_fallback]:
            if pid in seen:
                continue
            seen.add(pid); cand.append(pid)
            if len(cand) >= max_total:
                break
        if len(cand) < 12:
            for nb in KB["domain_relations"]["candidate_fallback_neighbors"].get(_dom_key(dom_hint), []):
                for actual_dom, pids in domain_sorted[vid].items():
                    if _dom_key(actual_dom) != nb:
                        continue
                    for pid in pids[:max_fallback]:
                        if pid in seen:
                            continue
                        seen.add(pid); cand.append(pid)
                        if len(cand) >= max_total:
                            break
                    if len(cand) >= max_total:
                        break
                if len(cand) >= max_total:
                    break
        if not cand:
            for pid in pl_by_vehicle.get(vid, [])[:max_fallback]:
                if pid in seen:
                    continue
                seen.add(pid); cand.append(pid)
                if len(cand) >= max_total:
                    break
    return cand[:max_total]


def eval_option(row_feat, leg_feat, pl: PLRec, std_pls_for_vid: Set[int]):
    toks, obj, dom_desc, has_loc, fast_present, fast_no_loc, desc_set = row_feat
    legacy_set, legacy_dom, legacy_conf = leg_feat
    dom_evidence = legacy_dom if (dom_desc in (UNKNOWN_NAME, CONFLICT_NAME) and legacy_conf >= 2 and legacy_dom not in (UNKNOWN_NAME, CONFLICT_NAME)) else dom_desc
    if fast_no_loc and (pl.pid not in std_pls_for_vid):
        return None
    desc_key = key_tokens(set(obj))
    overlap_set = desc_key & pl.key_subtoks
    strong_overlap = len([t for t in overlap_set if t not in WEAK_OVERLAP])
    weak_only = len(overlap_set) > 0 and strong_overlap == 0
    compat_key = 1 if overlap_set else 0
    body_dom = _dom_key(_canonical_dom_name("Body/Exterior"))
    int_dom = _dom_key(_canonical_dom_name("Interior/Trim"))
    if _dom_key(pl.dom) == _dom_key(_canonical_dom_name("Suspension/Steering")) and (desc_set & _marker("cooling_markers")) and not (desc_set & _marker("steering_markers")):
        return None
    strong_domains = KB["domain_relations"]["strong_domains"]
    if _dom_key(dom_evidence) in strong_domains and _dom_key(pl.dom) != _dom_key(dom_evidence):
        if not neighbor_allowed(dom_evidence, pl.dom, desc_set, set(pl.subtoks)):
            return None
    if _dom_key(dom_evidence) in {body_dom, int_dom} and _dom_key(pl.dom) in strong_domains:
        if not neighbor_allowed(dom_evidence, pl.dom, desc_set, set(pl.subtoks)):
            return None
    if legacy_conf >= 2 and legacy_dom not in (UNKNOWN_NAME, CONFLICT_NAME):
        if dom_desc in (UNKNOWN_NAME, CONFLICT_NAME) or weak_only:
            if legacy_dom != pl.dom and not neighbor_allowed(legacy_dom, pl.dom, desc_set, set(pl.subtoks)):
                return None
    if pl_guard_reject(desc_set, pl):
        return None
    if dom_evidence not in (UNKNOWN_NAME, CONFLICT_NAME):
        same_dom = (_dom_key(pl.dom) == _dom_key(dom_evidence))
    elif legacy_conf >= 2 and legacy_dom not in (UNKNOWN_NAME, CONFLICT_NAME):
        same_dom = (_dom_key(pl.dom) == _dom_key(legacy_dom))
    else:
        same_dom = False
    if strong_overlap >= 2:
        lr = 3
    elif strong_overlap == 1:
        lr = 2
    elif compat_key == 1:
        lr = 1
    elif same_dom:
        lr = 1
    else:
        return None
    if lr == 1 and legacy_conf >= 2 and legacy_dom not in (UNKNOWN_NAME, CONFLICT_NAME) and _dom_key(pl.dom) != _dom_key(legacy_dom) and compat_key == 0:
        return None
    legacy_score = legacy_conf * 2
    return (lr, strong_overlap, legacy_score, legacy_conf, compat_key)


def legacy_bonus(conf: int, compat_key: int) -> int:
    if conf >= 2 and compat_key == 1:
        return 6
    if conf >= 2 and compat_key == 0:
        return 2
    if conf == 1 and compat_key == 1:
        return 4
    if conf == 1 and compat_key == 0:
        return 1
    if conf == 0 and compat_key == 1:
        return 1
    return 0


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


# ============================
# SQL helpers
# ============================
def _split_schema_table(s: str) -> Tuple[str, str]:
    s = s.strip()
    if "." not in s:
        return "dbo", s
    a, b = s.split(".", 1)
    return a.strip("[]"), b.strip("[]")


def _qt(schema: str, table: str) -> str:
    return f"[{schema}].[{table}]"


def connect_sql(host: str, user: str, password: str, database: Optional[str] = None):
    kw = {"host": host, "user": user, "password": password, "tds_version": TDS_VERSION}
    if database:
        kw["database"] = database
    return pymssql.connect(**kw)


def list_databases(host: str, user: str, password: str) -> List[str]:
    with connect_sql(host, user, password, "master") as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT name FROM sys.databases WHERE state = 0 ORDER BY name")
            return [r[0] for r in cur.fetchall()]


def list_tables(host: str, user: str, password: str, database: str) -> List[str]:
    with connect_sql(host, user, password, database) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT TABLE_SCHEMA, TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE' ORDER BY TABLE_SCHEMA, TABLE_NAME"
            )
            pairs = cur.fetchall()
    return [f"{a}.{b}" for a, b in pairs]


def table_exists(conn, schema: str, table: str) -> bool:
    q = "SELECT 1 FROM sys.tables t JOIN sys.schemas s ON t.schema_id=s.schema_id WHERE s.name=%s AND t.name=%s"
    with conn.cursor() as cur:
        cur.execute(q, (schema, table))
        return cur.fetchone() is not None




def list_columns(conn, schema: str, table: str) -> Set[str]:
    q = (
        "SELECT c.name FROM sys.columns c "
        "JOIN sys.tables t ON c.object_id=t.object_id "
        "JOIN sys.schemas s ON t.schema_id=s.schema_id "
        "WHERE s.name=%s AND t.name=%s"
    )
    with conn.cursor() as cur:
        cur.execute(q, (schema, table))
        return {str(r[0]) for r in cur.fetchall()}


def assert_columns(conn, schema: str, table: str, required: List[str], ctx: str):
    cols = list_columns(conn, schema, table)
    cols_lower = {c.lower() for c in cols}
    missing = [c for c in required if c.lower() not in cols_lower]
    if missing:
        raise RuntimeError(f"{ctx} missing columns={missing}; table={schema}.{table}")

def drop_table(conn, schema: str, table: str):
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE {_qt(schema, table)}")
    conn.commit()


def select_into(conn, src_schema: str, src_table: str, dst_schema: str, dst_table: str):
    with conn.cursor() as cur:
        cur.execute(f"SELECT * INTO {_qt(dst_schema, dst_table)} FROM {_qt(src_schema, src_table)}")
    conn.commit()


def ensure_rowid_and_index(conn, schema: str, table: str) -> str:
    obj = _qt(schema, table)
    obj_name = f"{schema}.{table}"
    idx_rowid = (f"IX_{table}__rowid")[:128]
    idx_done = (f"IX_{table}__rowid_done")[:128]
    with conn.cursor() as cur:
        cur.execute(f"IF COL_LENGTH('{obj_name}','__rowid') IS NULL ALTER TABLE {obj} ADD [__rowid] INT IDENTITY(1,1) NOT NULL;")
        cur.execute(f"SELECT CASE WHEN COUNT_BIG(*)=COUNT_BIG(DISTINCT [__rowid]) THEN 1 ELSE 0 END FROM {obj}")
        uniq_rowid = bool(cur.fetchone()[0])
        if uniq_rowid:
            cur.execute(
                "SELECT 1 FROM sys.indexes i JOIN sys.tables t ON i.object_id=t.object_id JOIN sys.schemas s ON t.schema_id=s.schema_id WHERE s.name=%s AND t.name=%s AND i.name=%s",
                (schema, table, idx_rowid),
            )
            has = cur.fetchone() is not None
            if not has:
                cur.execute(f"CREATE UNIQUE NONCLUSTERED INDEX [{idx_rowid}] ON {obj}([__rowid]);")
            conn.commit()
            return "__rowid"

        cur.execute(f"IF COL_LENGTH('{obj_name}','__rowid_done') IS NULL ALTER TABLE {obj} ADD [__rowid_done] INT IDENTITY(1,1) NOT NULL;")
        cur.execute(
            "SELECT 1 FROM sys.indexes i JOIN sys.tables t ON i.object_id=t.object_id JOIN sys.schemas s ON t.schema_id=s.schema_id WHERE s.name=%s AND t.name=%s AND i.name=%s",
            (schema, table, idx_done),
        )
        has_done = cur.fetchone() is not None
        if not has_done:
            cur.execute(f"CREATE UNIQUE NONCLUSTERED INDEX [{idx_done}] ON {obj}([__rowid_done]);")
    conn.commit()
    return "__rowid_done"


def load_df(conn, schema: str, table: str, cols: List[str]) -> pd.DataFrame:
    sel = ", ".join([f"[{c}]" for c in cols])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*pandas only supports SQLAlchemy connectable.*")
        return pd.read_sql(f"SELECT {sel} FROM {_qt(schema, table)}", conn)


def _update_done_values(
    cur,
    schema: str,
    table: str,
    rowid_key: str,
    params_list: List[Tuple[str, str, int, int]],
    chunk_rows: int = CHUNK_VALUES,
    progress=None,  # progress(cur_n:int, total_n:int)
):
    if not params_list:
        if progress:
            progress(0, 0)
        return

    head = (
        f"UPDATE t SET t.[Category]=v.[Category], t.[SubCategory]=v.[SubCategory], t.[like]=v.[like] "
        f"FROM {_qt(schema, table)} t JOIN (VALUES "
    )
    tail = f") v([Category],[SubCategory],[like],[{rowid_key}]) ON t.[{rowid_key}]=v.[{rowid_key}]"
    n = len(params_list)

    done = 0
    for off in range(0, n, chunk_rows):
        chunk = params_list[off : off + chunk_rows]
        values_sql = ",".join(["(%s,%s,%s,%s)"] * len(chunk))
        flat = []
        for a, b, c, d in chunk:
            flat.extend((a, b, c, d))
        cur.execute(head + values_sql + tail, tuple(flat))

        done += len(chunk)
        if progress:
            progress(done, n)



# ============================
# Core job
# ============================
def run_job(host: str, user: str, password: str, database: str, parts_full: str, pl_full: str, overwrite_done: bool, progq: "queue.Queue"):
    ensure_kb_loaded()
    t0 = time.time()

    def push(stage, cur, total, msg=""):
        progq.put(("progress", stage, int(cur), int(total if total else 1), msg))

    try:
        ps, pt = _split_schema_table(parts_full)
        pls, plt = _split_schema_table(pl_full)
        done_schema, done_table = ps, f"{pt}{DONE_SUFFIX}"

        push("创建_Done表", 0, 1, f"prepare {done_schema}.{done_table}")
        conn = connect_sql(host, user, password, database)
        try:
            required_parts = ["VehicleId_Motor", "SubCategory_GPG", "PNCDesc", "PartNumber", "PartDescription", "value1", "SearchPartNumber", "Category", "SubCategory", "like"]
            required_pl = ["VehicleId_Motor", "Category", "SubCategory", "allready_Pick"]
            assert_columns(conn, ps, pt, required_parts, "parts source")
            assert_columns(conn, pls, plt, required_pl, "pl source")

            if table_exists(conn, done_schema, done_table):
                if not overwrite_done:
                    raise RuntimeError(f"Done table exists: {done_schema}.{done_table}")
                drop_table(conn, done_schema, done_table)
            select_into(conn, ps, pt, done_schema, done_table)
            rowid_key = ensure_rowid_and_index(conn, done_schema, done_table)
            push("创建_Done表", 1, 1, "done table ready")

            need_parts = [rowid_key, "VehicleId_Motor", "SubCategory_GPG", "PNCDesc", "PartNumber", "PartDescription", "value1", "SearchPartNumber", "Category", "SubCategory", "like"]
            need_pl = ["VehicleId_Motor", "Category", "SubCategory", "allready_Pick"]

            push("读取SQL", 0, 1, "loading parts/pl")
            parts = load_df(conn, done_schema, done_table, need_parts)
            pl = load_df(conn, pls, plt, need_pl)
            push("读取SQL", 1, 1, f"loaded parts={len(parts)} pl={len(pl)}")

            parts["value1_num"] = pd.to_numeric(parts["value1"], errors="coerce").fillna(0).astype(int)

            pl_recs, pl_by_vehicle, token_index, domain_sorted, std_pls = build_pl_index(pl)
            candidate_vids = set(pl_by_vehicle.keys())

            N = len(parts)
            rid_arr = parts[rowid_key].tolist()
            spn_arr = parts["SearchPartNumber"].fillna("").astype(str).tolist()
            vid_arr = parts["VehicleId_Motor"].fillna("").astype(str).map(lambda x: x.strip()).tolist()
            subgpg_arr = parts["SubCategory_GPG"].fillna("").astype(str).tolist()
            pnc_arr = parts["PNCDesc"].fillna("").astype(str).tolist()
            pdesc_arr = parts["PartDescription"].fillna("").astype(str).tolist()
            row_v1 = parts["value1_num"].tolist()
            row_partnum = parts["PartNumber"].fillna("").astype(str).tolist()

            spn_to_rows = collections.defaultdict(list)
            row_options = [[] for _ in range(N)]
            row_fast_no_loc = [False] * N
            row_dom_ev = [UNKNOWN_NAME] * N
            row_desc_set = [set() for _ in range(N)]

            push("构建候选", 0, N if N else 1, "building options")
            step = max(1, N // 200)  # 最多刷新约200次；N小会更细，避免开局跳半屏
            for i in range(N):
                spn = spn_arr[i]
                if spn:
                    spn_to_rows[spn].append(i)
                row_feat = desc_features(pnc_arr[i], pdesc_arr[i])
                toks, obj, dom_desc, has_loc, fast_present, fast_no_loc, dset = row_feat
                row_fast_no_loc[i] = fast_no_loc
                row_desc_set[i] = dset
                vid = vid_arr[i]
                if vid in candidate_vids:
                    leg_feat = legacy_features(subgpg_arr[i])
                    legacy_set, legacy_dom, legacy_conf = leg_feat
                    dom_hint = legacy_dom if (dom_desc in (UNKNOWN_NAME, CONFLICT_NAME) and legacy_conf >= 2 and legacy_dom not in (UNKNOWN_NAME, CONFLICT_NAME)) else dom_desc
                    dom_evidence = dom_hint
                    row_dom_ev[i] = dom_evidence
                    obj_key = key_tokens(set(obj))
                    cand = candidate_pls(vid, obj_key, dom_hint, fast_no_loc, pl_by_vehicle, token_index, domain_sorted, std_pls)
                    std_set = std_pls.get(vid, set())
                    opts = []
                    for pid in cand:
                        plr = pl_recs.get(pid)
                        if not plr:
                            continue
                        ev = eval_option(row_feat, leg_feat, plr, std_set)
                        if ev is None:
                            continue
                        lr, overlap, legacy_score, legacy_conf2, compat_key = ev
                        opts.append((pid, lr, overlap, legacy_score, legacy_conf2, compat_key))
                    if opts:
                        opts.sort(key=lambda x: (x[1], x[2], x[3], pl_recs[x[0]].remain, row_v1[i], pl_recs[x[0]].sub), reverse=True)
                        row_options[i] = opts[:12]

                # ✅ 自适应刷新
                if i == 0 or ((i + 1) % step == 0) or (i == N - 1):
                    push("构建候选", i + 1, N if N else 1, "")

            push("统计SPN", 0, 1, "aggregate spn")
            spn_ab_count = {}
            spn_abc_count = {}
            spn_max_v1 = {}
            items = list(spn_to_rows.items())
            M = len(items)
            for j, (spn, rows) in enumerate(items):
                ab = set()
                abc = set()
                mv = 0
                for rid in rows:
                    mv = max(mv, row_v1[rid])
                    for pid, lr, ov, ls, lconf, ckey in row_options[rid]:
                        if lr >= 2:
                            ab.add(pid)
                        abc.add(pid)
                spn_ab_count[spn] = len(ab)
                spn_abc_count[spn] = len(abc)
                spn_max_v1[spn] = mv
                if (j % 200 == 0) or (j == M - 1):
                    push("统计SPN", j + 1, M if M else 1, "")

            pl_remain = {pid: plr.remain for pid, plr in pl_recs.items()}
            selected_pl_for_row = {}
            selected_stage = {}
            picked_global = set()
            pl_used_spn = collections.defaultdict(set)

            spns_3a = [s for s, c in spn_ab_count.items() if c > 0]
            spns_3a.sort(key=lambda s: (spn_ab_count[s], -spn_max_v1[s], s))
            push("3A覆盖", 0, len(spns_3a) if spns_3a else 1, "3A selecting")
            for idx, spn in enumerate(spns_3a):
                if spn in picked_global:
                    continue
                best = None
                best_key = None
                best_sub = None
                best_rid = None
                for rid in spn_to_rows[spn]:
                    if rid in selected_pl_for_row:
                        continue
                    v1 = row_v1[rid]
                    for pid, lr, ov, ls, lconf, ckey in row_options[rid]:
                        if lr < 2 or pl_remain.get(pid, 0) <= 0:
                            continue
                        if spn and spn in pl_used_spn[pid]:
                            continue
                        key = (lr, ov, ls, pl_remain[pid], v1)
                        sub = pl_recs[pid].sub
                        if best_key is None or key > best_key or (key == best_key and sub < best_sub) or (key == best_key and sub == best_sub and rid < best_rid):
                            best = (rid, pid)
                            best_key = key
                            best_sub = sub
                            best_rid = rid
                if best is not None:
                    rid, pid = best
                    selected_pl_for_row[rid] = pid
                    selected_stage[rid] = "3A"
                    picked_global.add(spn)
                    pl_remain[pid] -= 1
                    if spn:
                        pl_used_spn[pid].add(spn)
                if (idx % 200 == 0) or (idx == len(spns_3a) - 1):
                    push("3A覆盖", idx + 1, len(spns_3a) if spns_3a else 1, "")
            if not spns_3a:
                push("3A覆盖", 1, 1, "empty")
            else:
                push("3A覆盖", len(spns_3a), len(spns_3a), "")

            spns_3b = [s for s, c in spn_abc_count.items() if c > 0 and s not in picked_global]
            spns_3b.sort(key=lambda s: (spn_abc_count[s], -spn_max_v1[s], s))
            push("3B降级覆盖", 0, len(spns_3b) if spns_3b else 1, "3B selecting")
            for idx, spn in enumerate(spns_3b):
                if spn in picked_global:
                    continue
                best = None
                best_key = None
                best_sub = None
                best_rid = None
                for rid in spn_to_rows[spn]:
                    if rid in selected_pl_for_row:
                        continue
                    v1 = row_v1[rid]
                    for pid, lr, ov, ls, lconf, ckey in row_options[rid]:
                        if lr < 1 or pl_remain.get(pid, 0) <= 0:
                            continue
                        if spn and spn in pl_used_spn[pid]:
                            continue
                        key = (lr, ov, ls, pl_remain[pid], v1)
                        sub = pl_recs[pid].sub
                        if best_key is None or key > best_key or (key == best_key and sub < best_sub) or (key == best_key and sub == best_sub and rid < best_rid):
                            best = (rid, pid)
                            best_key = key
                            best_sub = sub
                            best_rid = rid
                if best is not None:
                    rid, pid = best
                    selected_pl_for_row[rid] = pid
                    selected_stage[rid] = "3B"
                    picked_global.add(spn)
                    pl_remain[pid] -= 1
                    if spn:
                        pl_used_spn[pid].add(spn)
                if (idx % 200 == 0) or (idx == len(spns_3b) - 1):
                    push("3B降级覆盖", idx + 1, len(spns_3b) if spns_3b else 1, "")
            if not spns_3b:
                push("3B降级覆盖", 1, 1, "empty")
            else:
                push("3B降级覆盖", len(spns_3b), len(spns_3b), "")

            push("3C补满准备", 0, 1, "reverse candidates")
            pl_to_cands = collections.defaultdict(list)
            for rid, opts in enumerate(row_options):
                if not opts:
                    continue
                v1 = row_v1[rid]
                fnl = row_fast_no_loc[rid]
                spn = spn_arr[rid]
                for pid, lr, ov, ls, lconf, ckey in opts:
                    pl_to_cands[pid].append((lr, ov, ls, v1, fnl, rid, spn))
            for pid, cands in pl_to_cands.items():
                cands.sort(key=lambda x: (x[0], x[1], x[2], x[3], 0 if x[4] else 1), reverse=True)
            push("3C补满准备", 1, 1, "")

            pl_lr1_sink_used = collections.defaultdict(int)
            pl_ids_3c = [pid for pid, rem in pl_remain.items() if rem > 0]
            pl_ids_3c.sort(key=lambda pid: (-pl_remain[pid], pl_recs[pid].vid, pl_recs[pid].sub))
            push("3C补满", 0, len(pl_ids_3c) if pl_ids_3c else 1, "filling")
            for idx, pid in enumerate(pl_ids_3c):
                rem = pl_remain.get(pid, 0)
                if rem <= 0:
                    continue
                cands = pl_to_cands.get(pid, [])
                if not cands:
                    continue

                def try_pick(filter_lr, allow_fast_no_loc, allow_global_reuse, require_global_used):
                    nonlocal rem
                    for lr, ov, ls, v1, fnl, rid, spn in cands:
                        if rem <= 0:
                            break
                        if lr not in filter_lr:
                            continue
                        if rid in selected_pl_for_row:
                            continue
                        if spn and spn in pl_used_spn[pid]:
                            continue
                        if require_global_used:
                            if (not spn) or (spn not in picked_global):
                                continue
                        if (not allow_global_reuse) and spn and spn in picked_global:
                            continue
                        if lr == 1:
                            ev_dom = row_dom_ev[rid]
                            if ev_dom not in (UNKNOWN_NAME, CONFLICT_NAME):
                                if _dom_key(pl_recs[pid].dom) != _dom_key(ev_dom):
                                    if not neighbor_allowed(ev_dom, pl_recs[pid].dom, row_desc_set[rid], set(pl_recs[pid].subtoks)):
                                        continue
                            if pl_recs[pid].is_sink == 1 and pl_lr1_sink_used[pid] >= 1:
                                continue
                        if fnl and (not allow_fast_no_loc):
                            continue
                        selected_pl_for_row[rid] = pid
                        selected_stage[rid] = "3C"
                        pl_remain[pid] -= 1
                        rem -= 1
                        if spn:
                            pl_used_spn[pid].add(spn)
                            picked_global.add(spn)
                        if lr == 1 and pl_recs[pid].is_sink == 1:
                            pl_lr1_sink_used[pid] += 1

                try_pick({3, 2}, True, False, False)
                if rem > 0:
                    try_pick({1}, False, False, False)
                if rem > 0:
                    try_pick({1}, True, False, False)
                if rem > 0:
                    try_pick({3, 2}, False, True, True)
                if (idx % 200 == 0) or (idx == len(pl_ids_3c) - 1):
                    push("3C补满", idx + 1, len(pl_ids_3c) if pl_ids_3c else 1, "")

            push("写回&打分", 0, 1, "compute like")
            pl_selected_rows = collections.defaultdict(list)
            for rid, pid in selected_pl_for_row.items():
                pl_selected_rows[pid].append(rid)

            like_vals = {}
            fast_no_loc_selected = 0
            for pid, rids in pl_selected_rows.items():
                rids_sorted = sorted(rids, key=lambda rid: (row_v1[rid], row_partnum[rid]), reverse=True)
                n = len(rids_sorted)
                for j, rid in enumerate(rids_sorted):
                    rb = 0 if n <= 1 else int(round(5 * (1 - j / (n - 1))))
                    stage = selected_stage[rid]
                    lr = 1
                    conf = 0
                    compat_key = 0
                    for pid2, lr2, ov2, ls2, lconf2, ckey2 in row_options[rid]:
                        if pid2 == pid:
                            lr = lr2
                            conf = lconf2
                            compat_key = ckey2
                            break
                    base = 95 if lr == 3 else (80 if lr == 2 else 55)
                    like = base + legacy_bonus(conf, compat_key) + rb
                    like = clamp(like, 90, 100) if lr == 3 else (clamp(like, 70, 89) if lr == 2 else clamp(like, 40, 69))
                    if row_fast_no_loc[rid]:
                        like = min(like, 29 if stage == "3C" else 49)
                        fast_no_loc_selected += 1
                    like_vals[rid] = int(like)

            params = []
            for rid, pid in selected_pl_for_row.items():
                params.append((pl_recs[pid].cat, pl_recs[pid].sub, int(like_vals[rid]), int(rid_arr[rid])))

            if not params:
                push("写回SQL", 1, 1, "")
            else:
                total_rows = len(params)
                push("写回SQL", 0, total_rows, "")

                with conn.cursor() as cur:
                    _update_done_values(
                        cur,
                        done_schema,
                        done_table,
                        rowid_key,
                        params,
                        CHUNK_VALUES,
                        progress=lambda cur_n, tot_n: push("写回SQL", cur_n, tot_n if tot_n else 1, ""),
                    )
                conn.commit()
                push("写回SQL", total_rows, total_rows, "")

            X = len(spn_to_rows)
            Y = len(selected_pl_for_row)
            Z = len({spn_arr[rid] for rid in selected_pl_for_row if spn_arr[rid]})
            R = (Z / X) if X else 0.0
            summary = "\n".join(
                [
                    "Part 2) 总结（固定字段）",
                    f"- 输出表：{done_schema}.{done_table}",
                    f"- 表内去重SPN数量：{X}",
                    f"- 本次选中零件数量：{Y}",
                    f"- 本次选中去重SPN数量：{Z}",
                    f"- 覆盖率：{R:.4f}",
                    f"- Fastener无位置限定入选行数：{fast_no_loc_selected}",
                    f"Done. elapsed_s={time.time() - t0:.1f}",
                ]
            )
            progq.put(("done", summary))
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except Exception as e:
        progq.put(("error", str(e)))


def _safe_extract_zip(zf: zipfile.ZipFile, dest: Path):
    dest_resolved = dest.resolve()
    for member in zf.infolist():
        mpath = (dest / member.filename).resolve()
        try:
            ok = mpath.is_relative_to(dest_resolved)
        except AttributeError:
            ok = (dest_resolved == mpath) or (dest_resolved in mpath.parents)
        if not ok:
            raise RuntimeError(f"非法zip路径: {member.filename}")
    zf.extractall(dest)


# ============================
# UI
# ============================
class UI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("零件→PL 分配（SQL连接方式对齐一体化V2 + 输出_Done表）")
        self.root.geometry("980x600")
        self.root.resizable(False, False)

        self.q = queue.Queue()
        self.running = False
        self.logged_in = False
        self._last_stage = None
        self._write_last_cur = -1
        self._write_last_ts = 0.0
        self._stage_log_last = {}  # stage -> (last_cur:int, last_ts:float)
        self._stage_last_prog = {}   # stage -> (cur,total)
        self._stage_last_info = {}   # stage -> last non-ratio info (关键摘要候选)
        self._ui_last_ts = 0.0       # UI节流：最后刷新时间
        self._ui_last_cur = -1       # UI节流：最后刷新cur

        self.host = tk.StringVar(value="")
        self.user = tk.StringVar(value="")
        self.pwd = tk.StringVar(value="")
        self.db = tk.StringVar(value="")
        self.parts = tk.StringVar(value="")
        self.pl = tk.StringVar(value="")
        self.kb_path = tk.StringVar(value=os.environ.get("KB_DIR", DEFAULT_KB_DIR))
        self.kb_version = tk.StringVar(value="未加载")

        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="KB路径", width=8).grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(top, textvariable=self.kb_path, width=50).grid(row=0, column=1, columnspan=3, sticky="w", pady=3)
        tk.Button(top, text="选择文件夹", width=10, command=self.choose_kb_dir).grid(row=0, column=4, sticky="w", padx=(6, 0))
        tk.Button(top, text="选择zip", width=10, command=self.choose_kb_zip).grid(row=0, column=5, sticky="w", padx=(6, 0))
        self.btn_load_kb = tk.Button(top, text="加载KB", width=10, command=self.load_kb_action)
        self.btn_load_kb.grid(row=0, column=6, sticky="w", padx=(6, 0))

        ttk.Label(top, text="Host", width=8).grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(top, textvariable=self.host, width=30).grid(row=1, column=1, sticky="w", pady=3)
        ttk.Label(top, text="User", width=8).grid(row=1, column=2, sticky="w", padx=(12, 0))
        ttk.Entry(top, textvariable=self.user, width=18).grid(row=1, column=3, sticky="w")
        ttk.Label(top, text="Password", width=10).grid(row=1, column=4, sticky="w", padx=(12, 0))
        ttk.Entry(top, textvariable=self.pwd, width=18, show="*").grid(row=1, column=5, sticky="w")

        self.btn_login = tk.Button(top, text="登录/确认连接", width=14, command=self.login)
        self.btn_login.grid(row=1, column=6, sticky="w", padx=(12, 0))

        ttk.Label(top, text="Database", width=8).grid(row=2, column=0, sticky="w", pady=6)
        self.db_combo = ttk.Combobox(top, textvariable=self.db, width=27, state="disabled")
        self.db_combo.grid(row=2, column=1, sticky="w", pady=6)
        self.db_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_db_selected())

        self.btn_tables = tk.Button(top, text="加载表列表", width=12, state="disabled", command=self.load_tables)
        self.btn_tables.grid(row=2, column=2, sticky="w", padx=(12, 0))

        ttk.Label(top, text="PartsTable", width=8).grid(row=3, column=0, sticky="w", pady=3)
        self.parts_combo = ttk.Combobox(top, textvariable=self.parts, width=60, state="disabled")
        self.parts_combo.grid(row=3, column=1, columnspan=5, sticky="w", pady=3)

        ttk.Label(top, text="PLTable", width=8).grid(row=4, column=0, sticky="w", pady=3)
        self.pl_combo = ttk.Combobox(top, textvariable=self.pl, width=60, state="disabled")
        self.pl_combo.grid(row=4, column=1, columnspan=5, sticky="w", pady=3)

        self.btn_start = tk.Button(top, text="Start", width=10, state="disabled", command=self.start)
        self.btn_start.grid(row=4, column=6, sticky="w", padx=(12, 0))

        self.status = tk.StringVar(value="KB未加载")   # 只显示阶段名
        self.counter = tk.StringVar(value="")         # 显示 cur/total（*/ *）

        status_row = ttk.Frame(self.root)
        status_row.pack(fill="x", padx=12, pady=(2, 0))

        ttk.Label(status_row, textvariable=self.status, anchor="w").pack(side="left", fill="x", expand=True)
        ttk.Label(status_row, textvariable=self.counter, anchor="e").pack(side="right")

        self.pbar = ttk.Progressbar(self.root, orient="horizontal", mode="determinate", maximum=100)
        self.pbar.pack(fill="x", padx=12, pady=(2, 4))

        self.log = tk.Text(self.root, height=22, wrap="word")
        self.log.pack(fill="both", expand=True, padx=12, pady=(6, 10))
        self._log("Ready")
        self._log("KB可稍后加载；开始运行前必须加载KB")
        self._set_controls_for_kb(False)

        self.root.after(60, self.poll)

    def _kb_label(self) -> str:
        v = (self.kb_version.get() or "").strip()
        return f"KB:{v}" if (v and v != "未加载") else "KB未加载"

    def _set_controls_for_kb(self, loaded: bool):
        # KB是否加载不再控制登录/选库；只在运行时检查
        self.btn_login.config(state="normal")

        # 登录后：允许选择数据库 + 加载表
        if self.logged_in:
            self.db_combo.config(state="readonly")
            self.btn_tables.config(state="normal")
        else:
            self.db_combo.config(state="disabled")
            self.btn_tables.config(state="disabled")

        # 表列表加载后：允许选择Parts/PL + Start
        has_tables = bool(self.parts_combo["values"]) and bool(self.pl_combo["values"])
        if has_tables:
            self.parts_combo.config(state="readonly")
            self.pl_combo.config(state="readonly")
            self.btn_start.config(state="normal")
        else:
            self.parts_combo.config(state="disabled")
            self.pl_combo.config(state="disabled")
            self.btn_start.config(state="disabled")


    def choose_kb_dir(self):
        p = filedialog.askdirectory()
        if p:
            self.kb_path.set(p)

    def choose_kb_zip(self):
        p = filedialog.askopenfilename(filetypes=[("Zip", "*.zip")])
        if p:
            self.kb_path.set(p)

    def _resolve_kb_dir(self, p: str) -> str:
        p = (p or "").strip()
        if not p:
            raise RuntimeError("KB路径不能为空")
        path = Path(p)
        if path.is_file() and path.suffix.lower() == ".zip":
            cache_root = Path("./_kb_cache")
            cache_root.mkdir(parents=True, exist_ok=True)
            st = path.stat()
            digest = hashlib.md5(f"{path.resolve()}|{st.st_mtime_ns}|{st.st_size}".encode("utf-8")).hexdigest()
            out_dir = cache_root / f"{path.stem}_{digest}"
            if out_dir.exists():
                manifest = out_dir / "manifest.json"
                ok_manifest = False
                if manifest.exists():
                    try:
                        _read_json(manifest)
                        ok_manifest = True
                    except Exception:
                        ok_manifest = False
                if ok_manifest:
                    return str(out_dir)
                shutil.rmtree(out_dir, ignore_errors=True)
            tmp_dir = out_dir.with_name(out_dir.name + "_tmp")
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(path, "r") as zf:
                _safe_extract_zip(zf, tmp_dir)
            if (tmp_dir / "manifest.json").exists():
                tmp_dir.rename(out_dir)
                return str(out_dir)
            subdirs = [x for x in tmp_dir.iterdir() if x.is_dir()]
            if len(subdirs) == 1 and (subdirs[0] / "manifest.json").exists():
                subdirs[0].rename(out_dir)
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return str(out_dir)
            raise RuntimeError("zip解压后未找到manifest.json")
        if not path.exists() or not path.is_dir():
            raise RuntimeError("KB路径不存在或不是目录")
        return str(path)

    def load_kb_action(self):
        if self.running:
            return
        try:
            kb_dir = self._resolve_kb_dir(self.kb_path.get())
            kb = load_kb(kb_dir)
            apply_kb(kb)
            self.kb_version.set(kb["version"])
            self.status.set(f"{self._kb_label()} | KB已加载（可登录/选表；开始运行会检查KB）")
            self._log(f"KB loaded: {kb['version']}")
            self._set_controls_for_kb(True)
        except Exception as e:
            self._set_controls_for_kb(False)
            self.status.set("KB加载失败")
            messagebox.showerror("KB加载失败", str(e))

    def _log(self, s: str):
        ts = time.strftime("%H:%M:%S")
        self.log.insert("end", f"[{ts}] {s}\n")
        self.log.see("end")

    def login(self):
        if self.running:
            return
        host = self.host.get().strip()
        user = self.user.get().strip()
        if not host or not user:
            messagebox.showwarning("提示", "请先填写 Host / User / Password")
            return
        try:
            self.status.set(f"{self._kb_label()} | 登录中：连接 master 拉数据库列表...")
            dbs = list_databases(host, user, self.pwd.get())
            self.db_combo["values"] = dbs
            if dbs:
                self.db.set(dbs[0])
            self.logged_in = True
            self.status.set(f"{self._kb_label()} | 已登录：数据库数量={len(dbs)}（请选择 Database）")
            self.db_combo.config(state="readonly")
            self.btn_tables.config(state="normal")
            self._log(f"Login OK. databases={len(dbs)}")
            self._set_controls_for_kb(KB is not None)
        except Exception as e:
            self.logged_in = False
            self.status.set(f"{self._kb_label()} | 登录失败")
            self._set_controls_for_kb(KB is not None)
            messagebox.showerror("Login failed", str(e))

    def _on_db_selected(self):
        if not self.db.get().strip():
            return
        self.status.set(f"{self._kb_label()} | 已选择数据库：{self.db.get().strip()}（请点“加载表列表”）")

    def load_tables(self):
        if self.running or (not self.logged_in):
            return
        db = self.db.get().strip()
        if not db:
            messagebox.showwarning("提示", "请先选择 Database")
            return
        try:
            self.status.set(f"{self._kb_label()} | 加载表列表中...")
            tables = list_tables(self.host.get().strip(), self.user.get().strip(), self.pwd.get(), db)
            self.parts_combo["values"] = tables
            self.pl_combo["values"] = tables
            if tables:
                self.parts.set(tables[0])
                self.pl.set(tables[0])
            self.parts_combo.config(state="readonly")
            self.pl_combo.config(state="readonly")
            self.btn_start.config(state="normal")
            self.status.set(f"{self._kb_label()} | 表列表已加载：{len(tables)}（选择 PartsTable / PLTable）")
            self._log(f"Tables loaded: {len(tables)}")
            self._set_controls_for_kb(KB is not None)
        except Exception as e:
            self.status.set(f"{self._kb_label()} | 加载表失败")
            self._set_controls_for_kb(KB is not None)
            messagebox.showerror("Load tables failed", str(e))

    def start(self):
        if self.running:
            return
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        if KB is None:
            messagebox.showwarning("提示", "未加载KB：请先加载KB后再开始运行。")
            return
        db = self.db.get().strip()
        parts = self.parts.get().strip()
        pl = self.pl.get().strip()
        if not db or not parts or not pl:
            messagebox.showwarning("提示", "请选择 Database / PartsTable / PLTable")
            return

        ps, pt = _split_schema_table(parts)
        done = f"{ps}.{pt}{DONE_SUFFIX}"

        overwrite = False
        try:
            with connect_sql(self.host.get().strip(), self.user.get().strip(), self.pwd.get(), db) as conn:
                ds, dt = _split_schema_table(done)
                exists = table_exists(conn, ds, dt)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        if exists:
            overwrite = messagebox.askyesno("提示", f"目标输出表已存在：\n{done}\n\n是否覆盖重建？\nYes=删除并重建\nNo=取消")
            if not overwrite:
                return

        self.running = True
        self.pbar["value"] = 0
        self.status.set(f"{self._kb_label()} | 运行中...")
        self._log(f"DB={db} Parts={parts} PL={pl} Out={done} Overwrite={overwrite}")

        # reset UI progress cache (避免上次残留导致跳阶段/不显示)
        self._last_stage = None
        self._stage_last_prog.clear()
        self._stage_last_info.clear()
        self._ui_last_ts = 0.0
        self._ui_last_cur = -1
        self.counter.set("")
        self.pbar["value"] = 0

        th = threading.Thread(
            target=run_job,
            args=(self.host.get().strip(), self.user.get().strip(), self.pwd.get(), db, parts, pl, overwrite, self.q),
            daemon=True,
        )
        th.start()

    def poll(self):
        max_drain = 1200  # 多读点，避免队列积压导致“看起来跳过阶段”

        def _is_ratio(s: str) -> bool:
            s = (s or "").strip()
            return bool(s and re.fullmatch(r"\d+/\d+", s))

        def _remember(stage: str, cur: int, total: int, info: str):
            self._stage_last_prog[stage] = (cur, total)
            # 只记“关键摘要候选”：过滤纯比值，保留描述性信息
            if info and (not _is_ratio(info)):
                self._stage_last_info[stage] = info.strip()

        def _log_stage_end(stage: str):
            if not stage:
                return
            prog = self._stage_last_prog.get(stage)
            if not prog:
                return
            cur, total = prog
            info = (self._stage_last_info.get(stage) or "").strip()
            line = f"[{stage}] done {cur}/{total}"
            if info:
                line += f" | {info}"
            self._log(line)

        def _apply_ui(stage: str, cur: int, total: int, info: str):
            now = time.time()

            # 阶段切换：先打上一阶段摘要，再打新阶段 start
            if self._last_stage != stage:
                if self._last_stage is not None:
                    _log_stage_end(self._last_stage)

                self._last_stage = stage

                # 状态栏：只显示阶段名
                self.status.set(f"{self._kb_label()} | {stage}")

                # 日志：只打一条 start
                self._log(f"[{stage}] start")

                # reset throttle
                self._ui_last_ts = 0.0
                self._ui_last_cur = -1

            _remember(stage, cur, total, info)

            # 进度条：按 stage 内部百分比
            pct = 0 if total <= 0 else min(100.0, (cur / total) * 100.0)

            # UI节流：开始/结束强制更新；否则按时间/步进刷新
            step_need = max(1, (total // 200)) if total else 1  # 大表约200次可见刷新
            if (cur == 0) or (cur == total) or ((now - self._ui_last_ts) >= 0.05) or (abs(cur - self._ui_last_cur) >= step_need):
                self.pbar["value"] = pct
                self.counter.set(f"{cur}/{total}")  # ✅ */* 显示
                self._ui_last_ts = now
                self._ui_last_cur = cur

        try:
            for _ in range(max_drain):
                msg = self.q.get_nowait()

                if msg[0] == "progress":
                    _, stage, cur, total, info = msg
                    _apply_ui(stage, int(cur), int(total), info)
                    continue

                if msg[0] == "error":
                    self.running = False
                    if self._last_stage is not None:
                        _log_stage_end(self._last_stage)
                    self.status.set(f"{self._kb_label()} | 错误")
                    self.counter.set("")
                    self._log("ERROR: " + msg[1])
                    messagebox.showerror("错误", msg[1])
                    break

                if msg[0] == "done":
                    self.running = False
                    if self._last_stage is not None:
                        _log_stage_end(self._last_stage)
                    self.pbar["value"] = 100
                    self.counter.set("")
                    self.status.set(f"{self._kb_label()} | 完成")
                    self._log("DONE")
                    self._log(msg[1])
                    messagebox.showinfo("完成", msg[1])
                    break

        except queue.Empty:
            pass

        # 30ms：更顺滑，避免3A/3B/3C看起来“瞬间跳过”
        self.root.after(30, self.poll)


def main():
    UI().root.mainloop()


if __name__ == "__main__":
    main()
