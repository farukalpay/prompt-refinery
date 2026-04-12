#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import re
import sys
import time
import math
import hashlib
import sqlite3
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
warnings.filterwarnings(
    "ignore",
    message=r"urllib3 .* doesn't match a supported version!",
    category=Warning,
    module=r"requests(\..*)?",
)
import requests
from datasets import ClassLabel, load_dataset
from huggingface_hub import HfApi, hf_hub_url


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def first_env(*keys: str) -> str:
    for key in keys:
        value = os.getenv(key)
        if value is None:
            continue
        value = value.strip()
        if value:
            return value
    return ""


load_env_file(PROJECT_DIR / ".env")

LLM_API_KEY = first_env("OPENAI_API_KEY", "OPENROUTER_API_KEY", "LLM_API_KEY")
if not LLM_API_KEY:
    raise RuntimeError(
        "API key is missing. Set OPENAI_API_KEY in .env or environment variables."
    )

LLM_API_BASE_URL = first_env("LLM_API_BASE_URL") or "https://openrouter.ai/api/v1"
LLM_API_BASE_URL = LLM_API_BASE_URL.rstrip("/")
EMBEDDINGS_URL = f"{LLM_API_BASE_URL}/embeddings"
CHAT_COMPLETIONS_URL = f"{LLM_API_BASE_URL}/chat/completions"

# Cost-sensitive defaults:
EMBED_MODEL = first_env("EMBED_MODEL") or "openai/text-embedding-3-small"
REPAIR_MODEL = first_env("REPAIR_MODEL") or "mistralai/mistral-nemo"
INTENT_MODEL = first_env("INTENT_MODEL") or "openai/gpt-4o-mini"
POLISH_MODEL = first_env("POLISH_MODEL") or "openai/gpt-4o-mini"

PROMPT_DATASET = "fka/prompts.chat"
PROMPT_SPLIT = "train"

# MASSIVE is used as the structured slot example database
MASSIVE_DATASET = "AmazonScience/massive"
MASSIVE_CONFIGS = ["tr-TR", "en-US"]
MASSIVE_SPLIT = "train"
MASSIVE_PARQUET_REVISION = "refs/convert/parquet"

# Exact schemas: no column-guessing heuristics
PROMPTS_CHAT_COLUMNS = ["act", "prompt", "for_devs", "type", "contributor"]
MASSIVE_COLUMNS = [
    "id", "locale", "partition", "scenario", "intent", "utt",
    "annot_utt", "worker_id", "slot_method", "judgments"
]

APP_DIR = PROJECT_DIR / "runtime_db"
HF_CACHE_DIR = APP_DIR / "hf_cache"
INDEX_DIR = APP_DIR / "indices"
EXPORT_DIR = APP_DIR / "exports"
DB_PATH = APP_DIR / "runtime.sqlite3"

PROMPT_INDEX_FILE = INDEX_DIR / "prompt_index.npz"
SLOT_INDEX_FILE = INDEX_DIR / "slot_index.npz"
MEMORY_INDEX_FILE = INDEX_DIR / "memory_index.npz"

PROMPT_META_FILE = INDEX_DIR / "prompt_index_meta.json"
SLOT_META_FILE = INDEX_DIR / "slot_index_meta.json"
MEMORY_META_FILE = INDEX_DIR / "memory_index_meta.json"

LAST_RESULT_JSON = EXPORT_DIR / "last_result.json"
LAST_RESULT_TXT = EXPORT_DIR / "last_prompt.txt"

BATCH_SIZE = 64
REQUEST_TIMEOUT = 120
MAX_RETRIES = 6
EMBED_TEXT_CHUNK_CHARS = 3500
EMBED_TEXT_CHUNK_OVERLAP = 350

PROMPT_TOP_K = 6
SLOT_TOP_K = 14
MEMORY_TOP_K = 6

TEMPERATURE = 0.1
MAX_REPAIR_TOKENS = 1400
MAX_INTENT_TOKENS = 400
MAX_POLISH_TOKENS = 1200

_MASSIVE_PARQUET_FILE_CACHE: Dict[Tuple[str, str], List[str]] = {}


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class PromptCandidate:
    row_id: int
    act: str
    prompt: str
    for_devs: int
    record_type: str
    contributor: str
    score: float


@dataclass
class SlotSupport:
    row_id: int
    source: str
    locale: str
    intent: str
    utt: str
    slots: List[Dict[str, str]]
    score: float


@dataclass
class MemorySupport:
    row_id: int
    user_text: str
    chosen_prompt_row_id: int
    chosen_act: str
    final_prompt: str
    created_at: str
    score: float


@dataclass
class IntentSpec:
    objective: str
    deliverable_type: str
    audience: str
    language: str
    must_include: List[str]
    style_constraints: List[str]
    quality_targets: List[str]


# ============================================================
# UTILS
# ============================================================

def ensure_dirs() -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_user_text(x: str) -> str:
    return re.sub(r"\s+", " ", clean_text(x)).casefold()


def clean_list(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    out: List[str] = []
    for item in items:
        value = clean_text(item)
        if value:
            out.append(value)
    return out


def clip_text(text: str, max_chars: int) -> str:
    value = clean_text(text)
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "..."


def stable_hash(items: List[str]) -> str:
    h = hashlib.sha256()
    for item in items:
        h.update(item.encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim == 1:
        denom = np.linalg.norm(matrix) + 1e-12
        return matrix / denom
    denom = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    return matrix / denom


def batched(items: List[str], batch_size: int) -> List[List[str]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def chunk_text_for_embedding(text: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", clean_text(text))
    if not normalized:
        return [" "]

    if len(normalized) <= EMBED_TEXT_CHUNK_CHARS:
        return [normalized]

    step = max(1, EMBED_TEXT_CHUNK_CHARS - EMBED_TEXT_CHUNK_OVERLAP)
    chunks: List[str] = []
    for start in range(0, len(normalized), step):
        piece = normalized[start:start + EMBED_TEXT_CHUNK_CHARS]
        if not piece:
            continue
        chunks.append(piece)
        if start + EMBED_TEXT_CHUNK_CHARS >= len(normalized):
            break

    return chunks or [" "]


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_first_json_object(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("Empty model output")

    stripped = text.strip()

    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)

    # direct parse first
    try:
        return json.loads(stripped)
    except Exception:
        pass

    # bracket scan fallback
    start = stripped.find("{")
    if start == -1:
        raise ValueError("No JSON object found")

    depth = 0
    for i in range(start, len(stripped)):
        ch = stripped[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start:i + 1]
                return json.loads(candidate)

    raise ValueError("Could not extract JSON object")


def bool_to_int(x: Any) -> int:
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, np.integer)):
        return int(bool(x))
    s = clean_text(x).lower()
    return int(s in {"1", "true", "yes", "y"})


# ============================================================
# INPUT / OUTPUT
# ============================================================

def _get_user_input_gui() -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import scrolledtext
    except Exception:
        return None

    result = {"text": None}

    root = tk.Tk()
    root.title("Prompt Repair Input")
    root.geometry("900x500")

    label = tk.Label(
        root,
        text="Kullanıcı girdisini yapıştır ve 'Gönder'e bas:",
        font=("Arial", 12)
    )
    label.pack(pady=10)

    text_widget = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Consolas", 11))
    text_widget.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

    def submit() -> None:
        result["text"] = text_widget.get("1.0", tk.END).strip()
        root.destroy()

    button = tk.Button(root, text="Gönder", command=submit, font=("Arial", 12))
    button.pack(pady=10)

    root.mainloop()
    return result["text"]


def _show_output_gui(title: str, content: str) -> None:
    try:
        import tkinter as tk
        from tkinter import scrolledtext
    except Exception:
        return

    root = tk.Tk()
    root.title(title)
    root.geometry("1000x700")

    text_widget = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Consolas", 11))
    text_widget.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
    text_widget.insert("1.0", content)
    text_widget.configure(state="normal")

    root.mainloop()


def get_user_input() -> Tuple[str, bool]:
    """
    Returns (user_text, used_gui)
    """
    # 1) argv
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:]).strip()
        if text:
            return text, False

    # 2) stdin tty
    if sys.stdin and sys.stdin.isatty():
        text = input("Kullanıcı girdisini yaz:\n> ").strip()
        if text:
            return text, False

    # 3) GUI fallback for no-CLI launches
    gui_text = _get_user_input_gui()
    if gui_text:
        return gui_text, True

    raise RuntimeError(
        "Kullanıcı girdisi alınamadı. CLI yoksa GUI açılamadı. "
        "Masaüstü ortamında çalıştır veya argüman ver."
    )


# ============================================================
# API HTTP
# ============================================================

def api_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }


def post_json_with_retry(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    last_err: Optional[Exception] = None

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                url,
                headers=api_headers(),
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )

            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt == MAX_RETRIES - 1:
                    resp.raise_for_status()
                time.sleep(1.5 * (2 ** attempt))
                continue

            resp.raise_for_status()
            return resp.json()

        except requests.RequestException as exc:
            last_err = exc
            if attempt == MAX_RETRIES - 1:
                break
            time.sleep(1.5 * (2 ** attempt))

    raise RuntimeError(f"HTTP request failed after retries: {last_err}")


def _embed_batch(batch: List[str], model: str) -> List[np.ndarray]:
    payload = {
        "model": model,
        "input": batch,
    }
    data = post_json_with_retry(EMBEDDINGS_URL, payload)

    if "data" in data:
        ordered = sorted(data["data"], key=lambda x: x["index"])
        return [np.array(item["embedding"], dtype=np.float32) for item in ordered]

    if len(batch) > 1:
        mid = len(batch) // 2
        left = _embed_batch(batch[:mid], model)
        right = _embed_batch(batch[mid:], model)
        return left + right

    raise RuntimeError(f"Unexpected embedding response: {data}")


def embed_texts(texts: List[str], model: str = EMBED_MODEL) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)

    chunk_texts: List[str] = []
    owners: List[int] = []
    for idx, text in enumerate(texts):
        chunks = chunk_text_for_embedding(text)
        chunk_texts.extend(chunks)
        owners.extend([idx] * len(chunks))

    chunk_vectors: List[np.ndarray] = []
    for batch in batched(chunk_texts, BATCH_SIZE):
        chunk_vectors.extend(_embed_batch(batch, model))

    grouped: List[List[np.ndarray]] = [[] for _ in range(len(texts))]
    for owner, vec in zip(owners, chunk_vectors):
        grouped[owner].append(vec)

    vectors: List[np.ndarray] = []
    for vecs in grouped:
        if not vecs:
            raise RuntimeError("Embedding aggregation produced an empty vector group.")
        stacked = np.vstack(vecs).astype(np.float32)
        vectors.append(np.mean(stacked, axis=0))

    matrix = np.vstack(vectors).astype(np.float32)
    return l2_normalize(matrix)


def chat_json(
    messages: List[Dict[str, str]],
    model: str = REPAIR_MODEL,
    max_tokens: int = MAX_REPAIR_TOKENS,
    temperature: float = TEMPERATURE,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = post_json_with_retry(CHAT_COMPLETIONS_URL, payload)

    try:
        content = data["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError(f"Unexpected chat response: {data}") from exc

    try:
        return extract_first_json_object(content)
    except Exception:
        # one repair retry
        retry_messages = messages + [
            {
                "role": "assistant",
                "content": content,
            },
            {
                "role": "user",
                "content": (
                    "You returned invalid JSON. Return ONLY a valid JSON object. "
                    "Do not include markdown fences."
                ),
            },
        ]
        retry_payload = {
            "model": model,
            "messages": retry_messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        retry_data = post_json_with_retry(CHAT_COMPLETIONS_URL, retry_payload)
        retry_content = retry_data["choices"][0]["message"]["content"]
        return extract_first_json_object(retry_content)


# ============================================================
# SQLITE
# ============================================================

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;

        CREATE TABLE IF NOT EXISTS manifest (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS prompts (
            row_id INTEGER PRIMARY KEY,
            act TEXT NOT NULL,
            prompt TEXT NOT NULL,
            for_devs INTEGER NOT NULL,
            record_type TEXT NOT NULL,
            contributor TEXT NOT NULL,
            retrieval_text TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS slot_examples (
            row_id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            locale TEXT NOT NULL,
            intent TEXT NOT NULL,
            utt TEXT NOT NULL,
            slots_json TEXT NOT NULL,
            retrieval_text TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS memory (
            row_id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            normalized_user_text TEXT NOT NULL,
            user_text TEXT NOT NULL,
            chosen_prompt_row_id INTEGER NOT NULL,
            chosen_act TEXT NOT NULL,
            final_prompt TEXT NOT NULL,
            meta_json TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_memory_norm_text ON memory(normalized_user_text);
        """
    )
    conn.commit()


def get_manifest(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM manifest WHERE key = ?", (key,)).fetchone()
    return None if row is None else row["value"]


def set_manifest(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO manifest(key, value)
        VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )
    conn.commit()


# ============================================================
# DATASET LOADERS (EXACT SCHEMA, NO COLUMN GUESSING)
# ============================================================

def list_massive_parquet_files(config: str, split: str) -> List[str]:
    cache_key = (config, split)
    cached = _MASSIVE_PARQUET_FILE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    api = HfApi()
    files = api.list_repo_files(
        repo_id=MASSIVE_DATASET,
        repo_type="dataset",
        revision=MASSIVE_PARQUET_REVISION,
    )

    prefix = f"{config}/{split}/"
    parquet_files = sorted(
        file_path for file_path in files
        if file_path.startswith(prefix) and file_path.endswith(".parquet")
    )

    if not parquet_files:
        raise RuntimeError(
            f"No parquet files found for {MASSIVE_DATASET} config={config} split={split} "
            f"at revision={MASSIVE_PARQUET_REVISION}."
        )

    _MASSIVE_PARQUET_FILE_CACHE[cache_key] = parquet_files
    return parquet_files


def load_massive_split(config: str, split: str):
    """
    Loads MASSIVE using the standard HF loader first.
    If the environment rejects dataset scripts, falls back to HF's converted parquet revision.
    """
    try:
        return load_dataset(
            MASSIVE_DATASET,
            config,
            split=split,
            cache_dir=str(HF_CACHE_DIR),
        )
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc):
            raise

    parquet_files = list_massive_parquet_files(config=config, split=split)
    parquet_urls = [
        hf_hub_url(
            repo_id=MASSIVE_DATASET,
            repo_type="dataset",
            revision=MASSIVE_PARQUET_REVISION,
            filename=file_path,
        )
        for file_path in parquet_files
    ]

    return load_dataset(
        "parquet",
        data_files={split: parquet_urls},
        split=split,
        cache_dir=str(HF_CACHE_DIR),
    )


def validate_columns(actual: List[str], expected: List[str], dataset_name: str) -> None:
    if set(actual) != set(expected):
        raise RuntimeError(
            f"{dataset_name} schema mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )


def parse_massive_annot_utt(annot_utt: str) -> List[Dict[str, str]]:
    """
    This is not heuristic user parsing.
    It only parses MASSIVE's documented annotation format: [slot : entity]
    """
    out: List[Dict[str, str]] = []
    for m in re.finditer(r"\[(.*?)\s*:\s*(.*?)\]", annot_utt):
        slot_name = clean_text(m.group(1)).lower().replace("-", "_").replace(" ", "_")
        slot_value = clean_text(m.group(2))
        if slot_name and slot_value:
            out.append({"slot": slot_name, "value": slot_value})
    return out


def prompt_retrieval_text(act: str, prompt: str, record_type: str) -> str:
    return f"ACT: {act}\nTYPE: {record_type}\nPROMPT:\n{prompt}"


def slot_retrieval_text(source: str, locale: str, intent: str, utt: str, slots: List[Dict[str, str]]) -> str:
    slot_names = ", ".join(sorted({s["slot"] for s in slots}))
    return (
        f"SOURCE: {source}\n"
        f"LOCALE: {locale}\n"
        f"INTENT: {intent}\n"
        f"UTTERANCE: {utt}\n"
        f"SLOTS: {slot_names}"
    )


def build_prompts_table(conn: sqlite3.Connection) -> None:
    ds = load_dataset(
        PROMPT_DATASET,
        split=PROMPT_SPLIT,
        cache_dir=str(HF_CACHE_DIR),
    )

    validate_columns(ds.column_names, PROMPTS_CHAT_COLUMNS, PROMPT_DATASET)

    rows_to_insert: List[Tuple[int, str, str, int, str, str, str]] = []
    hash_lines: List[str] = []

    for idx, row in enumerate(ds):
        act = clean_text(row["act"])
        prompt = clean_text(row["prompt"])
        for_devs = bool_to_int(row["for_devs"])
        record_type = clean_text(row["type"])
        contributor = clean_text(row["contributor"])
        retrieval_text = prompt_retrieval_text(act, prompt, record_type)

        rows_to_insert.append(
            (idx, act, prompt, for_devs, record_type, contributor, retrieval_text)
        )
        hash_lines.append(
            json.dumps(
                [idx, act, prompt, for_devs, record_type, contributor],
                ensure_ascii=False
            )
        )

    current_sig = stable_hash(hash_lines)
    existing_sig = get_manifest(conn, "prompts_chat_signature")

    if existing_sig == current_sig:
        existing_count = conn.execute("SELECT COUNT(*) AS c FROM prompts").fetchone()["c"]
        if existing_count == len(rows_to_insert):
            return

    conn.execute("DELETE FROM prompts")
    conn.executemany(
        """
        INSERT INTO prompts(row_id, act, prompt, for_devs, record_type, contributor, retrieval_text)
        VALUES(?, ?, ?, ?, ?, ?, ?)
        """,
        rows_to_insert,
    )
    conn.commit()
    set_manifest(conn, "prompts_chat_signature", current_sig)


def decode_classlabel(value: Any, feature: Any) -> str:
    if isinstance(feature, ClassLabel):
        if isinstance(value, str):
            return value
        return feature.int2str(int(value))
    return clean_text(value)


def build_slot_examples_table(conn: sqlite3.Connection) -> None:
    rows_to_insert: List[Tuple[str, str, str, str, str, str]] = []
    hash_lines: List[str] = []

    for cfg in MASSIVE_CONFIGS:
        ds = load_massive_split(config=cfg, split=MASSIVE_SPLIT)

        validate_columns(ds.column_names, MASSIVE_COLUMNS, f"{MASSIVE_DATASET}/{cfg}")

        intent_feature = ds.features["intent"]

        for row in ds:
            utt = clean_text(row["utt"])
            annot_utt = clean_text(row["annot_utt"])
            intent = decode_classlabel(row["intent"], intent_feature)
            locale = clean_text(row["locale"])
            slots = parse_massive_annot_utt(annot_utt)

            if not utt or not slots:
                continue

            retrieval_text = slot_retrieval_text(
                source="MASSIVE",
                locale=locale,
                intent=intent,
                utt=utt,
                slots=slots,
            )

            slots_json = json.dumps(slots, ensure_ascii=False)
            rows_to_insert.append(
                ("MASSIVE", locale, intent, utt, slots_json, retrieval_text)
            )
            hash_lines.append(
                json.dumps(
                    ["MASSIVE", locale, intent, utt, slots],
                    ensure_ascii=False
                )
            )

    current_sig = stable_hash(hash_lines)
    existing_sig = get_manifest(conn, "slot_examples_signature")

    if existing_sig == current_sig:
        existing_count = conn.execute("SELECT COUNT(*) AS c FROM slot_examples").fetchone()["c"]
        if existing_count == len(rows_to_insert):
            return

    conn.execute("DELETE FROM slot_examples")
    conn.execute("DELETE FROM sqlite_sequence WHERE name = 'slot_examples'")
    conn.executemany(
        """
        INSERT INTO slot_examples(source, locale, intent, utt, slots_json, retrieval_text)
        VALUES(?, ?, ?, ?, ?, ?)
        """,
        rows_to_insert,
    )
    conn.commit()
    set_manifest(conn, "slot_examples_signature", current_sig)


# ============================================================
# INDEX BUILDERS
# ============================================================

def build_index_meta_signature(ids: List[int], texts: List[str], model: str) -> str:
    lines = [model, str(len(ids))]
    for i, t in zip(ids, texts):
        lines.append(f"{i}|{t}")
    return stable_hash(lines)


def build_or_load_index(
    ids: List[int],
    texts: List[str],
    model: str,
    index_file: Path,
    meta_file: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    signature = build_index_meta_signature(ids, texts, model)

    if index_file.exists() and meta_file.exists():
        try:
            meta = load_json(meta_file)
            if (
                meta.get("signature") == signature
                and meta.get("model") == model
                and meta.get("count") == len(ids)
            ):
                arr = np.load(index_file)
                return arr["ids"].astype(np.int64), arr["emb"].astype(np.float32)
        except Exception:
            pass

    embeddings = embed_texts(texts, model=model)
    np.savez_compressed(index_file, ids=np.array(ids, dtype=np.int64), emb=embeddings)
    save_json(meta_file, {
        "signature": signature,
        "model": model,
        "count": len(ids),
    })
    return np.array(ids, dtype=np.int64), embeddings


def load_prompt_index(conn: sqlite3.Connection) -> Tuple[np.ndarray, np.ndarray]:
    rows = conn.execute(
        "SELECT row_id, retrieval_text FROM prompts ORDER BY row_id ASC"
    ).fetchall()
    ids = [int(r["row_id"]) for r in rows]
    texts = [r["retrieval_text"] for r in rows]
    return build_or_load_index(ids, texts, EMBED_MODEL, PROMPT_INDEX_FILE, PROMPT_META_FILE)


def load_slot_index(conn: sqlite3.Connection) -> Tuple[np.ndarray, np.ndarray]:
    rows = conn.execute(
        "SELECT row_id, retrieval_text FROM slot_examples ORDER BY row_id ASC"
    ).fetchall()
    ids = [int(r["row_id"]) for r in rows]
    texts = [r["retrieval_text"] for r in rows]
    return build_or_load_index(ids, texts, EMBED_MODEL, SLOT_INDEX_FILE, SLOT_META_FILE)


def load_memory_index(conn: sqlite3.Connection) -> Tuple[np.ndarray, np.ndarray]:
    rows = conn.execute(
        """
        SELECT row_id, user_text, chosen_act, final_prompt
        FROM memory
        ORDER BY row_id ASC
        """
    ).fetchall()

    if not rows:
        return np.array([], dtype=np.int64), np.zeros((0, 1536), dtype=np.float32)

    ids = [int(r["row_id"]) for r in rows]
    texts = [
        f"USER: {clean_text(r['user_text'])}\nACT: {clean_text(r['chosen_act'])}\nFINAL_PROMPT:\n{clean_text(r['final_prompt'])}"
        for r in rows
    ]
    return build_or_load_index(ids, texts, EMBED_MODEL, MEMORY_INDEX_FILE, MEMORY_META_FILE)


# ============================================================
# RETRIEVAL
# ============================================================

def top_k_search(query: str, ids: np.ndarray, emb: np.ndarray, k: int) -> List[Tuple[int, float]]:
    if emb.shape[0] == 0:
        return []

    q = embed_texts([query], model=EMBED_MODEL)[0]
    scores = emb @ q
    order = np.argsort(-scores)[:k]
    return [(int(ids[i]), float(scores[i])) for i in order]


def fetch_prompt_candidates(
    conn: sqlite3.Connection,
    prompt_ids: np.ndarray,
    prompt_emb: np.ndarray,
    user_text: str,
) -> List[PromptCandidate]:
    hits = top_k_search(user_text, prompt_ids, prompt_emb, PROMPT_TOP_K)
    out: List[PromptCandidate] = []

    for row_id, score in hits:
        row = conn.execute(
            """
            SELECT row_id, act, prompt, for_devs, record_type, contributor
            FROM prompts
            WHERE row_id = ?
            """,
            (row_id,),
        ).fetchone()
        if row is None:
            continue
        out.append(
            PromptCandidate(
                row_id=int(row["row_id"]),
                act=clean_text(row["act"]),
                prompt=clean_text(row["prompt"]),
                for_devs=int(row["for_devs"]),
                record_type=clean_text(row["record_type"]),
                contributor=clean_text(row["contributor"]),
                score=score,
            )
        )
    return out


def fetch_slot_support(
    conn: sqlite3.Connection,
    slot_ids: np.ndarray,
    slot_emb: np.ndarray,
    user_text: str,
) -> List[SlotSupport]:
    hits = top_k_search(user_text, slot_ids, slot_emb, SLOT_TOP_K)
    out: List[SlotSupport] = []

    for row_id, score in hits:
        row = conn.execute(
            """
            SELECT row_id, source, locale, intent, utt, slots_json
            FROM slot_examples
            WHERE row_id = ?
            """,
            (row_id,),
        ).fetchone()
        if row is None:
            continue
        out.append(
            SlotSupport(
                row_id=int(row["row_id"]),
                source=clean_text(row["source"]),
                locale=clean_text(row["locale"]),
                intent=clean_text(row["intent"]),
                utt=clean_text(row["utt"]),
                slots=json.loads(row["slots_json"]),
                score=score,
            )
        )
    return out


def fetch_memory_support(
    conn: sqlite3.Connection,
    memory_ids: np.ndarray,
    memory_emb: np.ndarray,
    user_text: str,
) -> List[MemorySupport]:
    # exact text reuse first
    exact = conn.execute(
        """
        SELECT row_id, user_text, chosen_prompt_row_id, chosen_act, final_prompt, created_at
        FROM memory
        WHERE normalized_user_text = ?
        ORDER BY row_id DESC
        LIMIT 1
        """,
        (normalize_user_text(user_text),),
    ).fetchone()

    if exact is not None:
        return [
            MemorySupport(
                row_id=int(exact["row_id"]),
                user_text=clean_text(exact["user_text"]),
                chosen_prompt_row_id=int(exact["chosen_prompt_row_id"]),
                chosen_act=clean_text(exact["chosen_act"]),
                final_prompt=clean_text(exact["final_prompt"]),
                created_at=clean_text(exact["created_at"]),
                score=1.0,
            )
        ]

    hits = top_k_search(user_text, memory_ids, memory_emb, MEMORY_TOP_K)
    out: List[MemorySupport] = []

    for row_id, score in hits:
        row = conn.execute(
            """
            SELECT row_id, user_text, chosen_prompt_row_id, chosen_act, final_prompt, created_at
            FROM memory
            WHERE row_id = ?
            """,
            (row_id,),
        ).fetchone()
        if row is None:
            continue
        out.append(
            MemorySupport(
                row_id=int(row["row_id"]),
                user_text=clean_text(row["user_text"]),
                chosen_prompt_row_id=int(row["chosen_prompt_row_id"]),
                chosen_act=clean_text(row["chosen_act"]),
                final_prompt=clean_text(row["final_prompt"]),
                created_at=clean_text(row["created_at"]),
                score=score,
            )
        )
    return out


# ============================================================
# REPAIR LOGIC (LLM-DRIVEN, NOT RULE-BASED USER FILTERING)
# ============================================================

def build_repair_messages(
    user_text: str,
    prompt_candidates: List[PromptCandidate],
    slot_support: List[SlotSupport],
    memory_support: List[MemorySupport],
    intent_spec: IntentSpec,
) -> List[Dict[str, str]]:
    candidate_payload = [
        {
            "row_id": c.row_id,
            "act": c.act,
            "prompt": c.prompt,
            "for_devs": c.for_devs,
            "type": c.record_type,
            "contributor": c.contributor,
            "retrieval_score": round(c.score, 6),
        }
        for c in prompt_candidates
    ]

    slot_payload = [
        {
            "row_id": s.row_id,
            "source": s.source,
            "locale": s.locale,
            "intent": s.intent,
            "utt": s.utt,
            "slots": s.slots,
            "retrieval_score": round(s.score, 6),
        }
        for s in slot_support
    ]

    memory_payload = [
        {
            "row_id": m.row_id,
            "user_text": m.user_text,
            "chosen_prompt_row_id": m.chosen_prompt_row_id,
            "chosen_act": m.chosen_act,
            "final_prompt": m.final_prompt,
            "created_at": m.created_at,
            "retrieval_score": round(m.score, 6),
        }
        for m in memory_support
    ]

    system = (
        "You are a strict prompt-repair engine.\n"
        "Your task:\n"
        "1) Choose the best base prompt from the retrieved prompt candidates.\n"
        "2) Preserve the chosen prompt's wording, constraints, and style as much as possible.\n"
        "3) Use the user's input and intent specification to replace or adapt data-bearing/example parts.\n"
        "4) Use slot-support examples to understand what values are likely being mentioned in the user's input.\n"
        "5) Use memory examples when they are directly relevant; memory is preferred over invention.\n"
        "6) Do not invent a totally new prompt if a candidate already fits.\n"
        "7) Keep the base prompt language aligned to the user intent and base template language.\n"
        "8) The repaired prompt must be fully copy-paste ready: no unresolved placeholders, no TODO markers.\n"
        "9) Avoid pseudo-scientific or mystical phrasing.\n"
        "10) Output ONLY one JSON object.\n\n"
        "Return JSON with this exact schema:\n"
        "{\n"
        '  "chosen_prompt_row_id": <int>,\n'
        '  "chosen_act": <string>,\n'
        '  "repaired_prompt": <string>,\n'
        '  "used_values": [{"name": <string>, "value": <string>}],\n'
        '  "memory_reused": <true_or_false>,\n'
        '  "notes": <short string>\n'
        "}\n"
    )

    user = (
        "USER_INPUT:\n"
        f"{user_text}\n\n"
        "INTENT_SPEC_JSON:\n"
        f"{json.dumps(intent_spec.__dict__, ensure_ascii=False, indent=2)}\n\n"
        "PROMPT_CANDIDATES_JSON:\n"
        f"{json.dumps(candidate_payload, ensure_ascii=False, indent=2)}\n\n"
        "SLOT_SUPPORT_JSON:\n"
        f"{json.dumps(slot_payload, ensure_ascii=False, indent=2)}\n\n"
        "MEMORY_SUPPORT_JSON:\n"
        f"{json.dumps(memory_payload, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON only."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_intent_spec_messages(
    user_text: str,
    prompt_candidates: List[PromptCandidate],
) -> List[Dict[str, str]]:
    candidate_preview = [
        {
            "row_id": c.row_id,
            "act": c.act,
            "type": c.record_type,
            "prompt_preview": clip_text(c.prompt, 320),
            "retrieval_score": round(c.score, 6),
        }
        for c in prompt_candidates
    ]

    system = (
        "You extract user intent and output requirements for prompt generation.\n"
        "Return a compact JSON object with this schema:\n"
        "{\n"
        '  "objective": <string>,\n'
        '  "deliverable_type": <string>,\n'
        '  "audience": <string>,\n'
        '  "language": <string>,\n'
        '  "must_include": [<string>],\n'
        '  "style_constraints": [<string>],\n'
        '  "quality_targets": [<string>]\n'
        "}\n"
        "Rules:\n"
        "- Derive only from user input and retrieved context.\n"
        "- Be concrete, short, and technical.\n"
        "- No category labels or generic fluff.\n"
    )

    user = (
        "USER_INPUT:\n"
        f"{user_text}\n\n"
        "TOP_PROMPT_CANDIDATE_PREVIEW_JSON:\n"
        f"{json.dumps(candidate_preview, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON only."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def infer_intent_spec(user_text: str, prompt_candidates: List[PromptCandidate]) -> IntentSpec:
    try:
        data = chat_json(
            build_intent_spec_messages(user_text, prompt_candidates),
            model=INTENT_MODEL,
            max_tokens=MAX_INTENT_TOKENS,
            temperature=0,
        )
    except Exception:
        data = {}

    objective = clean_text(data.get("objective")) or clean_text(user_text)
    deliverable_type = clean_text(data.get("deliverable_type")) or "prompt"
    audience = clean_text(data.get("audience")) or "unspecified"
    language = clean_text(data.get("language")) or "same as user input"
    must_include = clean_list(data.get("must_include"))
    style_constraints = clean_list(data.get("style_constraints"))
    quality_targets = clean_list(data.get("quality_targets"))

    if not quality_targets:
        quality_targets = [
            "Fully specified output",
            "No unresolved placeholders",
            "Clear actionable wording",
        ]

    return IntentSpec(
        objective=objective,
        deliverable_type=deliverable_type,
        audience=audience,
        language=language,
        must_include=must_include,
        style_constraints=style_constraints,
        quality_targets=quality_targets,
    )


def build_polish_messages(
    user_text: str,
    chosen_act: str,
    repaired_prompt: str,
    intent_spec: IntentSpec,
) -> List[Dict[str, str]]:
    system = (
        "You are a prompt quality editor.\n"
        "Polish the draft prompt to maximize clarity, completeness, and execution readiness.\n"
        "Constraints:\n"
        "- Keep the original task and intent unchanged.\n"
        "- Keep the prompt practical and technically grounded.\n"
        "- Remove unresolved placeholders and template artifacts.\n"
        "- Avoid pseudo-scientific or mystical framing.\n"
        "Return JSON only with schema:\n"
        "{\n"
        '  "final_prompt": <string>,\n'
        '  "polish_notes": <short string>\n'
        "}\n"
    )

    user = (
        "USER_INPUT:\n"
        f"{user_text}\n\n"
        "CHOSEN_ACT:\n"
        f"{chosen_act}\n\n"
        "INTENT_SPEC_JSON:\n"
        f"{json.dumps(intent_spec.__dict__, ensure_ascii=False, indent=2)}\n\n"
        "DRAFT_PROMPT:\n"
        f"{repaired_prompt}\n\n"
        "Return JSON only."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def polish_repaired_prompt(
    user_text: str,
    chosen_act: str,
    repaired_prompt: str,
    intent_spec: IntentSpec,
) -> Tuple[str, str]:
    try:
        data = chat_json(
            build_polish_messages(user_text, chosen_act, repaired_prompt, intent_spec),
            model=POLISH_MODEL,
            max_tokens=MAX_POLISH_TOKENS,
            temperature=0.1,
        )
    except Exception as exc:
        return repaired_prompt, f"polish skipped: {clean_text(exc)}"

    final_prompt = clean_text(data.get("final_prompt"))
    polish_notes = clean_text(data.get("polish_notes"))
    if not final_prompt:
        final_prompt = repaired_prompt
    return final_prompt, polish_notes


def repair_prompt_with_model(
    user_text: str,
    prompt_candidates: List[PromptCandidate],
    slot_support: List[SlotSupport],
    memory_support: List[MemorySupport],
) -> Dict[str, Any]:
    if not prompt_candidates:
        raise RuntimeError("No prompt candidates found")

    intent_spec = infer_intent_spec(user_text, prompt_candidates)
    messages = build_repair_messages(
        user_text,
        prompt_candidates,
        slot_support,
        memory_support,
        intent_spec,
    )
    data = chat_json(messages, model=REPAIR_MODEL, max_tokens=MAX_REPAIR_TOKENS)

    # mild validation
    chosen_prompt_row_id = int(data.get("chosen_prompt_row_id", prompt_candidates[0].row_id))
    chosen_ids = {c.row_id for c in prompt_candidates}
    if chosen_prompt_row_id not in chosen_ids:
        chosen_prompt_row_id = prompt_candidates[0].row_id

    chosen_map = {c.row_id: c for c in prompt_candidates}
    chosen_act = clean_text(data.get("chosen_act")) or chosen_map[chosen_prompt_row_id].act
    repaired_prompt = clean_text(data.get("repaired_prompt"))

    if not repaired_prompt:
        repaired_prompt = chosen_map[chosen_prompt_row_id].prompt

    polished_prompt, polish_notes = polish_repaired_prompt(
        user_text=user_text,
        chosen_act=chosen_act,
        repaired_prompt=repaired_prompt,
        intent_spec=intent_spec,
    )

    used_values = data.get("used_values", [])
    if not isinstance(used_values, list):
        used_values = []

    memory_reused = bool(data.get("memory_reused", False))
    notes = clean_text(data.get("notes"))
    combined_notes = " | ".join(
        part for part in [notes, f"polish: {polish_notes}" if polish_notes else ""] if part
    )

    return {
        "chosen_prompt_row_id": chosen_prompt_row_id,
        "chosen_act": chosen_act,
        "repaired_prompt": polished_prompt,
        "used_values": used_values,
        "memory_reused": memory_reused,
        "notes": combined_notes,
        "intent_spec": intent_spec.__dict__,
    }


# ============================================================
# MEMORY
# ============================================================

def save_memory(
    conn: sqlite3.Connection,
    user_text: str,
    chosen_prompt_row_id: int,
    chosen_act: str,
    final_prompt: str,
    meta: Dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT INTO memory(
            created_at,
            normalized_user_text,
            user_text,
            chosen_prompt_row_id,
            chosen_act,
            final_prompt,
            meta_json
        )
        VALUES(?, ?, ?, ?, ?, ?, ?)
        """,
        (
            utc_now_iso(),
            normalize_user_text(user_text),
            user_text,
            int(chosen_prompt_row_id),
            chosen_act,
            final_prompt,
            json.dumps(meta, ensure_ascii=False),
        ),
    )
    conn.commit()


# ============================================================
# OUTPUT HELPERS
# ============================================================

def export_result(result: Dict[str, Any]) -> None:
    save_json(LAST_RESULT_JSON, result)
    LAST_RESULT_TXT.write_text(result["repaired_prompt"], encoding="utf-8")


def build_report_text(
    user_text: str,
    prompt_candidates: List[PromptCandidate],
    slot_support: List[SlotSupport],
    memory_support: List[MemorySupport],
    result: Dict[str, Any],
) -> str:
    lines: List[str] = []

    lines.append("=== USER INPUT ===")
    lines.append(user_text)
    lines.append("")

    lines.append("=== TOP PROMPT CANDIDATES ===")
    for c in prompt_candidates:
        lines.append(
            f"[row_id={c.row_id}] score={c.score:.4f} | act={c.act} | type={c.record_type} | contributor={c.contributor}"
        )
    lines.append("")

    lines.append("=== TOP SLOT SUPPORT ===")
    for s in slot_support:
        slot_names = ", ".join(sorted({x['slot'] for x in s.slots}))
        lines.append(
            f"[row_id={s.row_id}] score={s.score:.4f} | {s.source}/{s.locale} | intent={s.intent} | slots={slot_names}"
        )
        lines.append(f"  utt: {s.utt}")
    lines.append("")

    lines.append("=== MEMORY SUPPORT ===")
    if not memory_support:
        lines.append("(none)")
    else:
        for m in memory_support:
            lines.append(
                f"[row_id={m.row_id}] score={m.score:.4f} | chosen_act={m.chosen_act} | created_at={m.created_at}"
            )
            lines.append(f"  user_text: {m.user_text}")
    lines.append("")

    lines.append("=== FINAL RESULT ===")
    lines.append(json.dumps(result, ensure_ascii=False, indent=2))
    lines.append("")
    lines.append("=== REPAIRED PROMPT ===")
    lines.append(result["repaired_prompt"])

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ensure_dirs()
    conn = get_conn()
    init_db(conn)

    print("Local databases hazırlanıyor...")
    build_prompts_table(conn)
    build_slot_examples_table(conn)

    print("Indexler hazırlanıyor...")
    prompt_ids, prompt_emb = load_prompt_index(conn)
    slot_ids, slot_emb = load_slot_index(conn)
    memory_ids, memory_emb = load_memory_index(conn)

    user_text, used_gui = get_user_input()

    prompt_candidates = fetch_prompt_candidates(conn, prompt_ids, prompt_emb, user_text)
    slot_support = fetch_slot_support(conn, slot_ids, slot_emb, user_text)
    memory_support = fetch_memory_support(conn, memory_ids, memory_emb, user_text)

    result = repair_prompt_with_model(
        user_text=user_text,
        prompt_candidates=prompt_candidates,
        slot_support=slot_support,
        memory_support=memory_support,
    )

    result_meta = {
        "api_base_url": LLM_API_BASE_URL,
        "embed_model": EMBED_MODEL,
        "repair_model": REPAIR_MODEL,
        "intent_model": INTENT_MODEL,
        "polish_model": POLISH_MODEL,
        "prompt_top_k": PROMPT_TOP_K,
        "slot_top_k": SLOT_TOP_K,
        "memory_top_k": MEMORY_TOP_K,
        "prompt_candidate_ids": [c.row_id for c in prompt_candidates],
        "slot_support_ids": [s.row_id for s in slot_support],
        "memory_support_ids": [m.row_id for m in memory_support],
        "intent_spec": result["intent_spec"],
    }

    save_memory(
        conn=conn,
        user_text=user_text,
        chosen_prompt_row_id=result["chosen_prompt_row_id"],
        chosen_act=result["chosen_act"],
        final_prompt=result["repaired_prompt"],
        meta=result_meta,
    )

    # memory index may be outdated after this run; no need now, next run rebuilds or reuses
    full_result = {
        "created_at": utc_now_iso(),
        "user_text": user_text,
        "chosen_prompt_row_id": result["chosen_prompt_row_id"],
        "chosen_act": result["chosen_act"],
        "memory_reused": result["memory_reused"],
        "notes": result["notes"],
        "used_values": result["used_values"],
        "intent_spec": result["intent_spec"],
        "repaired_prompt": result["repaired_prompt"],
        "meta": result_meta,
    }

    export_result(full_result)
    report_text = build_report_text(user_text, prompt_candidates, slot_support, memory_support, full_result)

    print("\n=== REPAIRED PROMPT ===\n")
    print(full_result["repaired_prompt"])
    print(f"\nKaydedildi:\n- {LAST_RESULT_TXT}\n- {LAST_RESULT_JSON}\n- {DB_PATH}")

    if used_gui:
        _show_output_gui("Prompt Repair Result", report_text)


if __name__ == "__main__":
    main()
