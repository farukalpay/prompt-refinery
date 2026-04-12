from __future__ import annotations

import json
import os
import re
import time
import hashlib
import sqlite3
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 .* doesn't match a supported version!",
    category=Warning,
    module=r"requests(\..*)?",
)

import requests
from datasets import ClassLabel, get_dataset_config_names, load_dataset
from huggingface_hub import HfApi, hf_hub_url


PROMPT_DATASET = "fka/prompts.chat"
PROMPT_SPLIT = "train"

MASSIVE_DATASET = "AmazonScience/massive"
MASSIVE_SPLIT = "train"
MASSIVE_PARQUET_REVISION = "refs/convert/parquet"

PROMPTS_CHAT_COLUMNS = ["act", "prompt", "for_devs", "type", "contributor"]
MASSIVE_COLUMNS = [
    "id",
    "locale",
    "partition",
    "scenario",
    "intent",
    "utt",
    "annot_utt",
    "worker_id",
    "slot_method",
    "judgments",
]

DEFAULT_QUALITY_TARGETS: List[str] = [
    "Fully specified output",
    "No unresolved placeholders",
    "Clear actionable wording",
]

BATCH_SIZE = 64
REQUEST_TIMEOUT = 120
MAX_RETRIES = 6

EMBED_TEXT_CHUNK_CHARS = 3500
EMBED_TEXT_CHUNK_OVERLAP = 350

PROMPT_TOP_K = 6
SLOT_TOP_K = 14
MEMORY_TOP_K = 6

TEMPERATURE = 0.1
MAX_REPAIR_TOKENS = 1800

_MASSIVE_PARQUET_FILE_CACHE: Dict[Tuple[str, str], List[str]] = {}


@dataclass(frozen=True)
class RuntimePaths:
    project_dir: Path
    app_dir: Path
    hf_cache_dir: Path
    index_dir: Path
    export_dir: Path
    db_path: Path
    prompt_index_file: Path
    slot_index_file: Path
    memory_index_file: Path
    prompt_meta_file: Path
    slot_meta_file: Path
    memory_meta_file: Path
    last_result_json: Path
    last_result_txt: Path
    profile_file: Path

    @classmethod
    def from_project_dir(cls, project_dir: Path) -> "RuntimePaths":
        app_dir = project_dir / "runtime_db"
        index_dir = app_dir / "indices"
        export_dir = app_dir / "exports"
        return cls(
            project_dir=project_dir,
            app_dir=app_dir,
            hf_cache_dir=app_dir / "hf_cache",
            index_dir=index_dir,
            export_dir=export_dir,
            db_path=app_dir / "runtime.sqlite3",
            prompt_index_file=index_dir / "prompt_index.npz",
            slot_index_file=index_dir / "slot_index.npz",
            memory_index_file=index_dir / "memory_index.npz",
            prompt_meta_file=index_dir / "prompt_index_meta.json",
            slot_meta_file=index_dir / "slot_index_meta.json",
            memory_meta_file=index_dir / "memory_index_meta.json",
            last_result_json=export_dir / "last_result.json",
            last_result_txt=export_dir / "last_prompt.txt",
            profile_file=project_dir / "refinery_profile.json",
        )


@dataclass(frozen=True)
class RuntimeSettings:
    api_key: str
    api_base_url: str
    embeddings_url: str
    chat_completions_url: str
    embed_model: str
    repair_model: str
    quality_targets_env: Optional[List[str]]
    request_timeout: int = REQUEST_TIMEOUT
    max_retries: int = MAX_RETRIES
    batch_size: int = BATCH_SIZE
    prompt_top_k: int = PROMPT_TOP_K
    slot_top_k: int = SLOT_TOP_K
    memory_top_k: int = MEMORY_TOP_K
    max_repair_tokens: int = MAX_REPAIR_TOKENS
    temperature: float = TEMPERATURE

    @classmethod
    def from_env(cls, project_dir: Optional[Path] = None) -> "RuntimeSettings":
        base_dir = project_dir or Path.cwd()
        load_env_file(base_dir / ".env")

        api_key = first_env("OPENROUTER_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY")
        api_base_url = (first_env("LLM_API_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
        embed_model = first_env("EMBED_MODEL") or "openai/text-embedding-3-small"
        repair_model = first_env("REPAIR_MODEL") or "mistralai/mistral-nemo"

        env_targets = parse_quality_targets_json(first_env("QUALITY_TARGETS"))

        return cls(
            api_key=api_key,
            api_base_url=api_base_url,
            embeddings_url=f"{api_base_url}/embeddings",
            chat_completions_url=f"{api_base_url}/chat/completions",
            embed_model=embed_model,
            repair_model=repair_model,
            quality_targets_env=env_targets,
        )


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
    dominant_intents: List[str]
    relevant_slots: List[str]
    quality_targets: List[str]


class LLMClient(Protocol):
    def embed_texts(self, texts: Sequence[str], model: str) -> np.ndarray:
        ...

    def chat_json(
        self,
        messages: Sequence[Dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> Dict[str, Any]:
        ...


class HTTPModelClient:
    def __init__(self, settings: RuntimeSettings):
        self.settings = settings

    def _require_api_key(self) -> None:
        if not self.settings.api_key:
            raise RuntimeError(
                "API key is missing. Set OPENROUTER_API_KEY in .env or environment variables."
            )

    def _api_headers(self) -> Dict[str, str]:
        self._require_api_key()
        return {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }

    def _post_json_with_retry(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        last_err: Optional[Exception] = None
        for attempt in range(self.settings.max_retries):
            try:
                resp = requests.post(
                    url,
                    headers=self._api_headers(),
                    json=payload,
                    timeout=self.settings.request_timeout,
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    if attempt == self.settings.max_retries - 1:
                        resp.raise_for_status()
                    time.sleep(1.5 * (2**attempt))
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last_err = exc
                if attempt == self.settings.max_retries - 1:
                    break
                time.sleep(1.5 * (2**attempt))
        raise RuntimeError(f"HTTP request failed after retries: {last_err}")

    def _embed_batch(self, batch: Sequence[str], model: str) -> List[np.ndarray]:
        payload = {"model": model, "input": list(batch)}
        data = self._post_json_with_retry(self.settings.embeddings_url, payload)
        if "data" in data:
            ordered = sorted(data["data"], key=lambda item: item["index"])
            return [np.array(item["embedding"], dtype=np.float32) for item in ordered]

        # recursive split fallback for oversized batches
        if len(batch) > 1:
            mid = len(batch) // 2
            left = self._embed_batch(batch[:mid], model)
            right = self._embed_batch(batch[mid:], model)
            return left + right
        raise RuntimeError(f"Unexpected embedding response: {data}")

    def embed_texts(self, texts: Sequence[str], model: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        chunk_texts: List[str] = []
        owners: List[int] = []

        for idx, text in enumerate(texts):
            chunks = chunk_text_for_embedding(text)
            chunk_texts.extend(chunks)
            owners.extend([idx] * len(chunks))

        chunk_vectors: List[np.ndarray] = []
        for batch in batched(chunk_texts, self.settings.batch_size):
            chunk_vectors.extend(self._embed_batch(batch, model))

        grouped: List[List[np.ndarray]] = [[] for _ in range(len(texts))]
        for owner, vec in zip(owners, chunk_vectors):
            grouped[owner].append(vec)

        vectors: List[np.ndarray] = []
        for vecs in grouped:
            if not vecs:
                raise RuntimeError("Embedding aggregation produced an empty vector group.")
            vectors.append(np.mean(np.vstack(vecs).astype(np.float32), axis=0))

        return l2_normalize(np.vstack(vectors).astype(np.float32))

    def chat_json(
        self,
        messages: Sequence[Dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> Dict[str, Any]:
        payload = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = self._post_json_with_retry(self.settings.chat_completions_url, payload)

        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as exc:
            raise RuntimeError(f"Unexpected chat response: {data}") from exc

        try:
            return extract_first_json_object(content)
        except Exception:
            retry_messages = list(messages) + [
                {"role": "assistant", "content": content},
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
            retry_data = self._post_json_with_retry(self.settings.chat_completions_url, retry_payload)
            return extract_first_json_object(retry_data["choices"][0]["message"]["content"])


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
        normalized = value.strip()
        if normalized:
            return normalized
    return ""


def parse_quality_targets_json(raw_value: str) -> Optional[List[str]]:
    if not raw_value:
        return None
    try:
        parsed = json.loads(raw_value)
    except Exception:
        return None
    return clean_list(parsed) or None


def resolve_quality_targets(
    cli_targets: Optional[Sequence[str]],
    profile_path: Path,
    env_targets: Optional[Sequence[str]],
    default_targets: Optional[Sequence[str]] = None,
) -> List[str]:
    if cli_targets:
        cleaned = clean_list(list(cli_targets))
        if cleaned:
            return cleaned

    if profile_path.exists():
        try:
            data = json.loads(profile_path.read_text(encoding="utf-8"))
            profile_targets = clean_list(data.get("quality_targets"))
            if profile_targets:
                return profile_targets
        except Exception:
            pass

    if env_targets:
        cleaned_env = clean_list(list(env_targets))
        if cleaned_env:
            return cleaned_env

    return clean_list(list(default_targets or DEFAULT_QUALITY_TARGETS))


def ensure_dirs(paths: RuntimePaths) -> None:
    for directory in (paths.app_dir, paths.hf_cache_dir, paths.index_dir, paths.export_dir):
        directory.mkdir(parents=True, exist_ok=True)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_user_text(value: str) -> str:
    return re.sub(r"\s+", " ", clean_text(value)).casefold()


def clean_list(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    return [item for raw in items if (item := clean_text(raw))]


def stable_hash(items: Iterable[str]) -> str:
    hasher = hashlib.sha256()
    for item in items:
        hasher.update(item.encode("utf-8", errors="ignore"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim == 1:
        return matrix / (np.linalg.norm(matrix) + 1e-12)
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)


def batched(items: Sequence[str], batch_size: int) -> List[List[str]]:
    return [list(items[i : i + batch_size]) for i in range(0, len(items), batch_size)]


def chunk_text_for_embedding(text: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", clean_text(text))
    if not normalized:
        return [" "]
    if len(normalized) <= EMBED_TEXT_CHUNK_CHARS:
        return [normalized]

    step = max(1, EMBED_TEXT_CHUNK_CHARS - EMBED_TEXT_CHUNK_OVERLAP)
    chunks: List[str] = []
    for start in range(0, len(normalized), step):
        piece = normalized[start : start + EMBED_TEXT_CHUNK_CHARS]
        if not piece:
            continue
        chunks.append(piece)
        if start + EMBED_TEXT_CHUNK_CHARS >= len(normalized):
            break
    return chunks or [" "]


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_first_json_object(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("Empty model output")
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        return json.loads(stripped)
    except Exception:
        pass

    start = stripped.find("{")
    if start == -1:
        raise ValueError("No JSON object found")

    depth = 0
    for index in range(start, len(stripped)):
        if stripped[index] == "{":
            depth += 1
        elif stripped[index] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(stripped[start : index + 1])

    raise ValueError("Could not extract JSON object")


def bool_to_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(bool(value))
    return int(clean_text(value).lower() in {"1", "true", "yes", "y"})


def get_conn(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
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
        "INSERT INTO manifest(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
    conn.commit()


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
        file_path for file_path in files if file_path.startswith(prefix) and file_path.endswith(".parquet")
    )
    if not parquet_files:
        raise RuntimeError(
            f"No parquet files for {MASSIVE_DATASET} config={config} split={split}"
        )
    _MASSIVE_PARQUET_FILE_CACHE[cache_key] = parquet_files
    return parquet_files


def list_massive_configs() -> List[str]:
    try:
        configs = sorted(clean_list(get_dataset_config_names(MASSIVE_DATASET)))
        if configs:
            return configs
    except Exception:
        pass

    api = HfApi()
    files = api.list_repo_files(
        repo_id=MASSIVE_DATASET,
        repo_type="dataset",
        revision=MASSIVE_PARQUET_REVISION,
    )
    configs = sorted(
        {
            file_path.split("/", 1)[0]
            for file_path in files
            if file_path.endswith(".parquet") and "/" in file_path
        }
    )
    if not configs:
        raise RuntimeError(f"Could not discover configs for {MASSIVE_DATASET}")
    return configs


def load_massive_split(config: str, split: str, cache_dir: Path):
    try:
        return load_dataset(MASSIVE_DATASET, config, split=split, cache_dir=str(cache_dir))
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc):
            raise

    parquet_urls = [
        hf_hub_url(
            repo_id=MASSIVE_DATASET,
            repo_type="dataset",
            revision=MASSIVE_PARQUET_REVISION,
            filename=file_path,
        )
        for file_path in list_massive_parquet_files(config, split)
    ]
    return load_dataset(
        "parquet",
        data_files={split: parquet_urls},
        split=split,
        cache_dir=str(cache_dir),
    )


def validate_columns(actual: Sequence[str], expected: Sequence[str], dataset_name: str) -> None:
    if set(actual) != set(expected):
        raise RuntimeError(
            f"{dataset_name} schema mismatch.\nExpected: {list(expected)}\nActual:   {list(actual)}"
        )


def parse_massive_annot_utt(annot_utt: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for match in re.finditer(r"\[(.*?)\s*:\s*(.*?)\]", annot_utt):
        slot_name = clean_text(match.group(1)).lower().replace("-", "_").replace(" ", "_")
        slot_value = clean_text(match.group(2))
        if slot_name and slot_value:
            out.append({"slot": slot_name, "value": slot_value})
    return out


def prompt_retrieval_text(act: str, prompt: str, record_type: str) -> str:
    return f"ACT: {act}\nTYPE: {record_type}\nPROMPT:\n{prompt}"


def slot_retrieval_text(
    source: str,
    locale: str,
    intent: str,
    utt: str,
    slots: Sequence[Dict[str, str]],
) -> str:
    slot_names = ", ".join(sorted({slot["slot"] for slot in slots}))
    return (
        f"SOURCE: {source}\nLOCALE: {locale}\nINTENT: {intent}\n"
        f"UTTERANCE: {utt}\nSLOTS: {slot_names}"
    )


def decode_classlabel(value: Any, feature: Any) -> str:
    if isinstance(feature, ClassLabel):
        return value if isinstance(value, str) else feature.int2str(int(value))
    return clean_text(value)


def build_prompts_table(conn: sqlite3.Connection, cache_dir: Path) -> None:
    ds = load_dataset(PROMPT_DATASET, split=PROMPT_SPLIT, cache_dir=str(cache_dir))
    validate_columns(ds.column_names, PROMPTS_CHAT_COLUMNS, PROMPT_DATASET)

    rows: List[Tuple[int, str, str, int, str, str, str]] = []
    hash_lines: List[str] = []

    for idx, row in enumerate(ds):
        act = clean_text(row["act"])
        prompt = clean_text(row["prompt"])
        for_devs = bool_to_int(row["for_devs"])
        record_type = clean_text(row["type"])
        contributor = clean_text(row["contributor"])
        retrieval_text = prompt_retrieval_text(act, prompt, record_type)

        rows.append((idx, act, prompt, for_devs, record_type, contributor, retrieval_text))
        hash_lines.append(
            json.dumps([idx, act, prompt, for_devs, record_type, contributor], ensure_ascii=False)
        )

    signature = stable_hash(hash_lines)
    if get_manifest(conn, "prompts_chat_signature") == signature:
        row = conn.execute("SELECT COUNT(*) AS c FROM prompts").fetchone()
        if row and int(row["c"]) == len(rows):
            return

    conn.execute("DELETE FROM prompts")
    conn.executemany(
        "INSERT INTO prompts(row_id,act,prompt,for_devs,record_type,contributor,retrieval_text) "
        "VALUES(?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    set_manifest(conn, "prompts_chat_signature", signature)


def build_slot_examples_table(conn: sqlite3.Connection, cache_dir: Path) -> None:
    rows: List[Tuple[str, str, str, str, str, str]] = []
    hash_lines: List[str] = []

    for config in list_massive_configs():
        ds = load_massive_split(config=config, split=MASSIVE_SPLIT, cache_dir=cache_dir)
        validate_columns(ds.column_names, MASSIVE_COLUMNS, f"{MASSIVE_DATASET}/{config}")
        intent_feature = ds.features["intent"]

        for row in ds:
            utt = clean_text(row["utt"])
            annot_utt = clean_text(row["annot_utt"])
            intent = decode_classlabel(row["intent"], intent_feature)
            locale = clean_text(row["locale"])
            slots = parse_massive_annot_utt(annot_utt)
            if not utt or not slots:
                continue

            retrieval_text = slot_retrieval_text("MASSIVE", locale, intent, utt, slots)
            slots_json = json.dumps(slots, ensure_ascii=False)

            rows.append(("MASSIVE", locale, intent, utt, slots_json, retrieval_text))
            hash_lines.append(
                json.dumps(["MASSIVE", locale, intent, utt, slots], ensure_ascii=False)
            )

    signature = stable_hash(hash_lines)
    if get_manifest(conn, "slot_examples_signature") == signature:
        row = conn.execute("SELECT COUNT(*) AS c FROM slot_examples").fetchone()
        if row and int(row["c"]) == len(rows):
            return

    conn.execute("DELETE FROM slot_examples")
    conn.execute("DELETE FROM sqlite_sequence WHERE name = 'slot_examples'")
    conn.executemany(
        "INSERT INTO slot_examples(source,locale,intent,utt,slots_json,retrieval_text) "
        "VALUES(?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    set_manifest(conn, "slot_examples_signature", signature)


def index_signature(ids: Sequence[int], texts: Sequence[str], model: str) -> str:
    lines = [model, str(len(ids))]
    lines.extend(f"{row_id}|{text}" for row_id, text in zip(ids, texts))
    return stable_hash(lines)


def build_or_load_index(
    ids: Sequence[int],
    texts: Sequence[str],
    model: str,
    index_file: Path,
    meta_file: Path,
    llm_client: LLMClient,
) -> Tuple[np.ndarray, np.ndarray]:
    signature = index_signature(ids, texts, model)

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

    embeddings = llm_client.embed_texts(list(texts), model=model)
    np.savez_compressed(index_file, ids=np.array(ids, dtype=np.int64), emb=embeddings)
    save_json(meta_file, {"signature": signature, "model": model, "count": len(ids)})
    return np.array(ids, dtype=np.int64), embeddings


def load_prompt_index(
    conn: sqlite3.Connection,
    paths: RuntimePaths,
    embed_model: str,
    llm_client: LLMClient,
) -> Tuple[np.ndarray, np.ndarray]:
    rows = conn.execute("SELECT row_id, retrieval_text FROM prompts ORDER BY row_id").fetchall()
    ids = [int(row["row_id"]) for row in rows]
    texts = [row["retrieval_text"] for row in rows]
    return build_or_load_index(
        ids=ids,
        texts=texts,
        model=embed_model,
        index_file=paths.prompt_index_file,
        meta_file=paths.prompt_meta_file,
        llm_client=llm_client,
    )


def load_slot_index(
    conn: sqlite3.Connection,
    paths: RuntimePaths,
    embed_model: str,
    llm_client: LLMClient,
) -> Tuple[np.ndarray, np.ndarray]:
    rows = conn.execute("SELECT row_id, retrieval_text FROM slot_examples ORDER BY row_id").fetchall()
    ids = [int(row["row_id"]) for row in rows]
    texts = [row["retrieval_text"] for row in rows]
    return build_or_load_index(
        ids=ids,
        texts=texts,
        model=embed_model,
        index_file=paths.slot_index_file,
        meta_file=paths.slot_meta_file,
        llm_client=llm_client,
    )


def load_memory_index(
    conn: sqlite3.Connection,
    paths: RuntimePaths,
    embed_model: str,
    llm_client: LLMClient,
) -> Tuple[np.ndarray, np.ndarray]:
    rows = conn.execute(
        "SELECT row_id, user_text, chosen_act, final_prompt FROM memory ORDER BY row_id"
    ).fetchall()

    if not rows:
        return np.array([], dtype=np.int64), np.zeros((0, 1), dtype=np.float32)

    ids = [int(row["row_id"]) for row in rows]
    texts = [
        (
            f"USER: {clean_text(row['user_text'])}\n"
            f"ACT: {clean_text(row['chosen_act'])}\n"
            f"FINAL_PROMPT:\n{clean_text(row['final_prompt'])}"
        )
        for row in rows
    ]

    return build_or_load_index(
        ids=ids,
        texts=texts,
        model=embed_model,
        index_file=paths.memory_index_file,
        meta_file=paths.memory_meta_file,
        llm_client=llm_client,
    )


def top_k_search_from_vector(
    query_vector: np.ndarray,
    ids: np.ndarray,
    emb: np.ndarray,
    k: int,
) -> List[Tuple[int, float]]:
    if emb.shape[0] == 0:
        return []
    scores = emb @ query_vector
    order = np.argsort(-scores)[:k]
    return [(int(ids[i]), float(scores[i])) for i in order]


def fetch_prompt_candidates(
    conn: sqlite3.Connection,
    prompt_ids: np.ndarray,
    prompt_emb: np.ndarray,
    query_vector: np.ndarray,
    top_k: int,
) -> List[PromptCandidate]:
    hits = top_k_search_from_vector(query_vector, prompt_ids, prompt_emb, top_k)
    out: List[PromptCandidate] = []

    for row_id, score in hits:
        row = conn.execute(
            "SELECT row_id,act,prompt,for_devs,record_type,contributor "
            "FROM prompts WHERE row_id=?",
            (row_id,),
        ).fetchone()
        if not row:
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
    query_vector: np.ndarray,
    top_k: int,
) -> List[SlotSupport]:
    hits = top_k_search_from_vector(query_vector, slot_ids, slot_emb, top_k)
    out: List[SlotSupport] = []

    for row_id, score in hits:
        row = conn.execute(
            "SELECT row_id,source,locale,intent,utt,slots_json FROM slot_examples WHERE row_id=?",
            (row_id,),
        ).fetchone()
        if not row:
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
    query_vector: np.ndarray,
    top_k: int,
) -> List[MemorySupport]:
    exact = conn.execute(
        "SELECT row_id,user_text,chosen_prompt_row_id,chosen_act,final_prompt,created_at "
        "FROM memory WHERE normalized_user_text=? ORDER BY row_id DESC LIMIT 1",
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

    hits = top_k_search_from_vector(query_vector, memory_ids, memory_emb, top_k)
    out: List[MemorySupport] = []

    for row_id, score in hits:
        row = conn.execute(
            "SELECT row_id,user_text,chosen_prompt_row_id,chosen_act,final_prompt,created_at "
            "FROM memory WHERE row_id=?",
            (row_id,),
        ).fetchone()
        if not row:
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


def build_intent_spec_from_retrieval(
    user_text: str,
    prompt_candidates: Sequence[PromptCandidate],
    slot_support: Sequence[SlotSupport],
    quality_targets: Optional[Sequence[str]] = None,
) -> IntentSpec:
    if prompt_candidates:
        objective = prompt_candidates[0].act
        deliverable_type = prompt_candidates[0].record_type
    else:
        objective = user_text
        deliverable_type = "prompt"

    if prompt_candidates:
        dev_score = sum(candidate.for_devs * candidate.score for candidate in prompt_candidates)
        gen_score = sum((1 - candidate.for_devs) * candidate.score for candidate in prompt_candidates)
        audience = "developers" if dev_score > gen_score else "general"
    else:
        audience = "general"

    locale_scores: Dict[str, float] = {}
    for support in slot_support:
        locale_scores[support.locale] = locale_scores.get(support.locale, 0.0) + support.score
    language = max(locale_scores, key=locale_scores.get) if locale_scores else "en-US"

    intent_counter: Counter[str] = Counter()
    for support in slot_support:
        intent_counter[support.intent] += 1
    dominant_intents = [intent for intent, _ in intent_counter.most_common(5)]

    slot_counter: Counter[str] = Counter()
    for support in slot_support:
        for slot in support.slots:
            slot_counter[slot["slot"]] += 1
    relevant_slots = [slot_name for slot_name, _ in slot_counter.most_common(12)]

    resolved_targets = clean_list(list(quality_targets)) if quality_targets else list(DEFAULT_QUALITY_TARGETS)

    return IntentSpec(
        objective=objective,
        deliverable_type=deliverable_type,
        audience=audience,
        language=language,
        dominant_intents=dominant_intents,
        relevant_slots=relevant_slots,
        quality_targets=resolved_targets,
    )


def build_repair_messages(
    user_text: str,
    prompt_candidates: Sequence[PromptCandidate],
    slot_support: Sequence[SlotSupport],
    memory_support: Sequence[MemorySupport],
    intent_spec: IntentSpec,
) -> List[Dict[str, str]]:
    candidate_payload = [
        {
            "row_id": candidate.row_id,
            "act": candidate.act,
            "prompt": candidate.prompt,
            "for_devs": candidate.for_devs,
            "type": candidate.record_type,
            "contributor": candidate.contributor,
            "retrieval_score": round(candidate.score, 6),
        }
        for candidate in prompt_candidates
    ]

    slot_payload = [
        {
            "row_id": support.row_id,
            "source": support.source,
            "locale": support.locale,
            "intent": support.intent,
            "utt": support.utt,
            "slots": support.slots,
            "retrieval_score": round(support.score, 6),
        }
        for support in slot_support
    ]

    memory_payload = [
        {
            "row_id": memory.row_id,
            "user_text": memory.user_text,
            "chosen_prompt_row_id": memory.chosen_prompt_row_id,
            "chosen_act": memory.chosen_act,
            "final_prompt": memory.final_prompt,
            "created_at": memory.created_at,
            "retrieval_score": round(memory.score, 6),
        }
        for memory in memory_support
    ]

    system = (
        "You are a strict prompt-repair and quality-polish engine.\n"
        "You receive ONE user input plus retrieval context and produce ONE final prompt.\n\n"
        "REPAIR rules:\n"
        "1) Choose the best base prompt from the retrieved prompt candidates.\n"
        "2) Preserve the chosen prompt's wording, constraints, and style as much as possible.\n"
        "3) Use the user's input and intent specification to replace or adapt data-bearing/example parts.\n"
        "4) Use slot-support examples to understand what named entities and values the user is referencing.\n"
        "5) Use memory examples when directly relevant; memory is preferred over invention.\n"
        "6) Do not invent a totally new prompt if a candidate already fits.\n"
        "7) Keep the base prompt language aligned to the user intent and base template language.\n\n"
        "POLISH rules (apply in the same pass - do NOT output a draft):\n"
        "8) The output must be fully copy-paste ready: no unresolved placeholders, no TODO markers.\n"
        "9) Maximize clarity, completeness, and execution readiness.\n"
        "10) Remove any template artifacts, placeholder markers, or incomplete sections.\n"
        "11) Avoid pseudo-scientific or mystical phrasing.\n"
        "12) Ensure the final text is grammatically clean and professionally worded.\n\n"
        "Output ONLY one JSON object with this exact schema:\n"
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
        f"USER_INPUT:\n{user_text}\n\n"
        f"INTENT_SPEC_JSON:\n{json.dumps(intent_spec.__dict__, ensure_ascii=False, indent=2)}\n\n"
        f"PROMPT_CANDIDATES_JSON:\n{json.dumps(candidate_payload, ensure_ascii=False, indent=2)}\n\n"
        f"SLOT_SUPPORT_JSON:\n{json.dumps(slot_payload, ensure_ascii=False, indent=2)}\n\n"
        f"MEMORY_SUPPORT_JSON:\n{json.dumps(memory_payload, ensure_ascii=False, indent=2)}\n\n"
        "Return JSON only."
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def repair_prompt(
    llm_client: LLMClient,
    settings: RuntimeSettings,
    user_text: str,
    prompt_candidates: Sequence[PromptCandidate],
    slot_support: Sequence[SlotSupport],
    memory_support: Sequence[MemorySupport],
    quality_targets: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    if not prompt_candidates:
        raise RuntimeError("No prompt candidates found")

    intent_spec = build_intent_spec_from_retrieval(
        user_text=user_text,
        prompt_candidates=prompt_candidates,
        slot_support=slot_support,
        quality_targets=quality_targets,
    )

    messages = build_repair_messages(
        user_text=user_text,
        prompt_candidates=prompt_candidates,
        slot_support=slot_support,
        memory_support=memory_support,
        intent_spec=intent_spec,
    )

    data = llm_client.chat_json(
        messages=messages,
        model=settings.repair_model,
        max_tokens=settings.max_repair_tokens,
        temperature=settings.temperature,
    )

    chosen_prompt_row_id = int(data.get("chosen_prompt_row_id", prompt_candidates[0].row_id))
    valid_ids = {candidate.row_id for candidate in prompt_candidates}
    if chosen_prompt_row_id not in valid_ids:
        chosen_prompt_row_id = prompt_candidates[0].row_id

    chosen_map = {candidate.row_id: candidate for candidate in prompt_candidates}
    chosen_act = clean_text(data.get("chosen_act")) or chosen_map[chosen_prompt_row_id].act
    repaired_prompt = clean_text(data.get("repaired_prompt"))
    if not repaired_prompt:
        repaired_prompt = chosen_map[chosen_prompt_row_id].prompt

    used_values = data.get("used_values", [])
    if not isinstance(used_values, list):
        used_values = []

    return {
        "chosen_prompt_row_id": chosen_prompt_row_id,
        "chosen_act": chosen_act,
        "repaired_prompt": repaired_prompt,
        "used_values": used_values,
        "memory_reused": bool(data.get("memory_reused", False)),
        "notes": clean_text(data.get("notes")),
        "intent_spec": intent_spec.__dict__,
    }


def save_memory(
    conn: sqlite3.Connection,
    user_text: str,
    chosen_prompt_row_id: int,
    chosen_act: str,
    final_prompt: str,
    meta: Dict[str, Any],
) -> None:
    conn.execute(
        "INSERT INTO memory("
        "created_at, normalized_user_text, user_text, "
        "chosen_prompt_row_id, chosen_act, final_prompt, meta_json"
        ") VALUES(?,?,?,?,?,?,?)",
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


def export_result(paths: RuntimePaths, result: Dict[str, Any]) -> None:
    save_json(paths.last_result_json, result)
    paths.last_result_txt.write_text(result["repaired_prompt"], encoding="utf-8")


class RefineryEngine:
    def __init__(
        self,
        settings: Optional[RuntimeSettings] = None,
        paths: Optional[RuntimePaths] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        self.paths = paths or RuntimePaths.from_project_dir(Path.cwd())
        self.settings = settings or RuntimeSettings.from_env(self.paths.project_dir)
        self.llm_client = llm_client or HTTPModelClient(self.settings)

        self._conn: Optional[sqlite3.Connection] = None
        self._prompt_index: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._slot_index: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._memory_index: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._prepared = False

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = get_conn(self.paths.db_path)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def prepare(self) -> None:
        if self._prepared:
            return

        ensure_dirs(self.paths)
        init_db(self.conn)
        build_prompts_table(self.conn, self.paths.hf_cache_dir)
        build_slot_examples_table(self.conn, self.paths.hf_cache_dir)

        self._prompt_index = load_prompt_index(
            conn=self.conn,
            paths=self.paths,
            embed_model=self.settings.embed_model,
            llm_client=self.llm_client,
        )
        self._slot_index = load_slot_index(
            conn=self.conn,
            paths=self.paths,
            embed_model=self.settings.embed_model,
            llm_client=self.llm_client,
        )
        self._memory_index = load_memory_index(
            conn=self.conn,
            paths=self.paths,
            embed_model=self.settings.embed_model,
            llm_client=self.llm_client,
        )

        self._prepared = True

    def reload_memory_index(self) -> None:
        self._memory_index = load_memory_index(
            conn=self.conn,
            paths=self.paths,
            embed_model=self.settings.embed_model,
            llm_client=self.llm_client,
        )

    def resolve_targets(
        self,
        cli_targets: Optional[Sequence[str]] = None,
        profile_path: Optional[Path] = None,
    ) -> List[str]:
        effective_profile = profile_path or self.paths.profile_file
        return resolve_quality_targets(
            cli_targets=cli_targets,
            profile_path=effective_profile,
            env_targets=self.settings.quality_targets_env,
            default_targets=DEFAULT_QUALITY_TARGETS,
        )

    def run(
        self,
        user_text: str,
        quality_targets: Optional[Sequence[str]] = None,
        export_outputs: bool = True,
    ) -> Dict[str, Any]:
        normalized_user_text = clean_text(user_text)
        if not normalized_user_text:
            raise RuntimeError("user_text cannot be empty")

        self.prepare()

        if self._prompt_index is None or self._slot_index is None or self._memory_index is None:
            raise RuntimeError("Engine indices are not initialized")

        prompt_ids, prompt_emb = self._prompt_index
        slot_ids, slot_emb = self._slot_index
        memory_ids, memory_emb = self._memory_index

        # optimization: embed query once and reuse it across all retrieval spaces
        query_vector = self.llm_client.embed_texts(
            [normalized_user_text], model=self.settings.embed_model
        )[0]

        prompt_candidates = fetch_prompt_candidates(
            conn=self.conn,
            prompt_ids=prompt_ids,
            prompt_emb=prompt_emb,
            query_vector=query_vector,
            top_k=self.settings.prompt_top_k,
        )
        slot_support = fetch_slot_support(
            conn=self.conn,
            slot_ids=slot_ids,
            slot_emb=slot_emb,
            query_vector=query_vector,
            top_k=self.settings.slot_top_k,
        )
        memory_support = fetch_memory_support(
            conn=self.conn,
            memory_ids=memory_ids,
            memory_emb=memory_emb,
            user_text=normalized_user_text,
            query_vector=query_vector,
            top_k=self.settings.memory_top_k,
        )

        resolved_targets = (
            clean_list(list(quality_targets))
            if quality_targets is not None
            else self.resolve_targets(cli_targets=None)
        )

        result = repair_prompt(
            llm_client=self.llm_client,
            settings=self.settings,
            user_text=normalized_user_text,
            prompt_candidates=prompt_candidates,
            slot_support=slot_support,
            memory_support=memory_support,
            quality_targets=resolved_targets,
        )

        result_meta = {
            "api_base_url": self.settings.api_base_url,
            "embed_model": self.settings.embed_model,
            "repair_model": self.settings.repair_model,
            "prompt_top_k": self.settings.prompt_top_k,
            "slot_top_k": self.settings.slot_top_k,
            "memory_top_k": self.settings.memory_top_k,
            "quality_targets": resolved_targets,
            "prompt_candidate_ids": [candidate.row_id for candidate in prompt_candidates],
            "slot_support_ids": [support.row_id for support in slot_support],
            "memory_support_ids": [support.row_id for support in memory_support],
            "intent_spec": result["intent_spec"],
        }

        save_memory(
            conn=self.conn,
            user_text=normalized_user_text,
            chosen_prompt_row_id=result["chosen_prompt_row_id"],
            chosen_act=result["chosen_act"],
            final_prompt=result["repaired_prompt"],
            meta=result_meta,
        )

        full_result = {
            "created_at": utc_now_iso(),
            "user_text": normalized_user_text,
            "chosen_prompt_row_id": result["chosen_prompt_row_id"],
            "chosen_act": result["chosen_act"],
            "memory_reused": result["memory_reused"],
            "notes": result["notes"],
            "used_values": result["used_values"],
            "intent_spec": result["intent_spec"],
            "repaired_prompt": result["repaired_prompt"],
            "meta": result_meta,
        }

        if export_outputs:
            export_result(self.paths, full_result)

        # memory table changed; keep memory index in sync for next request in the same process
        self.reload_memory_index()

        return full_result


__all__ = [
    "DEFAULT_QUALITY_TARGETS",
    "HTTPModelClient",
    "IntentSpec",
    "LLMClient",
    "MemorySupport",
    "PromptCandidate",
    "RefineryEngine",
    "RuntimePaths",
    "RuntimeSettings",
    "SlotSupport",
    "build_intent_spec_from_retrieval",
    "build_report_text",
    "clean_list",
    "parse_massive_annot_utt",
    "resolve_quality_targets",
]
