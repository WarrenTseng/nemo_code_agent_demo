"""Knowledge loading utilities for coder_tool and the Planner.

Feature 1 — Static injection (always on when file exists):
    Reads ``knowledge/static.md`` (or ``$CODER_KNOWLEDGE_DIR/static.md``) and
    injects the entire file as additional context into coder_tool or the Planner.

Feature 2 — RAG retrieval (opt-in via CODER_RAG_ENABLED=true):
    Indexes all .md files under ``knowledge/RAG/`` using an OpenAI-compatible
    embedding API (NIM, vLLM, etc.).  On each invocation retrieves the top-k
    most relevant chunks for the task and injects only those.

Directory layout (relative to cwd / -w working dir):
    knowledge/
        static.md       ← Feature 1
        RAG/
            *.md        ← Feature 2

Environment variables:
    ENABLE_KNOWLEDGE            default: true  — master on/off switch for all knowledge
    CODER_KNOWLEDGE_DIR         default: knowledge
    CODER_RAG_ENABLED           default: false
    CODER_RAG_TOP_K             default: 5
    CODER_RAG_CHUNK_SIZE        default: 500
    CODER_RAG_CHUNK_OVERLAP     default: 50
    CODER_EMBEDDING_URL         default: CODER_URL → PLANNER_URL
    CODER_EMBEDDING_MODEL       default: nvidia/llama-nemotron-embed-1b-v2
    CODER_EMBEDDING_API_KEY     default: CODER_API_KEY → PLANNER_API_KEY → "none"
"""

import os
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from nemo_code_agent.utils.logger import get_logger

logger = get_logger(__name__)


def is_knowledge_enabled() -> bool:
    """Return True unless ENABLE_KNOWLEDGE is explicitly set to false/0/no."""
    return os.environ.get("ENABLE_KNOWLEDGE", "true").lower() not in ("false", "0", "no")


def _knowledge_dir() -> Path:
    """Resolve the knowledge base directory (relative to cwd at call time)."""
    return Path(os.environ.get("CODER_KNOWLEDGE_DIR", "knowledge"))


# ---------------------------------------------------------------------------
# Feature 1 — Static MD injection
# ---------------------------------------------------------------------------


def load_static_knowledge() -> str:
    """Return the content of static.md, or empty string if disabled/not found."""
    if not is_knowledge_enabled():
        return ""
    static_path = _knowledge_dir() / "static.md"
    if not static_path.exists():
        return ""
    try:
        content = static_path.read_text(encoding="utf-8").strip()
        logger.debug("Static knowledge loaded | path=%s | chars=%d", static_path, len(content))
        return content
    except Exception as exc:
        logger.warning("Failed to load static knowledge | path=%s | error=%s", static_path, exc)
        return ""


# ---------------------------------------------------------------------------
# Feature 2 — RAG retrieval
# ---------------------------------------------------------------------------


def _embedding_config() -> tuple[str, str, str]:
    """Return (url, model, api_key) from env vars."""
    url = (
        os.environ.get("CODER_EMBEDDING_URL")
        or os.environ.get("CODER_URL")
        or os.environ.get("PLANNER_URL")
        or ""
    )
    model = os.environ.get("CODER_EMBEDDING_MODEL", "nvidia/llama-nemotron-embed-1b-v2")
    api_key = (
        os.environ.get("CODER_EMBEDDING_API_KEY")
        or os.environ.get("CODER_API_KEY")
        or os.environ.get("PLANNER_API_KEY")
        or "none"
    )
    return url, model, api_key


class _RawEmbeddings:
    """Embeddings using the raw openai client so extra_body can be passed.

    langchain_openai.OpenAIEmbeddings does not forward model_kwargs to the
    embeddings.create() call, so we bypass it and use the openai SDK directly.
    This lets us pass NIM-specific fields like ``input_type`` via extra_body.

    Args:
        doc_input_type:   ``input_type`` value for embed_documents calls.
                          ``None`` → field is omitted (symmetric models).
        query_input_type: ``input_type`` value for embed_query calls.
                          ``None`` → field is omitted.
    """

    def __init__(
        self,
        doc_input_type: Optional[str] = None,
        query_input_type: Optional[str] = None,
    ) -> None:
        import openai

        url, model, api_key = _embedding_config()
        self._client = openai.OpenAI(base_url=url, api_key=api_key)
        self._model = model
        self._doc_input_type   = doc_input_type
        self._query_input_type = query_input_type

    def _embed(self, texts: list[str], input_type: Optional[str]) -> list[list[float]]:
        extra_body = {"input_type": input_type} if input_type else {}
        response = self._client.embeddings.create(
            model=self._model,
            input=texts,
            extra_body=extra_body,
        )
        # Sort by index to guarantee order matches input list
        return [d.embedding for d in sorted(response.data, key=lambda x: x.index)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, self._doc_input_type)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], self._query_input_type)[0]


class _ProgressEmbeddings:
    """Wraps _RawEmbeddings to show a Rich progress bar during embed_documents.

    Processes texts in batches so the progress bar advances incrementally.
    embed_query is passed through without a progress bar (single vector, fast).
    """

    _BATCH_SIZE = 16

    def __init__(self, inner: _RawEmbeddings) -> None:
        self._inner = inner

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        results: list[list[float]] = []
        total = len(texts)
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Embedding knowledge base…[/bold cyan]"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("embed", total=total)
            for i in range(0, total, self._BATCH_SIZE):
                batch = texts[i : i + self._BATCH_SIZE]
                results.extend(self._inner.embed_documents(batch))
                progress.advance(task, len(batch))
        return results

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)


def _get_embeddings(show_progress: bool = False):
    """Return the correct embeddings object based on CODER_EMBEDDING_INPUT_TYPE.

    - ``asymmetric`` (default): passage for docs, query for queries (NIM asymmetric)
    - ``symmetric``:            no input_type sent (OpenAI, local models)
    - any other string:         use that value for both doc and query embedding

    Args:
        show_progress: Wrap with _ProgressEmbeddings to show a Rich progress bar
                       during embed_documents (used when building the index).
    """
    mode = os.environ.get("CODER_EMBEDDING_INPUT_TYPE", "asymmetric").lower()
    if mode == "symmetric":
        inner = _RawEmbeddings(doc_input_type=None, query_input_type=None)
    elif mode == "asymmetric":
        inner = _RawEmbeddings(doc_input_type="passage", query_input_type="query")
    else:
        inner = _RawEmbeddings(doc_input_type=mode, query_input_type=mode)

    return _ProgressEmbeddings(inner) if show_progress else inner


class KnowledgeRetriever:
    """Lazy-initialised RAG retriever over ``knowledge/RAG/*.md`` files.

    The vector index is built on first use and automatically rebuilt when any
    source file is added or modified (mtime check).
    """

    def __init__(self) -> None:
        self._vectorstore = None
        self._indexed_mtime: dict[str, float] = {}
        self._rag_dir: Optional[Path] = None

    @staticmethod
    def _store_mode() -> str:
        return os.environ.get("CODER_RAG_STORE", "memory").lower()

    @staticmethod
    def _chroma_dir() -> Path:
        """Persistent Chroma DB directory, stored alongside the knowledge dir."""
        return _knowledge_dir() / ".chromadb"

    @staticmethod
    def _mtime_file() -> Path:
        """JSON file that tracks per-source mtime for the Chroma store."""
        return KnowledgeRetriever._chroma_dir() / "mtime.json"

    def _load_mtime_cache(self) -> dict[str, float]:
        mtime_file = self._mtime_file()
        if mtime_file.exists():
            try:
                import json
                return json.loads(mtime_file.read_text())
            except Exception:
                pass
        return {}

    def _save_mtime_cache(self, mtime: dict[str, float]) -> None:
        import json
        mtime_file = self._mtime_file()
        mtime_file.parent.mkdir(parents=True, exist_ok=True)
        mtime_file.write_text(json.dumps(mtime))

    def _needs_rebuild(self) -> bool:
        rag_dir = _knowledge_dir() / "RAG"
        if rag_dir != self._rag_dir:
            return True
        if not rag_dir.exists():
            return False

        # For Chroma: compare against persisted mtime cache
        if self._store_mode() == "chroma":
            cached = self._load_mtime_cache()
            for md_file in rag_dir.glob("*.md"):
                key = str(md_file)
                if key not in cached or md_file.stat().st_mtime != cached[key]:
                    return True
            return False

        # For memory: compare against in-process mtime dict
        if self._vectorstore is None:
            return True
        for md_file in rag_dir.glob("*.md"):
            key = str(md_file)
            if key not in self._indexed_mtime:
                return True
            if md_file.stat().st_mtime != self._indexed_mtime[key]:
                return True
        return False

    def _build_index(self) -> None:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        rag_dir = _knowledge_dir() / "RAG"
        self._rag_dir = rag_dir

        if not rag_dir.exists():
            logger.debug("RAG dir not found: %s", rag_dir)
            self._vectorstore = None
            return

        md_files = sorted(rag_dir.glob("*.md"))
        if not md_files:
            logger.debug("No .md files in RAG dir: %s", rag_dir)
            self._vectorstore = None
            return

        chunk_size    = int(os.environ.get("CODER_RAG_CHUNK_SIZE",    "500"))
        chunk_overlap = int(os.environ.get("CODER_RAG_CHUNK_OVERLAP", "50"))
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        docs = []
        new_mtime: dict[str, float] = {}
        for md_file in md_files:
            try:
                content = md_file.read_text(encoding="utf-8")
                chunks = splitter.create_documents(
                    [content],
                    metadatas=[{"source": md_file.name}],
                )
                docs.extend(chunks)
                new_mtime[str(md_file)] = md_file.stat().st_mtime
            except Exception as exc:
                logger.warning("Failed to load RAG file %s: %s", md_file.name, exc)

        if not docs:
            self._vectorstore = None
            return

        embeddings = _get_embeddings(show_progress=True)
        store_mode = self._store_mode()

        if store_mode == "chroma":
            try:
                from langchain_chroma import Chroma
            except ImportError:
                from langchain_community.vectorstores import Chroma  # type: ignore[no-redef]

            chroma_dir = str(self._chroma_dir())
            # Wipe and recreate to ensure clean state when files changed
            self._vectorstore = Chroma.from_documents(
                docs,
                embeddings,
                persist_directory=chroma_dir,
                collection_name="rag_knowledge",
            )
            self._save_mtime_cache(new_mtime)
            logger.info(
                "RAG index built (chroma) | dir=%s | files=%d | chunks=%d | db=%s",
                rag_dir, len(md_files), len(docs), chroma_dir,
            )
        else:
            from langchain_core.vectorstores import InMemoryVectorStore
            self._vectorstore = InMemoryVectorStore.from_documents(docs, embeddings)
            self._indexed_mtime = new_mtime
            logger.debug(
                "RAG index built (memory) | dir=%s | files=%d | chunks=%d",
                rag_dir, len(md_files), len(docs),
            )

    def _load_chroma(self) -> bool:
        """Load existing Chroma DB from disk without re-embedding. Returns True on success."""
        try:
            from langchain_chroma import Chroma
        except ImportError:
            try:
                from langchain_community.vectorstores import Chroma  # type: ignore[no-redef]
            except ImportError:
                return False

        chroma_dir = self._chroma_dir()
        if not chroma_dir.exists():
            return False
        try:
            embeddings = _get_embeddings()
            self._vectorstore = Chroma(
                persist_directory=str(chroma_dir),
                embedding_function=embeddings,
                collection_name="rag_knowledge",
            )
            logger.info("RAG index loaded from disk | db=%s", chroma_dir)
            return True
        except Exception as exc:
            logger.warning("Failed to load Chroma DB: %s", exc)
            return False

    def retrieve(self, query: str) -> str:
        """Return top-k relevant chunks for *query*, or empty string on failure."""
        if self._needs_rebuild():
            # Chroma: try loading from disk before re-embedding
            if self._store_mode() == "chroma" and self._vectorstore is None:
                if self._load_chroma():
                    self._rag_dir = _knowledge_dir() / "RAG"
                    # Check again — if files changed since last index, rebuild
                    if self._needs_rebuild():
                        self._build_index()
                else:
                    self._build_index()
            else:
                self._build_index()

        if self._vectorstore is None:
            return ""

        top_k = int(os.environ.get("CODER_RAG_TOP_K", "5"))
        try:
            results = self._vectorstore.similarity_search(query, k=top_k)
            if not results:
                return ""
            chunks = [
                f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
                for doc in results
            ]
            retrieved = "\n\n---\n\n".join(chunks)
            logger.debug(
                "RAG retrieved | query_len=%d | chunks=%d", len(query), len(results)
            )
            return retrieved
        except Exception as exc:
            logger.warning("RAG retrieval failed: %s", exc)
            return ""


# Module-level singleton — one index per process lifetime.
_retriever: Optional[KnowledgeRetriever] = None


def _get_retriever() -> KnowledgeRetriever:
    global _retriever
    if _retriever is None:
        _retriever = KnowledgeRetriever()
    return _retriever


# ---------------------------------------------------------------------------
# Public helper — used by coder_tool
# ---------------------------------------------------------------------------


def build_system_prompt(base_prompt: str) -> str:
    """Return the coder system prompt with static.md rules appended inline.

    Merging into the system prompt (rather than a separate SystemMessage)
    ensures the model treats the rules as first-class instructions.
    Feature 1 fires whenever knowledge/static.md exists.
    """
    static = load_static_knowledge()
    if not static:
        return base_prompt
    return (
        base_prompt
        + "\n\n## Project Rules (MANDATORY)\n\n"
        + "The following rules were configured by the user. "
        + "You MUST follow every rule without exception:\n\n"
        + static
    )


def build_planner_knowledge_messages(query: str) -> list[HumanMessage]:
    """Return knowledge context messages to inject into the Planner when coder_tool is disabled.

    Used by workflow.planner_node when ENABLE_CODER=false and ENABLE_KNOWLEDGE=true.
    Returns a list with a single HumanMessage containing static rules and/or RAG
    chunks, or an empty list if nothing is available.

    Args:
        query: The current user request (used as the RAG retrieval query).

    Returns:
        A list with one HumanMessage, or [] if knowledge is disabled/empty.
    """
    if not is_knowledge_enabled():
        return []

    parts: list[str] = []

    static = load_static_knowledge()
    if static:
        parts.append(
            "## Project Rules (MANDATORY)\n\n"
            "You MUST follow these rules in every line of code you write:\n\n"
            + static
        )

    rag_enabled = os.environ.get("CODER_RAG_ENABLED", "false").lower() in ("true", "1", "yes")
    if rag_enabled:
        retrieved = _get_retriever().retrieve(query)
        if retrieved:
            parts.append(
                "## Relevant Knowledge (RAG)\n\n"
                "The following snippets were retrieved from the knowledge base "
                "as most relevant to your task:\n\n"
                + retrieved
            )

    if not parts:
        return []

    return [HumanMessage(content="\n\n---\n\n".join(parts))]


def build_rag_messages(task: str) -> list[SystemMessage]:
    """Return a RAG SystemMessage if CODER_RAG_ENABLED=true, else empty list.

    Feature 2 fires only when ENABLE_KNOWLEDGE=true, CODER_RAG_ENABLED=true,
    and knowledge/RAG/ has files.
    """
    if not is_knowledge_enabled():
        return []
    rag_enabled = os.environ.get("CODER_RAG_ENABLED", "false").lower() in ("true", "1", "yes")
    if not rag_enabled:
        return []
    retrieved = _get_retriever().retrieve(task)
    if not retrieved:
        return []
    return [
        SystemMessage(
            content=f"## Relevant Knowledge (RAG)\n\nThe following snippets were "
                    f"retrieved from the knowledge base as most relevant to your task:\n\n"
                    f"{retrieved}"
        )
    ]
