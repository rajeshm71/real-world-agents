"""Vector store adapters for the policy-QA agent.

Two backends implement the `VectorStore` Protocol: `SqliteVecStore`
(primary) and `FaissStore` (fallback when sqlite-vec's extension load
fails). `get_store()` picks whichever loads without raising.

Split from `agent.py` so the RAG pipeline stays focused on
retrieval/generation and the backend-specific SQL and FAISS index
plumbing lives in one place. Adding a Chroma or Qdrant adapter is a
matter of writing one more class here that implements the same
Protocol.

Both backends store the same shape: chunks in a sqlite `chunks` table
(rowid + chunk_id + source_path + text + section) plus a vector index
(vec0 virtual table for sqlite-vec; IndexFlatIP for FAISS). Search
returns `[(Chunk, distance)]` in ranked order, distance being a
lower-is-better score in either backend.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Protocol

try:
    from .schemas import Chunk
except ImportError:
    from schemas import Chunk

_EMBEDDING_DIM = 384  # bge-small-en-v1.5 output dim


class VectorStore(Protocol):
    """Backend-agnostic vector store. See SqliteVecStore and FaissStore
    for the two current implementations."""

    def insert_batch(self, chunks: list[Chunk], embeddings) -> None: ...
    def search(self, query_embedding, top_k: int) -> list[tuple[Chunk, float]]: ...
    def close(self) -> None: ...
    @property
    def backend_name(self) -> str: ...
    @property
    def on_disk_size_bytes(self) -> int: ...


class SqliteVecStore:
    """sqlite-vec-backed store. Loadable sqlite extension. Preferred
    backend since everything lives in one .db file the caller can
    inspect with `sqlite3` from the command line."""

    backend_name = "sqlite-vec"

    def __init__(self, db_path: Path, dim: int = _EMBEDDING_DIM):
        self._path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._conn.enable_load_extension(True)
        import sqlite_vec

        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._conn.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                rowid INTEGER PRIMARY KEY,
                chunk_id TEXT UNIQUE NOT NULL,
                source_path TEXT NOT NULL,
                text TEXT NOT NULL,
                section TEXT
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS vecs USING vec0(
                embedding float[{dim}]
            );
            """
        )
        self._conn.commit()

    def insert_batch(self, chunks: list[Chunk], embeddings) -> None:
        import sqlite_vec

        cursor = self._conn.cursor()
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            cursor.execute(
                "INSERT OR IGNORE INTO chunks(chunk_id, source_path, text, section) "
                "VALUES (?, ?, ?, ?)",
                (chunk.chunk_id, chunk.source_path, chunk.text, chunk.section),
            )
            rowid = cursor.execute(
                "SELECT rowid FROM chunks WHERE chunk_id = ?", (chunk.chunk_id,)
            ).fetchone()[0]
            cursor.execute(
                "INSERT OR REPLACE INTO vecs(rowid, embedding) VALUES (?, ?)",
                (rowid, sqlite_vec.serialize_float32(embedding.tolist())),
            )
        self._conn.commit()

    def search(self, query_embedding, top_k: int) -> list[tuple[Chunk, float]]:
        import sqlite_vec

        # sqlite-vec's vec0 KNN queries require a LITERAL integer in
        # the LIMIT clause; a bind parameter (`LIMIT ?`) raises
        # "A LIMIT or 'k = ?' constraint is required". Coerce top_k
        # to int and interpolate it -- safe since it's an int, not a
        # user-provided string.
        top_k_int = int(top_k)
        if top_k_int < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k!r}")
        # The vec0 virtual table requires LIMIT to be applied ON THE
        # VEC TABLE directly, not on a joined result -- otherwise the
        # KNN constraint check fails. Push the search into a subquery,
        # then join to `chunks` for metadata.
        rows = self._conn.execute(
            f"""
            SELECT c.chunk_id, c.source_path, c.text, c.section, v.distance
            FROM (
                SELECT rowid, distance
                FROM vecs
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT {top_k_int}
            ) v JOIN chunks c ON v.rowid = c.rowid
            ORDER BY v.distance
            """,
            (sqlite_vec.serialize_float32(query_embedding.tolist()),),
        ).fetchall()
        return [
            (Chunk(chunk_id=cid, source_path=sp, text=t, section=sec), float(d))
            for cid, sp, t, sec, d in rows
        ]

    def close(self) -> None:
        self._conn.close()

    @property
    def on_disk_size_bytes(self) -> int:
        return self._path.stat().st_size if self._path.exists() else 0


class FaissStore:
    """Fallback backend: FAISS IndexFlatIP + a sibling sqlite metadata
    file. Two files on disk instead of one: `<path>` (sqlite) and
    `<path>.faiss` (FAISS index).

    Uses cosine similarity via normalized inner product -- fastembed
    embeddings are already unit-normalized so the dot product IS the
    cosine similarity. Higher inner product = more similar; we invert
    to `1.0 - score` before returning so the return-shape matches
    SqliteVecStore's lower-is-better distance."""

    backend_name = "faiss"

    def __init__(self, db_path: Path, dim: int = _EMBEDDING_DIM):
        import faiss

        self._path = db_path
        self._faiss_path = db_path.with_suffix(db_path.suffix + ".faiss")
        self._index = (
            faiss.read_index(str(self._faiss_path))
            if self._faiss_path.exists()
            else faiss.IndexFlatIP(dim)
        )
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT UNIQUE NOT NULL,
                source_path TEXT NOT NULL,
                text TEXT NOT NULL,
                section TEXT
            )
            """
        )
        self._conn.commit()

    def insert_batch(self, chunks: list[Chunk], embeddings) -> None:
        import numpy as np

        embeddings_arr = np.asarray(list(embeddings), dtype="float32")
        cursor = self._conn.cursor()
        for chunk in chunks:
            cursor.execute(
                "INSERT OR IGNORE INTO chunks(chunk_id, source_path, text, section) "
                "VALUES (?, ?, ?, ?)",
                (chunk.chunk_id, chunk.source_path, chunk.text, chunk.section),
            )
        self._conn.commit()
        self._index.add(embeddings_arr)

    def search(self, query_embedding, top_k: int) -> list[tuple[Chunk, float]]:
        import numpy as np

        query = np.asarray([query_embedding], dtype="float32")
        scores, indices = self._index.search(query, top_k)
        results: list[tuple[Chunk, float]] = []
        for idx, score in zip(indices[0], scores[0], strict=True):
            if idx < 0:
                continue
            row = self._conn.execute(
                "SELECT chunk_id, source_path, text, section FROM chunks WHERE rowid = ?",
                (int(idx) + 1,),  # sqlite rowids are 1-indexed; faiss row ids are 0-indexed
            ).fetchone()
            if row:
                cid, sp, t, sec = row
                results.append(
                    (Chunk(chunk_id=cid, source_path=sp, text=t, section=sec), 1.0 - float(score))
                )
        return results

    def close(self) -> None:
        import faiss

        faiss.write_index(self._index, str(self._faiss_path))
        self._conn.close()

    @property
    def on_disk_size_bytes(self) -> int:
        total = self._path.stat().st_size if self._path.exists() else 0
        if self._faiss_path.exists():
            total += self._faiss_path.stat().st_size
        return total


def get_store(index_path: Path, *, dim: int = _EMBEDDING_DIM) -> VectorStore:
    """Try sqlite-vec first; fall back to FAISS on any load failure
    (extension not compiled for this platform, path unwritable, etc.).
    Returns a store ready for insert_batch / search."""
    try:
        return SqliteVecStore(index_path, dim=dim)
    except (sqlite3.OperationalError, OSError, RuntimeError, ImportError):
        return FaissStore(index_path, dim=dim)
