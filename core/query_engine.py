"""
query_engine.py
────────────────
Executes queries against the CSV (structured data) and hybrid index (RAG).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from config_loader import cfg
from index.hybrid_searcher import HybridSearcher
from index.vector_store import VectorStore

_CSV_PATH = Path(cfg["csv_output"])


class QueryEngine:
    """
    Provides structured (CSV) and hybrid (BM25 + vector) query execution.
    """

    def __init__(self, hybrid_searcher: Optional[HybridSearcher] = None):
        self.hybrid = hybrid_searcher
        self.df: pd.DataFrame = self._load_csv()

    def _load_csv(self) -> pd.DataFrame:
        if not _CSV_PATH.exists():
            print(f"[query_engine] CSV not found: {_CSV_PATH}")
            return pd.DataFrame()
        df = pd.read_csv(_CSV_PATH)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        print(f"[query_engine] Loaded CSV: {len(df)} rows, columns={list(df.columns)}")
        return df

    # ── CSV Queries ──────────────────────────────────────────────────

    def query_programs(self, filters: dict) -> pd.DataFrame:
        """Filter programs by department and/or degree_level."""
        df = self.df.copy()
        if not len(df):
            return df
        dept = filters.get("department")
        if dept and "department" in df.columns:
            df = df[df["department"].astype(str).str.contains(dept, case=False, na=False)]
        level = filters.get("degree_level")
        if level and "degree_level" in df.columns:
            df = df[df["degree_level"].astype(str).str.lower().str.contains(level.lower(), na=False)]
        return df

    def to_summary_string(self, row: pd.Series) -> str:
        """Format a program row as a readable string."""
        parts = []
        for col in ["program_name", "degree_level", "department", "duration"]:
            val = row.get(col)
            if pd.notna(val) and str(val).strip() and str(val).strip() not in ("nan", ""):
                label = col.replace("_", " ").title()
                parts.append(f"  {label}: {val}")
        return "\n".join(parts)

    # ── Counting ──────────────────────────────────────────────────────

    def count_programs(self, filters: dict | None = None) -> dict:
        """Return aggregate counts from the programs CSV."""
        if len(self.df) == 0:
            return {"total": 0, "by_department": [], "by_level": []}
        df = self.df.copy()
        if filters:
            dept = filters.get("department")
            if dept and "department" in df.columns:
                df = df[df["department"].astype(str).str.contains(dept, case=False, na=False)]
            level = filters.get("degree_level")
            if level and "degree_level" in df.columns:
                df = df[df["degree_level"].astype(str).str.lower().str.contains(level.lower(), na=False)]
        total = len(df)
        by_dept = (
            df.groupby("department").size().sort_values(ascending=False).head(5).to_dict()
        ) if "department" in df.columns else {}
        by_level = (
            df.groupby("degree_level").size().to_dict()
        ) if "degree_level" in df.columns else {}
        return {"total": total, "by_department": by_dept, "by_level": by_level}

    def count_all_entities(self) -> dict:
        """
        Return counts for all countable entities in the known corpus.
        Uses RAG to estimate scholarship and supervisor counts.
        """
        result = {"programs": self.count_programs()}
        # Counts from CSV are precise; others come from RAG chunks
        # Supervisor count: chunks from phd_supervisors.txt
        if self.hybrid:
            sup_results = self.hybrid.search("list all PhD supervisors", top_k=10,
                                              source_filter="phd_supervisors")
            sup_names = set()
            for r in sup_results:
                for line in r["content"].split("\n"):
                    line = line.strip()
                    if line and not line.startswith("PhD") and not line.startswith("="):
                        sup_names.add(line)
            if sup_names:
                result["supervisors"] = len(sup_names)
        return result

    # ── Hybrid RAG Queries ───────────────────────────────────────────

    def search(self, query: str, top_k: int = 4,
               source_filter: str | None = None) -> list[dict]:
        """Hybrid search: BM25 pre-filter → vector re-rank."""
        if self.hybrid is None:
            return []
        return self.hybrid.search(query, top_k=top_k, source_filter=source_filter)
