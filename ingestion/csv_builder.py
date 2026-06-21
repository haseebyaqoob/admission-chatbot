"""
csv_builder.py
───────────────
Build programs.csv from academic_programmes.txt.

CSV scope: programs (department, degree level, program name, duration)
only. Eligibility, fees, deadlines, seats come from RAG on PDFs.
"""

import re
from pathlib import Path

import pandas as pd

from config_loader import cfg

_CORPUS_DIR    = Path(cfg["corpus_dir"])
_CSV_OUTPUT    = Path(cfg["csv_output"])

# Degree level detection from section headings
_DEGREE_PATTERN = re.compile(
    r"^(Bachelor|Master|PhD)\s*(?:\(([^)]*)\))?\s*:",
    re.IGNORECASE,
)

# Program line: starts with "  - " or "- "
_PROGRAM_LINE = re.compile(r"^\s*-\s*(.+)")

# Department header: "N. DEPARTMENT NAME" or "DEPARTMENT NAME"
_DEPT_HEADER = re.compile(r"^(?:\d+\.)?\s*([A-Z][A-Z &,\-]+)(?:\s+DEPARTMENT)?\s*$")


def parse() -> list[dict]:
    """
    Parse academic_programmes.txt into structured program rows.

    Format:
      N. DEPARTMENT NAME
      ------------------
      Bachelor (4 Years):
        - Program Name
        - Program Name (Specialization)

      Master (2.5 Years M.Engg.):
        - Program Name (+)
        ...

    Returns list of {department, faculty, degree_level, program_name, duration}
    """
    txt_path = _CORPUS_DIR / "academic_programmes.txt"
    if not txt_path.exists():
        print(f"[csv_builder] Warning: {txt_path} not found — skipping CSV build")
        return []

    lines = txt_path.read_text(encoding="utf-8").split("\n")
    programs: list[dict] = []
    current_dept: str = ""
    current_level: str = ""
    current_duration: str = ""

    for line in lines:
        stripped = line.strip()

        # Skip empty lines, decorative lines, key section, and the title
        if not stripped:
            continue
        if re.match(r"^[=\-]{5,}$", stripped):
            continue
        if stripped.startswith("Key:") or stripped.startswith("NED UNIVERSITY"):
            continue
        if stripped.startswith("(+") or stripped.startswith("(*") or stripped.startswith("(***"):
            continue

        # Check for department header: "N. DEPARTMENT NAME"
        # Looks like "1. CIVIL ENGINEERING", "2. URBAN & INFRASTRUCTURE ENGINEERING"
        dept_match = re.match(r"^(\d+)\.\s*(.+?)\s*$", stripped)
        if dept_match:
            current_dept = dept_match.group(2).strip().title()
            continue

        # Check for degree-level heading: "Bachelor (4 Years):", "Master (2.5 Years M.Engg.):"
        deg_match = _DEGREE_PATTERN.match(stripped)
        if deg_match:
            raw_level = deg_match.group(1).strip()
            duration_raw = deg_match.group(2).strip() if deg_match.group(2) else ""

            # Map to canonical level
            if raw_level.lower().startswith("bach"):
                current_level = "BE"
            elif raw_level.lower().startswith("master"):
                # Check sub-type from duration text
                dur_lower = duration_raw.lower()
                if "mem" in dur_lower:
                    current_level = "MEM"
                elif "ms" in dur_lower:
                    current_level = "MS"
                else:
                    current_level = "M.Engg"
            elif raw_level.lower().startswith("phd"):
                current_level = "PhD"
            else:
                current_level = raw_level

            current_duration = duration_raw
            continue

        # Check for program line: "  - Program Name"
        prog_match = _PROGRAM_LINE.match(stripped)
        if prog_match and current_dept:
            prog_name_raw = prog_match.group(1).strip()

            # Skip special lines like "PhD: Available in various research areas"
            if prog_name_raw.lower().startswith("phd:"):
                continue

            # Clean up markers: "(+)", "(*)", "(**)", "(***)"
            prog_name = re.sub(r"\s*\([\+\*]+\)\s*$", "", prog_name_raw).strip()

            # Extract specialization from parentheses
            spec_match = re.search(r"\((Specialization in .+?)\)", prog_name)
            specialization = spec_match.group(1) if spec_match else ""

            # Remove specialization from name for base program name
            prog_name_clean = re.sub(r"\s*\(Specialization in .+?\)", "", prog_name).strip()

            programs.append({
                "department":       current_dept,
                "faculty":          _map_faculty(current_dept),
                "degree_level":     current_level,
                "program_name":     prog_name_clean,
                "duration":         current_duration,
                "specializations":  specialization,
                "seats":            "",
                "eligibility_summary": "",
            })

    print(f"[csv_builder] Parsed {len(programs)} programs from academic_programmes.txt")
    return programs


def _map_faculty(dept: str) -> str:
    """Map department name to faculty."""
    dept_lower = dept.lower()
    if any(kw in dept_lower for kw in ["civil", "petroleum", "urban", "environmental", "earthquake"]):
        return "Faculty of Civil & Petroleum Engineering"
    elif any(kw in dept_lower for kw in ["mechanical", "automotive", "marine", "industrial", "manufacturing", "metallurgy", "textile"]):
        return "Faculty of Mechanical Engineering"
    elif any(kw in dept_lower for kw in ["electrical", "electronic", "computer", "telecommunication", "software", "information"]):
        return "Faculty of Electrical & Computer Engineering"
    elif any(kw in dept_lower for kw in ["chemical", "material", "polymer", "biomedical"]):
        return "Faculty of Chemical & Process Engineering"
    elif any(kw in dept_lower for kw in ["architecture", "planning", "design"]):
        return "Faculty of Architecture & Management Sciences"
    elif any(kw in dept_lower for kw in ["science", "mathematics", "physics", "chemistry"]):
        return "Faculty of Sciences"
    else:
        return "Other"


def build():
    """Build programs.csv from corpus files."""
    print("=" * 60)
    print("Building programs.csv...")

    programs = parse()

    if not programs:
        print("[csv_builder] No programs parsed — creating empty CSV with headers only.")
        df = pd.DataFrame(columns=[
            "department", "faculty", "degree_level", "program_name",
            "duration", "seats", "specializations", "eligibility_summary"
        ])
    else:
        df = pd.DataFrame(programs)

    _CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_CSV_OUTPUT, index=False, encoding="utf-8")
    print(f"[csv_builder] CSV written → {_CSV_OUTPUT} ({len(df)} rows)")

    if len(df) > 0:
        print(f"[csv_builder] Columns: {list(df.columns)}")
        print(f"[csv_builder] Sample departments: {df['department'].unique()[:5]}")

    return df
