import os, re, json, sys, argparse, hashlib, difflib
from pathlib import Path
from datetime import datetime, UTC
from datasets import disable_progress_bars
from typing import List, Dict, Any, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset

# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

# SRC = "AI-MO/NuminaMath-1.5"
SRC="annakosovskaia/NuminaMath-1.5-RL-Verifiable-cleaned"

disable_progress_bars()


# ───────────── helpers ────────────── #
def conv_id(seed: str) -> str:
    return f"{SRC}_{hashlib.sha256(seed.encode()).hexdigest()[:12]}"



def make_part(
    ptype: str, content: str = "", name: str = "", args: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    return {
        "type": ptype,
        "content": content,
        "name": name,
        "args": json.dumps(args, ensure_ascii=False) if args else "",
    }


def contains_links_or_images(text: str) -> bool:
    """Check if text contains links or images."""
    if not text:
        return False
    # Check for image links, http links, .com domains, etc.
    patterns = [
        r"!\[.*?\]\(.*?\)",  # Markdown images
        r"https?://",  # HTTP/HTTPS links
        r"\.com",  # .com domains
        r"\.org",  # .org domains
        r"\.net",  # .net domains
        r"\.edu",  # .edu domains
        r"cdn\.",  # CDN links
        r"mathpix\.com",  # Mathpix specifically
    ]
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def contains_figure_references(text: str) -> bool:
    """Check if text contains references to figures or diagrams."""
    if not text:
        return False
    patterns = [
        r"\bfigure\s+\d+",  # Figure 1, Figure 2, etc.
        r"\bfig\.\s*\d+",  # Fig. 1, Fig.2, etc.
        r"\bdiagram\s+\d+",  # Diagram 1, Diagram 2, etc.
        r"\bas shown in (the )?(figure|diagram|picture|image)",  # As shown in the figure
        r"\bin (the )?(figure|diagram|picture|image)\s+(above|below)",  # in the figure above
    ]
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def contains_chinese_characters(text: str) -> bool:
    """Check if text contains Chinese characters."""
    if not text:
        return False
    # Check for Chinese Unicode range
    return bool(re.search(r'[\u4e00-\u9fff]', text))


# ───────────── OCR prefix/cruft cleaning ────────────── #
# Compact, robust patterns for leading prefixes and points cleaning focused on the
# FIRST non-empty line (recursing if that line becomes a pure header and is removed).

# Leading title-like words (Problem, Question, ...), optionally with indices like 1, 1.2, A3, etc.
_TITLE_WORDS = (
    r"(?:problem|question|task|exercise|example|theorem|corollary|claim|"
    r"observation|remark)"
)

# Match leading enumerators or titles at line start, with optional separators right after
LEADING_PREFIX_RE = re.compile(
    rf"^(?!\s*\$)\s*(?:#{{1,6}}\s*)?"  # ignore pure LaTeX lines and markdown headers
    rf"(?:"
    rf"\d+(?:\.\d+){1,4}\*?[.)]"  # 21.17*. 3.2.1.
    rf"|\(?\d+\)?[.)]"  # 1) 1. (1)
    rf"|[IVXLCDM]+[.)]"  # IV) II.
    rf"|[A-Za-z]\d{1,3}[.)]"  # A) A1. B12)
    rf"|(?!(?:19|20))\d{1,3}\s+(?=[A-Z(])"  # 251 For ... ; avoid removing 19xx/20xx years
    rf"|[A-Z]{1,3}\d{1,3}\b"  # A8, NT5, G12 (no trailing punctuation)
    rf"|Q(?:uestion)?\s*:"  # Q: Question:
    rf"|{_TITLE_WORDS}\b(?:\s*[A-Za-z]?\d+(?:/\d+)?(?:\.\d+)*)?\b[:.]?"  # Problem 1: Exercise 1.2.3.
    rf"|[A-Z]/\d+:"  # D/5:
    rf")"
    rf"(?:\s*\([^)]*\))*[\s\-–—:.)\(]*",  # allow multiple parenthetical cruft segments
    re.IGNORECASE,
)

# Remove parenthesized points/pts/marks anywhere in the line (safe because it requires a number)
POINTS_ANYWHERE_RE = re.compile(
    r"\(\s*\d+\s*(?:points?|pts?|marks?)\s*\)",
    re.IGNORECASE,
)

# Clean leftover leading separators after prefix removal
LEADING_SEP_TRIM_RE = re.compile(r"^[\s\-–—:.)\(,]+")

# Fallback: remove bare numeric prefixes like "251 " at start (avoid 4-digit years 19xx/20xx)
NUMERIC_PREFIX_FALLBACK_RE = re.compile(r"^(?!(?:19|20)\d{2}\b)\s*\d{1,4}\s+(?=[A-Z])")

# Contest/year metadata keywords for safe removal
_META_KEYWORDS = (
    r"(?:test|contest|olympiad|round|exam|examination|training|team|shortlist|league|cup|"
    r"putnam|amc|aime|bmo|usamo|imo|korean|romanian|polish|turkish|canadian|"
    r"moldova|serbia|montenegro|zhejiang|indonesia|iran|girls|balkan|china|"
    r"national|competition|regional|province|provincial|municipal|pdr|american)"
)
# Remove any leading parentheses containing metadata keywords
LEADING_META_PAREN_RE = re.compile(
    rf"^\s*\([^)]*\b{_META_KEYWORDS}\b[^)]*\)\s*[\s\-–—:.)\(]*",
    re.IGNORECASE,
)
# Match bracket-only header lines with optional $: $[ Topic ]$
# But exclude lines with mathematical content (formulas with $...$ inside brackets)
# Only match short topic/category headers, not mathematical notation or long explanations
BRACKET_HEADER_RE = re.compile(
    r"^\s*\$?\[[^\[\]]{1,50}\]\$?(?:\s*\$?\[[^\[\]]{1,50}\]\$?)?\s*$"
)

# Remove parenthesized points/pts/marks anywhere in the line (safe because it requires a number)
POINTS_ANYWHERE_RE = re.compile(
    r"\(\s*\d+\s*(?:points?|pts?|marks?)\s*\)",
    re.IGNORECASE,
)

# Clean leftover leading separators after prefix removal
LEADING_SEP_TRIM_RE = re.compile(r"^[\s\-–—:.)\(]+")

# Fallback: remove bare numeric prefixes like "251 " at start (avoid 4-digit years 19xx/20xx)
NUMERIC_PREFIX_FALLBACK_RE = re.compile(r"^(?!(?:19|20)\d{2}\b)\s*\d{1,4}\s+(?=[A-Z])")

# Trailing contest/year parenthetical metadata (end-of-line)
TRAILING_YEAR_META_PAREN_RE = re.compile(
    rf"\s*\([^)]*(?:19|20)\d{{2}}[^)]*\)\s*$",
    re.IGNORECASE,
)
# Add leading contest/year meta parentheses removal at start
LEADING_YEAR_META_PAREN_RE = re.compile(
    rf"^\s*\(\s*(?:19|20)\d{{2}}[^)]*\b{_META_KEYWORDS}\b[^)]*\)\s*",
    re.IGNORECASE,
)

# Additional prefix patterns used in cleaning and mirrored by the colorizer
AUTHOR_ATTR_RE = re.compile(r'^[A-Z][a-z]+(?:\s+[A-Z])\s*:\s*')
I_TAG_RE = re.compile(r'\[i\](.*?)\[/i\]', re.IGNORECASE | re.DOTALL)  # match [i]...[/i] to extract content
DECIMAL_STAR_PREFIX_RE = re.compile(r'^\s*\d+\.\d+\*\.\s*')
PUTNAM_PROBLEM_PREFIX_RE = re.compile(
    r'^\s*\d+(?:st|nd|rd|th)\s+Putnam\s*\d+\s*Problem\s*\w+\s*',
    re.IGNORECASE,
)
CONTEST_PHRASE_PREFIX_RE = re.compile(
    r"^\s*(?:\d{1,3}(?:st|nd|rd|th)\s+)?(?:[A-Za-z][A-Za-z\-]+\s+){0,6}(?:19|20)\d{2}\s+"
    r"(?:problem|question|exam|olympiad|competition|round|shortlist)\b[^:]*:?\s*",
    re.IGNORECASE,
)
DECIMAL_DOT_CH_PREFIX_RE = re.compile(r'^\s*\d+・\d+\s*')  # Chinese-style decimal-dot prefixes like '6・167'
EXAMPLE_PREFIX_RE = re.compile(r'^\s*Example\s*\d+\.?\s*', re.IGNORECASE)  # remove 'Example N' prefixes

def _remove_trailing_year_meta_segment(line: str) -> str:
    """Remove only a trailing contest/year parenthetical segment.

    Cases handled safely:
    - Closed parenthesis at the end of the line: ... (2010 China ...)
    - Unclosed parenthesis that starts near the end of the line: ... (2010 China ...
      In this case remove from the last '(' to the end of line, but only if no ')' follows.
    """
    if not line:
        return line
    # Closed and at end-of-line
    m = TRAILING_YEAR_META_PAREN_RE.search(line)
    if m:
        return line[: m.start()].rstrip()
    # Unclosed segment from last '(' to end-of-line
    last_open = line.rfind("(")
    if last_open != -1:
        tail = line[last_open:]
        # Only if there is no closing ')' afterwards
        if ")" not in tail:
            if re.search(r"(?:19|20)\d{2}", tail) and re.search(_META_KEYWORDS, tail, re.IGNORECASE):
                return line[:last_open].rstrip()
    return line


def _has_trailing_year_meta_segment(line: str) -> bool:
    if not line:
        return False
    if TRAILING_YEAR_META_PAREN_RE.search(line):
        # Closed parenthetical at end
        return True
    last_open = line.rfind("(")
    if last_open != -1:
        tail = line[last_open:]
        if ")" not in tail:
            if re.search(r"(?:19|20)\d{2}", tail) and re.search(_META_KEYWORDS, tail, re.IGNORECASE):
                return True
    return False


def _strip_line_prefixes_once(line: str) -> str:
    """Strip a single leading OCR-style prefix and any numeric points markers.

    Applied iteratively to peel multiple stacked prefixes like "Problem 3: 1) —".
    """
    # Remove Example and Chinese decimal-dot prefixes
    line = EXAMPLE_PREFIX_RE.sub('', line)
    line = DECIMAL_DOT_CH_PREFIX_RE.sub('', line)
    # Remove author attributions like 'Lubshin D:'
    line = re.sub(r'^[A-Z][a-z]+(?:\s+[A-Z])\s*:\s*', '', line)
    # Remove decimal-star prefixes like '21.17*.'
    line = re.sub(r'^\s*\d+\.\d+\*\.\s*', '', line)
    # Remove ordinal Putnam problem prefixes like '40th Putnam 1979 Problem B5'
    line = re.sub(r'^\s*\d+(?:st|nd|rd|th)\s+Putnam\s*\d+\s*Problem\s*\w+\s*', '', line, flags=re.IGNORECASE)
    if not line:
        return line

    # Remove leading metadata parentheses like (PDR, ...)
    s0 = LEADING_META_PAREN_RE.sub("", line)
    # Remove any year-based metadata parentheses at start
    s0 = LEADING_YEAR_META_PAREN_RE.sub("", s0)
    # Remove points markers
    s = POINTS_ANYWHERE_RE.sub("", s0)

    # Remove one leading prefix instance
    line3 = LEADING_PREFIX_RE.sub("", s, count=1)
    # Trim leftover leading separators
    line4 = LEADING_SEP_TRIM_RE.sub("", line3)
    result = line4.lstrip()
    # If no structured prefix was removed, attempt fallback numeric-only removal
    if result == line:
        fb = NUMERIC_PREFIX_FALLBACK_RE.sub("", line.lstrip(), count=1)
        if fb != line.lstrip():
            return fb
    # Remove trailing contest/year meta parentheses (safe)
    result = _remove_trailing_year_meta_segment(result)

    # Remove leading meta parentheses like "(2010 China ...)" at start
    m = re.match(r"^\s*\(([^)]{0,200})\)\s*", result)
    if m:
        inner = m.group(1)
        if re.search(r"(?:19|20)\d{2}", inner) or re.search(_META_KEYWORDS, inner, re.IGNORECASE):
            result = result[m.end():]

    # Remove leading contest phrase like "40th Putnam 1979 Problem B5" at start
    pat = re.compile(
        r"^\s*(?:\d{1,3}(?:st|nd|rd|th)\s+)?(?:[A-Za-z][A-Za-z\-]+\s+){0,6}(?:19|20)\d{2}\s+"
        r"(?:problem|question|exam|olympiad|competition|round|shortlist)\b[^:]*:?\s*",
        re.IGNORECASE,
    )
    m2 = pat.match(result)
    if m2:
        result = result[m2.end():]

    return result


def remove_translation_metadata(text: str) -> str:
    # Remove ANSI color codes if present
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    """Remove translation-related boilerplate text from the end of the content."""
    if not text:
        return text

    # Patterns for translation metadata (case-insensitive)
    translation_patterns = [
        r'\n+\s*translat(?:e|ing) the (?:above )?text.*$',
        r'\n+\s*the translation maintains.*$',
        r'\n+\s*保留源文本.*$',  # Chinese: "Keep the source text..."
        r'\n+\s*---\s*$',  # Trailing separators after translation text
        r'\n+\s*the translation is provided as requested.*$',
        r'\n+\s*translation:\s*',
    ]

    cleaned = text
    for pattern in translation_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)

    # Remove trailing empty lines and whitespace
    # Additional cleaning: remove BBcode tags, hide tags, metadata lines
    # Remove BBcode tags [b] and [/b]
    cleaned = re.sub(r'\[/?b\]', '', cleaned)
    # Remove list bullet markers
    cleaned = cleaned.replace('[*]', '')
    # Remove hide tags entirely
    cleaned = re.sub(r'\[hide=.*?\].*?\[/hide\]', '', cleaned, flags=re.DOTALL)
    # Remove metadata lines like Time allowed, Each problem is worth, ## SOLUTIONS, Proposed by, Note:
    lines = cleaned.split('\n')
    filtered = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Preserve legitimate notes containing key phrases
        if "note:" in stripped.lower():
            keep_phrases = ["understood", "largest integer", "[t]"]
            if any(phrase in stripped.lower() for phrase in keep_phrases):
                filtered.append(line)
                continue
        if re.match(r'^(Time allowed:|Each problem is worth|##\s*SOLUTIONS|Proposed by|Promotion\s*\d+:|Strengthening\s*\d+:|Inference\s*\d+\.|Proposition\s*\d+\.|Corollary\s*\d+\.|Note:)', stripped, re.IGNORECASE):
            continue
        filtered.append(line)
    cleaned = '\n'.join(filtered).strip()
    return cleaned.rstrip()


def clean_italic_tags(text: str) -> str:
    """Clean [i]...[/i] tags properly and remove author attributions and metadata.

    - Remove [i]...[/i] entirely (including content) if it's on the last non-empty line
    - For other [i]...[/i] in content, remove only the tags and keep the content (e.g., "[i]move[/i]" -> "move")
    - Remove author attribution and metadata lines (e.g., "created by John Doe", "Fresh translation.")
    """
    if not text:
        return text

    lines = text.split('\n')

    # Find the index of the last non-empty line
    last_nonempty_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip():
            last_nonempty_idx = i
            break

    if last_nonempty_idx is None:
        return text

    # Author attribution and metadata patterns (case-insensitive)
    author_patterns = [
        r'\b(created|proposed|submitted|contributed|problem)\s+by\b',
        r'\bauthor\s*:',
        r'\b(fresh|new|original)\s+translation\.?$',
        r'^translation\.?$',
        r'^\([A-Z][^)]+\)\.?$',  # Author name in parentheses, e.g., "(Walther Janous)"
    ]

    # First, remove [i]...[/i] tags from ALL lines, keeping the content
    cleaned_lines = []
    for line in lines:
        # Remove tags but keep content for all lines
        line = I_TAG_RE.sub(r'\1', line)
        cleaned_lines.append(line)

    # Now check if the last non-empty line is an author attribution
    if last_nonempty_idx is not None:
        last_line = cleaned_lines[last_nonempty_idx].strip()
        is_author_line = False

        # Check standard patterns first
        for pattern in author_patterns:
            if re.search(pattern, last_line, re.IGNORECASE):
                is_author_line = True
                break

        # Also check if it looks like a short author name (1-4 words, under 50 chars)
        if not is_author_line and len(last_line) < 50:
            words = last_line.rstrip('.').split()
            if 1 <= len(words) <= 4:
                # Check if all words are capitalized names or initials
                all_caps = all(
                    (len(w) == 2 and w[0].isupper() and w[1] == '.') or  # Initial like "G."
                    (len(w) >= 2 and w[0].isupper() and w[1:].islower())  # Name like "Smith"
                    for w in words
                )
                if all_caps:
                    is_author_line = True

        if is_author_line:
            # Remove the author attribution line entirely
            cleaned_lines[last_nonempty_idx] = ''

    return '\n'.join(cleaned_lines).rstrip()


def strip_ocr_prefixes(text: str) -> str:
    """Remove OCR prefixes and points from the first non-empty line.

    If the first line becomes empty or just separators after cleanup, drop it and
    recurse so that secondary header lines (or lines starting with "(5 pts)") are
    also handled.
    """
    if not text:
        return text or ""

    lines: List[str] = text.splitlines()
    # Remove ANSI color codes from each line
    lines = [re.sub(r'\x1b\[[0-9;]*m', '', l) for l in lines]
    # Remove stray 'Example N.' inside LaTeX arrays
    lines = [re.sub(r'\\text\s*\{\s*Example\s*\d+\.?\s*', '', l) for l in lines]
    # Drop leading bracket-only header lines
    while lines and BRACKET_HEADER_RE.match(lines[0].strip().replace('$','')):
        lines.pop(0)
    # Drop leading markdown header lines (e.g., "## A1 MLD")
    while lines and re.match(r'^\s*#+\s*', lines[0]):
        lines.pop(0)

    # Find first non-empty line
    first_idx = next((i for i, ln in enumerate(lines) if ln.strip() != ""), None)
    if first_idx is None:
        return text.strip()

    # Strip up to 3 stacked prefixes from the first non-empty line
    line = lines[first_idx]
    for _ in range(3):
        new_line = _strip_line_prefixes_once(line)
        if new_line == line:
            break
        line = new_line

    # If the resulting first line is empty or just separators, drop it and recurse
    if not line.strip() or re.fullmatch(r"[-–—•·\s]*", line):
        cleaned = [ln for i, ln in enumerate(lines) if i != first_idx]
        return strip_ocr_prefixes("\n".join(cleaned).strip())

    # If the resulting first line contains no alphabetic characters and no LaTeX/math markers,
    # treat it as OCR noise (e.g., "8,9 |  |") and drop it, then recurse.
    if not re.search(r"[A-Za-z]", line) and not re.search(r"[\\$]", line):
        cleaned = [ln for i, ln in enumerate(lines) if i != first_idx]
        return strip_ocr_prefixes("\n".join(cleaned).strip())

    # Otherwise, replace the first non-empty line with the cleaned version
    lines[first_idx] = line

    # Drop any lines that are pure parenthetical metadata (e.g., contest/year)
    # Also remove trailing points metadata like "(10 points)"
    trailing_points_re = re.compile(r"\(\s*\d+\s*points?\s*\)$", re.IGNORECASE)
    filtered: List[str] = []
    for i, ln in enumerate(lines):
        if i == first_idx:
            filtered.append(ln)
            continue
        s = ln.strip()
        # remove $ for bracket matching
        if BRACKET_HEADER_RE.match(s.replace('$','')):
            continue
        # drop parenthetical meta-only lines
        # Remove trailing points lines
        if trailing_points_re.search(s):
            continue
        if s.startswith('(') and s.endswith(')'):
            inner = s[1:-1]
            if re.search(_META_KEYWORDS, inner, re.IGNORECASE) or re.search(r"(?:19|20)\d{2}", inner):
                continue
        filtered.append(ln)
    lines = filtered

    # Additional cleanup: remove any remaining subproblem identifiers like "A1.", "C2.", "Problem G5." etc.
    cleaned_lines = []
    # Pattern matches optional leading word (Problem, etc.) followed by optional letters/numbers and a dot or parenthesis
    subproblem_prefix_re = re.compile(r"^(?:Problem\s+)?(?:[A-Z]{1,3}\s*\d+|\d+\.\d+)(?:[\.)]|:)?\s+", re.IGNORECASE)
    for ln in lines:
        cleaned = subproblem_prefix_re.sub("", ln)
        # Additional removal for patterns like 'ALG 1.' with possible spaces
        cleaned = re.sub(r"^[A-Z]{1,4}\s*\d+\.[\s]*", "", cleaned)
        cleaned_lines.append(cleaned)
    # Remove lines that consist solely of leftover identifiers (e.g., "SAU", "BUL", "N3", "A 1.", "ALG 1.", "COM 2", "GEO 1.")
    identifier_line_re = re.compile(r"^[A-Z]{1,4}(?:\s+[A-Z]{1,4})?\s*\d+(?:[\.)])?$|^\d+(?:\.\d+)?(?:[\.)])?$")
    final_lines = []
    for ln in cleaned_lines:
        stripped = ln.strip()
        # If line matches solitary identifier pattern and contains no lowercase letters, drop it
        if stripped and identifier_line_re.fullmatch(stripped) and not re.search(r"[a-z]", stripped):
            continue
        final_lines.append(ln)
    return "\n".join(final_lines).strip()


def clean_problem_text(text: str) -> str:
    """Clean problem text, removing BBcode, metadata lines, and solution headers."""
    if not text:
        return text or ""
    
    # Apply cleaning steps
    text = clean_italic_tags(text)
    text = remove_translation_metadata(text)
    cleaned = strip_ocr_prefixes(text)
    if cleaned:
        # Remove inline italic tags like [i] and [/i]
        cleaned = I_TAG_RE.sub("", cleaned)
    
    # Ensure we never return None - return empty string instead
    return cleaned if cleaned is not None else ""


def clean_solution_text(text: str) -> str:
    """Clean solution text by removing headers and short prefixes."""
    if not text:
        return text

    # Remove translation metadata first
    text = remove_translation_metadata(text)

    # Remove markdown headers
    text = re.sub(r"^#+\s*", "", text)

    # Remove common solution prefixes
    prefixes_to_remove = [
        r"^Solution\s*\d*\.?\s*",
        r"^##\s*Solution\s*\d*\.?\s*",
        r"^#\s*Solution\s*\d*\.?\s*",
    ]

    for prefix in prefixes_to_remove:
        text = re.sub(prefix, "", text, flags=re.IGNORECASE)

    # Remove first sentence if it's very short (less than 20 chars)
    lines = text.split("\n")
    if lines and len(lines[0].strip()) < 20:
        # Check if it ends with a period
        first_line = lines[0].strip()
        if first_line.endswith(".") and len(first_line) < 20:
            lines = lines[1:]

    # Remove markdown headers at the end
    cleaned_lines = []
    for line in lines:
        # Skip lines that are just markdown headers (## Title)
        if re.match(r"^#+\s+[A-Za-z]+\s*$", line.strip()):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()

def colorise_text(text: str, original_text: str) -> str:
    """Colorize text to show differences from original.
    
    - Green for text that was added (in text but not in original_text)
    - Red for text that was removed (in original_text but not in text)
    """
    if not original_text:
        # If no original, return text in green (all added)
        return f"{GREEN}{text}{RESET}"
    if not text:
        # If text is empty but original isn't, return original in red (all removed)
        return f"{RED}{original_text}{RESET}"
    
    # Use SequenceMatcher to find differences
    matcher = difflib.SequenceMatcher(None, original_text, text)
    result_parts = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Unchanged text - no color
            result_parts.append(text[j1:j2])
        elif tag == 'delete':
            # Text removed from original - red
            result_parts.append(f"{RED}{original_text[i1:i2]}{RESET}")
        elif tag == 'insert':
            # Text added - green
            result_parts.append(f"{GREEN}{text[j1:j2]}{RESET}")
        elif tag == 'replace':
            # Text replaced - show removed in red, added in green
            result_parts.append(f"{RED}{original_text[i1:i2]}{RESET}")
            result_parts.append(f"{GREEN}{text[j1:j2]}{RESET}")
    
    return ''.join(result_parts)

# ───────────— sample parser ───────── #
def parse_sample(row: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a NuminaMath sample into our chat format."""

    # Extract fields
    problem = row.get("problem", "")
    solution = row.get("solution", "")
    answer = row.get("answer", "")
    problem_type = row.get("problem_type", "")
    question_type = row.get("question_type", "")

    # Clean problem and solution text, storing original problem text
    original_problem = problem
    problem = clean_problem_text(problem)
    if problem is None:
        print(original_problem, problem)
    solution = clean_solution_text(solution)
    original_problem = colorise_text(problem, original_problem)

    # Determine answer type and create appropriate response
    answer_lower = answer.lower() if answer else ""

    if answer_lower == "proof":
        # For proof answers, use the solution as assistant text
        assistant_text = solution
        reasoning = None
        verifiable_answer = None
    else:
        # For regular answers, use "The answer is: {answer}" as assistant text
        assistant_text = f"The answer is: {answer}"
        reasoning = solution
        verifiable_answer = answer

    # Create messages - assistant message with reasoning and response
    messages = []

    if assistant_text:
        parts = []

        # Add reasoning as THOUGHT if it exists
        if reasoning:
            parts.append(make_part("thought", reasoning))

        # Add the answer as response
        parts.append(make_part("response", assistant_text))

        messages.append({"role": "assistant", "parts": parts})

    return {
        "system": "",
        "functions": [],  # No functions for math problems
        "initial": {
            "role": "user",
            "content": problem,
            "metadata": {
                "problem_type": problem_type,
                "question_type": question_type,
                "answer": answer,
                "source": row.get("source", ""),
                "synthetic": row.get("synthetic", False),
                "verifiable_answer": verifiable_answer,
                "original_problem": original_problem,
            },
        },
        "messages": messages,
    }


# ───────────— map row ─────────────── #
def convert_row(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    p = parse_sample(row)
    return {
        "conversation_id": conv_id(p["initial"]["content"] + str(idx)),
        "dataset_source": SRC,
        "original_metadata": {
            "row_index": idx,
            "problem_type": row.get("problem_type", ""),
            "question_type": row.get("question_type", ""),
            "answer": row.get("answer", ""),
            "source": row.get("source", ""),
            "synthetic": row.get("synthetic", False),
            "problem_is_valid": row.get("problem_is_valid", ""),
            "solution_is_valid": row.get("solution_is_valid", ""),
            "verifiable_answer": p["initial"]["metadata"]["verifiable_answer"],
        },
        "system_prompt": {"content": p["system"], "metadata": {}},
        "initial_prompt": p["initial"],
        "available_functions": p["functions"],
        "conversation_branches": [{"messages": p["messages"]}],
        "created_timestamp": datetime.now(UTC).isoformat(),
    }


def process_split(ds: Dataset, num_proc: int) -> Dataset:
    # Filter first
    filtered_ds = ds.filter(
        lambda x: x["problem_is_valid"] == "Yes"
        and x["solution_is_valid"] == "Yes"
        and x["problem"] is not None
        and x["solution"] is not None,
        num_proc=num_proc,
        desc="Filtering samples",
    )

    filtered_ds = filtered_ds.filter(
        lambda x: not x["synthetic"]
        and x["answer"] is not None
        and x["answer"] != ""
        and x["answer"] != "notfound"
        and x["answer"] != "not found"
        and x["source"] not in ["synthetic_math", "orca_math", "metamath"],
        num_proc=num_proc,
        desc="Filtering samples",
    )

    filtered_ds = filtered_ds.filter(
        lambda x: not contains_links_or_images(x["problem"])
        and not contains_links_or_images(x["solution"]),
        num_proc=num_proc,
        desc="Filtering links and images",
    )

    # Filter out samples with figure/diagram references
    filtered_ds = filtered_ds.filter(
        lambda x: not contains_figure_references(x["problem"])
        and not contains_figure_references(x["solution"]),
        num_proc=num_proc,
        desc="Filtering figure references",
    )

    # Filter out samples with Chinese characters
    filtered_ds = filtered_ds.filter(
        lambda x: not contains_chinese_characters(x["problem"])
        and not contains_chinese_characters(x["solution"]),
        num_proc=num_proc,
        desc="Filtering Chinese characters",
    )

    filtered_ds = filtered_ds.filter(
        lambda x: x["solution"] != "", num_proc=num_proc, desc="Filtering samples"
    )

    filtered_ds = filtered_ds.filter(
        lambda x: not (
            "Solution" in x["problem"].splitlines()[0]
            or "Answer" in x["problem"].splitlines()[0]
        ),
        num_proc=num_proc,
        desc="Filtering samples",
    )

    filtered_ds = filtered_ds.filter(
        lambda x: not "[asy]" in x["problem"],
        num_proc=num_proc,
        desc="Filtering samples",
    )

    def map_fn(x):
        lines = x["problem"].splitlines()

        cutoff_idx = -1
        for i, line in enumerate(lines):
            if (
                "Answer:" in line
                or "Solution" in line
                or "## Answers and solutions:" in line
                or "Answer." in line
                or "Answer," in line
                or "## Correct answers:" in line
                or "## Answer" in line
                or "Answer (" in line
                or "Answer $" in line
                or "Answer" == line.strip()
                or "## Numerical Answer Problems" in line
                or "## - Answer -" in line
                or "Answer(" in line
                or f"{{Answer}}" in line
                or "【Answer】" in line
                or "Answer \\" in line
                or "[hide=Answer]" in line
            ):
                cutoff_idx = i
                break

        if cutoff_idx != -1:
            x["problem"] = "\n".join(lines[:cutoff_idx]).strip()

        return x

    filtered_ds = filtered_ds.map(map_fn, num_proc=num_proc, desc="Filtering samples")

    # Then convert
    converted = filtered_ds.map(
        convert_row, with_indices=True, num_proc=num_proc, desc="Converting NuminaMath", keep_in_memory=True
    )

    return converted


def subset(ds: Dataset, lim: Optional[int]):
    return ds if not lim or lim <= 0 or lim >= ds.num_rows else ds.select(range(lim))


def load_existing_metadata(input_path: Path) -> Optional[Dict[str, Any]]:
    """Load existing dataset metadata if it exists."""
    meta_file = Path(input_path) / "dataset_metadata.json"
    if meta_file.exists():
        try:
            with open(meta_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def save_dataset_and_metadata(
    dataset_dict: DatasetDict,
    output_path: Path,
    input_path: Path,
    args: argparse.Namespace,
):
    """Save converted dataset with processing metadata."""
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Save dataset
    dataset_dict.save_to_disk(str(output_path))

    # Load existing metadata or create new
    metadata = load_existing_metadata(input_path) or {}

    # Create processing entry
    processing_entry = {
        "operation": "convert_numinamath",
        "script": "convert_numinamath.py",
        "timestamp": datetime.now(UTC).isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "num_processes": args.num_proc,
        "limit": args.limit,
        "description": "Converted NuminaMath dataset with filtering, cleaning, and proper field mapping",
    }

    # Add to processing log
    if "processing_log" not in metadata:
        metadata["processing_log"] = []
    metadata["processing_log"].append(processing_entry)

    # Add format metadata if not already present
    if "format" not in metadata:
        metadata["format"] = "chat_format_v1"
    if "source_dataset" not in metadata:
        metadata["source_dataset"] = "AI-MO/NuminaMath-1.5"
    if "conversion_details" not in metadata:
        metadata["conversion_details"] = {
            "conversation_type": "mathematical_problem_solving",
            "problem_format": "text_based_math_problems",
            "solution_format": "step_by_step_explanations",
            "metadata_preservation": "full_original_metadata",
            "filtering": "removed_links_images_notfound",
            "cleaning": "removed_headers_short_prefixes",
            "field_mapping": "reasoning_verifiable_answer",
        }

    # Save metadata
    metadata_file = output_path / "dataset_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset saved to {output_path}")
    print(f"Metadata saved to {metadata_file}")


# ───────────— CLI / main ───────────── #
def cli():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input_path", default=SRC)
    p.add_argument("-o", "--output", required=False)
    p.add_argument("--num-proc", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main():
    a = cli()
    inp = Path(a.input_path)
    out = Path(
        a.output if not a.output.endswith("/") else a.output + inp.name + "-converted"
    )

    # if out.exists() and input(f"{out} exists. overwrite? [y/N]: ").lower() != "y":
    #     sys.exit(0)

    ds = load_dataset(str(inp))
    if not isinstance(ds, DatasetDict):
        ds = DatasetDict({"train": ds})

    out_ds = DatasetDict()
    for split, d in ds.items():
        print(f"{split}: {d.num_rows:,} rows")
        d = subset(d, a.limit)
        out_ds[split] = process_split(d, a.num_proc)

    save_dataset_and_metadata(out_ds, out, inp, a)


if __name__ == "__main__":
    main()
