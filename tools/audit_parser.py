from __future__ import annotations
import io, gzip, json, sys, pathlib
from typing import Iterator, Tuple, Optional, Dict, Any

class LineParseError(Exception):
    pass

def _iter_json_objects_from_string(s: str) -> Iterator[Dict[str, Any]]:
    """
    Robustly parse one or more JSON objects concatenated in a single string.
    Example:
      {"a":1}{"b":2}
      {"event":...},{"sha1":"..."}
    """
    dec = json.JSONDecoder()
    i, n = 0, len(s)
    while i < n:
        # Skip whitespace and stray commas
        while i < n and s[i] in " \t\r\n,":
            i += 1
        if i >= n: 
            break
        obj, end = dec.raw_decode(s, i)
        yield obj
        i = end

def parse_audit_line(line: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Returns (event_obj, sha1_hex) for a single audit line.
    If the line has only an event, sha1_hex is None.
    If malformed, returns (None, None) and lets caller decide logging policy.
    """
    try:
        # Handle the specific format: {"main_object"},"sha1":"hash_value"}
        if '},"sha1":"' in line and line.endswith('}'):
            # Split at the SHA1 separator
            parts = line.split('},"sha1":"', 1)
            if len(parts) == 2:
                json_part = parts[0] + '}'
                sha1_part = parts[1].rstrip('}"')
                
                # Parse the main JSON object
                event = json.loads(json_part)
                sha1 = sha1_part
                return event, sha1
        
        # Fallback: try to parse as multiple JSON objects
        objs = list(_iter_json_objects_from_string(line))
        if not objs:
            return None, None
        # Strategy:
        # - If there's a dict with a 'type' (event) and a dict with only 'sha1', pair them.
        # - If multiple events in same line (unexpected), use the first, prefer final sha1.
        event = None
        sha1 = None
        for obj in objs:
            if isinstance(obj, dict) and "sha1" in obj and len(obj) == 1:
                # Trailing checksum
                sha1 = obj.get("sha1")
            elif isinstance(obj, dict) and obj.get("type"):
                # Candidate event
                if event is None:
                    event = obj
        return event, sha1
    except json.JSONDecodeError:
        return None, None

def open_maybe_gz(path: str | pathlib.Path) -> io.TextIOBase:
    p = pathlib.Path(path)
    if p.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(p, "rb"), encoding="utf-8", newline="")
    return open(p, "r", encoding="utf-8", newline="")

def iter_audit_file(path: str | pathlib.Path, *, max_json_errors: int = 10) -> Iterator[Tuple[int, Dict[str, Any], Optional[str]]]:
    """
    Yields (line_num, event, sha1) for each well-formed event line.
    Silently skips lines that cannot be parsed as JSON (up to max_json_errors logged to stderr).
    """
    json_errs = 0
    with open_maybe_gz(path) as f:
        for ln, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            event, sha1 = parse_audit_line(line)
            if event is None:
                if json_errs < max_json_errors:
                    json_errs += 1
                    print(f"⚠️  Invalid JSON on line {ln}: {line[:140]}...", file=sys.stderr)
                continue
            yield ln, event, sha1
