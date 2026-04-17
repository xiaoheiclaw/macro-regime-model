"""One-off arxiv literature search for localize + forecast methodology.

Targets q-fin.* and econ.EM with keyword queries. Dedupes across queries,
dumps abstracts to docs/literature/methodology_search_<date>.md for review.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import httpx

ARXIV_API = "https://export.arxiv.org/api/query"
PROXY = "http://127.0.0.1:59527"
NS = {"atom": "http://www.w3.org/2005/Atom"}

QUERIES: dict[str, str] = {
    "analog-forecasting": 'abs:"analog forecasting" OR abs:"nearest neighbor forecasting"',
    "regime-switching-forecast": 'abs:"regime switching" AND abs:forecast AND (cat:q-fin.ST OR cat:q-fin.PM OR cat:q-fin.RM OR cat:econ.EM)',
    "hidden-markov-asset": '(abs:"hidden Markov" OR abs:"Markov switching") AND abs:asset AND (cat:q-fin.* OR cat:econ.*)',
    "manifold-finance": '(abs:"manifold learning" OR abs:"diffusion map" OR abs:"diffusion maps") AND (cat:q-fin.* OR cat:econ.*)',
    "scenario-generation": 'abs:"scenario generation" AND (cat:q-fin.PM OR cat:q-fin.RM OR cat:q-fin.CP)',
    "long-horizon-forecast": 'abs:"long horizon" AND abs:forecast AND (cat:q-fin.* OR cat:econ.EM)',
    "recession-business-cycle": '(abs:"recession prediction" OR abs:"business cycle" ) AND abs:forecast AND cat:econ.*',
    "regime-asset-allocation": 'abs:regime AND abs:"asset allocation" AND cat:q-fin.PM',
    "nowcasting-macro": 'abs:nowcasting AND cat:econ.EM',
    "foundation-model-timeseries": '(abs:TimesFM OR abs:Chronos OR abs:"foundation model") AND abs:forecast AND (cat:q-fin.* OR cat:stat.ML)',
    "transition-prob-macro": '(abs:"transition probability" OR abs:"transition matrix") AND abs:macro AND (cat:q-fin.* OR cat:econ.*)',
    "wasserstein-regime": 'abs:Wasserstein AND (abs:regime OR abs:cluster) AND (cat:q-fin.* OR cat:stat.ML)',
}


def search(query: str, max_results: int = 15) -> list[dict]:
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    with httpx.Client(proxy=PROXY, timeout=30) as c:
        resp = c.get(ARXIV_API, params=params)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
    items = []
    for entry in root.findall("atom:entry", NS):
        arxiv_id = (entry.findtext("atom:id", "", NS) or "").split("/abs/")[-1]
        title = (entry.findtext("atom:title", "", NS) or "").strip().replace("\n", " ")
        summary = (entry.findtext("atom:summary", "", NS) or "").strip()
        published = entry.findtext("atom:published", "", NS) or ""
        authors = [a.findtext("atom:name", "", NS) for a in entry.findall("atom:author", NS)]
        items.append({
            "id": arxiv_id,
            "title": title,
            "abstract": summary,
            "published": published[:10],
            "authors": authors,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
        })
    return items


def main() -> int:
    today = datetime.now().strftime("%Y-%m-%d")
    out_dir = Path(__file__).parent.parent / "docs" / "literature"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"methodology_search_{today}.md"

    seen: dict[str, dict] = {}
    per_topic: dict[str, list[str]] = {}
    failed: list[tuple[str, str]] = []

    for topic, q in QUERIES.items():
        print(f"[{topic}] {q[:80]}...", file=sys.stderr)
        try:
            items = search(q, max_results=15)
        except Exception as e:
            failed.append((topic, str(e)))
            print(f"  FAIL: {e}", file=sys.stderr)
            continue
        per_topic[topic] = []
        for item in items:
            if item["id"] not in seen:
                seen[item["id"]] = item
            per_topic[topic].append(item["id"])
        time.sleep(3)

    lines = [
        f"# Methodology literature search — {today}",
        "",
        f"Purpose: decide localize-and-forecast methodology for lab-macro-regime.",
        f"Unique papers: {len(seen)} across {len(per_topic)} queries.",
        "",
    ]
    if failed:
        lines.append("## Failed queries")
        for topic, err in failed:
            lines.append(f"- `{topic}`: {err}")
        lines.append("")

    lines.append("## Papers by topic")
    lines.append("")
    for topic, ids in per_topic.items():
        lines.append(f"### {topic}")
        lines.append(f"Query: `{QUERIES[topic]}`")
        lines.append("")
        for pid in ids:
            item = seen[pid]
            lines.append(f"- **[{item['title']}]({item['url']})** ({item['published']})")
            authors_short = ", ".join(item['authors'][:3]) + (" et al." if len(item['authors']) > 3 else "")
            lines.append(f"  - {authors_short}")
            abstract = item['abstract'].replace("\n", " ")[:400]
            lines.append(f"  - {abstract}...")
        lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"\nWritten: {out_path}", file=sys.stderr)
    print(f"Unique papers: {len(seen)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
