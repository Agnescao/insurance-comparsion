from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from sqlalchemy.orm import Session

from app.database import engine
from app.models import Plan
from app.services.hybrid_retriever import HybridRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate hybrid retrieval hit@k on labeled queries.")
    parser.add_argument("--dataset", required=True, help="JSONL path, each line: {query, relevant_plan_ids:[...]} ")
    parser.add_argument("--k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    rows = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    if not rows:
        print("empty dataset")
        return

    retriever = HybridRetriever()
    hit = 0
    with Session(engine) as db:
        plan_name_to_id = {p.name: p.plan_id for p in db.query(Plan).all()}
        for r in rows:
            query = r["query"]
            relevant = set(r.get("relevant_plan_ids", []))
            for plan_name in r.get("relevant_plan_names", []):
                pid = plan_name_to_id.get(plan_name)
                if pid:
                    relevant.add(pid)
            pred = retriever.discover_plan_ids(db, query=query, top_k=args.k)
            if relevant.intersection(pred):
                hit += 1
            print(f"query={query} pred={pred} relevant={list(relevant)}")

    score = hit / len(rows)
    print(f"hit@{args.k}={score:.4f} ({hit}/{len(rows)})")


if __name__ == "__main__":
    main()
