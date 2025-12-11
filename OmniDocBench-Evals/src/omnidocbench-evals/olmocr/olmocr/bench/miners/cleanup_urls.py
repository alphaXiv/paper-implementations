# Rewrites all URLs in a dataset.jsonl file using a sql lite database lookup
import argparse
import json
import re
import sqlite3
from typing import Optional


def parse_pdf_hash(pretty_pdf_path: str) -> Optional[str]:
    pattern = r"s3://ai2-s2-pdfs/([a-f0-9]{4})/([a-f0-9]+)\.pdf"
    match = re.match(pattern, pretty_pdf_path)
    if match:
        return match.group(1) + match.group(2)
    return None


def get_uri_from_db(db_path: str, pdf_hash: str) -> Optional[str]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT uri FROM pdf_mapping WHERE pdf_hash = ?", (pdf_hash,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rewrites all URLs in a dataset.jsonl file using a sql lite database lookup")
    parser.add_argument("jsonl", type=str, help="JSONL file containing s3 paths")
    parser.add_argument("--db", type=str, required=True, help="Path to sqlite database mapping internal s3 urls to external ones")
    parser.add_argument("--force", action="store_true", help="Path to sqlite database mapping internal s3 urls to external ones")
    args = parser.parse_args()

    data = []
    skipped = 0

    with open(args.jsonl, "r") as inpf:
        for row in inpf:
            if len(row.strip()) > 0:
                j = json.loads(row)

                assert j["url"]
                hash = parse_pdf_hash(j["url"])
                if hash:
                    url = get_uri_from_db(args.db, hash)

                    if url:
                        j["url"] = url
                        data.append(j)
                    else:
                        skipped += 1
                else:
                    data.append(j)

    print(data)

    print(f"{skipped} entries were skipped!")

    if not args.force:
        print("Now run with --force to write data")
        quit()

    with open(args.jsonl, "w") as inpf:
        for row in data:
            print(json.dumps(row), file=inpf)
