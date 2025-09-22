import os
import time
import html
import re
import json
import gc
import logging
from pathlib import Path
import base64
import fitz  # PyMuPDF
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from agno.agent import Agent
from agno.models.google import Gemini

load_dotenv()

# ========== CONFIG ==========
PDF_PATH = os.getenv("PDF_PATH", "dewa_noc.pdf")
IMAGES_DIR = os.getenv("IMAGES_DIR", "pdf_pages")
CHECKPOINT_FILE = os.getenv("CHECKPOINT_FILE", "checkpoint.txt")
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
RATE_SLEEP_SEC = int(os.getenv("RATE_SLEEP_SEC", "8"))  # gentle default
RENDER_DPI = int(os.getenv("RENDER_DPI", "144"))        # 144 ~= 2x zoom
MAX_PAGES = int(os.getenv("MAX_PAGES", "0"))            # 0 = no limit

# ===========================
print(f"pdf_path: {PDF_PATH}, images_dir: {IMAGES_DIR}, checkpoint_file: {CHECKPOINT_FILE}")
logging.basicConfig(
    filename="gemini_pdf_qa_generator.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\\n", " ")
    text = text.replace("\u00a0", " ")
    text = text.replace("\\/", "/")
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def path_to_file_uri(path: Path) -> str:
    # Works on Windows & POSIX
    return path.resolve().as_uri()

def load_checkpoint() -> int:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            val = f.read().strip()
            if val.isdigit():
                return int(val)
    return 0

def save_checkpoint(idx: int) -> None:
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        f.write(str(idx))

def safe_run_agent(agent: Agent, messages):
    try:
        return agent.run(messages)
    except Exception as e:
        emsg = str(e).lower()
        if any(k in emsg for k in ["quota", "rate", "permission", "429"]):
            logging.critical(f"🚫 Rate/Quota error: {e}")
        else:
            logging.error(f"❌ LLM error: {e}")
        raise

def build_messages(page_text: str, image_path: Path | None):
    prompt_text = f"""
You are a helpful assistant. Generate exactly 3 high-quality question–answer pairs
from this DEWA NOC page. If both text and diagram are present, ground the Q&A in BOTH.
If only one modality exists, use that modality only.

Return STRICT JSON (no prose) in this shape:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]

Text (may be empty):
\"\"\"{page_text or ""}\"\"\"
""".strip()

    if image_path and image_path.exists():
        # ✅ Agno Gemini takes [text, image_path] directly
        return [prompt_text, image_path.as_posix()]
    else:
        return prompt_text

def parse_pairs(raw: str):
    """
    Parse model output to a list of {question, answer}.
    Try strict JSON first, then a lenient fallback.
    """
    raw = raw.strip().strip("`")
    raw = raw.replace("```json", "").replace("```", "").strip()
    if raw.startswith("json\n"):
        raw = raw[5:].strip()

    # Try strict JSON
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Try to extract JSON-like block between first '[' and last ']'
    try:
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            snippet = raw[start:end+1]
            return json.loads(snippet)
    except Exception:
        pass

    # Final fallback (safe-ish) — ast.literal_eval to avoid eval risks
    import ast
    try:
        return ast.literal_eval(raw)
    except Exception as e:
        raise ValueError(f"Could not parse model output as JSON. Output was:\n{raw}") from e

def render_page_image(page: fitz.Page, out_dir: Path, page_num: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    # matrix by DPI
    zoom = RENDER_DPI / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    out_path = out_dir / f"page_{page_num}.png"
    pix.save(out_path.as_posix())
    # free pixmap memory
    del pix
    return out_path


def encode_image_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main():


    # Prepare agent
    agent = Agent(
        model=Gemini(id=GEMINI_MODEL_ID),
        markdown=True,
    )

    pdf_path = Path(PDF_PATH)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    start_index = load_checkpoint()
    logging.info(f"⏳ Resuming from page index {start_index} (0-based)")

    with fitz.open(pdf_path.as_posix()) as doc:
        total_pages = doc.page_count
        logging.info(f"📘 PDF opened: {pdf_path.name} with {total_pages} pages")

        # Optional: limit pages (debug/runs)
        last_page = total_pages if MAX_PAGES <= 0 else min(total_pages, start_index + MAX_PAGES)

        for idx in range(start_index, last_page):
            page_num = idx + 1
            logging.info(f"📄 Processing page {page_num}/{total_pages}")
            try:
                page = doc[idx]

                # Extract text for THIS page
                text = clean_text(page.get_text("text") or "")

                # Always render a page image (keeps multimodal consistent)
                image_path = render_page_image(page, Path(IMAGES_DIR), page_num)

                # image_uri = path_to_file_uri(image_path) if image_path.exists() else None

                # Build multimodal message
                input_data  = build_messages(text, image_path)
                print(f"input_data : {input_data }")
                # Try multimodal first
                try:
                    response = safe_run_agent(agent, input_data )
                except Exception as e:
                    # Fallback: try text-only if multimodal failed (e.g., image loading issue)
                    logging.warning(f"⚠️ Multimodal failed on page {page_num}. Retrying text-only. Error: {e}")
                    messages_text_only = build_messages(text, None)
                    response = safe_run_agent(agent, messages_text_only)

                raw = response.content if hasattr(response, "content") else str(response)
                pairs = parse_pairs(raw)

                qa_rows = []
                for p in pairs:
                    q = (p.get("question") or "").strip()
                    a = (p.get("answer") or "").strip()
                    if not q and not a:
                        continue
                    qa_rows.append({
                        "question": q,
                        "answer": a,
                        "context": text,
                        "page_num": page_num,
                        "image_path": image_path.as_posix()
                    })

                if not qa_rows:
                    logging.warning(f"⚠️ No QA parsed on page {page_num}. Skipping save.")
                else:
                    # Save to Hugging Face (per-page incremental push)
                    ds = Dataset.from_dict({
                        "question": [r["question"] for r in qa_rows],
                        "answer":   [r["answer"]   for r in qa_rows],
                        "context":  [r["context"]  for r in qa_rows],
                        "page_num": [r["page_num"] for r in qa_rows],
                        "image_path": [r["image_path"] for r in qa_rows],
                    })
                    ds.save_to_disk("gemini_pdf_qa_dataset")  # local snapshot (overwrites each page)
                    # ds.push_to_hub(HF_DATASET_ID, token=os.getenv("HF_TOKEN"))

                    # Append/maintain CSV + JSONL (streaming-friendly)
                    df = pd.DataFrame(qa_rows)
                    if os.path.exists("qa_pairs.csv"):
                        df.to_csv("qa_pairs.csv", mode="a", index=False, header=False)
                    else:
                        df.to_csv("qa_pairs.csv", index=False)
                    with open("qa_pairs.jsonl", "a", encoding="utf-8") as f:
                        for r in qa_rows:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")

                    logging.info(f"✅ Saved {len(qa_rows)} QA rows for page {page_num}")

                # ✅ checkpoint AFTER successful page
                save_checkpoint(idx + 1)

                # free memory
                del page
                gc.collect()

                # be nice to the API
                time.sleep(RATE_SLEEP_SEC)

            except Exception as e:
                logging.error(f"❌ Failed on page {page_num}: {e}")
                # do NOT advance checkpoint; resume will retry this page
                break

    logging.info("🏁 Done (or paused due to error).")

if __name__ == "__main__":
    main()
