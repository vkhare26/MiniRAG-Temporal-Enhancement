import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minirag import MiniRAG
from minirag.llm import hf_embed
from minirag.utils import EmbeddingFunc
from minirag.operate import chunking_by_token_size
from transformers import AutoModel, AutoTokenizer
import httpx
import json
import time
import logging
import asyncio
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("minirag").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

async def gpt_4o_mini_complete(prompt, max_tokens=200, hashing_kv=None, history_messages=None):
    api_key = "sk-zkOFrPfxbtucG60ybpANlQ"  # Hardcoded, consider using os.environ for security
    if not api_key:
        logger.error("LITELLM_API_KEY is not set")
        raise ValueError("API key not found")
    url = "https://cmu.litellm.ai/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    messages = history_messages if history_messages else [{"role": "user", "content": prompt}]
    if history_messages and isinstance(history_messages, list):
        messages = history_messages + [{"role": "user", "content": prompt}]
    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    timeout = httpx.Timeout(30.0)
    retries = 3
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(1 << attempt)
        except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(1 << attempt)
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG")
    parser.add_argument("--model", type=str, default="GPT4o")
    parser.add_argument("--workingdir", type=str, default="./LiHua-World")
    parser.add_argument("--datapath", type=str, default="/content/MiniRAG/LiHua-World")
    parser.add_argument("--start_index", type=int, default=0, help="Index to start processing from (0-based, corresponds to week number - 1)")
    args = parser.parse_args()
    return args

args = get_args()

if args.model == "GPT4o":
    LLM_MODEL = "gpt-4o-mini"
else:
    print("Invalid model name")
    exit(1)

WORKING_DIR = args.workingdir
DATA_PATH = args.datapath
START_INDEX = args.start_index  # Now interpreted as starting week number - 1 (e.g., 0 = week1)
CHECKPOINT_FILE = os.path.join(WORKING_DIR, "checkpoint.json")

print("USING LLM:", LLM_MODEL)
print("USING WORKING DIR:", WORKING_DIR)
print("DATA PATH:", DATA_PATH)
print("STARTING FROM WEEK:", START_INDEX + 1)  # Display as week number

if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

def custom_chunking(
    text,
    split_by_character=None,
    split_by_character_only=False,
    overlap=100,
    chunk_size=1200,
    model_name="gpt-4o-mini"
):
    return chunking_by_token_size(
        content=text,
        overlap_token_size=overlap,
        max_token_size=chunk_size,
        tiktoken_model=model_name
    )

rag = MiniRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete,
    llm_model_max_token_size=200,
    llm_model_name=LLM_MODEL,
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=1000,
        func=lambda texts: hf_embed(
            texts,
            tokenizer=AutoTokenizer.from_pretrained(EMBEDDING_MODEL),
            embed_model=AutoModel.from_pretrained(EMBEDDING_MODEL),
        ),
    ),
    chunking_func=custom_chunking
)

def find_txt_files(root_path):
    txt_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".txt"):
                full_path = os.path.join(root, file)
                txt_files.append(full_path)
    
    # Sort by week number extracted from directory name
    def get_week_number(filepath):
        # Extract week number from path (e.g., "week1", "week10")
        parts = filepath.split(os.sep)
        for part in parts:
            if part.startswith("week"):
                try:
                    return int(part.replace("week", ""))
                except ValueError:
                    return float('inf')  # Put non-numeric weeks at the end
        return float('inf')  # Files without "week" go last
    
    txt_files.sort(key=get_week_number)
    return txt_files

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"last_index": START_INDEX - 1, "last_file": None}

def save_checkpoint(index, filepath):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"last_index": index, "last_file": filepath}, f)
    logger.info(f"Checkpoint saved: index {index}, file {filepath}")

WEEK_LIST = find_txt_files(DATA_PATH)
logger.info(f"Found {len(WEEK_LIST)} .txt files in {DATA_PATH}")
if not WEEK_LIST:
    logger.warning("No .txt files found. Check datapath.")
else:
    checkpoint = load_checkpoint()
    start_index = max(checkpoint["last_index"] + 1, START_INDEX)
    for i, WEEK in enumerate(WEEK_LIST[start_index:], start=start_index):
        week_num = WEEK.split('week')[1].split(os.sep)[0] if 'week' in WEEK else "unknown"
        logger.info(f"Processing week {week_num} ({i}/{len(WEEK_LIST)}): {WEEK}")
        with open(WEEK, 'r', encoding='utf-8') as f:
            content = f.read()
            logger.info(f"Content size: {len(content)} chars")
            try:
                rag.insert(content)
                save_checkpoint(i, WEEK)
            except (httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                logger.warning(f"Network error processing {WEEK}: {str(e)}")
                save_checkpoint(i - 1, WEEK_LIST[i - 1] if i > START_INDEX else None)
                continue
            except Exception as e:
                logger.error(f"Failed to process {WEEK}: {str(e)}")
                save_checkpoint(i - 1, WEEK_LIST[i - 1] if i > START_INDEX else None)
                raise