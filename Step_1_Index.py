import pandas as pd
import argparse
import logging
import asyncio
import httpx
from minirag import MiniRAG
from minirag.llm import hf_model_complete, hf_embed
from minirag.utils import EmbeddingFunc
from minirag.base import QueryParam
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import CrossEncoder

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the cross-encoder for re-ranking
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

async def gpt_4o_mini_complete(prompt, max_tokens=200, hashing_kv=None, history_messages=None, system_prompt=None):
    api_key = "sk-zkOFrPfxbtucG60ybpANlQ"
    if not api_key:
        logger.error("LITELLM_API_KEY is not set")
        raise ValueError("API key not found")
    url = "https://cmu.litellm.ai/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    messages = []
    if system_prompt is None:
        system_prompt = "You are a precise question-answering assistant. Provide a detailed and accurate answer based on the context provided. Use the timeline and events in the context to inform your response. If the evidence is ambiguous, state that clearly and avoid speculation."
    messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    timeout = httpx.Timeout(60.0)
    retries = 3
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
        except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            logger.warning(f"Timeout on attempt {attempt + 1}/{retries}: {str(e)}")
            if attempt == retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (502, 503, 504):
                logger.warning(f"Server error {e.response.status_code} on attempt {attempt + 1}/{retries}")
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
            else:
                logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
                raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise

def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG QA")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--workingdir", type=str, default="./LiHua-World")
    parser.add_argument("--querypath", type=str, default="/Users/vkhare26/Documents/anlp/Minirag_repo/MiniRAG/dataset/LiHua-World/qa/query_set.csv")
    parser.add_argument("--outputpath", type=str, default="/Users/vkhare26/Documents/anlp/Minirag_repo/MiniRAG/dataset/LiHua-World/qa/results.csv")
    args = parser.parse_args()
    return args

args = get_args()

if args.model == "TinyLLaMA":
    LLM_MODEL = "TinyLLaMA/TinyLLaMA-1.1B-Chat-v1.0"
    llm_func = hf_model_complete
elif args.model == "GLM":
    LLM_MODEL = "THUDM/glm-edge-1.5b-chat"
    llm_func = hf_model_complete
elif args.model == "gpt-4o-mini":
    LLM_MODEL = "gpt-4o-mini"
    llm_func = gpt_4o_mini_complete
else:
    print("Invalid model name")
    exit(1)

WORKING_DIR = args.workingdir
QUERY_PATH = args.querypath
OUTPUT_PATH = args.outputpath
print("USING LLM:", LLM_MODEL)
print("USING WORKING DIR:", WORKING_DIR)
print("QUERY PATH:", QUERY_PATH)
print("OUTPUT PATH:", OUTPUT_PATH)

MODE = "mini"

rag = MiniRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_func,
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
)

logger.info(f"Loading query set from {QUERY_PATH}")
query_df = pd.read_csv(QUERY_PATH)
query_df = query_df.dropna(subset=["Question"])
query_df = query_df[query_df["Question"].str.strip() != ""]

queries = query_df["Question"].tolist()
gold_answers = query_df["Gold Answer"].tolist()
evidence_list = query_df["Evidence"].tolist()

queries_with_context = []
for idx, row in query_df.iterrows():
    question = row["Question"]
    evidence = row["Evidence"]
    query_with_context = f"{question} (Context: Events occurred at {evidence})"
    queries_with_context.append(query_with_context)

async def run_queries():
    generated_answers = []
    query_param = QueryParam(mode=MODE, top_k=10)
    for i, query in enumerate(queries_with_context):
        logger.info(f"[MiniRAG] Processing query {i+1}/{len(queries_with_context)}: {query}")
        try:
            # Retrieve top-K candidates with documents
            result = await rag.aquery(query, param=query_param, return_docs=True)
            docs = result["docs"]
            
            if not docs:
                answer = "No relevant documents found."
            else:
                # Re-rank the retrieved documents using the cross-encoder
                query_doc_pairs = [[query, doc["content"]] for doc in docs]
                scores = cross_encoder.predict(query_doc_pairs)
                
                # Sort documents by score and select the top one
                ranked_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
                top_doc = ranked_docs[0][1]["content"] if ranked_docs else "No relevant documents found."
                top_score = ranked_docs[0][0] if ranked_docs else 0
                logger.info(f"[Re-Ranking] Top document score: {top_score:.4f}")
                
                # Generate the final answer using the top re-ranked document
                prompt = f"Based on the following context, answer the question in detail:\nContext: {top_doc}\nQuestion: {query}"
                answer = await gpt_4o_mini_complete(prompt)
            
            logger.info(f"[MiniRAG] Generated answer: {answer}")
            generated_answers.append(answer)
        except Exception as e:
            logger.error(f"[MiniRAG] Error processing query '{query}': {str(e)}")
            generated_answers.append("Error: " + str(e))
        
        if (i + 1) % 50 == 0 or (i + 1) == len(queries_with_context):
            exact_matches, accuracy, error_rate = evaluate_answers(
                generated_answers[:i+1], gold_answers[:i+1]
            )
            print(f"\nIntermediate Evaluation after {i+1} queries:")
            print(f"{'Method':<15} {'acc↑':<10} {'err↓':<10}")
            print("-" * 35)
            print(f"{'MiniRAG':<15} {accuracy:.2%}    {error_rate:.2%}")
    
    return generated_answers

def evaluate_answers(generated_answers, gold_answers):
    def contains_yes_no(text, gold):
        text_lower = text.lower()
        gold_lower = gold.lower()
        if gold_lower == "yes":
            return "yes" in text_lower
        elif gold_lower == "no":
            return "no" in text_lower
        return False

    exact_matches = []
    for gen, gold in zip(generated_answers, gold_answers):
        # Since gold answers are "Yes"/"No", check if the generated answer contains the same sentiment
        match = contains_yes_no(gen, gold)
        exact_matches.append(1 if match else 0)

    accuracy = sum(exact_matches) / len(exact_matches) if exact_matches else 0
    error_rate = 1 - accuracy
    return exact_matches, accuracy, error_rate

loop = asyncio.get_event_loop()
generated_answers = loop.run_until_complete(run_queries())
exact_matches, accuracy, error_rate = evaluate_answers(generated_answers, gold_answers)

print("\nPer-Query Results for LiHuaWorld (MiniRAG with gpt-4o-mini):")
print(f"{'Query':<80} {'Evidence':<40} {'Gold Answer':<20} {'Generated Answer':<50} {'Exact Match':<10}")
print("-" * 200)
for i, (query, evidence, gold, gen, match) in enumerate(zip(queries, evidence_list, gold_answers, generated_answers, exact_matches)):
    display_query = (query[:77] + "...") if len(query) > 77 else query
    display_evidence = (evidence[:37] + "...") if len(evidence) > 37 else evidence
    display_gen = (gen[:47] + "...") if len(gen) > 47 else gen
    print(f"{display_query:<80} {display_evidence:<40} {gold:<20} {display_gen:<50} {match:<10}")

print("\nSummary Metrics for LiHuaWorld (MiniRAG with gpt-4o-mini):")
print(f"{'Method':<15} {'acc↑':<10} {'err↓':<10}")
print("-" * 35)
print(f"{'MiniRAG':<15} {accuracy:.2%}    {error_rate:.2%}")

results_df = pd.DataFrame({
    "Question": queries,
    "Evidence": evidence_list,
    "Gold Answer": gold_answers,
    "Generated Answer": generated_answers,
    "Exact Match": exact_matches,
    "Method": "MiniRAG"
})
results_df.to_csv(OUTPUT_PATH, index=False)
logger.info(f"Results saved to {OUTPUT_PATH}")