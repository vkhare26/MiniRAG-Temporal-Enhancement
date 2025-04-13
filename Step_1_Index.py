
import sys
import os

from minirag import MiniRAG
from minirag.llm import hf_model_complete, hf_embed
from minirag.utils import EmbeddingFunc
from minirag.base import QueryParam  # Import QueryParam
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import argparse
import logging
import asyncio
import openai  # Import OpenAI package

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI API key (set your API key here or via environment variable)
openai.api_key = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")

# Custom LLM function for gpt-4o-mini using OpenAI API
def openai_model_complete(prompt, model="gpt-4o-mini", max_tokens=200):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        logger.error(f"Error in OpenAI API call: {str(e)}")
        return f"Error: {str(e)}"

def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG QA")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--workingdir", type=str, default="./LiHua-World")
    parser.add_argument("--querypath", type=str, default="/content/MiniRAG/dataset/LiHua-World/qa/query_set.csv")
    parser.add_argument("--outputpath", type=str, default="/content/MiniRAG/dataset/LiHua-World/qa/results.csv")
    args = parser.parse_args()
    return args

args = get_args()

# Select the model and corresponding LLM function
if args.model == "TinyLLaMA":
    LLM_MODEL = "TinyLLaMA/TinyLLaMA-1.1B-Chat-v1.0"
    llm_func = hf_model_complete
elif args.model == "GLM":
    LLM_MODEL = "THUDM/glm-edge-1.5b-chat"
    llm_func = hf_model_complete
elif args.model == "gpt-4o-mini":
    LLM_MODEL = "gpt-4o-mini"
    llm_func = openai_model_complete
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

# Set mode to MiniRAG only
MODE = "mini"  # Corresponds to MiniRAG in the table

# Initialize MiniRAG
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

# Load the query set
logger.info(f"Loading query set from {QUERY_PATH}")
query_df = pd.read_csv(QUERY_PATH)

# Clean the dataset: Remove incomplete rows
query_df = query_df.dropna(subset=["Question"])
query_df = query_df[query_df["Question"].str.strip() != ""]

# Extract queries and answers
queries = query_df["Question"].tolist()
gold_answers = query_df["Answer"].tolist()

# Add evidence as context to the queries
queries_with_context = []
for idx, row in query_df.iterrows():
    question = row["Question"]
    evidence = row["Evidence"]
    query_with_context = f"{question} (Context: Events occurred at {evidence})"
    queries_with_context.append(query_with_context)

# Run QA for MiniRAG
async def run_queries():
    generated_answers = []
    query_param = QueryParam(mode=MODE, top_k=5)
    for i, query in enumerate(queries_with_context):
        logger.info(f"[MiniRAG] Processing query {i+1}/{len(queries_with_context)}: {query}")
        try:
            answer = await rag.aquery(query, param=query_param)
            logger.info(f"[MiniRAG] Generated answer: {answer}")
            generated_answers.append(answer)
        except Exception as e:
            logger.error(f"[MiniRAG] Error processing query '{query}': {str(e)}")
            generated_answers.append("Error: " + str(e))
    return generated_answers

# Evaluate answers
def evaluate_answers(generated_answers, gold_answers):
    def normalize_answer(answer):
        answer = answer.strip().lower()
        if answer in ["yes", "no"]:
            return answer
        return answer

    exact_matches = []
    for gen, gold in zip(generated_answers, gold_answers):
        gen_norm = normalize_answer(gen)
        gold_norm = normalize_answer(gold)
        if gold_norm in ["yes", "no"]:
            match = gen_norm == gold_norm
        else:
            match = gold_norm in gen_norm
        exact_matches.append(1 if match else 0)

    accuracy = sum(exact_matches) / len(exact_matches) if exact_matches else 0
    error_rate = 1 - accuracy
    return exact_matches, accuracy, error_rate

# Run the queries and evaluate
loop = asyncio.get_event_loop()
generated_answers = loop.run_until_complete(run_queries())
exact_matches, accuracy, error_rate = evaluate_answers(generated_answers, gold_answers)

# Print per-query results
print("\nPer-Query Results for LiHuaWorld (MiniRAG with gpt-4o-mini):")
print(f"{'Query':<80} {'Gold Answer':<20} {'Generated Answer':<30} {'Exact Match':<10}")
print("-" * 140)
for i, (query, gold, gen, match) in enumerate(zip(queries, gold_answers, generated_answers, exact_matches)):
    # Truncate query for display if too long
    display_query = (query[:77] + "...") if len(query) > 77 else query
    print(f"{display_query:<80} {gold:<20} {gen:<30} {match:<10}")

# Print summary metrics
print("\nSummary Metrics for LiHuaWorld (MiniRAG with gpt-4o-mini):")
print(f"{'Method':<15} {'acc↑':<10} {'err↓':<10}")
print("-" * 35)
print(f"{'MiniRAG':<15} {accuracy:.2%}    {error_rate:.2%}")

# Save the results
results_df = pd.DataFrame({
    "Question": queries,
    "Answer": gold_answers,
    "Generated Answer": generated_answers,
    "Exact Match": exact_matches,
    "Method": "MiniRAG"
})
results_df.to_csv(OUTPUT_PATH, index=False)
logger.info(f"Results saved to {OUTPUT_PATH}")