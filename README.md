# Temporal Reasoning Enhancement in MiniRAG Systems 

## Overview

This project, conducted at Carnegie Mellon University in 2025, enhances the MiniRAG system to improve temporal reasoning on the LiHua-World dataset using GPT-4o-mini. By integrating a cross-encoder (ms-marco-MiniLM-L-6-v2) for re-ranking and a context fusion mechanism, the system prioritizes temporally relevant passages, achieving a 4% accuracy improvement (from 54% to 58%) and a 4.2% F1 score increase (from 57% to 62%) over the MiniRAG baseline on 200 temporal queries.

## Methodology





Data Processing: Chunked input text into 1200-token segments with 100-token overlap, stored in a JSON-based vector store.



Heterogeneous Graph Indexing: Built a graph linking events, dates, and objects for semantic retrieval.



Retrieval & Re-ranking: Used graph-based cosine similarity for initial retrieval, followed by cross-encoder re-ranking to prioritize semantic and temporal relevance.



Context Fusion: Combined top-scoring passages, weighted by relevance, into a token-budgeted context block (500 characters per document).



Answer Generation: Prompted GPT-4o-mini to generate "Yes/No + justification" answers for temporal queries (e.g., "Did Event X happen before Event Y?").

## Results





Accuracy: Improved from 54% to 58% (4% gain).



F1 Score: Increased from 57% to 62% (4.2% gain).



Query Performance: Correctly answered 117 out of 200 queries, compared to 109 by the baseline.



Error Reduction: Reduced error rate by 4%, with remaining errors tied to implicit time expressions and token truncation.
