"""
Advanced — PageIndex vs Traditional Vector RAG (Self-Hosted)
=========================================================================

Side-by-side comparison of PageIndex (reasoning-based) vs traditional
vector RAG (embedding-based) on the SAME document and queries.

This demonstrates WHY reasoning-based retrieval produces better results
for structured documents.

Note: This example simulates vector RAG by chunking + embedding-based
ranking (using OpenAI embeddings) to show the contrast. No vector DB needed.
"""

import sys
import os
import json
import asyncio
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from openai import OpenAI
from utils.helpers import (
    get_openai_key,
    print_header,
    print_step,
    print_result,
    save_json,
    create_node_mapping,
    ensure_dir,
    register_artifact,
    console,
    PAGEINDEX_SRC,
)

sys.path.insert(0, PAGEINDEX_SRC)
from pageindex.page_index_md import md_to_tree
from pageindex.utils import remove_fields


def call_llm(prompt: str, model: str = "gpt-4o") -> str:
    client = OpenAI(api_key=get_openai_key())
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def get_embeddings(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """Get OpenAI embeddings for a list of texts."""
    client = OpenAI(api_key=get_openai_key())
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


# ─────────────────────────────────────────────────────────
# Traditional Vector RAG (simulated, no vector DB)
# ─────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    """Split text into overlapping chunks (traditional RAG approach)."""
    words = text.split()
    chunks = []
    start = 0
    chunk_id = 0
    while start < len(words):
        end = start + chunk_size
        chunk_content = " ".join(words[start:end])
        chunks.append({"id": chunk_id, "text": chunk_content, "start_word": start})
        chunk_id += 1
        start += chunk_size - overlap
    return chunks


def vector_rag(document_text: str, query: str, top_k: int = 3) -> dict:
    """Simulate traditional vector RAG: chunk → embed → similarity search → answer."""

    # Step 1: Chunk the document
    chunks = chunk_text(document_text, chunk_size=300, overlap=50)

    # Step 2: Embed all chunks + the query
    all_texts = [c["text"] for c in chunks] + [query]
    embeddings = get_embeddings(all_texts)

    chunk_embeddings = embeddings[:-1]
    query_embedding = embeddings[-1]

    # Step 3: Rank chunks by cosine similarity
    for i, chunk in enumerate(chunks):
        chunk["similarity"] = cosine_similarity(query_embedding, chunk_embeddings[i])

    chunks.sort(key=lambda c: c["similarity"], reverse=True)
    top_chunks = chunks[:top_k]

    # Step 4: Generate answer from top-k chunks
    context = "\n\n".join(
        f"[Chunk {c['id']}, similarity={c['similarity']:.3f}]\n{c['text']}"
        for c in top_chunks
    )

    answer = call_llm(
        f"Answer based on context:\nQuestion: {query}\nContext:\n{context}\nAnswer:"
    )

    return {
        "method": "Vector RAG",
        "num_chunks": len(chunks),
        "top_k": top_k,
        "retrieved_chunks": [
            {"id": c["id"], "similarity": round(c["similarity"], 3), "preview": c["text"][:100]}
            for c in top_chunks
        ],
        "answer": answer,
    }


# ─────────────────────────────────────────────────────────
# PageIndex Reasoning-Based RAG
# ─────────────────────────────────────────────────────────

def pageindex_rag(structure: list, query: str) -> dict:
    """PageIndex reasoning-based RAG: tree search → extract → answer."""

    # Step 1: Reasoning-based tree search
    tree_compact = remove_fields(structure, fields=["text"])
    search_prompt = f"""Find nodes that answer the question.
Question: {query}
Tree: {json.dumps(tree_compact, indent=2)}
Reply in JSON: {{"thinking": "<reasoning>", "node_list": ["id1"]}}"""

    raw = call_llm(search_prompt)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        result = json.loads(match.group()) if match else {"thinking": "", "node_list": []}

    # Step 2: Extract context
    node_map = create_node_mapping(structure)
    context_parts = []
    retrieved = []
    for nid in result.get("node_list", []):
        if nid in node_map and node_map[nid].get("text"):
            node = node_map[nid]
            context_parts.append(f"[{node.get('title', '?')}]\n{node['text']}")
            retrieved.append({"node_id": nid, "title": node.get("title", "?")})

    # Step 3: Generate answer
    context = "\n\n".join(context_parts)
    answer = call_llm(
        f"Answer based on context:\nQuestion: {query}\nContext:\n{context}\nAnswer:"
    )

    return {
        "method": "PageIndex RAG",
        "reasoning": result.get("thinking", ""),
        "retrieved_nodes": retrieved,
        "answer": answer,
    }


def main():
    print_header("Example 08: PageIndex vs Traditional Vector RAG")

    # ─── Create a document that highlights the differences ───
    print_step(1, "Create test document with cross-references and structure")

    doc_path = os.path.join("data", "sample_docs", "system_architecture.md")
    ensure_dir(os.path.dirname(doc_path))

    doc_content = """# Distributed System Architecture Guide

## 1. System Overview

This guide covers the architecture of our distributed microservices platform.
The system handles 50,000 requests per second across 12 services.
For authentication details, see Section 3. For database design, see Section 4.
Performance benchmarks are documented in Section 6.

## 2. Service Communication

### 2.1 Synchronous Communication
Services communicate via gRPC for low-latency calls. REST APIs are used for
external-facing endpoints. Average latency: 5ms for gRPC, 15ms for REST.

### 2.2 Asynchronous Communication
Apache Kafka handles event streaming between services. We use the outbox pattern
for reliable event publishing. Message throughput: 100K messages/second.
Dead letter queues handle failed message processing (see Section 5 for monitoring).

## 3. Authentication & Authorization

### 3.1 Authentication Flow
Users authenticate via OAuth 2.0 with JWT tokens. Token lifetime: 15 minutes.
Refresh tokens expire after 7 days. The auth service (referenced in Section 2)
handles token issuance and validation.

### 3.2 Authorization
Role-based access control (RBAC) with fine-grained permissions.
Roles: admin, editor, viewer. Permission checks happen at the API gateway
(see Section 2.1 for gateway details).

## 4. Database Architecture

### 4.1 Primary Database
PostgreSQL 15 with read replicas. Sharding strategy: hash-based on tenant_id.
Connection pooling via PgBouncer (max 500 connections per shard).

### 4.2 Caching Layer
Redis cluster for session data and frequently accessed resources.
Cache invalidation: write-through for critical data, TTL-based for non-critical.
Cache hit rate: 94% (see Section 6 for performance details).

### 4.3 Search Engine
Elasticsearch for full-text search and analytics queries.
Index size: 500GB across 5 shards. Refresh interval: 1 second.

## 5. Monitoring & Alerting

### 5.1 Observability Stack
- Metrics: Prometheus + Grafana (see Section 6 for key metrics)
- Logs: ELK Stack (Elasticsearch, Logstash, Kibana)
- Traces: Jaeger for distributed tracing across services (Section 2)

### 5.2 Alert Rules
- P1: Service down or error rate > 5% → PagerDuty → 5min response
- P2: Latency > 100ms (p99) → Slack notification
- P3: Cache hit rate < 90% (see Section 4.2) → Daily report

## 6. Performance Benchmarks

### 6.1 Load Testing Results
- Throughput: 50,000 RPS at p99 latency < 50ms
- Auth service (Section 3): 10,000 token validations/second
- Database (Section 4): 5,000 write operations/second
- Message queue (Section 2.2): 100,000 messages/second

### 6.2 Scalability
Horizontal scaling tested up to 100 nodes. Linear throughput increase
up to 80 nodes, then diminishing returns due to coordination overhead.
Auto-scaling triggers at 70% CPU utilization.
"""

    with open(doc_path, "w") as f:
        f.write(doc_content)
    register_artifact(doc_path, kind="markdown", note="Comparison test document")

    # Read the full text for vector RAG
    with open(doc_path) as f:
        full_text = f.read()

    # Generate tree for PageIndex RAG
    print_step(2, "Generate PageIndex tree")
    tree_result = asyncio.run(md_to_tree(
        md_path=doc_path,
        if_thinning=False,
        if_add_node_summary="yes",
        summary_token_threshold=200,
        model="gpt-4o-2024-11-20",
        if_add_node_text="yes",
        if_add_node_id="yes",
    ))
    structure = tree_result["structure"]

    # ─── Run comparison queries ───
    queries = [
        # Query 1: Requires following cross-references
        "What is the cache hit rate and how does it relate to the alerting rules?",
        # Query 2: Requires understanding document structure
        "Give me a complete picture of how authentication works end-to-end, including which services are involved.",
        # Query 3: Requires gathering info from multiple sections
        "What are all the performance numbers mentioned across the entire system?",
    ]

    for i, query in enumerate(queries, 1):
        print_step(2 + i, f"Query {i}: {query}")

        # Run both methods
        console.print("\n[bold blue]── Vector RAG ──[/bold blue]")
        v_result = vector_rag(full_text, query)
        console.print(f"Chunks created: {v_result['num_chunks']}")
        console.print(f"Top-{v_result['top_k']} chunks retrieved:")
        for c in v_result["retrieved_chunks"]:
            console.print(f"  [{c['similarity']:.3f}] {c['preview']}...")
        console.print(f"\n[bold]Vector RAG Answer:[/bold]")
        print_result(v_result["answer"])

        console.print("\n[bold magenta]── PageIndex RAG ──[/bold magenta]")
        p_result = pageindex_rag(structure, query)
        console.print(f"Reasoning: {p_result['reasoning']}")
        console.print(f"Retrieved nodes:")
        for n in p_result["retrieved_nodes"]:
            console.print(f"  [green]✓[/green] {n['title']} (id={n['node_id']})")
        console.print(f"\n[bold]PageIndex Answer:[/bold]")
        print_result(p_result["answer"])

        console.print("[dim]─" * 60 + "[/dim]")

    # ─── Summary ───
    console.print("\n[bold green]Comparison Summary:[/bold green]")
    console.print("")
    console.print("  [bold]Vector RAG[/bold]")
    console.print("  ✓ Simple setup, no tree generation step")
    console.print("  ✓ Works on any text (no structure needed)")
    console.print("  ✓ Fast retrieval (single embedding comparison)")
    console.print("  ✗ Chunks break logical sections apart")
    console.print("  ✗ Cannot follow cross-references")
    console.print("  ✗ Similarity ≠ relevance (matches surface words)")
    console.print("")
    console.print("  [bold]PageIndex RAG[/bold]")
    console.print("  ✓ Preserves document structure and logic")
    console.print("  ✓ LLM understands WHAT to retrieve (not just similar text)")
    console.print("  ✓ Follows cross-references between sections")
    console.print("  ✓ Full explainability (reasoning trace)")
    console.print("  ✗ Requires structured documents")
    console.print("  ✗ Tree generation costs time + API calls")
    console.print("  ✗ Each retrieval requires LLM reasoning (slower)\n")


if __name__ == "__main__":
    main()