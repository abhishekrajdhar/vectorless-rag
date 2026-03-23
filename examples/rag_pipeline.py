"""
Traditional RAG: chunk → embed → store in vector DB → query → cosine similarity → top-k
PageIndex RAG:   document → tree index → LLM reasons over tree → extract context → answer
"""

import sys
import os
import json
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from openai import OpenAI
from utils.helpers import (
    get_openai_key,
    print_header,
    print_step,
    print_result,
    print_tree_rich,
    save_json,
    load_json,
    create_node_mapping,
    console,
    PAGEINDEX_SRC,
)
from utils.downloader import download_sample

sys.path.insert(0, PAGEINDEX_SRC)
from pageindex import page_index_main, config
from pageindex.utils import remove_fields


def call_llm(prompt: str, model: str = "gpt-4o", api_key: str = None) -> str:
    """Simple synchronous LLM call."""
    client = OpenAI(api_key=api_key or get_openai_key())
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def generate_or_load_tree(pdf_path: str, cache_path: str) -> dict:
    """Generate tree index or load from cache if available."""
    if os.path.exists(cache_path):
        console.print(f"[dim]Loading cached tree from {cache_path}[/dim]")
        return load_json(cache_path)

    console.print("[cyan]Generating tree index (this takes 1-3 min)...[/cyan]")
    opt = config(
        model="gpt-4o-2024-11-20",
        toc_check_page_num=20,
        max_page_num_each_node=10,
        max_token_num_each_node=20000,
        if_add_node_id="yes",
        if_add_node_summary="yes",
        if_add_doc_description="no",
        if_add_node_text="yes",
    )
    result = page_index_main(pdf_path, opt)
    save_json(result, cache_path)
    return result


def tree_search(query: str, structure: list, model: str = "gpt-4o") -> dict:
    """
    REASONING-BASED TREE SEARCH — the core of PageIndex RAG.

    Instead of embedding the query and computing cosine similarity against chunks,
    we show the LLM the tree structure (with summaries, no text) and ask it to
    REASON about which sections would contain the answer.
    """
    # Remove text fields to keep the prompt compact — summaries are enough for reasoning
    tree_compact = remove_fields(structure, fields=["text"])

    search_prompt = f"""You are given a question and a tree-structured index of a document.
Each node has a node_id, title, and summary describing its contents.
Your task is to identify ALL nodes whose content is needed to answer the question.

Think carefully about:
- Which sections would logically contain this information
- Whether the answer might span multiple sections
- Whether background/context sections are needed too

Question: {query}

Document tree structure:
{json.dumps(tree_compact, indent=2)}

Reply in this exact JSON format (no other text):
{{
    "thinking": "<Your step-by-step reasoning about which sections are relevant>",
    "node_list": ["node_id_1", "node_id_2"]
}}"""

    result = call_llm(search_prompt, model=model)

    # Parse JSON from LLM response
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", result, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"thinking": "Failed to parse", "node_list": []}


def extract_context(node_ids: list, node_map: dict) -> str:
    """Extract full text from selected nodes to build context."""
    sections = []
    for nid in node_ids:
        if nid in node_map:
            node = node_map[nid]
            text = node.get("text", "")
            if text:
                title = node.get("title", "Untitled")
                page = node.get("start_index", "?")
                sections.append(f"--- {title} (Page {page}) ---\n{text}")
    return "\n\n".join(sections)


def generate_answer(query: str, context: str, model: str = "gpt-4o") -> str:
    """Generate final answer from retrieved context."""
    prompt = f"""Answer the following question based ONLY on the provided context.
Include specific details from the source material.
Cite section titles and page numbers where you found the information.

Question: {query}

Context:
{context}

Answer:"""
    return call_llm(prompt, model=model)


def main():
    print_header("Example 03: Complete Vectorless RAG Pipeline (Self-Hosted)")

    # ─── Step 1: Prepare document ───
    print_step(1, "Download and prepare document")
    pdf_path = download_sample("attention_paper")

    # ─── Step 2: Generate or load tree index ───
    print_step(2, "Generate tree index (or load from cache)")
    cache_path = "results/01_attention_paper_tree.json"
    result = generate_or_load_tree(pdf_path, cache_path)
    structure = result["structure"]

    print_tree_rich(structure, title="Document Tree (used for reasoning)")

    # Build node lookup map
    node_map = create_node_mapping(structure)
    console.print(f"[bold]Total nodes available:[/bold] {len(node_map)}")

    # ─── Step 3: Ask questions! ───
    queries = [
        "How does multi-head attention work and why is it better than single attention?",
        "What training data and hardware were used? How long did training take?",
        "What are the BLEU scores achieved and how do they compare to previous work?",
    ]

    for i, query in enumerate(queries, 1):
        print_step(2 + i, f"Query {i}: {query}")

        # ── PHASE 1: Reasoning-based tree search ──
        console.print("[cyan]Phase 1: LLM reasoning over tree structure...[/cyan]")
        search_result = tree_search(query, structure)

        console.print(f"[bold]Reasoning:[/bold] {search_result.get('thinking', 'N/A')}")
        console.print(f"[bold]Selected nodes:[/bold] {search_result['node_list']}")

        for nid in search_result["node_list"]:
            if nid in node_map:
                node = node_map[nid]
                console.print(
                    f"  [green]✓[/green] {node.get('title', '?')} "
                    f"(pages {node.get('start_index', '?')}-{node.get('end_index', '?')})"
                )

        # ── PHASE 2: Extract context ──
        console.print("[cyan]Phase 2: Extracting full text from selected nodes...[/cyan]")
        context = extract_context(search_result["node_list"], node_map)
        console.print(f"[dim]Context size: {len(context)} characters[/dim]")

        # ── PHASE 3: Generate answer ──
        console.print("[cyan]Phase 3: Generating answer...[/cyan]")
        answer = generate_answer(query, context)
        print_result(answer)

    # ─── Summary ───
    console.print("\n[bold green]Key takeaways:[/bold green]")
    console.print("  1. No vector database, no embeddings, no chunking")
    console.print("  2. LLM reasons over document structure to find relevant sections")
    console.print("  3. Every retrieval is explainable (you can see the reasoning)")
    console.print("  4. Sections stay logically coherent (no chunking artifacts)")
    console.print("  5. The tree index is just a JSON file — no infrastructure needed\n")


if __name__ == "__main__":
    main()