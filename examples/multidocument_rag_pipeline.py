"""

Query across MULTIPLE documents simultaneously. The LLM first decides
which documents are relevant, then searches within those documents.

Use case: You have a collection of papers/reports and want to ask
questions that may require information from several of them.

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


def call_llm(prompt: str, model: str = "gpt-4o") -> str:
    client = OpenAI(api_key=get_openai_key())
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def generate_or_load_tree(pdf_path: str, cache_path: str) -> dict:
    if os.path.exists(cache_path):
        console.print(f"[dim]Loaded cached: {cache_path}[/dim]")
        return load_json(cache_path)

    console.print(f"[cyan]Generating tree for {os.path.basename(pdf_path)}...[/cyan]")
    opt = config(
        model="gpt-4o-2024-11-20",
        toc_check_page_num=20,
        max_page_num_each_node=10,
        max_token_num_each_node=20000,
        if_add_node_id="yes",
        if_add_node_summary="yes",
        if_add_doc_description="yes",  # Important for multi-doc: helps LLM pick relevant docs
        if_add_node_text="yes",
    )
    result = page_index_main(pdf_path, opt)
    save_json(result, cache_path)
    return result


def main():
    print_header("Example 04: Multi-Document RAG (Self-Hosted)")

    # ─── Step 1: Build document registry ───
    print_step(1, "Download and index multiple documents")

    docs = {
        "attention": {
            "sample": "attention_paper",
            "cache": "results/04_attention_tree.json",
        },
        "rag": {
            "sample": "rag_paper",
            "cache": "results/04_rag_tree.json",
        },
    }

    registry = {}
    for doc_key, doc_info in docs.items():
        pdf_path = download_sample(doc_info["sample"])
        tree = generate_or_load_tree(pdf_path, doc_info["cache"])
        registry[doc_key] = tree
        console.print(
            f"  [green]✓[/green] {tree.get('doc_name', doc_key)} — "
            f"{len(tree.get('structure', []))} top-level sections"
        )

    # ─── Step 2: Build document-level index ───
    print_step(2, "Build document-level index for first-pass selection")

    doc_index = []
    for doc_key, tree in registry.items():
        structure = tree.get("structure", [])
        # Get top-level section titles for overview
        section_titles = [s.get("title", "") for s in structure]
        doc_index.append({
            "doc_key": doc_key,
            "doc_name": tree.get("doc_name", doc_key),
            "doc_description": tree.get("doc_description", ""),
            "top_sections": section_titles[:10],
        })

    console.print(f"[bold]Document registry:[/bold] {len(doc_index)} documents")
    for d in doc_index:
        console.print(f"  • {d['doc_name']}: {d.get('doc_description', 'N/A')[:100]}")

    # ─── Step 3: Two-phase retrieval ───
    print_step(3, "Two-phase retrieval: document selection → node search")

    query = "How do attention mechanisms relate to retrieval-augmented generation? What are the key innovations in both areas?"

    console.print(f"\n[bold]Query:[/bold] {query}")

    # Phase 1: Select relevant documents
    console.print("\n[cyan]Phase 1: Selecting relevant documents...[/cyan]")
    doc_select_prompt = f"""You are given a question and a registry of documents.
Select which documents are needed to answer the question.

Question: {query}

Document registry:
{json.dumps(doc_index, indent=2)}

Reply in JSON format:
{{
    "thinking": "<reasoning about which documents are relevant>",
    "selected_docs": ["doc_key_1", "doc_key_2"]
}}"""

    raw_doc_selection = call_llm(doc_select_prompt)
    try:
        doc_selection = json.loads(raw_doc_selection)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_doc_selection, re.DOTALL)
        if match:
            doc_selection = json.loads(match.group())
        else:
            doc_selection = {
                "thinking": "Fallback: parse failed, selected all documents.",
                "selected_docs": list(registry.keys()),
            }
    console.print(f"[bold]Selected documents:[/bold] {doc_selection['selected_docs']}")
    console.print(f"[bold]Reasoning:[/bold] {doc_selection['thinking']}")

    # Phase 2: Search within selected documents
    console.print("\n[cyan]Phase 2: Searching within selected documents...[/cyan]")

    all_context_parts = []
    for doc_key in doc_selection["selected_docs"]:
        if doc_key not in registry:
            continue

        tree = registry[doc_key]
        structure = tree["structure"]
        tree_compact = remove_fields(structure, fields=["text"])

        search_prompt = f"""You are given a question and a tree-structured index of a document called "{tree.get('doc_name', doc_key)}".
Find all nodes needed to answer the question.

Question: {query}

Document tree:
{json.dumps(tree_compact, indent=2)}

Reply in JSON:
{{
    "thinking": "<reasoning>",
    "node_list": ["node_id_1", "node_id_2"]
}}"""

        try:
            result = json.loads(call_llm(search_prompt))
        except json.JSONDecodeError:
            result = {"node_list": []}

        node_map = create_node_mapping(structure)
        for nid in result.get("node_list", []):
            if nid in node_map:
                node = node_map[nid]
                text = node.get("text", "")
                if text:
                    all_context_parts.append(
                        f"[From: {tree.get('doc_name', doc_key)}]\n"
                        f"--- {node.get('title', '?')} (Page {node.get('start_index', '?')}) ---\n{text}"
                    )
                    console.print(
                        f"  [green]✓[/green] [{doc_key}] {node.get('title', '?')}"
                    )

    # Phase 3: Synthesize answer from all documents
    console.print("\n[cyan]Phase 3: Synthesizing cross-document answer...[/cyan]")

    context = "\n\n".join(all_context_parts)
    answer_prompt = f"""Answer the question using context from MULTIPLE documents.
Clearly indicate which document each piece of information comes from.

Question: {query}

Context from multiple documents:
{context}

Answer:"""

    answer = call_llm(answer_prompt)
    print_result(answer)

    console.print("\n[bold green]Key takeaways:[/bold green]")
    console.print("  1. Two-phase retrieval: pick docs first, then search within them")
    console.print("  2. Doc descriptions help the LLM quickly identify relevant documents")
    console.print("  3. Cross-document synthesis produces richer, more complete answers")
    console.print("  4. Each tree is independent — add/remove docs without re-indexing\n")


if __name__ == "__main__":
    main()