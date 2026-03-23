"""
Example 01: Basic Tree Generation from PDF (Self-Hosted)
=========================================================

This is the foundational example. PageIndex transforms a PDF into a
hierarchical tree-structured index — replacing vector embeddings entirely.

What you'll learn:
- How to generate a tree index from a PDF locally using OpenAI
- How the tree structure maps to the document's logical sections
- How to inspect, visualize, and save the tree

How it works:
1. PageIndex reads the PDF and extracts text page by page
2. It scans the first N pages looking for a Table of Contents
3. If found, it maps TOC entries to physical page numbers
4. If not found, it asks the LLM to generate a hierarchical structure
5. It verifies page mappings and recursively splits large sections
6. Optionally generates summaries for each node

Cost: ~$0.05-0.50 depending on document size (OpenAI API calls)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.helpers import (
    get_openai_key,
    print_header,
    print_step,
    print_tree_rich,
    save_json,
    console,
    PAGEINDEX_SRC,
)
from utils.downloader import download_sample

# Import PageIndex self-hosted modules
sys.path.insert(0, PAGEINDEX_SRC)
from pageindex import page_index_main, config
from pageindex.utils import print_toc, structure_to_list


def main():
    print_header("Example 01: Basic Tree Generation from PDF (Self-Hosted)")

    # Verify OpenAI key is available
    get_openai_key()
    console.print("[green]OpenAI API key found.[/green]")

    # ─── Step 1: Download a sample PDF ───
    print_step(1, "Download sample PDF from arXiv")
    pdf_path = download_sample("attention_paper")

    # ─── Step 2: Configure tree generation options ───
    print_step(2, "Configure PageIndex options")
    opt = config(
        model="gpt-4o-2024-11-20",       # LLM model for tree generation
        toc_check_page_num=20,            # Pages to scan for TOC
        max_page_num_each_node=10,        # Max pages per tree node
        max_token_num_each_node=20000,    # Max tokens per node
        if_add_node_id="yes",             # Add unique IDs to each node
        if_add_node_summary="yes",        # Generate AI summaries per node
        if_add_doc_description="no",      # Skip doc-level description
        if_add_node_text="yes",           # Include raw text in nodes (needed for RAG!)
    )
    console.print("[dim]Options configured. Key setting: if_add_node_text=yes (required for RAG)[/dim]")

    # ─── Step 3: Generate the tree index ───
    print_step(3, "Generate tree index (this makes several OpenAI API calls)")
    console.print("[cyan]Processing... this may take 1-3 minutes depending on document size[/cyan]")

    result = page_index_main(pdf_path, opt)

    console.print("[green]Tree generation complete![/green]")

    # ─── Step 4: Explore the result structure ───
    print_step(4, "Explore the generated tree")

    # The result is a dict with: doc_name, structure (list of top-level nodes)
    console.print(f"\n[bold]Document:[/bold] {result.get('doc_name', 'N/A')}")
    structure = result.get("structure", [])
    console.print(f"[bold]Top-level sections:[/bold] {len(structure)}")

    # Count total nodes
    all_nodes = structure_to_list(structure)
    console.print(f"[bold]Total nodes:[/bold] {len(all_nodes)}")

    # Show a sample node's fields
    if all_nodes:
        sample = all_nodes[0]
        console.print(f"\n[bold]Sample node fields:[/bold]")
        for key in sample:
            if key == "nodes":
                console.print(f"  • nodes: {len(sample.get('nodes', []))} children")
            elif key == "text":
                console.print(f"  • text: {len(sample.get('text', ''))} chars")
            elif key == "summary":
                console.print(f"  • summary: {str(sample.get('summary', ''))[:100]}...")
            else:
                console.print(f"  • {key}: {sample[key]}")

    # ─── Step 5: Visualize the tree ───
    print_step(5, "Visualize the tree structure")
    print_tree_rich(structure, title="Attention Is All You Need — Document Tree")

    # Also show the simple text TOC
    console.print("\n[bold]Plain text TOC:[/bold]")
    print_toc(structure)

    # ─── Step 6: Save for reuse ───
    print_step(6, "Save tree to JSON (reuse in other examples)")
    output_path = "results/01_attention_paper_tree.json"
    save_json(result, output_path)

    console.print(f"\n[bold green]Done![/bold green]")
    console.print("This tree replaces vector embeddings + vector DB in traditional RAG.")
    console.print("Next: Run example 02 for markdown, or example 03 to do RAG queries.\n")

    return result


if __name__ == "__main__":
    main()