"""
PageIndex also works with Markdown documents. This is useful for:
- Documentation sites and wikis
- Notes and knowledge bases
- Any structured text content

How it works (different from PDF):
1. Parses # headers to determine hierarchy (no LLM needed for structure!)
2. Assigns text between headers to each node
3. Optional thinning: merges very small sections into parents
4. LLM generates summaries for each node (optional but recommended)
"""

import sys
import os
import asyncio

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
from utils.downloader import create_sample_markdown

sys.path.insert(0, PAGEINDEX_SRC)
from pageindex.page_index_md import md_to_tree
from pageindex.utils import structure_to_list


async def generate_tree(md_path: str, with_thinning: bool = False):
    """Generate a tree from a markdown file."""
    tree = await md_to_tree(
        md_path=md_path,
        if_thinning=with_thinning,
        min_token_threshold=5000,
        if_add_node_summary="yes",
        summary_token_threshold=200,
        model="gpt-4o-2024-11-20",
        if_add_doc_description="no",
        if_add_node_text="yes",
        if_add_node_id="yes",
    )
    return tree


def main():
    print_header("Example 02: Tree Generation from Markdown (Self-Hosted)")

    get_openai_key()
    console.print("[green]OpenAI API key found.[/green]")

    # ─── Step 1: Create sample markdown ───
    print_step(1, "Create sample markdown document")
    md_path = create_sample_markdown()

    # ─── Step 2: Generate tree WITHOUT thinning ───
    print_step(2, "Generate tree from markdown (no thinning)")
    console.print("[cyan]Processing... LLM generates summaries for each section[/cyan]")

    result = asyncio.run(generate_tree(md_path, with_thinning=False))
    structure = result.get("structure", [])

    all_nodes = structure_to_list(structure)
    console.print(f"[bold]Total nodes (no thinning):[/bold] {len(all_nodes)}")
    print_tree_rich(structure, title="ML Pipelines Doc — Full Tree")

    save_json(result, "results/02_markdown_tree.json")

    # ─── Step 3: Generate tree WITH thinning ───
    print_step(3, "Generate tree WITH thinning (merges small sections)")
    console.print("[dim]Thinning merges nodes below the token threshold into parents[/dim]")

    result_thin = asyncio.run(generate_tree(md_path, with_thinning=True))
    structure_thin = result_thin.get("structure", [])

    all_nodes_thin = structure_to_list(structure_thin)
    console.print(f"[bold]Total nodes (with thinning):[/bold] {len(all_nodes_thin)}")
    print_tree_rich(structure_thin, title="ML Pipelines Doc — Thinned Tree")

    save_json(result_thin, "results/02_markdown_tree_thinned.json")

    # ─── Step 4: Compare ───
    print_step(4, "Compare: thinning reduces tree size")
    console.print(f"  Without thinning: {len(all_nodes)} nodes")
    console.print(f"  With thinning:    {len(all_nodes_thin)} nodes")
    console.print(
        "  Thinning is useful when you have many small sections that"
        " would waste LLM reasoning steps during retrieval.\n"
    )

    # ─── Step 5: Key differences from PDF ───
    print_step(5, "Key differences: Markdown vs PDF tree generation")
    console.print("  • Markdown: Structure comes from # headers (no LLM needed for structure)")
    console.print("  • PDF: LLM must analyze pages to discover/verify structure")
    console.print("  • Markdown: Uses line_num instead of page numbers")
    console.print("  • PDF: Uses start_index/end_index (physical page numbers)")
    console.print("  • Markdown: Much faster and cheaper (fewer LLM calls)")
    console.print("  • Both: LLM generates summaries for each node\n")

    return result


if __name__ == "__main__":
    main()