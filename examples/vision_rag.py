"""
Vision RAG — OCR-Free Document Q&A (Self-Hosted)
=============================================================

Instead of extracting text from PDFs, Vision RAG sends PAGE IMAGES
directly to a vision-capable LLM (GPT-4o). This is powerful for:
- Documents with complex layouts, tables, figures
- Scanned PDFs where text extraction is poor
- Documents where visual context matters

Pipeline:
1. Generate tree index (same as before, text-based)
2. LLM reasons over tree to find relevant sections → page numbers
3. Render those pages as IMAGES
4. Send images + query to vision LLM for answer

Requirements: GPT-4o or another vision-capable model
"""

import sys
import os
import json
import base64
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import fitz  # PyMuPDF
from openai import OpenAI
from utils.helpers import (
    get_openai_key,
    print_header,
    print_step,
    print_result,
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


def call_vision_llm(query: str, images_b64: list[str], model: str = "gpt-4o") -> str:
    """Send query + page images to a vision-capable LLM."""
    client = OpenAI(api_key=get_openai_key())

    content = [{"type": "text", "text": query}]
    for i, img_b64 in enumerate(images_b64):
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_b64}",
                "detail": "high",
            },
        })

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        max_tokens=2000,
    )
    return response.choices[0].message.content.strip()


def pdf_pages_to_images(pdf_path: str, page_numbers: list[int], dpi: int = 200) -> list[str]:
    """Render specific PDF pages as base64-encoded PNG images."""
    doc = fitz.open(pdf_path)
    images = []
    for page_num in page_numbers:
        # page_num is 1-indexed, fitz is 0-indexed
        if 1 <= page_num <= len(doc):
            page = doc[page_num - 1]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            images.append(img_b64)
            console.print(f"  [green]✓[/green] Rendered page {page_num} ({len(img_bytes) // 1024} KB)")
    doc.close()
    return images


def generate_or_load_tree(pdf_path: str, cache_path: str) -> dict:
    if os.path.exists(cache_path):
        console.print(f"[dim]Loaded cached: {cache_path}[/dim]")
        return load_json(cache_path)

    console.print(f"[cyan]Generating tree...[/cyan]")
    opt = config(
        model="gpt-4o-2024-11-20",
        toc_check_page_num=20,
        max_page_num_each_node=10,
        max_token_num_each_node=20000,
        if_add_node_id="yes",
        if_add_node_summary="yes",
        if_add_doc_description="no",
        if_add_node_text="no",  # Not needed for vision RAG — we use images instead!
    )
    result = page_index_main(pdf_path, opt)
    save_json(result, cache_path)
    return result


def main():
    print_header("Example 05: Vision RAG — OCR-Free Document Q&A")

    # ─── Step 1: Prepare ───
    print_step(1, "Download document")
    pdf_path = download_sample("attention_paper")

    # ─── Step 2: Generate tree (text not needed, just structure + summaries) ───
    print_step(2, "Generate tree index (structure only, no text extraction)")
    cache_path = "results/05_vision_tree.json"
    result = generate_or_load_tree(pdf_path, cache_path)
    structure = result["structure"]

    node_map = create_node_mapping(structure)
    console.print(f"[bold]Tree nodes:[/bold] {len(node_map)}")

    # ─── Step 3: Tree search to find relevant sections ───
    print_step(3, "Reasoning-based tree search")
    query = "What does the architecture diagram of the Transformer look like? Describe the model architecture in detail."
    console.print(f"[bold]Query:[/bold] {query}")

    tree_compact = remove_fields(structure, fields=["text"])
    search_prompt = f"""You are given a question and a tree-structured index of a document.
Find all nodes whose pages would contain the answer (including figures/diagrams).

Question: {query}

Document tree:
{json.dumps(tree_compact, indent=2)}

Reply in JSON:
{{
    "thinking": "<reasoning>",
    "node_list": ["node_id_1", "node_id_2"]
}}"""

    raw_search = call_llm(search_prompt)
    try:
        search_result = json.loads(raw_search)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_search, re.DOTALL)
        search_result = json.loads(match.group()) if match else {"thinking": "", "node_list": []}
    console.print(f"[bold]Selected nodes:[/bold] {search_result['node_list']}")

    # ─── Step 4: Get page numbers from selected nodes ───
    print_step(4, "Extract page numbers from selected nodes")
    page_numbers = set()
    for nid in search_result["node_list"]:
        if nid in node_map:
            node = node_map[nid]
            start = node.get("start_index", 0)
            end = node.get("end_index", start)
            for p in range(start, end + 1):
                page_numbers.add(p)

    page_numbers = sorted(page_numbers)
    console.print(f"[bold]Pages to render:[/bold] {page_numbers}")

    # ─── Step 5: Render pages as images ───
    print_step(5, "Render PDF pages as images")
    images = pdf_pages_to_images(pdf_path, page_numbers)
    console.print(f"[bold]Rendered {len(images)} page images[/bold]")

    # ─── Step 6: Send images + query to vision LLM ───
    print_step(6, "Send page images to vision LLM for answer")
    console.print("[cyan]GPT-4o is analyzing the page images...[/cyan]")

    vision_query = f"""I'm showing you {len(images)} pages from a research paper.
Answer this question based on what you see in these pages (including any figures, tables, or diagrams):

{query}

Provide a detailed answer referencing specific visual elements you can see."""

    answer = call_vision_llm(vision_query, images)
    print_result(answer)

    # ─── Summary ───
    console.print("\n[bold green]Vision RAG advantages:[/bold green]")
    console.print("  1. No OCR errors — the LLM sees the actual rendered pages")
    console.print("  2. Works with figures, tables, charts, diagrams")
    console.print("  3. Handles complex layouts that text extraction would mangle")
    console.print("  4. Tree search narrows down to just the relevant pages (saves tokens)")
    console.print("  5. Combine with text RAG for hybrid approach\n")
    console.print("[bold yellow]Trade-offs:[/bold yellow]")
    console.print("  • Higher token cost (images use more tokens than text)")
    console.print("  • Requires vision-capable model (GPT-4o, Claude, etc.)")
    console.print("  • Slower than pure text RAG\n")


if __name__ == "__main__":
    main()