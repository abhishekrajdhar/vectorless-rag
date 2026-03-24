"""
Different Document Types & Use Cases (Self-Hosted)
===============================================================

PageIndex works best on well-structured documents. This example
demonstrates RAG on different types of documents:
1. Academic paper (arXiv) — has abstract, sections, references
2. Markdown documentation — has clear heading hierarchy
3. Custom document — create your own structured content

Each type shows different tree structures and retrieval patterns.
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
    print_tree_rich,
    save_json,
    load_json,
    create_node_mapping,
    ensure_dir,
    register_artifact,
    console,
    PAGEINDEX_SRC,
)
from utils.downloader import download_sample, create_sample_markdown

sys.path.insert(0, PAGEINDEX_SRC)
from pageindex import page_index_main, config
from pageindex.page_index_md import md_to_tree
from pageindex.utils import remove_fields, structure_to_list


def call_llm(prompt: str, model: str = "gpt-4o") -> str:
    client = OpenAI(api_key=get_openai_key())
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def do_rag_query(query: str, structure, doc_name: str):
    """Run a complete RAG query against a tree structure."""
    console.print(f"\n[bold]Query:[/bold] {query}")

    # Tree search
    tree_compact = remove_fields(
        structure if isinstance(structure, list) else [structure],
        fields=["text"],
    )
    search_prompt = f"""Find nodes in this document tree that answer the question.

Question: {query}

Document tree:
{json.dumps(tree_compact, indent=2)}

Reply in JSON:
{{"thinking": "<reasoning>", "node_list": ["node_id_1"]}}"""

    raw = call_llm(search_prompt)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        result = json.loads(match.group()) if match else {"node_list": []}

    node_map = create_node_mapping(structure)
    context_parts = []
    for nid in result.get("node_list", []):
        if nid in node_map:
            node = node_map[nid]
            text = node.get("text", "")
            if text:
                context_parts.append(f"--- {node.get('title', '?')} ---\n{text}")

    if not context_parts:
        console.print("[yellow]No relevant nodes found.[/yellow]")
        return

    context = "\n\n".join(context_parts)
    answer = call_llm(
        f"Answer based on context from '{doc_name}'.\nQuestion: {query}\nContext:\n{context}\nAnswer:"
    )
    print_result(answer)


def main():
    print_header("Example 06: Different Document Types & Use Cases")

    # ════════════════════════════════════════════════════
    # Type 1: Academic Paper (PDF)
    # ════════════════════════════════════════════════════
    print_step(1, "Type 1: Academic Paper (PDF from arXiv)")
    console.print("[dim]Academic papers typically have: abstract, intro, methods, results, references[/dim]")

    pdf_path = download_sample("rag_paper")
    cache_path = "results/06_rag_paper_tree.json"

    if os.path.exists(cache_path):
        pdf_result = load_json(cache_path)
    else:
        console.print("[cyan]Generating tree for RAG paper...[/cyan]")
        opt = config(
            model="gpt-4o-2024-11-20",
            toc_check_page_num=20,
            max_page_num_each_node=10,
            max_token_num_each_node=20000,
            if_add_node_id="yes",
            if_add_node_summary="yes",
            if_add_node_text="yes",
            if_add_doc_description="yes",
        )
        pdf_result = page_index_main(pdf_path, opt)
        save_json(pdf_result, cache_path)

    pdf_structure = pdf_result["structure"]
    console.print(f"[bold]Academic paper tree:[/bold] {len(structure_to_list(pdf_structure))} nodes")
    print_tree_rich(pdf_structure, title="RAG Paper — Academic Structure")

    do_rag_query(
        "What retrieval method does RAG use and how does it combine retrieval with generation?",
        pdf_structure,
        "RAG Paper",
    )

    # ════════════════════════════════════════════════════
    # Type 2: Technical Documentation (Markdown)
    # ════════════════════════════════════════════════════
    print_step(2, "Type 2: Technical Documentation (Markdown)")
    console.print("[dim]Markdown docs have explicit heading hierarchy — tree comes directly from headers[/dim]")

    md_path = create_sample_markdown()
    md_cache = "results/06_markdown_tree.json"

    if os.path.exists(md_cache):
        md_result = load_json(md_cache)
    else:
        console.print("[cyan]Generating tree for markdown doc...[/cyan]")
        md_result = asyncio.run(md_to_tree(
            md_path=md_path,
            if_thinning=False,
            if_add_node_summary="yes",
            summary_token_threshold=200,
            model="gpt-4o-2024-11-20",
            if_add_doc_description="no",
            if_add_node_text="yes",
            if_add_node_id="yes",
        ))
        save_json(md_result, md_cache)

    md_structure = md_result["structure"]
    console.print(f"[bold]Markdown doc tree:[/bold] {len(structure_to_list(md_structure))} nodes")
    print_tree_rich(md_structure, title="ML Pipelines — Markdown Structure")

    do_rag_query(
        "What are the best practices for hyperparameter tuning and model evaluation?",
        md_structure,
        "ML Pipelines Documentation",
    )

    # ════════════════════════════════════════════════════
    # Type 3: Custom FAQ / Knowledge Base (Markdown)
    # ════════════════════════════════════════════════════
    print_step(3, "Type 3: Custom FAQ Knowledge Base")

    faq_path = os.path.join("data", "sample_docs", "company_faq.md")
    ensure_dir(os.path.dirname(faq_path))

    if not os.path.exists(faq_path):
        faq_content = """# Acme Corp FAQ & Knowledge Base

## Products

### Widget Pro
Widget Pro is our flagship product for enterprise customers. It supports up to 10,000
concurrent users and includes advanced analytics, custom dashboards, and SSO integration.
Pricing starts at $500/month for up to 100 users, with volume discounts available.

### Widget Lite
Widget Lite is designed for small teams (up to 25 users). It includes core features
like task management, basic reporting, and email integration. Pricing is $49/month flat.

### Widget API
Our REST API allows programmatic access to all Widget features. Rate limits are
1000 requests/minute for Pro and 100 requests/minute for Lite. API documentation
is available at docs.acme.com/api.

## Billing & Pricing

### Payment Methods
We accept all major credit cards, ACH transfers, and wire transfers for annual plans.
Invoicing is available for Enterprise customers on annual contracts.

### Refund Policy
We offer a 30-day money-back guarantee for new subscriptions. Refunds for annual
plans are prorated based on remaining months. Contact billing@acme.com for refunds.

### Upgrading Plans
You can upgrade from Lite to Pro at any time. Your billing will be prorated for the
remainder of the current billing cycle. Downgrades take effect at the next billing cycle.

## Technical Support

### Getting Help
- Email: support@acme.com (24h response time)
- Chat: Available Mon-Fri 9am-5pm EST via our website
- Phone: 1-800-ACME (Enterprise customers only)
- Community forum: community.acme.com

### System Requirements
Widget Pro requires: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+.
Mobile apps available for iOS 15+ and Android 12+.
API requires TLS 1.2 or higher.

### Data & Security
All data is encrypted at rest (AES-256) and in transit (TLS 1.3).
We are SOC 2 Type II certified and GDPR compliant.
Data is stored in AWS us-east-1 with daily backups retained for 30 days.
Enterprise customers can request custom data residency regions.

## Company

### About Us
Acme Corp was founded in 2019. We have 150 employees across offices in
San Francisco, London, and Singapore. We serve over 5,000 customers globally.

### Careers
We're always hiring! Check acme.com/careers for open positions.
We offer competitive salary, equity, unlimited PTO, and remote-first culture.
"""
        with open(faq_path, "w") as f:
            f.write(faq_content)
        absolute = register_artifact(faq_path, kind="markdown", note="Generated FAQ knowledge base")
        console.print(f"[green]Created FAQ document:[/green] {absolute}")
    else:
        register_artifact(faq_path, kind="markdown", note="Existing FAQ knowledge base")

    faq_cache = "results/06_faq_tree.json"
    if os.path.exists(faq_cache):
        faq_result = load_json(faq_cache)
    else:
        console.print("[cyan]Generating tree for FAQ...[/cyan]")
        faq_result = asyncio.run(md_to_tree(
            md_path=faq_path,
            if_thinning=False,
            if_add_node_summary="yes",
            summary_token_threshold=200,
            model="gpt-4o-2024-11-20",
            if_add_doc_description="no",
            if_add_node_text="yes",
            if_add_node_id="yes",
        ))
        save_json(faq_result, faq_cache)

    faq_structure = faq_result["structure"]
    console.print(f"[bold]FAQ tree:[/bold] {len(structure_to_list(faq_structure))} nodes")
    print_tree_rich(faq_structure, title="Company FAQ — Knowledge Base")

    do_rag_query(
        "What's the pricing difference between Pro and Lite, and what security certifications do you have?",
        faq_structure,
        "Acme Corp FAQ",
    )

    # ─── Summary ───
    console.print("\n[bold green]Document type comparison:[/bold green]")
    console.print("  • Academic PDF:  LLM discovers structure → deep hierarchical trees")
    console.print("  • Markdown docs: Headers define structure → exact hierarchy, fast indexing")
    console.print("  • FAQ/KB:        Flat sections → shallow trees, precise retrieval")
    console.print("\n[bold]Best suited for PageIndex:[/bold]")
    console.print("  ✓ Long, well-structured documents (reports, papers, manuals)")
    console.print("  ✓ Documents with table of contents")
    console.print("  ✓ Documents where section boundaries carry meaning")
    console.print("  ✗ Short snippets or unstructured text blobs\n")


if __name__ == "__main__":
    main()