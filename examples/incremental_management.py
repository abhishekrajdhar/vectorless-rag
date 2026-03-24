"""
Incremental Document Management (Self-Hosted)
==========================================================

Real-world RAG systems need to handle documents that are added, updated,
or removed over time. Since PageIndex trees are just JSON files, this is
straightforward — no vector DB re-indexing needed.

"""

import sys
import os
import json
import time
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
    load_json,
    create_node_mapping,
    ensure_dir,
    register_artifact,
    console,
    PAGEINDEX_SRC,
)

sys.path.insert(0, PAGEINDEX_SRC)
from pageindex.page_index_md import md_to_tree
from pageindex.utils import remove_fields


REGISTRY_PATH = "results/07_document_registry.json"
TREES_DIR = "results/07_trees"


def call_llm(prompt: str, model: str = "gpt-4o") -> str:
    client = OpenAI(api_key=get_openai_key())
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


class DocumentRegistry:
    """Simple file-based document registry for managing multiple tree indexes."""

    def __init__(self, registry_path: str = REGISTRY_PATH, trees_dir: str = TREES_DIR):
        self.registry_path = registry_path
        self.trees_dir = trees_dir
        ensure_dir(trees_dir)
        self.registry = self._load_registry()

    def _load_registry(self) -> dict:
        if os.path.exists(self.registry_path):
            return load_json(self.registry_path)
        return {"documents": {}, "version": 1}

    def _save_registry(self):
        save_json(self.registry, self.registry_path)

    def add_document(self, doc_id: str, source_path: str, tree_result: dict):
        """Add a document to the registry."""
        tree_path = os.path.join(self.trees_dir, f"{doc_id}.json")
        save_json(tree_result, tree_path)

        self.registry["documents"][doc_id] = {
            "source_path": source_path,
            "tree_path": tree_path,
            "doc_name": tree_result.get("doc_name", doc_id),
            "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "node_count": len(self._count_nodes(tree_result.get("structure", []))),
        }
        self._save_registry()
        console.print(f"[green]Added:[/green] {doc_id} ({self.registry['documents'][doc_id]['node_count']} nodes)")

    def remove_document(self, doc_id: str):
        """Remove a document from the registry."""
        if doc_id in self.registry["documents"]:
            tree_path = self.registry["documents"][doc_id]["tree_path"]
            if os.path.exists(tree_path):
                os.remove(tree_path)
            del self.registry["documents"][doc_id]
            self._save_registry()
            console.print(f"[red]Removed:[/red] {doc_id}")
        else:
            console.print(f"[yellow]Not found:[/yellow] {doc_id}")

    def update_document(self, doc_id: str, source_path: str, tree_result: dict):
        """Update an existing document (re-index)."""
        self.remove_document(doc_id)
        self.add_document(doc_id, source_path, tree_result)
        console.print(f"[cyan]Updated:[/cyan] {doc_id}")

    def list_documents(self) -> dict:
        return self.registry["documents"]

    def get_tree(self, doc_id: str) -> dict:
        if doc_id not in self.registry["documents"]:
            raise KeyError(f"Document '{doc_id}' not in registry")
        return load_json(self.registry["documents"][doc_id]["tree_path"])

    def get_all_trees(self) -> dict[str, dict]:
        return {doc_id: self.get_tree(doc_id) for doc_id in self.registry["documents"]}

    def _count_nodes(self, structure) -> list:
        nodes = []
        if isinstance(structure, list):
            for s in structure:
                nodes.extend(self._count_nodes(s))
        elif isinstance(structure, dict):
            nodes.append(structure)
            for child in structure.get("nodes", []):
                nodes.extend(self._count_nodes(child))
        return nodes


def create_doc(filename: str, content: str) -> str:
    """Create a markdown document."""
    path = os.path.join("data", "sample_docs", filename)
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        f.write(content)
    register_artifact(path, kind="markdown", note="Registry source document")
    return path


def index_markdown(path: str) -> dict:
    """Generate tree from markdown."""
    return asyncio.run(md_to_tree(
        md_path=path,
        if_thinning=False,
        if_add_node_summary="yes",
        summary_token_threshold=200,
        model="gpt-4o-2024-11-20",
        if_add_node_text="yes",
        if_add_node_id="yes",
    ))


def query_registry(registry: DocumentRegistry, query: str):
    """Run a RAG query across all documents in the registry."""
    all_trees = registry.get_all_trees()

    if not all_trees:
        console.print("[yellow]No documents in registry.[/yellow]")
        return

    # Build document index
    doc_summaries = []
    for doc_id, tree in all_trees.items():
        structure = tree.get("structure", [])
        titles = []
        if isinstance(structure, list):
            titles = [s.get("title", "") for s in structure[:5]]
        doc_summaries.append({
            "doc_id": doc_id,
            "doc_name": tree.get("doc_name", doc_id),
            "sections": titles,
        })

    # Select docs
    select_prompt = f"""Select documents relevant to the question.
Question: {query}
Documents: {json.dumps(doc_summaries, indent=2)}
Reply in JSON: {{"selected": ["doc_id_1"]}}"""

    try:
        selected = json.loads(call_llm(select_prompt)).get("selected", [])
    except json.JSONDecodeError:
        selected = list(all_trees.keys())

    # Search within selected docs
    all_context = []
    for doc_id in selected:
        if doc_id not in all_trees:
            continue
        tree = all_trees[doc_id]
        structure = tree.get("structure", [])
        tree_compact = remove_fields(structure, fields=["text"])

        search_prompt = f"""Find relevant nodes.
Question: {query}
Tree: {json.dumps(tree_compact, indent=2)}
Reply in JSON: {{"node_list": ["id1"]}}"""

        raw = call_llm(search_prompt)
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            result = json.loads(match.group()) if match else {"node_list": []}

        node_map = create_node_mapping(structure)
        for nid in result.get("node_list", []):
            if nid in node_map and node_map[nid].get("text"):
                all_context.append(
                    f"[{tree.get('doc_name', doc_id)}] {node_map[nid].get('title', '')}\n"
                    f"{node_map[nid]['text']}"
                )

    if all_context:
        answer = call_llm(
            f"Answer based on context:\nQuestion: {query}\nContext:\n" + "\n\n".join(all_context)
        )
        print_result(answer)
    else:
        console.print("[yellow]No relevant content found.[/yellow]")


def main():
    print_header("Example 07: Incremental Document Management")

    # ─── Step 1: Create a fresh registry ───
    print_step(1, "Initialize document registry")
    registry = DocumentRegistry()
    console.print(f"[bold]Current documents:[/bold] {len(registry.list_documents())}")

    # ─── Step 2: Add first document ───
    print_step(2, "Add first document: Python Best Practices")
    doc1_path = create_doc("python_best_practices.md", """# Python Best Practices

## Code Style

### PEP 8
Follow PEP 8 for consistent code style. Use 4 spaces for indentation, limit lines to 79 characters,
and use snake_case for functions and variables. Use tools like black and flake8 for automated formatting.

### Type Hints
Use type hints for function signatures. They improve code readability and enable static analysis
with tools like mypy. Example: def greet(name: str) -> str: return f"Hello {name}"

## Testing

### Unit Tests
Write unit tests using pytest. Aim for at least 80% code coverage. Test edge cases and error
conditions. Use parametrized tests for testing multiple inputs.

### Integration Tests
Test component interactions with integration tests. Use fixtures for setup/teardown.
Mock external services to keep tests fast and reliable.

## Performance

### Profiling
Use cProfile or py-spy for performance profiling. Identify bottlenecks before optimizing.
Premature optimization is the root of all evil.

### Async Programming
Use asyncio for I/O-bound tasks. Use concurrent.futures for CPU-bound parallel processing.
Consider using uvloop for better async performance.
""")

    tree1 = index_markdown(doc1_path)
    registry.add_document("python_practices", doc1_path, tree1)

    # ─── Step 3: Add second document ───
    print_step(3, "Add second document: API Design Guide")
    doc2_path = create_doc("api_design.md", """# API Design Guide

## REST Principles

### Resource Naming
Use nouns for resources: /users, /orders, /products. Use plural forms. Nest related resources:
/users/123/orders. Keep URLs lowercase and use hyphens for multi-word resources.

### HTTP Methods
- GET: Retrieve resources (idempotent)
- POST: Create new resources
- PUT: Full resource replacement
- PATCH: Partial resource update
- DELETE: Remove resources

### Status Codes
- 200 OK: Successful GET/PUT/PATCH
- 201 Created: Successful POST
- 204 No Content: Successful DELETE
- 400 Bad Request: Client error
- 401 Unauthorized: Authentication required
- 404 Not Found: Resource doesn't exist
- 500 Internal Server Error: Server-side error

## Authentication

### API Keys
Simple but limited. Pass in header: X-API-Key. Rotate keys regularly. Never expose in URLs.

### OAuth 2.0
Use OAuth 2.0 for user-delegated access. Support authorization code flow for web apps
and client credentials flow for server-to-server communication.

## Versioning

### URL Versioning
Include version in URL: /api/v1/users. Simple and explicit. Easy to deprecate old versions.

### Header Versioning
Use Accept header: Accept: application/vnd.api+json;version=1. Cleaner URLs but harder to test.
""")

    tree2 = index_markdown(doc2_path)
    registry.add_document("api_design", doc2_path, tree2)

    # ─── Step 4: List all documents ───
    print_step(4, "List all documents in registry")
    for doc_id, info in registry.list_documents().items():
        console.print(
            f"  • [bold]{doc_id}[/bold]: {info['doc_name']} "
            f"({info['node_count']} nodes, indexed {info['indexed_at']})"
        )

    # ─── Step 5: Query across all documents ───
    print_step(5, "Query across all documents")
    query_registry(registry, "What are the best practices for testing Python code and API authentication?")

    # ─── Step 6: Update a document ───
    print_step(6, "Update a document (add new section)")
    doc1_updated_path = create_doc("python_best_practices.md", """# Python Best Practices

## Code Style

### PEP 8
Follow PEP 8 for consistent code style. Use 4 spaces for indentation, limit lines to 79 characters,
and use snake_case for functions and variables.

### Type Hints
Use type hints for function signatures. They improve code readability and enable static analysis with mypy.

## Testing

### Unit Tests
Write unit tests using pytest. Aim for at least 80% code coverage.

### Integration Tests
Test component interactions. Use fixtures for setup/teardown.

## Performance

### Profiling
Use cProfile or py-spy for performance profiling.

### Async Programming
Use asyncio for I/O-bound tasks. Use concurrent.futures for CPU-bound parallel processing.

## NEW: Dependency Management

### Poetry vs pip
Use Poetry for modern dependency management. It handles virtual environments, dependency resolution,
and lockfiles automatically. Alternative: pip + pip-tools for simpler projects.

### Virtual Environments
Always use virtual environments. Never install packages globally.
Use python -m venv .venv or Poetry's built-in venv management.
""")

    tree1_updated = index_markdown(doc1_updated_path)
    registry.update_document("python_practices", doc1_updated_path, tree1_updated)

    # ─── Step 7: Remove a document ───
    print_step(7, "Remove API design document from registry")
    registry.remove_document("api_design")

    console.print(f"\n[bold]Final registry:[/bold] {len(registry.list_documents())} document(s)")
    for doc_id, info in registry.list_documents().items():
        console.print(f"  • {doc_id}: {info['doc_name']} ({info['node_count']} nodes)")

    console.print("\n[bold green]Key takeaways:[/bold green]")
    console.print("  1. Trees are just JSON files — add/remove/update is trivial")
    console.print("  2. No vector DB re-indexing — just regenerate the tree for changed docs")
    console.print("  3. Registry pattern keeps track of all indexed documents")
    console.print("  4. Only re-index what changed (incremental updates)")
    console.print("  5. Each tree is independent — no global index to rebuild\n")


if __name__ == "__main__":
    main()