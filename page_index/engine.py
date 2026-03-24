from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .providers import LLMProvider, get_provider, MockProvider
from . import page_index_md, utils

logger = logging.getLogger(__name__)


class PageIndexEngine:
    """Engine implementing a vectorless RAG reasoning pipeline.

    Pipeline steps implemented:
      1) Document parsing (pages, sections)
      2) Symbolic index (topic -> pages)
      3) Query decomposition (intent, topics)
      4) Candidate selection (keyword + metadata)
      5) LLM-guided selection with reasoning
      6) Context construction (token-aware)
      7) Answer generation with source attribution

    The engine is deterministic by default (low temperature) and does not use embeddings.
    """

    def __init__(self, provider: Optional[LLMProvider] = None, model: Optional[str] = None):
        self.provider = provider or MockProvider()
        self.model = model or ""
        # Simple in-memory caches
        self._doc_cache: Dict[str, Any] = {}
        self._summary_cache: Dict[Tuple[str, int], str] = {}

    # ----------------------------- Document layer -----------------------------
    async def parse_markdown(self, md_path: str) -> Dict[str, Any]:
        """Deprecated name kept for backward compatibility. Use parse_document instead."""
        return await self.parse_document(md_path)

    async def parse_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Parse a PDF into true page-level structures.

        Uses utilities in `utils` (PyPDF2 or PyMuPDF) to extract per-page text.
        Returns: {doc_name, pages: [{page, title, sections, content}], structure: []}
        """
        # Use utils.get_pdf_name/get_pdf_title and get_page_tokens to extract pages
        try:
            # get_page_tokens returns list of (text, token_length)
            page_list = utils.get_page_tokens(pdf_path)
        except Exception as e:
            logger.exception("Failed to extract pages from PDF: %s", e)
            # fallback: attempt to read full text
            try:
                full = utils.extract_text_from_pdf(pdf_path)
                page_list = [(full, utils.count_tokens(full, model=self.model))]
            except Exception:
                page_list = []

        doc_name = utils.get_pdf_name(pdf_path)
        pages = []
        for i, (page_text, token_len) in enumerate(page_list, start=1):
            pages.append({
                "page": i,
                "title": f"Page {i}",
                "sections": [],
                "content": page_text or "",
            })

        doc = {"doc_name": doc_name, "pages": pages, "structure": []}
        self._doc_cache[pdf_path] = doc
        return doc

    async def parse_document(self, path: str) -> Dict[str, Any]:
        """Unified parser for markdown and PDF documents.

        Dispatches based on file extension. MD uses md_to_tree; PDF uses true page parsing.
        """
        if path.lower().endswith('.pdf'):
            return await self.parse_pdf(path)
        # assume markdown otherwise
        return await page_index_md.md_to_tree(md_path=path, if_thinning=False, min_token_threshold=None, if_add_node_summary='no', summary_token_threshold=200, model=self.model, if_add_doc_description='no', if_add_node_text='no', if_add_node_id='yes')

    def build_symbolic_index(self, doc: Dict[str, Any]) -> Dict[str, List[int]]:
        """Build a lightweight symbolic index: topic -> list of pages.

        This uses simple keyword extraction from titles and section names.
        """
        index = defaultdict(list)
        for page in doc.get('pages', []):
            page_no = page['page']
            # Keywords from title + sections
            words = []
            if page.get('title'):
                words.extend(re.findall(r"\w+", page['title'].lower()))
            for s in page.get('sections', []):
                words.extend(re.findall(r"\w+", (s or '').lower()))
            # dedupe words
            for w in set(words):
                index[w].append(page_no)
        return dict(index)

    # ----------------------------- Retrieval steps -----------------------------
    def decompose_query(self, query: str) -> Dict[str, Any]:
        """Lightweight query decomposition: extract keywords and (optionally) ask LLM

        Returns: {intent, topics, constraints}
        """
        # quick heuristic extraction
        topics = re.findall(r"\w+", query.lower())
        topics = [t for t in topics if len(t) > 2]

        # LLM-assisted decomposition (low temp, deterministic)
        template = {"intent": "...", "topics": [], "constraints": []}
        prompt = (
            "Extract intent, short topics (as a JSON list), and constraints from the user query.\n"
            f"Query:\n{query}\n"
            "Reply with JSON in the shape: " + json.dumps(template)
        )
        try:
            response = self.provider.generate(prompt, model=self.model)
            parsed = utils.extract_json(response)
            if isinstance(parsed, dict):
                # merge heuristic topics when LLM misses
                parsed.setdefault('topics', topics[:5])
                return parsed
        except Exception as e:
            logger.debug("LLM decomposition failed: %s", e)
        return {"intent": query, "topics": topics[:5], "constraints": []}

    def select_candidates(self, doc: Dict[str, Any], symbolic_index: Dict[str, List[int]], topics: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Candidate page selection using metadata + keyword matching + heuristic scoring.

        No embeddings used. Returns list of page dicts with simple score and match reason.
        """
        scores = defaultdict(float)
        reasons = defaultdict(list)
        for t in topics:
            pages = symbolic_index.get(t.lower(), [])
            for p in pages:
                scores[p] += 1.0
                reasons[p].append(f"keyword:{t}")

        # also try title substring matches
        for page in doc.get('pages', []):
            for t in topics:
                if t.lower() in (page.get('title') or '').lower():
                    p = page['page']
                    scores[p] += 1.5
                    reasons[p].append(f"title_match:{t}")

        # Collect and return top-k pages
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        result = []
        for p, sc in ranked[:top_k]:
            page_obj = next((pg for pg in doc['pages'] if pg['page'] == p), None)
            result.append({"page": p, "score": sc, "page_obj": page_obj, "reasons": reasons[p]})
        return result

    def llm_guided_selection(self, query: str, candidates: List[Dict[str, Any]], explain: bool = True) -> Dict[str, Any]:
        """Ask the LLM to pick best pages and justify.

        Input: shortlist of candidate pages (titles + short content)
        Output: selected_pages list and LLM reasoning
        """
        # Build assistant prompt with low temp reasoning instructions
        items_text = "\n\n".join([f"Page {c['page']}: {c['page_obj']['title']} -- reasons: {c['reasons']}\nSummary: { (c['page_obj']['content'] or '')[:400]}" for c in candidates])
        template = {"selected": [{"page": 0, "why": "..."}], "reasoning": "..."}
        prompt = (
            "You are given a user query and a list of candidate pages with titles and short summaries.\n"
            f"Query: {query}\nCandidates:\n{items_text}\n"
            "Task: Select the most relevant pages (1-3) and for each page output why it is relevant.\n"
            "Reply with JSON of the shape: " + json.dumps(template)
        )
        try:
            response = self.provider.generate(prompt, model=self.model)
            parsed = utils.extract_json(response)
            if isinstance(parsed, dict) and parsed.get('selected'):
                return {'selected': parsed.get('selected'), 'reasoning': parsed.get('reasoning', ''), 'raw': response}
        except Exception as e:
            logger.debug("LLM guided selection failed: %s", e)
        # Fallback: pick top 2 by score
        fallback = [{'page': c['page'], 'why': ';'.join(c.get('reasons', []))} for c in candidates[:2]]
        return {'selected': fallback, 'reasoning': 'fallback selection based on heuristic scores', 'raw': ''}

    def construct_context(self, doc: Dict[str, Any], selected_pages: List[int], token_limit: int = 2000) -> Tuple[str, List[Dict[str, Any]]]:
        """Assemble page contents for generation, with naive token-aware truncation."""
        parts = []
        sources = []
        total_tokens = 0
        for p in selected_pages:
            page_obj = next((pg for pg in doc['pages'] if pg['page'] == p), None)
            if not page_obj:
                continue
            text = page_obj.get('content', '')
            tokens = utils.count_tokens(text, model=self.model)
            # If adding this would exceed token_limit, truncate proportionally
            if total_tokens + tokens > token_limit:
                # simple truncation by characters
                remaining = max(0, token_limit - total_tokens)
                approx_chars = remaining * 4
                text = text[:approx_chars]
                tokens = utils.count_tokens(text, model=self.model)
            parts.append(f"<page:{p}>\nTitle: {page_obj.get('title')}\n{text}")
            sources.append({'page': p, 'title': page_obj.get('title')})
            total_tokens += tokens
            if total_tokens >= token_limit:
                break
        context = "\n\n".join(parts)
        return context, sources

    def generate_answer(self, query: str, context: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Final generation with enforced JSON output and source attribution."""
        template = {"answer": "...", "sources": [{"page": 1, "reason": "..."}], "reasoning": "step-by-step"}
        prompt = (
            "You are an assistant. Use the provided context (pages with titles and contents) to answer the user query.\n"
            f"Query: {query}\nContext:\n{context}\n\n"
            "Reply JSON with the shape: " + json.dumps(template) + "\n"
            "Be concise, do not hallucinate, and cite pages in sources."
        )
        response = self.provider.generate(prompt, model=self.model)
        parsed = utils.extract_json(response)
        if isinstance(parsed, dict) and parsed.get('answer'):
            # Ensure sources are present
            parsed.setdefault('sources', sources)
            return parsed
        # fallback
        return {"answer": response[:1000], "sources": sources, "reasoning": "fallback raw text"}

    # ----------------------------- High-level API -----------------------------
    async def answer(self, md_path: str, query: str, top_k: int = 5, token_limit: int = 2000) -> Dict[str, Any]:
        doc = self._doc_cache.get(md_path) or await self.parse_markdown(md_path)
        symbolic = self.build_symbolic_index(doc)
        decomposition = self.decompose_query(query)
        topics = decomposition.get('topics', [])
        candidates = self.select_candidates(doc, symbolic, topics, top_k=top_k)
        guided = self.llm_guided_selection(query, candidates)
        selected_pages = [s['page'] for s in guided['selected']]
        context, sources = self.construct_context(doc, selected_pages, token_limit=token_limit)
        answer = self.generate_answer(query, context, sources)
        # Return structured trace
        return {
            'query': query,
            'decomposition': decomposition,
            'symbolic_index': symbolic,
            'candidates': candidates,
            'guided_selection': guided,
            'context_sources': sources,
            'answer': answer,
        }
