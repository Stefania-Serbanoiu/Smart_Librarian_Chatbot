# backend/rag.py
from typing import List, Dict, Any, Tuple, Optional
from openai import OpenAI
import json
from .config import OPENAI_API_KEY, CHAT_MODEL, DEFAULT_TOP_K, BAD_WORDS
from .db import search
from .tools import get_summary_by_title


client = OpenAI(api_key=OPENAI_API_KEY)


def contains_bad_language(text: str) -> bool:
    """
    Returns True if the input text contains any word found in BAD_WORDS.
    The check is token-based (splits by whitespace) and normalizes tokens
    by stripping common punctuation and lowercasing before comparison
    """
    words = {w.strip(".,!?").lower() for w in text.split()}
    return any(bad in words for bad in BAD_WORDS)


def _ctx_from_hits(hits: List[Dict[str, Any]]) -> str:
    """
    Builds a human-readable context string from retrieval hits
    Args:
        hits: Retrieval results containing 'metadata.title' and 'document'
    Returns:
        A formatted context string
    """
    return "\n\n".join(
        f"[#{i+1}] {h['metadata']['title']}\n{h['document']}" for i, h in enumerate(hits)
    )


def _titles_from_hits(hits: List[Dict[str, Any]]) -> List[str]:
    """
    Extracts the list of titles from retrieval hits
    Args:
        hits: Retrieval results with 'metadata.title'
    Returns:
        A list of titles in the order of the hits
    """
    return [h["metadata"]["title"] for h in hits]


# Function Calling schema for get_summary_by_title() method
_TOOLS_SCHEMA = [{
    "type": "function",
    "function": {
        "name": "get_summary_by_title",
        "description": "Returns the full summary for an exact (case-insensitive) title from the local source.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "The exact book title"}
            },
            "required": ["title"]
        }
    }
}]


def _finalize_with_json(messages: List[Dict[str, Any]], num_recs: int) -> List[Tuple[str, str, str]]:
    """
    Ask the model to return STRICT JSON: [{title, rationale, detailed_summary}] and parse it

    Args:
        messages: The message list (system/user/assistant/tool) accumulated so far.
        num_recs: Expected number of recommendation items
    Returns:
        A list of (title, rationale, detailed_summary) tuples with at most `num_recs` items
        or empty list if parsing fails
    """
    messages_for_json = messages + [{
        "role": "system",
        "content": (
            "Formatează rezultatul STRICT ca JSON, fără text suplimentar: "
            f"[{{\"title\": str, \"rationale\": str, \"detailed_summary\": str}}] cu exact {num_recs} elemente."
        ),
    }]

    final = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages_for_json,
        temperature=0.2,
    )
    content = final.choices[0].message.content or "[]"
    try:
        data = json.loads(content)
        results: List[Tuple[str, str, str]] = []
        for item in data:
            title = str(item.get("title", "")).strip()
            rationale = str(item.get("rationale", "")).strip()
            detailed = str(item.get("detailed_summary", "")).strip()
            if title:
                results.append((title, rationale, detailed))
        # Enforce exact length if the model produced more
        return results[:num_recs]
    except Exception:
        # Defensive fallback: do not break the flow if JSON is invalid
        return []


def recommend_multiple_with_tool(query: str, top_k: int, num_recs: int) -> List[Tuple[str, str, str]]:
    """
    Multi-item recommendation using OpenAI Function Calling over a RAG shortlist.

    Args:
        query: User interests / query string
        top_k: Number of RAG hits to retrieve
        num_recs: Number of distinct recommendations to return

    Returns:
        A list of (title, rationale, detailed_summary) tuples or empty if no hits
    """
    hits = search(query, top_k=top_k)
    if not hits:
        return []

    allowed_titles = _titles_from_hits(hits)
    books_context = _ctx_from_hits(hits)

    system_msg = {
        "role": "system",
        "content": (
            "Ești un recomandator de cărți atent la temele cerute. Răspunde în română. "
            "NU inventa titluri. Alege doar din lista de titluri eligibile."
        ),
    }
    user_msg = {
        "role": "user",
        "content": (
            "Utilizatorul dorește recomandări multiple.\n"
            f"Alege EXACT {num_recs} titluri DISTINCTE doar din lista de titluri eligibile.\n"
            "Pentru fiecare titlu pe care îl alegi, te rog să chemi funcția get_summary_by_title(title) "
            "cu titlul exact (case-sensitive preferabil, dar acceptă și case-insensitive).\n\n"
            f"Interese: {query}\n\n"
            f"Titluri eligibile: {allowed_titles}\n\n"
            "Context RAG (pentru înțelegere, nu pentru halucinații de titluri):\n"
            f"{books_context}\n"
        ),
    }

    messages: List[Dict[str, Any]] = [system_msg, user_msg]

    # First, model selects titles and requests tool_calls
    first = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        tools=_TOOLS_SCHEMA,
        tool_choice="auto",
        temperature=0.4,
    )

    assistant_msg = first.choices[0].message
    messages.append({
        "role": "assistant",
        "content": assistant_msg.content or "",
        "tool_calls": assistant_msg.tool_calls,
    })

    tool_calls = assistant_msg.tool_calls or []

    # Execute requested tools + respond to every tool_call_id
    used_titles: List[str] = []
    executed = 0

    for call in tool_calls:
        fn = getattr(call, "function", None)
        fn_name = getattr(fn, "name", None) if fn else None

        # Unsupported or missing tool
        if fn_name != "get_summary_by_title":
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": fn_name or "unknown_tool",
                "content": "Unsupported tool.",
            })
            continue

        # Limit reached: acknowledge but don’t execute
        if executed >= num_recs:
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": "get_summary_by_title",
                "content": "Skipped: limit reached.",
            })
            continue

        # Parse arguments safely
        try:
            args = json.loads(getattr(fn, "arguments", "") or "{}")
        except Exception:
            args = {}
        title = str(args.get("title", "")).strip()

        # Validate & deduplicate titles, but still respond to the tool_call
        if not title or title not in allowed_titles or title in used_titles:
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": "get_summary_by_title",
                "content": "Ineligible or duplicate title.",
            })
            continue

        # Execute tool
        detailed = get_summary_by_title(title)
        used_titles.append(title)
        executed += 1

        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "name": "get_summary_by_title",
            "content": detailed,
        })


    results = _finalize_with_json(messages, num_recs=num_recs) # request JSON

    # Fallback: if JSON failed, build minimal results from executed titles
    if not results and used_titles:
        for t in used_titles:
            detailed = get_summary_by_title(t)
            rationale = "High thematic match based on RAG."
            results.append((t, rationale, detailed))

    return results[:num_recs]


def run_recommendation_pipeline_multi(
    query: str, top_k: int = DEFAULT_TOP_K, num_recs: int = 1, language_filter: bool = True
) -> Optional[List[Tuple[str, str, str]]]:
    """
    Entry point for the multi-recommendation pipeline with optional language filtering
    Args:
        query: User interests / query string
        top_k: Number of RAG hits to retrieve
        num_recs: Number of distinct recommendations to return
        language_filter: If True, checks for bad language and blocks when detected
    Returns:
        None if blocked by language_filter; otherwise a list of (title, rationale, detailed_summary)
    """
    if language_filter and contains_bad_language(query):
        return None  # flagging inadequate language
    recs = recommend_multiple_with_tool(query, top_k=top_k, num_recs=num_recs)
    return recs
