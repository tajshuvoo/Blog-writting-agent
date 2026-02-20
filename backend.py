from __future__ import annotations

import operator
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field
from db import save_blog
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults

from dotenv import load_dotenv


load_dotenv()
# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str

    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand after this section.",
    )
    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=6,
        description="3–6 concrete, non-overlapping subpoints to cover in this section.",
    )
    target_words: int = Field(..., description="Target word count for this section (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # keep if Tavily provides; DO NOT rely on it
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: List[str] = Field(default_factory=list)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)



class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)
    final: str


# -----------------------------
# 2) LLM
# -----------------------------
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-14B-Instruct",
    temperature=0,
    max_new_tokens=1000,
    top_p=0.9
)
llm = ChatHuggingFace(llm=llm)


# -----------------------------
# HF-Compatible Router (Single Cell)
# -----------------------------
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage



# -----------------------------
# Parser
# -----------------------------
router_parser = PydanticOutputParser(pydantic_object=RouterDecision)


# -----------------------------
# Router Prompt
# -----------------------------
ROUTER_PROMPT = PromptTemplate(
    template=(
        "You are a routing module for a technical blog planner.\n\n"
        "Decide whether web research is needed BEFORE planning.\n\n"
        "Modes:\n"
        "- closed_book (needs_research=false): evergreen fundamentals.\n"
        "- hybrid (needs_research=true): mostly evergreen but benefits from up-to-date tools/models/examples.\n"
        "- open_book (needs_research=true): volatile topics (latest, rankings, pricing, policy, weekly updates).\n\n"
        "If needs_research=true:\n"
        "- Output 3–10 high-signal, specific search queries.\n"
        "- Reflect time constraints like 'this week' or 'latest' in the queries if present.\n"
        "- Avoid generic queries like just 'AI' or 'LLM'.\n\n"
        "Output MUST be strictly valid JSON.\n"
        "Do NOT include markdown.\n"
        "Do NOT include explanations.\n\n"
        "Topic: {topic}\n\n"
        "{format_instructions}"
    ),
    input_variables=["topic"],
    partial_variables={
        "format_instructions": router_parser.get_format_instructions()
    }
)


# -----------------------------
# Router Node (HF Compatible)
# -----------------------------
def router_node(state: dict) -> dict:
    topic = state["topic"]
    prompt = ROUTER_PROMPT.invoke({"topic": topic})

    for attempt in range(3):

        result = llm.invoke([
            SystemMessage(content="Return strictly valid JSON. No markdown."),
            HumanMessage(content=prompt.to_string())
        ]).content

        try:
            decision = router_parser.parse(result)

            return {
                "needs_research": decision.needs_research,
                "mode": decision.mode,
                "queries": decision.queries,
            }

        except Exception as e:
            print(f"[Router Parse Failed - Attempt {attempt+1}] {e}")

    raise ValueError("Router failed to produce valid JSON after retries.")


# -----------------------------
# Routing Function
# -----------------------------
def route_next(state: dict) -> str:
    return "research" if state["needs_research"] else "orchestrator"



# -----------------------------
# 4) Research Node (HF Compatible)
# -----------------------------

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults

research_parser = PydanticOutputParser(pydantic_object=EvidencePack)

# -----------------------------
# Tavily Wrapper
# -----------------------------
def _tavily_search(query: str, max_results: int = 5):

    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({"query": query})

    normalized = []
    for r in results or []:
        normalized.append(
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "published_at": r.get("published_date") or r.get("published_at"),
                "snippet": r.get("content") or r.get("snippet"),
                "source": r.get("source"),
            }
        )

    return normalized


# -----------------------------
# Prompt
# -----------------------------
RESEARCH_PROMPT = PromptTemplate(
    template=(
        "You are a research synthesizer for technical writing.\n\n"
        "Given raw web search results, produce a deduplicated list of EvidenceItem objects.\n\n"
        "Rules:\n"
        "- Only include items with a non-empty url.\n"
        "- Prefer authoritative sources (official docs, company blogs, reputable outlets).\n"
        "- Keep published_at only if explicitly present in raw data. Otherwise set null.\n"
        "- Keep snippets concise.\n"
        "- Deduplicate strictly by URL.\n\n"
        "Output MUST be strictly valid JSON.\n"
        "Do NOT include markdown.\n"
        "Do NOT include explanations.\n\n"
        "Raw results:\n{raw_results}\n\n"
        "{format_instructions}"
    ),
    input_variables=["raw_results"],
    partial_variables={
        "format_instructions": research_parser.get_format_instructions()
    }
)


# -----------------------------
# Research Node
# -----------------------------
def research_node(state: State) -> dict:

    queries = state.get("queries", []) or []
    raw_results = []

    for q in queries[:10]:
        raw_results.extend(_tavily_search(q, max_results=6))

    if not raw_results:
        return {"evidence": []}

    prompt = RESEARCH_PROMPT.invoke(
        {"raw_results": str(raw_results)}
    )

    for attempt in range(3):

        result = llm.invoke([
            SystemMessage(content="Return strictly valid JSON. No markdown."),
            HumanMessage(content=prompt.to_string())
        ]).content

        try:
            pack = research_parser.parse(result)

            # Hard dedup safety
            dedup = {}
            for e in pack.evidence:
                if e.url:
                    dedup[e.url] = e

            return {"evidence": list(dedup.values())}

        except Exception as e:
            print(f"[Research Parse Failed - Attempt {attempt+1}] {e}")

    raise ValueError("Research node failed after 3 attempts.")



# -----------------------------
# 5) Orchestrator (HF Safe)
# -----------------------------

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

plan_parser = PydanticOutputParser(pydantic_object=Plan)

ORCH_TEMPLATE = PromptTemplate(
    template=(
        "You are a senior technical writer and developer advocate.\n"
        "Your job is to produce a highly actionable outline for a technical blog post.\n\n"

        "Hard requirements:\n"
        "- Create 5–9 sections (tasks).\n"
        "- Each task must include:\n"
        "  1) goal (1 sentence)\n"
        "  2) 3–6 bullets that are concrete, specific, and non-overlapping\n"
        "  3) target word count (120–550)\n\n"

        "Quality bar:\n"
        "- Assume reader is a developer.\n"
        "- Bullets must be actionable: build/compare/measure/verify/debug.\n"
        "- Include at least TWO of these across the plan:\n"
        "  * minimal code sketch (set requires_code=True)\n"
        "  * edge cases / failure modes\n"
        "  * performance/cost considerations\n"
        "  * security/privacy considerations (if relevant)\n"
        "  * debugging/observability tips\n\n"

        "Grounding rules:\n"
        "- Mode closed_book: evergreen only.\n"
        "- Mode hybrid: use evidence for fresh examples. Mark those sections "
        "requires_research=True and requires_citations=True.\n"
        "- Mode open_book:\n"
        "  * Set blog_kind='news_roundup'\n"
        "  * Every section summarizes events + implications\n"
        "  * No tutorials unless explicitly requested\n"
        "  * If evidence is insufficient, create plan stating insufficient sources\n\n"

        "Output MUST be strictly valid JSON.\n"
        "Do NOT include markdown.\n"
        "Do NOT include explanations.\n\n"

        "Topic: {topic}\n"
        "Mode: {mode}\n\n"
        "Evidence (may be empty; use only for fresh claims):\n"
        "{evidence}\n\n"

        "{format_instructions}"
    ),
    input_variables=["topic", "mode", "evidence"],
    partial_variables={
        "format_instructions": plan_parser.get_format_instructions()
    }
)


def orchestrator_node(state: State) -> dict:

    evidence = state.get("evidence", []) or []
    mode = state.get("mode", "closed_book")

    prompt = ORCH_TEMPLATE.invoke(
        {
            "topic": state["topic"],
            "mode": mode,
            "evidence": str([e.model_dump() for e in evidence][:16]),
        }
    )

    for attempt in range(3):

        result = llm.invoke([
            SystemMessage(content="Return strictly valid JSON. No markdown."),
            HumanMessage(content=prompt.to_string())
        ]).content

        try:
            plan = plan_parser.parse(result)
            return {"plan": plan}

        except Exception as e:
            print(f"[Orchestrator Parse Failed - Attempt {attempt+1}] {e}")

    raise ValueError("Failed to generate valid Plan JSON after retries.")



# -----------------------------
# 6) Fanout
# -----------------------------
def fanout(state: State):
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]




# -----------------------------
# 7) Worker (write one section)
# -----------------------------
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Hard constraints:
- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
- Stay close to Target words (±15%).
- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).
- Start with a '## <Section Title>' heading.

Scope guard:
- If blog_kind == "news_roundup": do NOT turn this into a tutorial/how-to guide.
  Do NOT teach web scraping, RSS, automation, or "how to fetch news" unless bullets explicitly ask for it.
  Focus on summarizing events and implications.

Grounding policy:
- If mode == open_book:
  - Do NOT introduce any specific event/company/model/funding/policy claim unless it is supported by provided Evidence URLs.
  - For each event claim, attach a source as a Markdown link: ([Source](URL)).
  - Only use URLs provided in Evidence. If not supported, write: "Not found in provided sources."
- If requires_citations == true:
  - For outside-world claims, cite Evidence URLs the same way.
- Evergreen reasoning is OK without citations unless requires_citations is true.

Code:
- If requires_code == true, include at least one minimal, correct code snippet relevant to the bullets.

Style:
- Short paragraphs, bullets where helpful, code fences for code.
- Avoid fluff/marketing. Be precise and implementation-oriented.
"""

def worker_node(payload: dict) -> dict:
    
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    topic = payload["topic"]
    mode = payload.get("mode", "closed_book")

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}".strip()
            for e in evidence[:20]
        )

    section_md = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {topic}\n"
                    f"Mode: {mode}\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    return {"sections": [(task.id, section_md)]}



# -----------------------------
# 8) Reducer (merge + save)
# -----------------------------
def reducer_node(state: State) -> dict:
    plan = state["plan"]

    # Order sections
    ordered_sections = [
        md for _, md in sorted(state["sections"], key=lambda x: x[0])
    ]

    body = "\n\n".join(ordered_sections).strip()
    final_md = f"# {plan.blog_title}\n\n{body}\n"

    # ---- Build DB payload (minimal + clean) ----
    db_payload = {
        "plan": plan.model_dump() if hasattr(plan, "model_dump") else plan,
        "evidence": [
            e.model_dump() if hasattr(e, "model_dump") else e
            for e in state.get("evidence", [])
        ],
        "final": final_md,
    }

    # Save JSON (NOT just markdown)
    save_blog(plan.blog_title, db_payload)

    # Return for UI
    return db_payload

# -----------------------------
# 9) Build graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_node)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()
app
  