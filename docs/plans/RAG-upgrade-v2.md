# Agentic RAG Implementation Plan — v2

> **Status:** Ready for Phase 1 implementation
> **Date:** 2026-06-27
> **Target:** Refactor `/lore` slash command from retrieve-then-ask to agentic tool-calling

---

## 1. Architecture Overview

### Current System (Before)
```
User → /lore "what did we talk about last week?"
  └── Bot calls RAG service /retrieve(query="what did we talk about last week?")
        └── ChromaDB vector search → top 5 chunks
              └── Bot stuffs chunks into prompt → sends to llama.cpp lore model
                    └── Model answers → Bot sends to user
```

### Target System (After)
```
User → /lore "what did we talk about last week?"
  └── Bot sends to brain-dense via proxy:
        - System prompt with current date + available channels
        - User question
        - Tool schemas: search_discord_history, search_channel_history, summarize_channel
          └── LLM decides: "I need to search history" → returns tool_calls
                └── Bot executes tool → calls RAG service /retrieve(query="...", channel_name="...", start_date="...")
                      └── Results appended as role="tool" message
                            └── Bot sends updated history back to brain-dense (up to 3 rounds)
                                  └── LLM synthesizes final answer → Bot sends to user with intermediate progress messages
```

### Component Roles
| Component | Role | Changes |
|-----------|------|---------|
| **Discord Bot** ("The Brain") | Orchestrates tool-calling loop, defines schemas, executes tools, manages UX | Major refactor of `/lore`, new `agent_tools.py` |
| **FastAPI Proxy** ("Traffic Cop") | Pass-through gateway for LLM requests | Add `brain-dense` to swappable models set |
| **Llama.cpp Server** ("The Engine") | Runs brain-dense (Qwen3.6-27B) with tool-calling capability | No changes — already configured in models.ini |
| **RAG Service** | ChromaDB vector search + ingestion | Extend `/retrieve` with channel/date filters |
| **ChromaDB** | Vector store with metadata | No schema changes — keep ISO timestamps as-is |

---

## 2. Model Configuration

### Tool-Calling Model: `brain-dense`
- **GGUF:** `Qwen3.6-27B-Q4_K_M.gguf` (from `unsloth/Qwen3.6-27B-GGUF`)
- **VRAM:** ~18 GB
- **Context:** 96,000 tokens
- **Why:** Strong tool-calling capabilities from Qwen3.6 instruct family, fits in swappable slot
- **Temperature for tool-calling:** Override to `0.1`–`0.3` via proxy request (lower than default 0.65)

### Action Required
Add `brain-dense` to `proxy/config.py` → `SWAPPABLE_MODELS` set so the proxy routes requests correctly.

---

## 3. Tool Definitions

### Tool 1: `search_discord_history`
Searches ALL channels for relevant conversation history.

```json
{
  "name": "search_discord_history",
  "description": "Search Discord server history across all channels for conversations related to a topic or question.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The search query — what topic, event, or conversation to find."
      },
      "start_date": {
        "type": "string",
        "description": "Optional ISO 8601 date string (e.g. '2026-06-20'). Only return results after this date."
      },
      "end_date": {
        "type": "string",
        "description": "Optional ISO 8601 date string. Only return results before this date."
      },
      "top_k": {
        "type": "integer",
        "description": "Number of chunks to retrieve. Default 5, max 20."
      }
    },
    "required": ["query"]
  }
}
```

**Execution:** Call RAG service `/retrieve` with `query`, optional `start_date`/`end_date` filters.

---

### Tool 2: `search_channel_history`
Searches a SPECIFIC channel for relevant conversation history.

```json
{
  "name": "search_channel_history",
  "description": "Search Discord history within a specific channel for conversations related to a topic.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The search query."
      },
      "channel_name": {
        "type": "string",
        "description": "Exact channel name to search (e.g. 'general', 'support')."
      },
      "start_date": {
        "type": "string",
        "description": "Optional ISO 8601 date string."
      },
      "end_date": {
        "type": "string",
        "description": "Optional ISO 8601 date string."
      },
      "top_k": {
        "type": "integer",
        "description": "Number of chunks to retrieve. Default 5, max 20."
      }
    },
    "required": ["query", "channel_name"]
  }
}
```

**Execution:** Call RAG service `/retrieve` with `query` + `channel_name` filter.

---

### Tool 3: `summarize_channel`
Gets a recent activity summary of a channel (retrieves top-K chunks with no specific query, just channel context).

```json
{
  "name": "summarize_channel",
  "description": "Get recent activity from a Discord channel to understand what's been happening there.",
  "parameters": {
    "type": "object",
    "properties": {
      "channel_name": {
        "type": "string",
        "description": "Exact channel name to summarize."
      },
      "start_date": {
        "type": "string",
        "description": "Optional ISO 8601 date string. Only get activity after this date."
      },
      "top_k": {
        "type": "integer",
        "description": "Number of recent chunks to retrieve. Default 10, max 20."
      }
    },
    "required": ["channel_name"]
  }
}
```

**Execution:** Call RAG service `/retrieve` with a generic query (e.g., "recent conversations") + `channel_name` filter + higher `top_k`.

---

## 4. System Prompt Template

Injected per-request by the bot. Provides real-world context the LLM needs:

```
You are a Discord server lore assistant. Your job is to answer questions about
server history, in-jokes, decisions, and conversations using the tools available to you.

Today is {current_date}. The current time is {current_time}.

Available channels: {channel_list}

Rules:
- Use tools to search history when a question requires factual information.
- You may call multiple tools if needed (e.g., search different channels).
- Always cite which channel(s) your answer comes from.
- If you find no relevant results, say so honestly rather than guessing.
- Keep answers concise and conversational — this is Discord, not a research paper.
```

---

## 5. Execution Loop (Core Logic)

Implemented in `discord-bot/agent_tools.py` as `run_agent_loop()`.

### Flow
```
1. BUILD REQUEST:
   - System prompt (with date + channels)
   - User's question as role="user"
   - Tool schemas attached to request

2. SEND TO BRAIN-DENSE via proxy:
   proxy_client.chat(model="brain-dense", messages=..., tools=...)

3. EVALUATE RESPONSE:
   ┌─────────────────────────────────────────────────────┐
   │ LLM returns tool_calls?                             │
   │                                                     │
   │ YES → Go to step 4                                  │
   │                                                     │
   │ NO  → LLM gave final answer → Send to user ✓        │
   └─────────────────────────────────────────────────────┘

4. APPEND ASSISTANT TOOL_CALL MESSAGE:
   conversation_history.append({
       "role": "assistant",
       "tool_calls": [ { "id": "...", "function": {...} } ]
   })
   # CRITICAL: LLM must see its own tool call request in history

5. EXECUTE EACH TOOL:
   For each tool_call:
     a. Parse function name + arguments (json.loads with error handling)
     b. Call appropriate RAG service endpoint
     c. Format results as tool response
     d. Send Discord message: "🔍 Searching #{channel}..."

6. APPEND TOOL RESULTS:
   conversation_history.append({
       "role": "tool",
       "tool_call_id": "...",
       "content": "<retrieved chunks from ChromaDB>"
   })

7. CHECK ROUND LIMIT:
   If tool_call_rounds >= MAX_ROUNDS (3):
     Force LLM to answer with what it has.
   Else:
     Go to step 2 with updated conversation_history.

8. FINAL ANSWER:
   Send synthesized response to Discord user.
```

### UX Flow on Discord
```
User: /lore "what happened during the server migration last month?"

Bot: 🔍 Searching Discord history...
[1-2 seconds]
Bot: 🔍 Searching #announcements for migration details...
[1-2 seconds]
Bot: Based on the conversation in #announcements on June 15th, here's what happened:

The server migration was planned by @User1 to move from...

(@User2 noted that DNS propagation took longer than expected,
causing about 4 hours of downtime.)
```

---

## 6. Implementation Phases

### Phase 1: Foundation ✅ (Starting Here)
- [ ] **1a.** Add `brain-dense` to `proxy/config.py` → `SWAPPABLE_MODELS`
- [ ] **1b.** Extend RAG service `/retrieve` endpoint with optional filters:
  - `channel_name: Optional[str]`
  - `start_date: Optional[str]` (ISO 8601)
  - `end_date: Optional[str]` (ISO 8601)
- [ ] **1b-i.** Update `rag/retrieve.py` → `build_lore_context()` to accept and apply filters via ChromaDB `where` clause
- [ ] **1b-ii.** Update `rag/main.py` → `RetrieveRequest` Pydantic model with new optional fields
- [ ] **1c.** Update `discord-bot/rag_client.py` → `retrieve()` method to pass filter params

### Phase 2: Agent Core (New File)
- [ ] **2a.** Create `discord-bot/agent_tools.py`:
  - Tool schema definitions (all 3 tools as JSON dicts)
  - `build_system_prompt()` — injects current date + channel list
  - `execute_tool(tool_name, tool_args)` — dispatches to RAG service
  - `run_agent_loop(user_question, channel_id, proxy_client, rag_client)` — the core loop
- [ ] **2b.** Add config constants to `discord-bot/config.py`:
  - `AGENT_MODEL: str = "brain-dense"`
  - `AGENT_MAX_ROUNDS: int = 3`
  - `AGENT_TEMPERATURE: float = 0.1`

### Phase 3: Refactor `/lore` Command
- [ ] **3a.** Rewrite `/lore` slash command in `discord-bot/bot.py`:
  - Replace retrieve-then-ask with agent loop
  - Send intermediate progress messages ("🔍 Searching...")
  - Use `brain-dense` model instead of `lore` model
  - Handle tool call rounds (up to 3)
- [ ] **3b.** Add `proxy_client.chat()` support for tools parameter (or extend existing method)

### Phase 4: Config & Polish
- [ ] **4a.** Verify proxy pass-through handles `tools`, `tool_calls` fields without stripping
- [ ] **4b.** Test end-to-end: `/lore "what did we discuss about X?"`
- [ ] **4c.** Add logging for tool calls and round counts

---

## 7. File Change Summary

| File | Action | Description |
|------|--------|-------------|
| `proxy/config.py` | Edit | Add `"brain-dense"` to `SWAPPABLE_MODELS` |
| `rag/main.py` | Edit | Extend `RetrieveRequest` with filter fields |
| `rag/retrieve.py` | Edit | Add `where` clause support in `build_lore_context()` |
| `discord-bot/rag_client.py` | Edit | Pass filter params to `/retrieve` |
| `discord-bot/config.py` | Edit | Add agent config constants |
| `discord-bot/agent_tools.py` | **Create** | Tool schemas, system prompt builder, execution loop |
| `discord-bot/bot.py` | Edit | Refactor `/lore` command for agentic flow |
| `discord-bot/proxy_client.py` | Edit | Add `tools` parameter support to `chat()` method |

---

## 8. Known Constraints & Gotchas

- **Proxy pass-through:** Currently forwards raw body — should handle tool-calling fields, but needs verification in Phase 4
- **Temperature override:** Brain-dense defaults to 0.65 in models.ini, but agent loop should send `temperature=0.1` per-request for deterministic tool calling
- **Context window:** Brain-dense has 96K context — plenty for prompt + tools + retrieved chunks + final answer
- **ChromaDB metadata:** Already stores `channel_name`, `timestamp_start`, `timestamp_end` as ISO strings — ChromaDB supports string comparison in `where` clauses (e.g., `{"channel_name": {"$eq": "general"}}`)
- **Model swap latency:** Brain-dense is ~18 GB and may need to be loaded into VRAM on first request (evicting the currently loaded model). First `/lore` after a swap will take 10-30 seconds.
