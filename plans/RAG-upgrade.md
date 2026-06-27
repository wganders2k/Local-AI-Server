
---

# Implementation Guide: Agentic RAG for Discord via Llama.cpp

## 1. Architecture Overview
*   **The Discord Bot ("The Brain"):** Handles user interactions, holds the ChromaDB client, defines the tools, injects context (current date/time), and manages the back-and-forth execution loop.
*   **FastAPI Proxy ("The Traffic Cop"):** Acts purely as a pass-through gateway for LLM requests. It manages queues and system resources but contains *zero* logic regarding Discord or ChromaDB.
*   **Llama.cpp Server ("The Engine"):** Runs an instruct/tool-capable local model and communicates with the proxy via an OpenAI-compatible API.

## 2. Data Ingestion & ChromaDB Upgrades
To make tool-calling effective, the underlying vector data must be cleaned and structured.
*   **Mandatory Metadata:** Update your ingestion pipeline to append metadata to every document. Most importantly, store dates as **Unix Epoch integers** (e.g., `timestamp: 1718040000`). This is required for ChromaDB’s `$gte` (greater than) and `$lte` (less than) filters. Add `channel_name` and `author` as well.
*   **Noise Filtering:** Do not embed messages under 3–4 words unless they contain links, attachments, or are part of a concatenated thread.
*   **Context Windowing (Highly Recommended):** When fetching a match from ChromaDB, use the message ID to also grab the 2–3 preceding messages. Discord data is fragmented; retrieving surrounding messages provides the LLM with the missing context (e.g., what "it" refers to when someone says "restart it").

## 3. The "No-Harness" Implementation Logic
You will use the official `openai` Python SDK (specifically `AsyncOpenAI`) inside your Discord bot, pointing the `base_url` to your FastAPI proxy. 

### A. Tool Schema Definition
Define a static JSON dictionary representing the tools the LLM is allowed to use. 
*   **Tool Name:** `search_discord_history`
*   **Parameters:** `query` (string, required), `channel_name` (string, optional), `start_timestamp` (integer, optional).

### B. Dynamic System Prompting
LLMs do not natively know the current date or your server's channels. Every time a user invokes the slash command, your bot must prepend a system prompt containing the **current real-world context**:
> *"You are a helpful assistant. Today is June 27, 2026. The current Unix timestamp is [Insert Timestamp]. The available channels are 'general', 'support', etc. Use tools to search history if needed."*
*(This allows the LLM to successfully translate a user asking about "yesterday" into a mathematical Unix timestamp for your ChromaDB filter).*

### C. The Execution Loop (The Core Logic)
Because you are not using a framework, you must implement the tool-calling loop manually in your bot's command handler:
1.  **Initial Request:** Send the user's prompt + System Prompt + Tool Schema to the FastAPI proxy.
2.  **Evaluate Response:** Check if the LLM's response contains `tool_calls`.
    *   *If NO:* The LLM answered directly. Send the text to the Discord user.
    *   *If YES:* Proceed to Step 3.
3.  **Append Assistant Request:** Add the LLM's raw tool-call request to your temporary conversation history array. *(Crucial: The LLM must see its own request in the history).*
4.  **Execute the Tool:** 
    *   Parse the LLM's JSON arguments (using `json.loads` or a simple Pydantic model to catch hallucinated formats safely).
    *   Execute the ChromaDB query using the parsed `query` and mapping the LLM's requested date/channel to ChromaDB `where` filters.
5.  **Append Tool Results:** Add the retrieved ChromaDB documents to the conversation history as a new message with the role `"tool"`.
6.  **Final Request:** Send the updated conversation history back to the FastAPI proxy. The LLM will read the database results and synthesize the final answer for the user.

## 4. System Requirements & Gotchas

*   **FastAPI Proxy Validation:** Ensure your proxy is configured to blindly accept and pass through the `tools`, `tool_choice`, and `tool_calls` JSON keys. If the proxy strictly validates against an outdated schema, it will strip the tool data.
*   **Model Selection:** You **must** run a model fine-tuned for tool calling (e.g., `Llama-3.1-8B-Instruct`, `Hermes-2-Pro`, `Mistral-Nemo-Instruct`). Standard base models will ignore your JSON schemas.
*   **Temperature:** Hardcode the `temperature` parameter to a very low value (e.g., `0.1` to `0.3`) during the tool-calling phase to prevent the model from hallucinating invalid JSON structures.
*   **Context Size:** Ensure the `llama.cpp` server is launched with a sufficiently large context window (`-c` flag, ideally 4096 or 8192) so it does not truncate the prompt, the retrieved database results, and the final output.