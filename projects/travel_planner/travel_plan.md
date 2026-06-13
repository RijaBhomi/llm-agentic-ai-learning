# Travel Planner (Agentic Tool Calling)

Welcome to the **Autonomous AI Agents** section of my learning repository! This project transitions from static Retrieval-Augmented Generation (RAG) workflows into building conversational interfaces that seamlessly map natural language intents to real-world execution parameters.

This command-line intelligent travel planner is built using the new `google-genai` SDK and `gemini-2.5-flash`. Instead of relying on simulated data or static text, the agent is equipped with native Python tools that query live endpoints to retrieve real-time global weather and active financial exchange rates.

---

## System Architecture

[ Multi-Turn User Dialogue Stream ]
                   │
                   ▼
┌───────────────────────────────────────────┐
│       Stateful Chat Session Context       │ <─── Keeps track of locations,
│  (Persistent Memory & Token Accounting)   │      budgets, & previous details
└───────────────────────────────────────────┘
│
▼ Passes History Matrix
┌───────────────────────────────────────────┐
│               Gemini Model                │ ─── Reads Docstrings ───> [ Tool Menu ]
└───────────────────────────────────────────┘
│
│ (Interceptors Intent & Requests Run)
▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Local Python Execution Loop                         │
│  1. wttr.in (Live Weather API)                                              │
│  2. open.er-api.com (Cross-Currency Exchange Math)                         │
│  3. budget_breakdown_calculator (Financial Allocation Formula)               │
└─────────────────────────────────────────────────────────────────────────────┘
│
▼ (Submits JSON Data Payload Back to Session)
┌───────────────────────────────────────────┐
│               Gemini Model                │ ─── Synthesizes Data ───> [ Contextual Reply ]
└───────────────────────────────────────────┘


The engine acts as a centralized **Decision Engine**. It parses natural language across multiple messages, updates its memory state, selects the appropriate tool from its configuration layout, and instructs the local runtime environment to execute it when sufficient parameters are met.

---

## 🚀 Key Engineering Concepts Mastered

### 1. Function Calling & Tool Schemas
Native Python functions are passed directly into the Gemini configuration array. The model scans function signatures, variable type hints (`amount: float`), and docstring descriptions to map natural language intents directly to back-end parameters.

### 2. Multi-Turn Stateful Memory (`client.chats`)
Instead of managing manual conversation history arrays, the pipeline utilizes persistent chat state managers. The agent retains context across multiple turns, enabling users to split their specifications (e.g., providing a destination in message one, and a budget in message two) while maintaining seamless context tracking.

### 3. Live API Integration & Cross-Currency Math
* **Real-Time Weather:** Intersects location strings and hooks into `wttr.in` to retrieve live temperatures and conditions globally, passing them to a rule-based algorithm that determines packing recommendations.
* **Global Cross-Currency Conversion:** Implements a cross-currency mathematical formula to convert budgets starting from **Nepalese Rupees (NPR)** into any foreign currency code (like `JPY` or `EUR`) using free, live USD-base financial tracking tiers.

### 4. Dynamic Token Guardrails & Trimming
Monitors the chat history footprint dynamically using native token counting utilities (`client.models.count_tokens`). Includes a custom sliding-window guardrail that safely auto-trims ancient user/model message exchanges when memory sizes cross safety thresholds, keeping the agent fast and stable.

---
