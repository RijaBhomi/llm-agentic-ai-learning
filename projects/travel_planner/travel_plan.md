# Travel Planner (Agentic Tool Calling)

Welcome to the **Autonomous AI Agents** section of my learning repository! This project transitions from static Retrieval-Augmented Generation (RAG) workflows into building conversational interfaces that seamlessly map natural language intents to real-world execution parameters.

This command-line intelligent travel planner is built using the new `google-genai` SDK and `gemini-2.5-flash`. Instead of relying on simulated data or static text, the agent is equipped with native Python tools that query live endpoints to retrieve real-time global weather and active financial exchange rates.

---

## System Architecture

[ User Casual Request ]
│
▼
┌─────────────────┐      Reads Docstrings      ┌──────────────────────────────┐
│  Gemini Model   │ ────────────────────────>  │   Available Python Tools     │
└─────────────────┘                            └──────────────────────────────┘
│                                                    │
│ (Intercepts Intent & Halts)                        │ Matches Schema
▼                                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Local Python Execution Loop                         │
│  1. wttr.in (Live Weather API)                                              │
│  2. open.er-api.com (Cross-Currency Exchange Math)                         │
│  3. budget_breakdown_calculator (Financial Allocation Formula)               │
└─────────────────────────────────────────────────────────────────────────────┘
│
▼ (Submits JSON Data Payload)
┌─────────────────┐
│  Gemini Model   │ ───> Synthesizes Final Context ───> [ Beautiful Itinerary ]
└─────────────────┘


The agent acts as a centralized **Decision Engine**. It does not execute local machine code directly; rather, it parses the user's natural language, selects the appropriate tool from its configuration menu, and requests the local runtime to execute it.

---

##  Key Engineering Concepts Mastered

### 1. Function Calling & Tool Schemas
Native Python functions are passed directly into the Gemini configuration array. The model scans function signatures, variable type hints (`amount: float`), and docstring descriptions to map natural language intents to background code.

### 2. Live API Integration & Fallbacks
* **Real-Time Weather:** Intersects location strings and hooks into `wttr.in` to retrieve live temperatures and conditions, passing them to a rule-based algorithm that determines packing recommendations.
* **Global Cross-Currency Conversion:** Implements a cross-currency mathematical formula to convert budgets starting from **Nepalese Rupees (NPR)** into any foreign currency code (like `JPY` or `EUR`) using free, live USD-base financial tracking tiers.

### 3. Dynamic Multi-Step Routing
The execution loop supports parallel and sequential function resolution. The model can request multiple tool executions simultaneously, absorb the generated JSON data packets, and pipe those results into subsequent calculations (such as the budget allocation matrix).

---
