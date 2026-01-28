[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/tailvar/theAIE/blob/master/theaiengineer-curriculum/capstone04/Capstone/notebook/incident_mcp_colab_demo.ipynb)

←**[Back to main README](../../../README.md)**

# Incident Command Agent (MCP Capstone)
## Overview
This project implements an <b>Incident Command Agent</b> as a Capstone demonstration of the <b>Model Context Protocol</b>. The purpose of the project is not to build a production ready incident-response system, but to show how a <b>Large Language Model (LLM)</b> can be embedded safely and credibly inside a larger engineering system.

At a high level, the agent observes a situation, decides what to do next, executes actions through well-defined tools, and records what happened for future steps. The system is deliberately structured to be auditable, reproducible and extensible. Rather than treating the LLM as an all-knowing authority, it is used narrowly as a <b>planner</b> within a controlled loop.

The Incident Command scenario provides a concrete and realistic example of a system which uses framing-alerts, telemetry, runbooks, diagnostics and handoff notes-but the underlying architecture is general and applicable to many other domains.

## Project Root and Runtime Contract
A key architectural decision in this Capstone is the explicit separation between <b>code</b> and <b>runtime</b>.

All runtime-visible files - tools, resources, runbooks, telemetry snapshots, persistent memory, prompts and generated artifacts - are resolved relative to a single directory known as the <b>project root</b>. This root is defined in the environment variable:

`INCIDENT_MCP_ROOT`

Both the agent and the MCP server rely on this variable to locate:

    -   `config/` - tool and resources schema
    -   `data/` - alerts, telemetry, runbooks and persistent memory
    -   `prompts/' - planner system prompts
    -   `atrifacts` - telemetry logs, traces and handoff documents

The design ensures that:

    -   The MCP server remains deterministic and inspectable
    -   The same codebase can run locally, in CL or in Colab
    -   Replays are portable and reproducible across environments
Importantly, <b>no runtime state is baked into the Python package itself</b>. The agent is portable, the "world" it reasons about is external explicit and versionable.

## Conceptual Model

The core of the system is a simple decision loop: <b>Observe -> Plan -> Act -> Learn.</b>

During the <b>Observation</b> phase, the agent reads structured information about the current state of the world. In this project, that world consists of alert snapshots, recent telemetry, runbook documents, and previously recorded memory entries. Observation is read-only and side-effect free.

In the <b>Planning</b> phase, the agent uses an LLM to choose the single best next action. The model is not asked to solve the entire problem or generate a final answer. Instead, it selects one action from a constrained set of available tools and produces a structured response describing that action. Budgets and guardrails are enforced so that planning cannot loop indefinetely or exceed predefined limits.

In the <b>Action</b> phase, the agent does not directly manipulate the environment. Instead, it requests tht the MCP server execute a specific tool. The server validates inputs against schemas, performs the requested operation deterministically, and returns a structured result. This separation ensures that untrusted reasoning never directly cause side effects.

In the <b>Learning</b> phase, the outcome of the action is summarised into a compact memory entry. These "deltas" provide freash context for subsequent planning steps without accumulating unbounded state.

This loops repeats until the agent decides to finish, typically by producing a concise human-readable incident handoff.

## Architecture and Design

The system is intentionally split into two components:

The <b>Client (agent)</b> is responsible for orchestration. It implements the <b>Observe -> Plan -> Act -> Learn</b> loop, enforces budgets and records telemetry. The LLM is used exclusively within this component to decide which action to take next.

The <b>Server (tool stack)</b> represents the external world. It exposes <b>Resources</b> (alerts, telemetry, runbooks, memory) and <b>Tools</b> (retrieval, diagnostics, summarisation) via MCP using JSON-RPC 2.0. the server is deterministic, validates all inputs and persists state safely. The server exposes a set of structured tools, validates inputs against JSON schemas, executes actions and persists both execution traces and memory to disk. Each tool call is measured for basic latency and logged as a durable artifact, giving a clear audit trail f what the agent decided, what it did and what it learned.

Communication between the Client and the Server is fully structured and protocol-driven. Every request and response includes correlation identifiers and can be logged and replayed. This design makes the system testable, debuggable and auditable.

 ## Planning Modes: 'LLM' and 'Rules'
A key feature of this Capstone is that the <b>planning phase is a replaceable component</b>. The agent supports two planning modes that exercise the same MCP protocol, tools, memory surfaces and telemetry.

### LLM Based Planning
In LLM mode, the agent uses a Large Language Model (Anthropic or OpenAI) to choose the next action. The model is constrained by:

    -   A strict system prompt
    -   A list of available tools ahd schemas
    -   Explicit budgets (steps, tool calls, tokens, cost)
The model must output a single JSON object describing either a tool call or a finish decision. This mirrors how LLMs are used in real systems: as <b>constrained planners</b>, not autonomous executors.

LLM planning is useful when:

    -   Context is ambiguous or incomplete
    -   You want generalisation beyond fixed rules
    -   You are experimenting with reasoning quelity or prompts

### Rules Based Planning
In rules mode, the agent makes decisions using a deterministic Python method instead of an LLM. The rules encode a simple but realistic triage flow: run diagnostics, summarise the incident and finish when appropriate.

Rules mode exists to demonstarte that <b>MCP is not tied to language models</b>. It enables zero-cost runs, deterministic behavious, debugging, and replayable tests while exercising the same architecture as LLM mode.

## Client-Server Communication (stdio and JSON-RPC)
Although the Incident Command Agent appears to be a single program  it is in fact composed of two independent processes: an <b>agent (client)</b> and an <b>MCP Server</b>. These processes communicate using a simple, explicit protocol built on JSON-RPC 2.0 and standard input/output streams.

In this project the agent launches the server as a subprocess and connects to it using operating system pipes. The agent writes JSON-encoded requests to the server's standard input, and the server replies with JSON-encoded responses on standard output. Each message occupies exactly one line of JSON. This strict one-request / one response pattern keeps the interaction deterministic and easy to reason about.

A critical design rule is that <b>only protocol messages are written to stdout</b>. All human-readable logs, banners and diagnostics are written to stderr instead. This seperation ensures that the protocol stream remains clean, machine-readable and replayable. Any transscipt captured from stdout alone can be deterministically re-executed for debugging or evaluation.

Using stdio pipes rather than WebSockets or HTTP is an intentional choice for the Capstone. Pipes eliminate networking complexity, avoid background services and port management, and make every interaction visible and auditable. Importantly the <b>Model Context protocol</b> itself is transport agnostic: the same JSON-RPC messages could be sent over WebSockets or HTTP without changing the agent logic, tools or memory model. The stdio approach simply makes the mechanics of the protocol as transparent as possible for demonstration and assessment purposes.

## Configuration
The project is designed to be run locally using standard Python tooling:
    
    1.  Create and activate a Python 3.10+ virtual environment.
    2.  cd theaiengineer-curriculum/capstone04/Capstone/src
    3.  Set up the project root:
            `export INCIDENT_MCP_ROOT=$(pwd)` -> this can also live in your .env file
    4.  Install dependencies (conda)
            `bash
            `conda env create -f environment.yml
            `conda activate theAIE
    5.  In a terminal, Start the MCP server:
            `python -m incident_mcp server`
    6.  In a second terminal, run the agent:
            *   LLM mode:
                    `export PLANNER_BACKEND=anthropic` # or openai
                    `python -m incident_mcp agent --root $(pwd)
            *   Rules mode:
                    `export PLANNER_BACKEND=rules`
                    `python -m incident_mcp agent --root $(pwd)`
Each run produces:
    
    *   Telemetry logs:         `artifacts/runs.jsonl`
    *   Replay traces:          `artifacts/traces/run-<id..jsonl`
    *   Persistent memory:      `data/memory/memory.jsonl`
    *   Human handoff notes:    `artifacts/handoffs/handoff_<run_id>.md`
These outputs are identical regardless of the planning mode (LLM or Rules)
## Optional Colab Path (Demo Only)

Local execution is the canonical path for this Capstone. Colab is supported only as an optional demonstration environment.

In stdio mode, the agent starts the MCP server as a subprocess and communicates with it over operating-system pipes (stdin/stdout). This makes the protocol extremely transparent (one JSON-RPC message per line) and avoids any networking or port management. The provided Colab notebook runs the agent end-to-end in this same stdio configuration, so you do **not** need to start a separate server process in Colab.

Colab runtimes are ephemeral, so by default all artifacts (telemetry, traces, memory, handoff notes) disappear when the runtime resets. If you want persistence, mount Google Drive and point the project’s `artifacts/` and `data/` directories into Drive so logs and memory survive restarts. If you do this, keep the on-disk layout identical to local runs so replay and inspection work the same way.

A more advanced Colab variant is to run the MCP server as a network service (WebSockets) inside Colab and connect to it from a local agent via a public tunnel (cloudflared/ngrok). This is useful if you explicitly want “server in Colab, agent on your laptop”, but it adds moving parts: port forwarding, tunnel auth, and long-running process constraints. For the Capstone, stdio remains the recommended transport because it is simpler, auditable, and transport-agnostic with respect to MCP (the same JSON-RPC messages can be carried over pipes, WebSockets, or HTTP).


## Extending the project

Although framed as an incident-response agent, this architecture is intentionally general. By changing the exposed resources and tools, the same client can be reused for other domains such as quantitative Research or data science experimentation. The key idea is the disciplined separation between observation, decision making, execution and learning, The project demonstrates how to engineer AI systems that are understandable and trustworth rather than opaque or purely prompt driven. 

Natural extensions to the minimal resource controls implemented would include enforcing timeouts on tool execution, token and cost budgets for LLM calls and latency limits to prevent runaway or slow agents. These controls would make the system more robust and production ready, while preserving the same clear separation between reasoning, tool use and persistent memory.

## Summary

<b>This Capstone demonstrates how to embed a language model inside a controlled, protocol-driven system that can reason, act and remember safely. The empasis is not on the intelligence of the model itself, but on the surrounding structure that makes its behaviour reliable, inspectable and extensible.</b>
