# Incident Command Agent (MCP Capstone)
## Overview
This project implements an <b>Incident Command Agent</b> as a Capstone demonstration of the <b>Model Context Protocol</b>. The purpose of the project is not to build a production ready incident-response system, but to show how a <b>Large Language Model (LLM)</b> can be embedded safely and credibly inside a larger engineering system.

At a high level, the agent observes a situation, decides what to do next, executes actions through well-defined tools, and records what happened for future steps. The system is deliberately structured to be auditable, reproducible and extensible. Rather than treating the LLM as an all-knowing authority, it is used narrowly as a <b>planner</b> within a controlled loop.

The Incident Command scenario provides a concrete and realistic example of a system which uses framing-alerts, telemetry, runbooks, diagnostics and handoff notes-but the underlying architecture is general and applicable to many other domains.

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

The <b>Server (tool stack)</b> represents the external world. It exposes <b>Resources</b> (alerts, telemetry, runbooks, memory) and <b>Tools</b> (retrieval, diagnostics, summarisation) via MCP using JSON-RPC 2.0. the server is deterministic, validates all inputs and persists state safely.

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

When the agent needs something from the server-such as discovering avaialble tools, reading a resource, or executing a tool-it constructs a JSON-RPC request containing a unique identifier, a method name and a structured parameter object. This request is serialised to JSON, written to the servers stdin, and immediately flushed so it is delivered without delay. The ah=gent then blocks, waiting to read a single line of JSON from the servers stdout.

On the server side, a simple event loop continuously reads newline-delimited JSON messages from stdin. Each message is parsed, validated, and dispatched to a handler based on the requested method. The sever executes dterministic logic, constructs a structured result (or error), and writes a JSON-RPC response back to stdout using the same request identifier. This identifier allows the agent to safely correlate responses to requests, eben though the JSON-RPC is designed to support more general messaging patters.

A critical design rule is that <b>only protocol messages are written to stdout</b>. All human-readable logs, banners and diagnostics are written to stderr instead. This seperation ensures that the protocol stream remains clean, machine-readable and replayable. Any transscipt captured from stdout alone can be deterministically re-executed for debugging or evaluation.

Using stdio pipes rather than WebSockets or HTTP is an intentional choice for the Capstone. Pipes eliminate networking complexity, avoid background services and port management, and make every interaction visible and auditable. Importantly the <b>Model Context protocol</b> itself is transport agnostic: the same JSON-RPC messages could be sent over WebSockets or HTTP without changing the agent logic, tools or memory model. The stdio approach simply makes the mechanics of the protocol as transparent as possible for demonstration and assessment purposes.

## Why MCP and Structured Tools

MCP provides a strict contract between the agent and its tools. By forcing all interactions to go through explicit method calls with schemas, the system avoids immplicit side effects and hidden state. This makes it possible to replay past runs, evaluate changes to the planner or tools, and reason clearly about system behaviour.

Crucially, the protocol allows the language model to be swapped, tuned or even placed with rule-based logic without changing the rest of the system. The LLM is therefore a component, not the system itself.

## Incident Command Scenario

The chosen scenario models an on-call incident workflow. An alert is emitted by a platform, indicting a problem such as CrashLoopBackOff or elevated error rate. The agent gathers context, consults runbooks, executes diagnostics, and ultimately produces a concise handoff suitable for a human engineer.

The scenario is intentionally familiar and practical, but it serves primarily as a narrative wrapper around the underlying agent architecture. The same pattern could be applied to research workflows, compliance checks, data analysis or experimentation pipelines.

## Alignment with Capstone Requirements

The project satisfies the Capstone requirements in the following ways:

The server exposes multiple <b>Resources</b> (alerts, telemetry, runbooks, memory) and multiple <b>Tools</b> (retrieval, diagnostics, summarisation) via <b>MCP</b>.

The client implements a complete <b>Observe -> Plan -> Act -> Learn</b> loop with explicit budget enforcement and guardrails. All interactions are logged with structured telementry, including request/response transcripts and high-level run events, enabling replay and audit. Artifacts such as memory snapshots and incident handoff documents are written to disk to demonstrate persistence and human readable outputs. Configurations and secrets are externalised via environment variables, ensuring reproducibility and safe handling of credentials.

## Configuration
All configuration is provided via the <b>environment variables</b>. For local development it is recommended to use a .env file at the project root (ignored by git). The agent automatically reads configuration from the environment file, regardless of whether variables were set by .env or shell export.

## Running the Demo

The project is designed to be run locally using standard Python tooling.

1. Create and activate a Pythin 3.10+ virtual environment. Set the protocol root: export INCIDENT_MCP_ROOT=$(pwd) This can live in .env file.
2. Install dependencies.
3. Configure the environment variables in the .env file (LLM keys, budgets).
4. Start the MCP Server - python -m incident_mcp server
5. Run the Agent (LLM MODE) - export PLANNER_BACKEND=Anthropic (or openai) python -m incident_mcp agent --root $(pwd), OR...
6. Run the Agent (RULES MODE) - export PLANNER_BACKEND=rules python -m incident_mcp agent --root $(pwd).

Each run produces the following artifacts:

    -   Telemetry logs: `artifacts/runs.jsonl`
    -   Replay traces: `artifacts/traces/run-<id>.jsonl`
    -   Persistent memory: `data/memory/memory.jsonl`
    -   Human handoff notes: `artifacts/handoffs/handoff_<run_id>.md`

These outputs are identical regardless of the planning mode (RULES or LLM).

## Extending the project

Although framed as an incident-response agent, this architecture is intentionally general. By changing the exposed resources and tools, the same client can be reused for other domains such as quantitative Research or data science experimentation. The key idea is the disciplined separation between observation, decision making, execution and learning, The project demonstrates how to engineer AI systems that are understandable and trustworth rather than opaque or purely prompt driven.

## Summary

<b>This Capstone demonstrates how to embed a language model inside a controlled, protocol-driven system that can reason, act and remember safely. The empasis is not on the intelligence of the model itself, but on the surrounding structure that makes its behaviour reliable, inspectable and extensible.</b>
