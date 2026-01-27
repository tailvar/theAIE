# Capstone 4 — Incident Command Agent (MCP)

This capstone builds a structured, protocol-driven AI agent system using the
Model Context Protocol (MCP). The agent follows an Observe → Plan → Act → Learn
loop and interacts with tools and resources in a controlled, auditable way.

➡ Full implementation lives here:  
**[Capstone 4 source folder](../theaiengineer-curriculum/capstone04/Capstone/)**

Key entry point:  
**[Incident MCP source](../theaiengineer-curriculum/capstone04/Capstone/incident_mcp/)**

### Set up the project root:
        `export INCIDENT_MCP_ROOT=$(pwd)` -> this can also live in your .env file
### Install dependencies
### In a terminal, start the MCP server:
        `python -m incident_mcp server`

- In a second terminal, run the agent in LLM mode:

                `export PLANNER_BACKEND=anthropic` # or openai
                `python -m incident_mcp agent --root $(pwd)`
- or ib Rules mode:
 
                `export PLANNER_BACKEND=rules`
                `python -m incident_mcp agent --root $(pwd)`