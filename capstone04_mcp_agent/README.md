# Capstone 4 — Incident Command Agent (MCP)

←**[Back to main README](../README.md)**


This capstone builds a structured, protocol-driven AI agent system using the
Model Context Protocol (MCP). The agent follows an Observe → Plan → Act → Learn
loop and interacts with tools and resources in a controlled, auditable way.

➡ Full implementation lives here:  
**[Capstone 4 source folder and README](../theaiengineer-curriculum/capstone04/Capstone/)**

Key entry point - run from command line at this directory:  
**[Incident MCP source](../theaiengineer-curriculum/capstone04/Capstone/src/)**

### Set up the project root:
        `export INCIDENT_MCP_ROOT=$(pwd)` -> this can also live in your .env file
### Install dependencies
### In a terminal, start the MCP server:
        `python -m incident_mcp server`

- In a second terminal, run the agent in LLM mode:

                `export PLANNER_BACKEND=anthropic` # or openai
                `python -m incident_mcp agent --root $(pwd)`
- or alternatively, run the agent in Rules mode:
 
                `export PLANNER_BACKEND=rules`
                `python -m incident_mcp agent --root $(pwd)`

### Alternatively use the incident MCP demo notebook
**[Incident MCP source](../theaiengineer-curriculum/capstone04/Capstone/notebook/incident_mcp_colab_demo.ipynb)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/tailvar/theAIE/blob/master/theaiengineer-curriculum/capstone04/Capstone/notebook/incident_mcp_colab_demo.ipynb)