WORLD={
    "alert":"CPU spike on service X",
    "runbook":"if CPU spike, check processes and recent deploys",
    "diagnostic_result":None
}

MEMORY=[]

def observe():
    return {
        "alert":WORLD["alert"],
        "runbook":WORLD["runbook"],
    }

def plan(observation):
    if WORLD["diagnostic_result"] is None:
        return "run_diagnostic"
    else:
        return "summarize"

def act(action):
    if action == "run_diagnostic":
        WORLD["diagnostic_result"] = "Found runaway process"
        return "Diagnstic complete"
    elif action == "summarize":
        return f"Handoff: {WORLD['diagnostic_result']}"

def learn(result):
    MEMORY.append(result)

def run():
    for step in range(3):
        obs = observe()
        action = plan(obs)
        result = act(action)
        learn(result)
        print(f"Step {step}: {action} -> {result}")

if __name__ == "__main__":
    run()

