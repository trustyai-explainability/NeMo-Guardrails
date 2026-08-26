# 🛡️ ToolGuard

**ToolGuard** is a runtime policy enforcement engine designed to secure
AI systems that use tool-calling architectures.

Unlike traditional text-based guardrails, ToolGuard focuses specifically
on **tool execution safety**.

------------------------------------------------------------------------

## 🎯 Core Idea

ToolGuard enforces security and usage policies **at the moment a tool is
invoked**.

It acts as a control layer between the LLM and the actual tool
execution, ensuring that:

-   Tools are only called when allowed\
-   Tool arguments conform to defined policies\
-   Privileged tools cannot be misused\
-   Sensitive operations require proper authorization via correct arguments \
-   The LLM cannot escalate privileges through creative prompt
    manipulation

In simple terms:

> ToolGuard ensures that Large Language Models use tools correctly,
> safely, and within defined boundaries.

------------------------------------------------------------------------

## 🔐 What Problem It Solves

Modern LLM applications often allow models to:

-   Call APIs\
-   Trigger backend actions\
-   Modify data\
-   Execute code\
-   Access external systems

Without enforcement, this introduces risks such as:

-   Prompt injection leading to dangerous tool calls\
-   Unauthorized administrative operations\
-   Data exfiltration\
-   Policy bypass via indirect reasoning

ToolGuard prevents these risks by validating:

1.  **Tool name**
2.  **Tool arguments**
3.  **User role / context (optional via args)**
4.  **Execution intent**
5.  **Tool output (optional post-validation)**

Tool execution does **not** happen unless ToolGuard approves it.
------------------------------------------------------------------------

## ⚙️ Key Capabilities

-   ✅ Tool-level policy enforcement\
-   ✅ Argument schema validation\
-   ✅ Runtime plugin policies via Toolguard-Forge\
-   ✅ Blocking tool calls\
-   ✅ Audit logging of decisions

------------------------------------------------------------------------

Components for demo :
- Toolguard-Forge main server :
    1. Configuration time : loading generated policy enforce plugin - via Admin endpoint
    2. Runtime : Doing toolcall evaluation
  https://github.ibm.com/MLT/toolguard-forge

- Clinic MCP server  -  simple Clinic application
    https://github.com/vz-ibm/clinic-mcp-server

- Nemoguard server
   this repo

- Optional MCP Gateway server


------------------------------------------------------------------------

## 📦 Installation ToolGuard-Forge

``` bash
git clone https://github.ibm.com/MLT/toolguard-forge.git
cd toolguard

```

Run the server as docker:
  https://github.ibm.com/MLT/toolguard-forge/tree/master/docker_files

Install plugin generated for Clinic policy
- Link to plugin zip file
- How to install plugin

------------------------------------------------------------------------

## 📦 Installation Clinic MCP Server

git clone https://github.com/vz-ibm/clinic-mcp-server.git


## Running Clinic MCP Server
https://github.com/vz-ibm/clinic-mcp-server/blob/main/README.md
------------------------------------------------------------------------

## 📦 Running NemoGuard with ToolGuard configuration

config.yaml
```yaml
rails:
  config: {}
  tool_output:
    flows:
      - toolguard block tool output
```


Colang v1:
```yaml
    define bot inform unsafe tool call
        "Tool call not allowed."


    define subflow toolguard block tool output
    $tg = execute toolguard_check(tool_calls=$tool_calls)

    if not $tg.allowed
        if $config.enable_rails_exceptions
            create event OutputRailException(message=$tg.reason)
        else
            bot refuse to respond
        stop

```

actions.py
 /examples/configs/toolguard/actions.py


------------------------------------------------------------------------

## 📚 Research Paper

For the theoretical foundation and evaluation methodology behind
ToolGuard, see:

**ToolGuard: Securing Tool-Calling LLM Systems via Runtime Policy
Enforcement**\
[https://arxiv.org/pdf/2507.16459 (link)](https://arxiv.org/pdf/2507.16459)

The paper discusses:

-   Threat modeling for tool-calling LLMs\
-   Policy design strategies\
-   Experimental results\
-   Attack simulations\
-   Formal security guarantees

------------------------------------------------------------------------

## 🏁 Summary

ToolGuard provides a dedicated enforcement layer for tool-calling AI
systems.

It ensures that:

-   The LLM cannot misuse tools\
-   High-risk operations are protected\
-   Policies are enforced deterministically\
-   AI systems remain safe even under adversarial prompts

ToolGuard is built for production-grade AI systems where tool access
equals real-world impact.
