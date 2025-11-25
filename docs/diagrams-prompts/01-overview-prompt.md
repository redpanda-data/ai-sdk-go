# Diagram 1: High-Level Overview

## Goal
Create a diagram that answers: **"What is this SDK and where does it sit?"**

Show the SDK's place in the developer's stack and the main components at a glance.

---

## Layout

Use a **vertical stack layout** (top to bottom represents the call hierarchy).

---

## Components (top to bottom)

### 1. YOUR APPLICATION (Entry Point)
- Position: Top of the diagram
- Shows the call: `Runner.Run(ctx, userID, sessionID, msg)`
- Color: Dark gray/charcoal (#2D3748)

### 2. RUNNER (Session Orchestrator)
- Position: Below Your Application
- Responsibilities:
  - Loads/creates sessions
  - Creates InvocationMetadata
  - Forwards events to caller
  - Saves session on completion
- Color: Primary brand color (#3182CE blue or #805AD5 purple)

### 3. SESSION STORE
- Position: Side component, connected to Runner
- Operations: `Load(sessionID)`, `Save(state)`, `Delete(sessionID)`
- Implementations: In-Memory | Database
- Color: Green/teal (#38A169) - implies persistence/state

### 4. AGENT (LLMAgent) - Central Component
- Position: Below Runner
- **Show what DEFINES an agent (inputs/configuration):**
  - **Required:**
    - Name
    - System Prompt
    - Model (LLM Provider)
  - **Optional:**
    - Tool Registry
    - Interceptors
    - Max Turns (default: 25)
    - Tool Concurrency (default: 3)
    - Description (for agent-as-tool)
- Visual: Show as "Agent Configuration" or "What defines an Agent" section within the box
- Color: Primary brand color (#3182CE or #805AD5)

### 5. SDK EXTERNAL INTEGRATIONS
- Position: Below Agent, arranged side by side

**LLM PROVIDER:**
- Label: "Model Interface"
- Pluggable providers (show as stacked chips with "..." for extensibility):
  - OpenAI
  - Anthropic
  - Gemini
  - Custom...
- Color: Orange/amber (#DD6B20) - external API boundary
- Add plug icon to indicate pluggability

**TOOL REGISTRY:**
- Tool types:
  - Built-in
  - MCP
  - Functions (Code)
- Note: Tools execute concurrently
- Color: Blue-gray (#4A5568)
- Add plug icon to indicate pluggability

**INTERCEPTORS:**
- Types:
  - Turn
  - Model
  - Tool
- Label as "middleware"
- Color: Light purple (#B794F4)

---

## Caption
> "The Runner coordinates sessions and delegates to an LLMAgent, which calls LLM models and executes tools. Everything below your application is handled by the SDK."

---

## Style Guide

### Color Palette
| Element | Color | Hex |
|---------|-------|-----|
| User Application | Charcoal | #2D3748 |
| SDK Core (Runner, Agent) | Blue or Purple | #3182CE or #805AD5 |
| Session Store | Green | #38A169 |
| External APIs (Providers) | Orange | #DD6B20 |
| Tool Registry | Blue-gray | #4A5568 |
| Interceptors | Light Purple | #B794F4 |
| Background | Light Gray | #F7FAFC |

### Typography
- **Component titles**: Inter/SF Pro, Bold, 16-18px
- **Labels/descriptions**: Inter/SF Pro, Medium, 12-14px
- **Code references**: JetBrains Mono/SF Mono, 11-12px

### Visual Elements
- Rounded rectangles for all components
- Solid arrows for control flow (2px stroke, dark color)
- Plug/socket icons next to LLM Provider and Tool Registry
- Container box around SDK components with light gray background (#F7FAFC)

### Spacing
- 16-24px padding inside boxes
- 32-48px spacing between components
- Align to 8px grid

---

## Reference: Agent Constructor
```go
agent := llmagent.New(
    name,           // Required: Agent name
    systemPrompt,   // Required: System prompt
    model,          // Required: LLM Model
    // Optional:
    llmagent.WithTools(registry),
    llmagent.WithInterceptors(...),
    llmagent.WithMaxTurns(25),
    llmagent.WithToolConcurrency(3),
    llmagent.WithDescription(desc),
)
```

---

## Output
- Modern, clean aesthetic (Vercel/Supabase/Stripe style)
- SVG format for scalability
- Include micro-legend explaining symbols
- High contrast for accessibility
