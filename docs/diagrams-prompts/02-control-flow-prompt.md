# Diagram 2: Control Flow & Interceptors

## Goal
Create a diagram that answers: **"What happens when I send a request?"**

Show the request lifecycle and where interceptors hook into the flow.

---

## Layout

Use a **horizontal flow** (left to right) with interceptors shown as "middleware pills" on the arrows.

---

## Flow Structure

### Main Flow (left to right)
```
Your App → Runner → [Turn Interceptor] → Agent (Loop) → LLM Provider / Tool Registry
```

### Components

**YOUR APP**
- Position: Far left
- Color: Charcoal (#2D3748)

**RUNNER**
- Position: After Your App
- Connected to Session Store (side connection for Load/Save)
- Color: Primary (#3182CE or #805AD5)

**SESSION STORE**
- Position: Below or beside Runner
- Operations: Load/Save
- Color: Green (#38A169)

**AGENT (Loop)**
- Position: Center, after Runner
- Show circular arrow to indicate turn-based loop
- Color: Primary (#3182CE or #805AD5)

**LLM PROVIDER**
- Position: Right side, below Agent
- Connected to External API (OpenAI, Anthropic, etc.)
- Color: Orange (#DD6B20)

**TOOL REGISTRY**
- Position: Right side, below Agent (parallel to LLM Provider)
- Connected to External APIs (MCP, Custom)
- Color: Blue-gray (#4A5568)

---

## Interceptor Placement (CRITICAL)

Show interceptors as **small rounded pills/badges sitting on the arrows**:

### TurnInterceptor
- **Location: Between Runner and Agent** (wraps the agent loop)
- **NOT between Your App and Runner**
- Wraps entire agent turns
- Use cases: logging, early stopping, context window management
- Color: Light purple (#B794F4)

### ModelInterceptor
- **Location: Between Agent and LLM Provider**
- Wraps LLM generation calls
- Use cases: request validation, response caching, redaction
- Color: Light purple (#B794F4)

### ToolInterceptor
- **Location: Between Agent and Tool Registry**
- Wraps tool execution
- Use cases: retries, monitoring, security enforcement
- Color: Light purple (#B794F4)

---

## The Agent Loop

Show a **circular arrow** around the Agent box with these steps:
1. Check max turns
2. Call LLM (via Model)
3. Process response
4. Execute tools (if any)
5. Add results to session
6. Repeat or complete

---

## External APIs

Position at far right:
- **External API** (OpenAI, Anthropic, etc.) - connected from LLM Provider
- **External APIs** (MCP, Custom) - connected from Tool Registry

---

## Annotation
Include text: "Interceptors are optional middleware that wrap calls at each level"

---

## Caption
> "Requests flow from your app through the Runner to the Agent. Interceptors can hook into Turn, Model, and Tool calls to add cross-cutting behavior like logging, caching, or security."

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
- Rounded rectangles for components
- Solid arrows for control flow (2px stroke, dark color)
- **Interceptor pills**: Small rounded rectangles with white text on light purple background
- Chain link or filter funnel icon near interceptors
- Circular arrow around Agent to show loop

### Spacing
- 16-24px padding inside boxes
- 32-48px spacing between components
- Align to 8px grid

---

## Output
- Modern, clean aesthetic (Vercel/Supabase/Stripe style)
- SVG format for scalability
- Include micro-legend explaining line types and interceptor symbols
- High contrast for accessibility
