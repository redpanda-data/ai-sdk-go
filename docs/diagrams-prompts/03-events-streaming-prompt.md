# Diagram 3: Events & Streaming

## Goal
Create a diagram that answers: **"How do events stream back to my app?"**

Show how events stream back to the application during execution.

---

## Layout

Use an **"Event Rail"** metaphor - a horizontal timeline showing events flowing back.

---

## Structure

### Top Section: SDK EXECUTION
- Show components that emit events: Runner, Agent, LLM, Tools, Session
- Arrange horizontally in a container box
- Arrow pointing down with label: "All components emit events"
- Color: Light gray container (#F7FAFC)

### Middle Section: EVENT STREAM (Iterator)
- Show as a horizontal rail/timeline flowing left to right
- Events appear in temporal order on the rail
- Use dashed line for the rail (indicates streaming/async)
- Color: Light blue (#4299E1)

### Bottom Section: YOUR APPLICATION
- Show code snippet in terminal/monospace style
- Color: Charcoal (#2D3748)

---

## Event Sequence (left to right on the rail)

Show events in this order with distinct icons:

1. **StatusEvent** (turn started)
   - Icon: Diamond shape
   - Color: Blue (#3182CE)

2. **StatusEvent** (model call)
   - Icon: Diamond shape
   - Color: Blue (#3182CE)

3. **AssistantDeltaEvent** (streaming)
   - Icon: Chat bubble
   - Show multiple bubbles to indicate streaming
   - Color: Blue (#4299E1)

4. **ToolRequestEvent** (execute)
   - Icon: Wrench
   - Color: Blue-gray (#4A5568)

5. **ToolResponseEvent** (result)
   - Icon: Outbox/arrow-out
   - Color: Blue-gray (#4A5568)

6. **ErrorEvent** (recoverable error)
   - Icon: Warning triangle
   - Color: Orange (#DD6B20)
   - Mark as "optional" or show with dashed outline

7. **InvocationEndEvent** (complete)
   - Icon: Checkmark or stop sign
   - Color: Green (#38A169)

---

## Event Categories Reference

| Event Type | Icon | Purpose |
|------------|------|---------|
| StatusEvent | Diamond | Execution phase transitions |
| AssistantDeltaEvent | Chat bubble | Streaming text tokens |
| MessageEvent | Document | Complete assistant message |
| ToolRequestEvent | Wrench | Tool invocation request |
| ToolResponseEvent | Outbox | Tool execution result |
| ErrorEvent | Warning triangle | Recoverable errors |
| InvocationEndEvent | Checkmark | Terminal event with FinishReason |

---

## Status Stages (sub-events of StatusEvent)

Show as a small legend or annotation:
- `StatusStageTurnStarted` - New turn beginning
- `StatusStageModelCall` - LLM being called
- `StatusStageToolExec` - Tools executing
- `StatusStageInputRequired` - Waiting for user input
- `StatusStageTurnCompleted` - Turn finished
- `StatusStageRunCompleted` / `RunFailed` / `RunCanceled` - Terminal states

---

## Code Snippet (Bottom Section)

Show in terminal/monospace style box:

```go
for event, err := range runner.Run(ctx, userID, sessionID, msg) {
    switch e := event.(type) {
    case *agent.StatusEvent:         // Turn started, model call, etc.
    case *agent.AssistantDeltaEvent: // Streaming text tokens
    case *agent.MessageEvent:        // Complete assistant message
    case *agent.ToolRequestEvent:    // Tool being called
    case *agent.ToolResponseEvent:   // Tool execution result
    case *agent.ErrorEvent:          // Recoverable error
    case *agent.InvocationEndEvent:  // Terminal event with usage
    }
}
```

---

## Caption
> "The SDK uses Go 1.22+ iterators for streaming. Events flow back to your app as they occur - from status updates and streaming text to tool executions. Every run ends with InvocationEndEvent containing the final FinishReason and cumulative TokenUsage."

---

## Style Guide

### Color Palette
| Element | Color | Hex |
|---------|-------|-----|
| User Application | Charcoal | #2D3748 |
| SDK Core (Runner, Agent) | Blue or Purple | #3182CE or #805AD5 |
| Session Store | Green | #38A169 |
| External APIs | Orange | #DD6B20 |
| Tool Events | Blue-gray | #4A5568 |
| Event Stream | Light Blue | #4299E1 |
| Errors | Orange | #DD6B20 |
| Success/Complete | Green | #38A169 |
| Background | Light Gray | #F7FAFC |

### Typography
- **Component titles**: Inter/SF Pro, Bold, 16-18px
- **Event labels**: Inter/SF Pro, Medium, 12-14px
- **Code**: JetBrains Mono/SF Mono, 11-12px
- **Captions**: Inter, Regular, 11px, Gray

### Visual Elements
- Dashed arrows for event/data flow (2px stroke, accent color)
- Event rail: Thick dashed line with gradient showing flow direction
- Events as colored badges/icons on the rail
- Timeline metaphor: left = earlier, right = later
- Code snippet in dark terminal-style box

### Spacing
- 16-24px padding inside boxes
- 32-48px spacing between components
- Align to 8px grid

---

## Output
- Modern, clean aesthetic (Vercel/Supabase/Stripe style)
- SVG format for scalability
- Include micro-legend mapping icons to event types
- High contrast for accessibility
