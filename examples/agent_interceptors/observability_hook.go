package main

import (
	"context"
	"iter"
	"log"
	"sync"
	"time"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

// ObservabilityInterceptor demonstrates ModelInterceptor for tracking model call metrics.
// It tracks:
// - Model call counts
// - Token usage
// - Call latencies
//
// This interceptor is thread-safe and can be reused across multiple invocations.
type ObservabilityInterceptor struct {
	mu         sync.Mutex
	modelCalls int
}

// NewObservabilityInterceptor creates a new observability interceptor.
func NewObservabilityInterceptor() *ObservabilityInterceptor {
	return &ObservabilityInterceptor{}
}

// InterceptModel implements agent.ModelInterceptor.
// It wraps model calls to track latency and usage.
func (h *ObservabilityInterceptor) InterceptModel(
	_ context.Context,
	info *agent.ModelCallInfo,
	next agent.ModelCallHandler,
) agent.ModelCallHandler {
	return &observabilityModelHandler{
		interceptor: h,
		next:        next,
		modelInfo:   info,
	}
}

// observabilityModelHandler wraps model generation calls.
type observabilityModelHandler struct {
	interceptor *ObservabilityInterceptor
	next        agent.ModelCallHandler
	modelInfo   *agent.ModelCallInfo
}

// Generate implements synchronous model generation with metrics.
func (h *observabilityModelHandler) Generate(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	h.interceptor.mu.Lock()
	h.interceptor.modelCalls++
	callNum := h.interceptor.modelCalls
	h.interceptor.mu.Unlock()

	start := time.Now()
	log.Printf("[Observability] Model call #%d started - model=%s provider=%s session=%s turn=%d messages=%d",
		callNum,
		h.modelInfo.Model.Name(),
		h.modelInfo.Model.Provider(),
		h.modelInfo.Inv.Session().ID,
		h.modelInfo.Inv.Turn(),
		len(h.modelInfo.Req.Messages))

	resp, err := h.next.Generate(ctx, req)

	duration := time.Since(start)
	if err != nil {
		log.Printf("[Observability] Model call #%d failed after %v: %v", callNum, duration, err)
		return nil, err
	}

	log.Printf("[Observability] Model call #%d completed in %v", callNum, duration)
	if resp.Usage != nil {
		log.Printf("[Observability] Tokens: input=%d output=%d total=%d",
			resp.Usage.InputTokens, resp.Usage.OutputTokens, resp.Usage.TotalTokens)
	}

	return resp, nil
}

// GenerateEvents implements streaming model generation with metrics.
func (h *observabilityModelHandler) GenerateEvents(ctx context.Context, req *llm.Request) iter.Seq2[llm.Event, error] {
	h.interceptor.mu.Lock()
	h.interceptor.modelCalls++
	callNum := h.interceptor.modelCalls
	h.interceptor.mu.Unlock()

	start := time.Now()
	log.Printf("[Observability] Streaming model call #%d started - model=%s provider=%s session=%s turn=%d messages=%d",
		callNum,
		h.modelInfo.Model.Name(),
		h.modelInfo.Model.Provider(),
		h.modelInfo.Inv.Session().ID,
		h.modelInfo.Inv.Turn(),
		len(h.modelInfo.Req.Messages))

	return func(yield func(llm.Event, error) bool) {
		var eventCount int

		for evt, err := range h.next.GenerateEvents(ctx, req) {
			if evt != nil {
				eventCount++
			}

			if !yield(evt, err) {
				return
			}
		}
		duration := time.Since(start)
		log.Printf("[Observability] Streaming call #%d completed in %v (%d events)",
			callNum, duration, eventCount)
	}
}
