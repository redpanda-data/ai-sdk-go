// Copyright 2026 Redpanda Data, Inc.
//
// Licensed as a Redpanda Enterprise feature, governed by the Redpanda
// Community License Agreement included in the file licenses/RCL.md
//
// Use of this software requires a Redpanda Enterprise license.

package engine

import (
	"container/heap"
	"time"
)

type timerKind int

const (
	timerLease timerKind = iota
	timerRetry
	timerWait
	timerTimeout
)

type timer struct {
	at    time.Time
	runID string
	kind  timerKind
	// token pins lease timers to an attempt; toolCallID pins wait timers.
	token      string
	toolCallID string
}

type timerHeap []timer

func (h *timerHeap) Len() int           { return len(*h) }
func (h *timerHeap) Less(i, j int) bool { return (*h)[i].at.Before((*h)[j].at) }
func (h *timerHeap) Swap(i, j int)      { (*h)[i], (*h)[j] = (*h)[j], (*h)[i] }

func (h *timerHeap) Push(x any) {
	if t, ok := x.(timer); ok {
		*h = append(*h, t)
	}
}

func (h *timerHeap) Pop() any {
	old := *h
	n := len(old)
	t := old[n-1]
	*h = old[:n-1]

	return t
}

func (h *timerHeap) push(t timer) { heap.Push(h, t) }

func (h *timerHeap) pop() timer {
	t, _ := heap.Pop(h).(timer)

	return t
}

func (h *timerHeap) peek() (timer, bool) {
	if len(*h) == 0 {
		return timer{}, false
	}

	return (*h)[0], true
}

// due pops every timer at or before now.
func (h *timerHeap) due(now time.Time) []timer {
	var out []timer

	for {
		t, ok := h.peek()
		if !ok || t.at.After(now) {
			return out
		}

		out = append(out, h.pop())
	}
}
