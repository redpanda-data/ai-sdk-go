// Copyright 2026 Redpanda Data, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package uimessagestream

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strconv"
	"time"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

// uiMessage is the wire shape of a UI message in the GET-history response —
// what useChat accepts as an initial messages array.
type uiMessage struct {
	ID    string        `json:"id"`
	Role  string        `json:"role"`
	Parts []messagePart `json:"parts"`
}

type chatHistoryResponse struct {
	ID        string      `json:"id"`
	UpdatedAt time.Time   `json:"updatedAt,omitzero"` //nolint:tagliatelle // response format is camelCase
	Messages  []uiMessage `json:"messages"`
}

type chatSummary struct {
	ID        string         `json:"id"`
	UpdatedAt time.Time      `json:"updatedAt,omitzero"` //nolint:tagliatelle // response format is camelCase
	Metadata  map[string]any `json:"metadata,omitempty"`
}

type chatListResponse struct {
	Chats         []chatSummary `json:"chats"`
	NextPageToken string        `json:"nextPageToken"` //nolint:tagliatelle // response format is camelCase
}

// projectUIMessages renders persisted model messages as UI messages, the
// inverse of the streaming path's chunk assembly: contiguous runs of assistant
// messages and their tool-result user messages merge into ONE assistant UI
// message with a step-start part per step — exactly the shape useChat builds
// from one streamed response — while genuine user messages split the runs.
//
// Tool error text is routed through onError, exactly as the streaming path
// does: a resumed page must not see server-side detail the live stream
// sanitized.
func projectUIMessages(msgs []llm.Message, onError ErrorMapper) []uiMessage {
	out := make([]uiMessage, 0, len(msgs))

	var (
		open    *uiMessage     // assistant UI message accumulating steps
		pending map[string]int // toolCallId -> index into open.Parts
	)

	flush := func() {
		if open == nil {
			return
		}

		// A tool request whose result never arrived (interrupted run) must not
		// render as an eternal spinner; close it the way the streaming path's
		// closePendingTools does.
		for _, idx := range pending {
			if open.Parts[idx].State == "input-available" {
				open.Parts[idx].State = "output-error"
				open.Parts[idx].ErrorText = onError(errors.New("tool call did not complete"))
			}
		}

		out = append(out, *open)
		open = nil
		pending = nil
	}

	for _, m := range msgs {
		switch {
		case m.Role == llm.RoleUser && isToolResultMessage(m):
			// Tool results belong to the open assistant message's current step;
			// they do not end the assistant turn.
			if open == nil {
				continue // orphan results (should not happen); skip
			}

			for _, p := range m.Content {
				tr, ok := p.(*llm.ToolResponsePart)
				if !ok {
					continue
				}

				idx, ok := pending[tr.ID]
				if !ok {
					continue // orphan response; mirror the streaming path's skip
				}

				if tr.IsError {
					open.Parts[idx].State = "output-error"
					open.Parts[idx].ErrorText = onError(errors.New(toolErrorText(tr.Result)))
				} else {
					open.Parts[idx].State = "output-available"
					open.Parts[idx].Output = tr.Result
				}

				delete(pending, tr.ID)
			}

		case m.Role == llm.RoleUser:
			flush()

			text := m.TextContent()
			if text == "" {
				continue
			}

			out = append(out, uiMessage{Role: "user", Parts: []messagePart{{Type: "text", Text: text}}})

		case m.Role == llm.RoleAssistant:
			if open == nil {
				open = &uiMessage{Role: "assistant"}
				pending = make(map[string]int)
			}

			open.Parts = append(open.Parts, messagePart{Type: "step-start"})

			for _, p := range m.Content {
				switch part := p.(type) {
				case *llm.TextPart:
					open.Parts = append(open.Parts, messagePart{Type: "text", Text: part.Text, State: "done"})
				case *llm.ReasoningPart:
					open.Parts = append(open.Parts, messagePart{Type: "reasoning", Text: part.Text, State: "done"})
				case *llm.ToolRequestPart:
					pending[part.ID] = len(open.Parts)
					open.Parts = append(open.Parts, messagePart{
						Type: "dynamic-tool", ToolName: part.Name, ToolCallID: part.ID,
						Input: part.Arguments, State: "input-available",
					})
				}
			}
		}
		// System messages are never persisted (the agent owns its prompt); any
		// other role is skipped.
	}

	flush()

	// Synthetic stable ids: sessions persist model messages, which carry none.
	for i := range out {
		out[i].ID = fmt.Sprintf("msg-%d", i)
	}

	return out
}

// isToolResultMessage reports whether the message carries only tool responses —
// the user-role envelope llmagent persists tool results in.
func isToolResultMessage(m llm.Message) bool {
	if len(m.Content) == 0 {
		return false
	}

	for _, p := range m.Content {
		if _, ok := p.(*llm.ToolResponsePart); !ok {
			return false
		}
	}

	return true
}

func (h *chatHandler) handleGet(w http.ResponseWriter, r *http.Request) {
	chatID := r.PathValue("id")

	key, err := h.resolveKey(r, chatID)
	if err != nil {
		http.Error(w, "forbidden", http.StatusForbidden)
		return
	}

	sess, err := h.store.Load(r.Context(), key)
	if errors.Is(err, session.ErrNotFound) {
		http.Error(w, "chat not found", http.StatusNotFound)
		return
	}

	if err != nil {
		h.cfg.logger.Error("failed to load session", "sessionId", key, "error", err)
		http.Error(w, "failed to load session", http.StatusInternalServerError)

		return
	}

	// Respond with the client-visible chat id, not the storage key.
	writeJSON(w, chatHistoryResponse{
		ID:        chatID,
		UpdatedAt: sess.UpdatedAt,
		Messages:  projectUIMessages(sess.Messages, h.cfg.onError),
	})
}

func (h *chatHandler) handleDelete(w http.ResponseWriter, r *http.Request) {
	key, err := h.resolveKey(r, r.PathValue("id"))
	if err != nil {
		http.Error(w, "forbidden", http.StatusForbidden)
		return
	}

	// Take the keyed lock so a concurrent run's saves cannot resurrect the
	// session mid-delete (within this process).
	unlock := h.locks.lock(key)
	defer unlock()

	if err := h.store.Delete(r.Context(), key); err != nil {
		h.cfg.logger.Error("failed to delete session", "sessionId", key, "error", err)
		http.Error(w, "failed to delete session", http.StatusInternalServerError)

		return
	}

	w.WriteHeader(http.StatusNoContent)
}

func (h *chatHandler) handleList(w http.ResponseWriter, r *http.Request) {
	// With a custom session key, storage keys are tenant-scoped and List would
	// enumerate every tenant's chats; listing must live behind the app's own
	// authorization there.
	if h.cfg.sessionKey != nil {
		http.Error(w, "listing is disabled when a session key function is configured; expose an app-level list API", http.StatusNotImplemented)
		return
	}

	req := &session.ListRequest{PageToken: r.URL.Query().Get("pageToken")}

	if s := r.URL.Query().Get("pageSize"); s != "" {
		n, err := strconv.ParseInt(s, 10, 32)
		if err != nil || n < 0 {
			http.Error(w, "invalid pageSize", http.StatusBadRequest)
			return
		}

		req.PageSize = int32(n)
	}

	resp, err := h.store.List(r.Context(), req)
	if errors.Is(err, session.ErrListNotSupported) {
		http.Error(w, "the session store does not support listing", http.StatusNotImplemented)
		return
	}

	if err != nil {
		h.cfg.logger.Error("failed to list sessions", "error", err)
		http.Error(w, "failed to list sessions", http.StatusInternalServerError)

		return
	}

	chats := make([]chatSummary, 0, len(resp.Sessions))
	for _, s := range resp.Sessions {
		chats = append(chats, chatSummary{ID: s.ID, UpdatedAt: s.UpdatedAt, Metadata: s.Metadata})
	}

	writeJSON(w, chatListResponse{Chats: chats, NextPageToken: resp.NextPageToken})
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")

	// A failure here means the client is gone or the response is already
	// committed; there is no useful recovery.
	if err := json.NewEncoder(w).Encode(v); err != nil {
		slog.Debug("failed to encode response", "error", err)
	}
}
