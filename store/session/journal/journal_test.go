// Copyright 2026 Redpanda Data, Inc.
//
// Licensed as a Redpanda Enterprise feature, governed by the Redpanda
// Community License Agreement included in the file licenses/RCL.md
//
// Use of this software requires a Redpanda Enterprise license.

package journal_test

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/testcontainers/testcontainers-go/modules/redpanda"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/store/session/journal"
)

func TestStore_AppendOnlyLog(t *testing.T) { //nolint:paralleltest // container
	if testing.Short() {
		t.Skip("skipping integration test")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	container, err := redpanda.Run(ctx, "redpandadata/redpanda:latest")
	require.NoError(t, err)

	defer func() { _ = container.Terminate(context.Background()) }()

	brokers, err := container.KafkaSeedBroker(ctx)
	require.NoError(t, err)

	store, err := journal.New(ctx, "sessions-journal", journal.Options{Brokers: []string{brokers}})
	require.NoError(t, err)

	defer store.Close()

	_, err = store.Load(ctx, "missing")
	require.ErrorIs(t, err, session.ErrNotFound)

	st := &session.State{
		ID:       "s1",
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("hello"))},
		Metadata: map[string]any{"user": "u1"},
	}
	require.NoError(t, store.Save(ctx, st))

	// Append-only save writes one record per new message.
	st.Messages = append(st.Messages, llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("hi")))
	require.NoError(t, store.Save(ctx, st))

	loaded, err := store.Load(ctx, "s1")
	require.NoError(t, err)
	assert.Len(t, loaded.Messages, 2)
	assert.Equal(t, "u1", loaded.Metadata["user"])
	assert.False(t, loaded.UpdatedAt.IsZero())

	hist := store.History("s1")
	require.Len(t, hist, 2, "snapshot for the first save, one message record for the delta")
	assert.Equal(t, "snapshot", hist[0].Type)
	assert.Equal(t, "message", hist[1].Type)
	assert.Equal(t, 1, hist[1].Seq)

	// Compaction-style rewrite (history pruned) forces a snapshot record but
	// the journal still holds the pruned message.
	st.Messages = st.Messages[1:]
	require.NoError(t, store.Save(ctx, st))

	loaded, err = store.Load(ctx, "s1")
	require.NoError(t, err)
	assert.Len(t, loaded.Messages, 1)
	assert.Equal(t, "hi", loaded.Messages[0].TextContent())

	hist = store.History("s1")
	require.Len(t, hist, 3)
	assert.Equal(t, "snapshot", hist[2].Type)
	require.Len(t, hist[2].Messages, 1)
	assert.Equal(t, "hi", hist[2].Messages[0].TextContent(), "the snapshot holds the rewritten history")
	require.Len(t, hist[0].Messages, 1)
	assert.Equal(t, "hello", hist[0].Messages[0].TextContent(), "the pruned message survives in the log")

	// A second store on the same topic sees the same state.
	other, err := journal.New(ctx, "sessions-journal", journal.Options{Brokers: []string{brokers}})
	require.NoError(t, err)

	defer other.Close()

	fromOther, err := other.Load(ctx, "s1")
	require.NoError(t, err)
	assert.Len(t, fromOther.Messages, 1)

	list, err := other.List(ctx, &session.ListRequest{PageSize: 10})
	require.NoError(t, err)
	require.Len(t, list.Sessions, 1)
	assert.Equal(t, "s1", list.Sessions[0].ID)

	require.NoError(t, store.Delete(ctx, "s1"))
	_, err = store.Load(ctx, "s1")
	require.ErrorIs(t, err, session.ErrNotFound)
}
