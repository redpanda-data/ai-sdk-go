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

package session_test

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/store/session"
)

// saveSpaced saves the given IDs in order, spacing the writes so each gets a
// strictly greater UpdatedAt (the store stamps time.Now()). Returns the store.
func saveSpaced(t *testing.T, ids ...string) *session.InMemoryStore {
	t.Helper()

	store := session.NewInMemoryStore()
	for _, id := range ids {
		require.NoError(t, store.Save(context.Background(), &session.State{ID: id}))
		time.Sleep(2 * time.Millisecond)
	}

	return store
}

func TestInMemoryStore_SaveStampsUpdatedAt(t *testing.T) {
	t.Parallel()

	store := session.NewInMemoryStore()

	before := time.Now()
	// The caller's UpdatedAt is ignored; the store stamps the write time.
	require.NoError(t, store.Save(context.Background(), &session.State{
		ID:        "s1",
		UpdatedAt: time.Unix(0, 0),
	}))

	after := time.Now()

	loaded, err := store.Load(context.Background(), "s1")
	require.NoError(t, err)
	require.False(t, loaded.UpdatedAt.Before(before), "UpdatedAt should be >= before")
	require.False(t, loaded.UpdatedAt.After(after), "UpdatedAt should be <= after")
}

func TestInMemoryStore_ListOrdersByUpdatedAtDescThenID(t *testing.T) {
	t.Parallel()
	store := saveSpaced(t, "alpha", "bravo", "charlie")

	resp, err := store.List(context.Background(), &session.ListRequest{})
	require.NoError(t, err)
	require.Empty(t, resp.NextPageToken)

	ids := summaryIDs(resp.Sessions)
	// Newest write first.
	require.Equal(t, []string{"charlie", "bravo", "alpha"}, ids)
}

func TestInMemoryStore_ListPaginates(t *testing.T) {
	t.Parallel()
	store := saveSpaced(t, "a", "b", "c", "d", "e")

	var got []string

	token := ""
	for range 10 {
		resp, err := store.List(context.Background(), &session.ListRequest{PageSize: 2, PageToken: token})
		require.NoError(t, err)
		require.LessOrEqual(t, len(resp.Sessions), 2)
		got = append(got, summaryIDs(resp.Sessions)...)

		token = resp.NextPageToken
		if token == "" {
			break
		}
	}

	// Every session shows up exactly once, newest first, across pages.
	require.Equal(t, []string{"e", "d", "c", "b", "a"}, got)
}

func TestInMemoryStore_ListEmpty(t *testing.T) {
	t.Parallel()

	store := session.NewInMemoryStore()

	resp, err := store.List(context.Background(), &session.ListRequest{})
	require.NoError(t, err)
	require.Empty(t, resp.Sessions)
	require.Empty(t, resp.NextPageToken)
}

func TestInMemoryStore_ListNilRequest(t *testing.T) {
	t.Parallel()
	store := saveSpaced(t, "only")

	resp, err := store.List(context.Background(), nil)
	require.NoError(t, err)
	require.Equal(t, []string{"only"}, summaryIDs(resp.Sessions))
}

func TestInMemoryStore_ListRejectsBadPageToken(t *testing.T) {
	t.Parallel()
	store := saveSpaced(t, "a")

	_, err := store.List(context.Background(), &session.ListRequest{PageToken: "not-a-number"})
	require.Error(t, err)
}

func TestInMemoryStore_ListOmitsMessagesKeepsMetadata(t *testing.T) {
	t.Parallel()

	store := session.NewInMemoryStore()
	require.NoError(t, store.Save(context.Background(), &session.State{
		ID:       "s1",
		Metadata: map[string]any{"title": "hello"},
	}))

	resp, err := store.List(context.Background(), &session.ListRequest{})
	require.NoError(t, err)
	require.Len(t, resp.Sessions, 1)
	require.Equal(t, "hello", resp.Sessions[0].Metadata["title"])

	// Mutating the returned metadata must not corrupt stored state.
	resp.Sessions[0].Metadata["title"] = "mutated"
	loaded, err := store.Load(context.Background(), "s1")
	require.NoError(t, err)
	require.Equal(t, "hello", loaded.Metadata["title"])
}

func TestState_CloneCopiesUpdatedAt(t *testing.T) {
	t.Parallel()

	ts := time.Date(2026, 7, 1, 10, 0, 0, 0, time.UTC)
	original := &session.State{ID: "s1", UpdatedAt: ts}

	clone := original.Clone()
	require.Equal(t, ts, clone.UpdatedAt)
}

func summaryIDs(summaries []session.Summary) []string {
	ids := make([]string, len(summaries))
	for i, s := range summaries {
		ids[i] = s.ID
	}

	return ids
}
