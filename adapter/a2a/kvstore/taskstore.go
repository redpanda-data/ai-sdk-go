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

package kvstore

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2apb"
	"github.com/a2aproject/a2a-go/a2apb/pbconv"
	"github.com/a2aproject/a2a-go/a2asrv"
	commonkvstore "github.com/redpanda-data/common-go/kvstore"
	"github.com/redpanda-data/common-go/kvstore/memdb"
	"github.com/twmb/franz-go/pkg/sr"
)

// Compile-time interface check.
var _ a2asrv.TaskStore = (*KVTaskStore)(nil)

// pageToken is the internal structure for pagination tokens.
// Tokens are JSON marshaled and base64 encoded to make them opaque to clients.
type pageToken struct {
	SortKey string `json:"s"` // The sortKey to continue from
}

// encodePageToken encodes a sortKey into an opaque base64 page token.
func encodePageToken(sortKey string) (string, error) {
	if sortKey == "" {
		return "", nil
	}

	token := pageToken{SortKey: sortKey}

	jsonData, err := json.Marshal(token)
	if err != nil {
		return "", fmt.Errorf("failed to marshal page token: %w", err)
	}

	return base64.URLEncoding.EncodeToString(jsonData), nil
}

// decodePageToken decodes a base64 page token into a sortKey.
// Returns an error if the token is invalid or malformed.
func decodePageToken(encoded string) (string, error) {
	if encoded == "" {
		return "", nil
	}

	jsonData, err := base64.URLEncoding.DecodeString(encoded)
	if err != nil {
		return "", fmt.Errorf("invalid page token: not valid base64: %w", err)
	}

	var token pageToken
	if err := json.Unmarshal(jsonData, &token); err != nil {
		return "", fmt.Errorf("invalid page token: malformed structure: %w", err)
	}

	if token.SortKey == "" {
		return "", errors.New("invalid page token: missing sortKey")
	}

	return token.SortKey, nil
}

// KVTaskStore provides A2A task storage backed by Kafka via commonkvstore.
// Writes are persisted to Kafka and block until visible in reads.
// Multiple instances share state through Kafka topic consumption.
//
// Tasks are stored with task_id as the key (enabling Kafka log compaction).
// An in-memory sorted index maintains time-ordering for efficient List() operations.
type KVTaskStore struct {
	client *commonkvstore.ResourceClient[*a2a.Task]

	mu     sync.RWMutex
	index  map[a2a.TaskID]string // task_id -> sortKey (for tracking/removal on update)
	sorted []string              // sortKeys in descending time order
}

// NewKVTaskStore creates a new A2A task store backed by Kafka.
//
// The topic parameter specifies the Kafka topic for task storage.
// Use commonkvstore.WithBrokers, commonkvstore.WithReplicationFactor, commonkvstore.WithKafkaOptions,
// and commonkvstore.WithLogger to configure the underlying client.
//
// The context is used only for initialization (topic creation, metadata fetches,
// and bootstrap sync). It is not retained after NewKVTaskStore returns.
//
// NewKVTaskStore blocks until the consumer has caught up to the current high watermark,
// ensuring reads are consistent immediately after creation.
//
// The caller MUST call Close() to release resources when done.
func NewKVTaskStore(ctx context.Context, topic string, opts ...commonkvstore.ClientOption) (*KVTaskStore, error) {
	storage, err := memdb.New()
	if err != nil {
		return nil, err
	}

	store := &KVTaskStore{
		index:  make(map[a2a.TaskID]string),
		sorted: nil,
	}

	serde := &taskSerde{}

	// Set up hooks to maintain the in-memory sorted index
	onSet := commonkvstore.WrapOnSet(serde, func(_ []byte, task *a2a.Task) {
		store.mu.Lock()
		defer store.mu.Unlock()

		newSortKey := makeSortKey(task.Status.Timestamp, task.ID)

		// Remove old sortKey if task already exists with different timestamp
		if oldSortKey, exists := store.index[task.ID]; exists && oldSortKey != newSortKey {
			store.removeSortKeyLocked(oldSortKey)
		}

		// Update index and insert new sortKey if not already present
		if store.index[task.ID] != newSortKey {
			store.index[task.ID] = newSortKey
			store.insertSortKeyLocked(newSortKey)
		}
	})

	onDelete := func(key []byte) {
		taskID := a2a.TaskID(key)

		store.mu.Lock()
		defer store.mu.Unlock()

		if sortKey, exists := store.index[taskID]; exists {
			store.removeSortKeyLocked(sortKey)
			delete(store.index, taskID)
		}
	}

	// Prepend hook options
	allOpts := append([]commonkvstore.ClientOption{
		commonkvstore.WithOnSetHook(onSet),
		commonkvstore.WithOnDeleteHook(onDelete),
	}, opts...)

	client, err := commonkvstore.NewClient(ctx, topic, storage, allOpts...)
	if err != nil {
		return nil, err
	}

	store.client = commonkvstore.NewResourceClient(client, serde)

	return store, nil
}

// NewKVTaskStoreWithSchemaRegistry creates a new A2A task store backed by Kafka with Schema Registry integration.
//
// This constructor uses protobuf serialization with Confluent Schema Registry wire format.
// The schema is automatically registered with the Schema Registry on first write.
//
// Parameters:
//   - ctx: Context for initialization only (not retained)
//   - topic: Kafka topic for task storage
//   - srClient: Schema Registry client for schema registration
//   - opts: Additional kvstore options (brokers, replication factor, etc.)
//
// All agents share the schema subject "redpanda-a2a-task-value" to avoid Schema Registry
// accumulation of identical per-agent schemas. Each agent still uses its own topic for data isolation.
//
// Example:
//
//	srClient, _ := sr.NewClient(sr.URLs("http://localhost:8081"))
//	store, err := a2a.NewKVTaskStoreWithSchemaRegistry(
//	    ctx,
//	    "a2a-tasks",
//	    srClient,
//	    commonkvstore.WithBrokers("localhost:9092"),
//	)
func NewKVTaskStoreWithSchemaRegistry(
	ctx context.Context,
	topic string,
	srClient *sr.Client,
	opts ...commonkvstore.ClientOption,
) (*KVTaskStore, error) {
	storage, err := memdb.New()
	if err != nil {
		return nil, err
	}

	store := &KVTaskStore{
		index:  make(map[a2a.TaskID]string),
		sorted: nil,
	}

	// Create protobuf serde with Schema Registry support
	// Uses shared subject to avoid per-agent schema accumulation in Schema Registry.
	// All agents use identical a2apb.Task schema - no need for per-topic subjects.
	// Uses a2a_task.proto which contains only Task and its dependencies,
	// ensuring Task is at message index 0.
	subject := "redpanda-a2a-task-value"

	srSerde, err := commonkvstore.Proto(
		func() *a2apb.Task { return &a2apb.Task{} },
		commonkvstore.WithSchemaRegistry(
			srClient,
			subject,
			a2aTaskProtoSchema,
		),
	)
	if err != nil {
		return nil, fmt.Errorf("create proto serde: %w", err)
	}

	// Wrap with conversion layer (a2a.Task ↔ a2apb.Task)
	serde := &taskSRSerde{inner: srSerde}

	// Set up hooks to maintain the in-memory sorted index
	onSet := commonkvstore.WrapOnSet(serde, func(_ []byte, task *a2a.Task) {
		store.mu.Lock()
		defer store.mu.Unlock()

		newSortKey := makeSortKey(task.Status.Timestamp, task.ID)

		// Remove old sortKey if task already exists with different timestamp
		if oldSortKey, exists := store.index[task.ID]; exists && oldSortKey != newSortKey {
			store.removeSortKeyLocked(oldSortKey)
		}

		// Update index and insert new sortKey if not already present
		if store.index[task.ID] != newSortKey {
			store.index[task.ID] = newSortKey
			store.insertSortKeyLocked(newSortKey)
		}
	})

	onDelete := func(key []byte) {
		taskID := a2a.TaskID(key)

		store.mu.Lock()
		defer store.mu.Unlock()

		if sortKey, exists := store.index[taskID]; exists {
			store.removeSortKeyLocked(sortKey)
			delete(store.index, taskID)
		}
	}

	// Prepend hook options
	allOpts := append([]commonkvstore.ClientOption{
		commonkvstore.WithOnSetHook(onSet),
		commonkvstore.WithOnDeleteHook(onDelete),
	}, opts...)

	client, err := commonkvstore.NewClient(ctx, topic, storage, allOpts...)
	if err != nil {
		return nil, err
	}

	store.client = commonkvstore.NewResourceClient(client, serde)

	return store, nil
}

// Save stores a task.
// Blocks until the write is visible in this client's reads.
//
// Note: The event and prev parameters are accepted for interface compatibility
// but are not used. The kvstore doesn't support optimistic concurrency control.
// The returned TaskVersion is always 0.
func (s *KVTaskStore) Save(ctx context.Context, task *a2a.Task, _ a2a.Event, _ a2a.TaskVersion) (a2a.TaskVersion, error) {
	// Key is just task_id - enables Kafka log compaction
	err := s.client.Put(ctx, []byte(task.ID), task)
	return 0, err
}

// Get retrieves a task by ID.
// Returns a2a.ErrTaskNotFound if the task does not exist.
//
// Note: The returned TaskVersion is always 0 since kvstore doesn't track versions.
func (s *KVTaskStore) Get(ctx context.Context, taskID a2a.TaskID) (*a2a.Task, a2a.TaskVersion, error) {
	task, err := s.client.Get(ctx, []byte(taskID))
	if errors.Is(err, commonkvstore.ErrNotFound) {
		return nil, 0, a2a.ErrTaskNotFound
	}

	if err != nil {
		return nil, 0, err
	}

	return task, 0, nil
}

// List retrieves tasks matching the criteria in the request.
// Tasks are returned in descending order by last update time.
//
// # Consistency Model
//
// List provides snapshot-at-copy consistency: it copies the sorted index under lock,
// then releases the lock and reads task data. This creates a race condition where:
//
//  1. Tasks in the snapshot may have been deleted before we read them
//     (handled by skipping ErrNotFound entries)
//  2. Tasks in the snapshot may have been updated with newer timestamps/state
//     (the returned data reflects the current state, not snapshot-time state)
//  3. Newly added tasks after the snapshot won't appear in results
//  4. The sort order reflects snapshot-time, but task data is current-time
//
// This is acceptable for the A2A list operation which doesn't require strict
// snapshot isolation. The alternative (holding RLock during all kvstore reads)
// would block all concurrent writes for the entire List operation.
func (s *KVTaskStore) List(ctx context.Context, req *a2a.ListTasksRequest) (*a2a.ListTasksResponse, error) {
	pageSize := req.PageSize
	if pageSize <= 0 {
		pageSize = 50
	}

	if pageSize > 100 {
		pageSize = 100
	}

	s.mu.RLock()
	sortedCopy := make([]string, len(s.sorted))
	copy(sortedCopy, s.sorted)
	s.mu.RUnlock()

	// Find starting position for pagination
	startIdx := 0

	if req.PageToken != "" {
		// Decode and validate the page token
		sortKey, err := decodePageToken(req.PageToken)
		if err != nil {
			return nil, fmt.Errorf("invalid page token: %w", err)
		}

		// Use binary search since sortedCopy is sorted (O(log n) vs O(n)).
		// If found, idx is the exact position of the token's task.
		// If not found (task was deleted between pages), idx is where it would be inserted.
		// This insertion point is correct because:
		// - Sort keys are unique (timestamp:taskID format)
		// - The insertion point is the first key > deleted key
		// - This is exactly where the next page should start
		// Thus pagination remains consistent even when tasks are deleted between requests.
		startIdx, _ = slices.BinarySearch(sortedCopy, sortKey)
	}

	var tasks []*a2a.Task
	var nextPageToken string

	for i := startIdx; i < len(sortedCopy); i++ {
		sortKey := sortedCopy[i]
		taskID := taskIDFromSortKey(sortKey)

		task, err := s.client.Get(ctx, []byte(taskID))
		if err != nil {
			if errors.Is(err, commonkvstore.ErrNotFound) {
				// Task was deleted between snapshot and read (see consistency model in function doc)
				continue
			}

			return nil, err
		}

		// Filter by ContextID
		if req.ContextID != "" && task.ContextID != req.ContextID {
			continue
		}

		// Filter by Status
		if req.Status != "" && task.Status.State != req.Status {
			continue
		}

		// Filter by LastUpdatedAfter
		if req.LastUpdatedAfter != nil && task.Status.Timestamp != nil {
			if task.Status.Timestamp.Before(*req.LastUpdatedAfter) {
				continue
			}
		}

		// Clone task to avoid modifying stored data
		taskCopy := *task
		taskCopy.History = append([]*a2a.Message(nil), task.History...)
		taskCopy.Artifacts = append([]*a2a.Artifact(nil), task.Artifacts...)

		// Trim history if requested
		if req.HistoryLength > 0 && len(taskCopy.History) > req.HistoryLength {
			taskCopy.History = taskCopy.History[len(taskCopy.History)-req.HistoryLength:]
		}

		// Exclude artifacts if not requested
		if !req.IncludeArtifacts {
			taskCopy.Artifacts = nil
		}

		tasks = append(tasks, &taskCopy)

		if len(tasks) >= pageSize {
			// Set token to the next item (first item of next page)
			if i+1 < len(sortedCopy) {
				encoded, err := encodePageToken(sortedCopy[i+1])
				if err != nil {
					return nil, fmt.Errorf("failed to encode page token: %w", err)
				}

				nextPageToken = encoded
			}

			break
		}
	}

	return &a2a.ListTasksResponse{
		Tasks: tasks,
		// TotalSize is the UNFILTERED total task count in the store at snapshot time.
		// This does NOT reflect any ContextID, Status, or LastUpdatedAfter filters.
		// Computing filtered count would require iterating all tasks which is expensive.
		// Consumers should use len(Tasks) for current page count and NextPageToken
		// to determine if more results exist.
		TotalSize:     len(sortedCopy),
		PageSize:      pageSize,
		NextPageToken: nextPageToken,
	}, nil
}

// Close shuts down the kvstore client and releases resources.
func (s *KVTaskStore) Close() error {
	return s.client.Raw().Close()
}

// insertSortKeyLocked inserts a sortKey in sorted order (descending by time).
// Caller must hold mu.
func (s *KVTaskStore) insertSortKeyLocked(sortKey string) {
	// Binary search for insertion point (sorted ascending, which gives descending time)
	idx, found := slices.BinarySearch(s.sorted, sortKey)

	if found {
		return // Already exists
	}

	s.sorted = slices.Insert(s.sorted, idx, sortKey)
}

// removeSortKeyLocked removes a sortKey from the sorted slice.
// Caller must hold mu.
func (s *KVTaskStore) removeSortKeyLocked(sortKey string) {
	idx, found := slices.BinarySearch(s.sorted, sortKey)

	if found {
		s.sorted = slices.Delete(s.sorted, idx, idx+1)
	}
}

// makeSortKey creates a sort key for time-ordered indexing.
// Format: {inverted_timestamp}:{task_id}
// Inverted timestamp ensures descending order with ascending key scan.
//
// Note: This assumes timestamps are non-negative (post-1970 Unix epoch).
// Negative timestamps would cause inverted values > MaxInt64, which would
// break sort ordering. This is acceptable since task timestamps should
// always be recent/current time.
func makeSortKey(timestamp *time.Time, taskID a2a.TaskID) string {
	var ts int64

	if timestamp != nil {
		ts = timestamp.UnixNano()
	}

	// Invert timestamp so ascending sort gives descending time order.
	// MaxInt64 - ts works for all reasonable timestamps (post-1970).
	inverted := math.MaxInt64 - ts

	return fmt.Sprintf("%020d:%s", inverted, taskID)
}

// taskIDFromSortKey extracts the task ID from a sort key.
func taskIDFromSortKey(sortKey string) a2a.TaskID {
	idx := strings.Index(sortKey, ":")

	if idx == -1 {
		return a2a.TaskID(sortKey)
	}

	return a2a.TaskID(sortKey[idx+1:])
}

// taskSerde handles JSON serialization of a2a.Task.
type taskSerde struct{}

func (*taskSerde) Serialize(t *a2a.Task) ([]byte, error) {
	return json.Marshal(t)
}

func (*taskSerde) Deserialize(b []byte) (*a2a.Task, error) {
	var t a2a.Task
	if err := json.Unmarshal(b, &t); err != nil {
		return nil, err
	}

	return &t, nil
}

// taskSRSerde combines conversion and Schema Registry serialization for A2A tasks.
// It converts a2a.Task ↔ a2apb.Task using the pbconv package, then delegates to SR serde.
type taskSRSerde struct {
	inner commonkvstore.Serde[*a2apb.Task]
}

// Serialize converts a2a.Task to proto, then serializes with SR wire format.
func (s *taskSRSerde) Serialize(task *a2a.Task) ([]byte, error) {
	pb, err := pbconv.ToProtoTask(task)
	if err != nil {
		return nil, fmt.Errorf("convert to proto: %w", err)
	}

	return s.inner.Serialize(pb)
}

// Deserialize decodes SR wire format, unmarshals proto, then converts to a2a.Task.
func (s *taskSRSerde) Deserialize(b []byte) (*a2a.Task, error) {
	pb, err := s.inner.Deserialize(b)
	if err != nil {
		return nil, fmt.Errorf("deserialize proto: %w", err)
	}

	return pbconv.FromProtoTask(pb)
}
