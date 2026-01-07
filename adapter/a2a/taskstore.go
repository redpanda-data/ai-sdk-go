package a2a

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/redpanda-data/common-go/kvstore"
	"github.com/redpanda-data/common-go/kvstore/memdb"
)

// Compile-time interface check.
var _ a2asrv.TaskStore = (*KVTaskStore)(nil)

// KVTaskStore provides A2A task storage backed by Kafka via kvstore.
// Writes are persisted to Kafka and block until visible in reads.
// Multiple instances share state through Kafka topic consumption.
//
// Tasks are stored with task_id as the key (enabling Kafka log compaction).
// An in-memory sorted index maintains time-ordering for efficient List() operations.
type KVTaskStore struct {
	client *kvstore.ResourceClient[*a2a.Task]

	mu     sync.RWMutex
	index  map[a2a.TaskID]string // task_id -> sortKey (for tracking/removal on update)
	sorted []string              // sortKeys in descending time order
}

// NewKVTaskStore creates a new A2A task store backed by Kafka.
//
// The topic parameter specifies the Kafka topic for task storage.
// Use kvstore.WithBrokers, kvstore.WithReplicationFactor, kvstore.WithKafkaOptions,
// and kvstore.WithLogger to configure the underlying client.
//
// The context is used only for initialization (topic creation, metadata fetches,
// and bootstrap sync). It is not retained after NewKVTaskStore returns.
//
// NewKVTaskStore blocks until the consumer has caught up to the current high watermark,
// ensuring reads are consistent immediately after creation.
//
// The caller MUST call Close() to release resources when done.
func NewKVTaskStore(ctx context.Context, topic string, opts ...kvstore.ClientOption) (*KVTaskStore, error) {
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
	onSet := kvstore.WrapOnSet(serde, func(_ []byte, task *a2a.Task) {
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
	allOpts := append([]kvstore.ClientOption{
		kvstore.WithOnSetHook(onSet),
		kvstore.WithOnDeleteHook(onDelete),
	}, opts...)

	client, err := kvstore.NewClient(ctx, topic, storage, allOpts...)
	if err != nil {
		return nil, err
	}

	store.client = kvstore.NewResourceClient(client, serde)

	return store, nil
}

// Save stores a task.
// Blocks until the write is visible in this client's reads.
func (s *KVTaskStore) Save(ctx context.Context, task *a2a.Task) error {
	// Key is just task_id - enables Kafka log compaction
	return s.client.Put(ctx, []byte(task.ID), task)
}

// Get retrieves a task by ID.
// Returns a2a.ErrTaskNotFound if the task does not exist.
func (s *KVTaskStore) Get(ctx context.Context, taskID a2a.TaskID) (*a2a.Task, error) {
	task, err := s.client.Get(ctx, []byte(taskID))
	if errors.Is(err, kvstore.ErrNotFound) {
		return nil, a2a.ErrTaskNotFound
	}

	if err != nil {
		return nil, err
	}

	return task, nil
}

// List retrieves tasks matching the criteria in the request.
// Tasks are returned in descending order by last update time.
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
		for i, sk := range sortedCopy {
			if sk == req.PageToken {
				startIdx = i + 1 // Start after the token
				break
			}
		}
	}

	var tasks []*a2a.Task
	var nextPageToken string

	for i := startIdx; i < len(sortedCopy); i++ {
		sortKey := sortedCopy[i]
		taskID := taskIDFromSortKey(sortKey)

		task, err := s.client.Get(ctx, []byte(taskID))
		if err != nil {
			if errors.Is(err, kvstore.ErrNotFound) {
				continue // Task was deleted, skip
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
			nextPageToken = sortKey
			break
		}
	}

	return &a2a.ListTasksResponse{
		Tasks:         tasks,
		TotalSize:     len(tasks),
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
func makeSortKey(timestamp *time.Time, taskID a2a.TaskID) string {
	var ts int64

	if timestamp != nil {
		ts = timestamp.UnixNano()
	}

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
