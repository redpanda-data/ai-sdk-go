package kvstore

import _ "embed"

// a2aTaskProtoSchema contains a minimal A2A Task proto definition for Schema Registry.
// This includes only the Task message and its dependencies, stripped of all RPC definitions
// and unused messages. This ensures Task is the first (index 0) message in the proto file,
// avoiding message index configuration complexity.
//
//go:embed proto/a2a_task.proto
var a2aTaskProtoSchema string
