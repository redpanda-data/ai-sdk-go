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

import _ "embed"

// a2aTaskProtoSchema contains a minimal A2A Task proto definition for Schema Registry.
// This includes only the Task message and its dependencies, stripped of all RPC definitions
// and unused messages. This ensures Task is the first (index 0) message in the proto file,
// avoiding message index configuration complexity.
//
//go:embed proto/a2a_task.proto
var a2aTaskProtoSchema string
