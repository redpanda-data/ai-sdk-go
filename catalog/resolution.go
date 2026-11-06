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

package catalog

// ResolutionPolicy controls whether an offering's metadata may be used for
// unlisted version-stamped IDs. It does not change exact ID or alias lookup.
type ResolutionPolicy string

const (
	// ResolutionPolicyVersionStamp preserves legacy resolution of digit/date
	// suffixes after an offering ID or alias. It is the zero-value policy.
	ResolutionPolicyVersionStamp ResolutionPolicy = ""

	// ResolutionPolicyExact permits only the authored ID and exact aliases.
	// Use it when unlisted suffixed IDs may denote distinct deployments or
	// products whose metadata and prices have not been verified.
	ResolutionPolicyExact ResolutionPolicy = "exact"
)
