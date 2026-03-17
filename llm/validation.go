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

package llm

// ConfigValidator defines the interface for provider configuration validation.
// All provider configurations should implement this interface to ensure
// consistent validation behavior across the SDK.
type ConfigValidator interface {
	// Validate checks if the configuration is valid and returns an error if not.
	// This should check all required fields and parameter constraints.
	Validate() error

	// ApplyDefaults sets default values for optional configuration parameters.
	// This should be called before validation to ensure consistent behavior.
	ApplyDefaults()
}
