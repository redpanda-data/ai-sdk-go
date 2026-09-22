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

// Publisher is the vendor that published a model, independent of the
// provider serving it. Values are lowercase vendor names. The Publisher*
// consts name the ones in use across the catalogs today, and a name that
// is not among them is still accepted.
type Publisher string

const (
	PublisherAmazon    Publisher = "amazon"
	PublisherAnthropic Publisher = "anthropic"
	PublisherGoogle    Publisher = "google"
	PublisherMeta      Publisher = "meta"
	PublisherMistral   Publisher = "mistral"
	PublisherOpenAI    Publisher = "openai"
)
