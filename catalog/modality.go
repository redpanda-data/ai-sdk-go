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

// Modality is one content kind an offering's API accepts in requests or
// produces in responses. Wire-level, not model architecture: no model
// has a "document" modality — providers serve PDF input as vision (each
// page rasterized to an image, plus extracted text) — but whether an
// offering accepts document parts is a real per-host fact, and not
// derivable from vision (Bedrock's Converse document block covers
// text-only models via extraction). A typed list extends without
// touching existing entries, where capability booleans would need a new
// field per modality.
type Modality string

const (
	ModalityText  Modality = "text"
	ModalityImage Modality = "image"
	ModalityAudio Modality = "audio"
	ModalityVideo Modality = "video"
	// ModalityDocument: the offering accepts document parts (PDF and
	// similar) natively; the provider handles parsing server-side.
	ModalityDocument Modality = "document"
)

// Modalities lists what an offering consumes and produces. Empty slices
// mean text-only; New normalizes them so resolved offerings always carry
// explicit lists.
//
// Modalities describe the PROVIDER offering, not this SDK's transport:
// llm.Part currently expresses text, tool, and reasoning parts only, so a
// catalogued modality is not proof that a request carrying it can be
// built yet. Routing decisions must not treat it as one.
type Modalities struct {
	Input  []Modality
	Output []Modality
}
