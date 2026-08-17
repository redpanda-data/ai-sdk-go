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

// Modality is one input or output content kind a model accepts or
// produces. A typed list extends without touching existing entries,
// where capability booleans (SupportsVision, SupportsAudio, ...) require
// a new field per modality.
type Modality string

const (
	ModalityText     Modality = "text"
	ModalityImage    Modality = "image"
	ModalityAudio    Modality = "audio"
	ModalityVideo    Modality = "video"
	ModalityDocument Modality = "document" // PDF and similar document inputs
)

// Modalities lists what an offering consumes and produces. Empty slices
// mean text-only; New normalizes them so resolved offerings always carry
// explicit lists.
type Modalities struct {
	Input  []Modality
	Output []Modality
}
