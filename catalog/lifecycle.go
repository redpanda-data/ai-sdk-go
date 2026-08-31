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

import "time"

// Stage is a lifecycle stage of one provider's offering.
//
// Only StagePreview and StageGA may be authored on an Entry. Deprecation
// and retirement are derived from the Lifecycle dates by View, so a date
// passing can never leave a stale authored flag behind — New rejects
// authored StageDeprecated and StageRetired.
type Stage string

const (
	// StagePreview marks an offering the provider labels preview, beta,
	// or experimental.
	StagePreview Stage = "preview"

	// StageGA marks a generally available offering. It is the default:
	// an empty authored Stage normalizes to StageGA.
	StageGA Stage = "ga"

	// StageDeprecated is derived: Lifecycle.Deprecated is set and has
	// arrived at the View's date.
	StageDeprecated Stage = "deprecated"

	// StageRetired is derived: Lifecycle.Retires is set and has arrived
	// at the View's date. Requests to retired offerings fail at the
	// provider; the catalog keeps the entry so historical usage stays
	// priceable and the failure stays explainable.
	StageRetired Stage = "retired"
)

// Lifecycle is the schedule of ONE provider's offering of a model.
//
// It is deliberately per offering, not per model: Anthropic documents
// that partner platforms (Amazon Bedrock, Google Cloud) set their own
// retirement schedules, so the same model can be GA on one host and
// retired on another.
//
// All dates are date-only time.Time values (midnight UTC, construct
// with MustDate); New rejects anything finer. Zero means "not set".
type Lifecycle struct {
	// Stage is the authored stage: StagePreview or StageGA (empty
	// normalizes to StageGA). Later stages are derived from the dates
	// below.
	Stage Stage

	// Available is when this provider started serving the offering.
	// Zero means "available, exact date unknown" — catalog membership
	// already implies availability, and a missing date must not force
	// authors to invent one.
	Available time.Time

	// Deprecated is the date the provider announced or applied
	// deprecation. Zero means not deprecated.
	Deprecated time.Time

	// Retires is the provider's EXACT announced shutdown date. The
	// boundary is inclusive: the offering is retired ON this date,
	// matching provider wording ("requests to retired models will
	// fail"). Zero means no shutdown is scheduled. A published "not
	// sooner than" floor is a lower bound, not a shutdown date — leave
	// Retires unset until the provider announces an exact date.
	Retires time.Time

	// ReplacedBy is the provider's announced recommended replacement,
	// as an offering ID in the same catalog. New validates it resolves.
	// Derived succession (Successor) is computed from Series ordering
	// and never authored.
	ReplacedBy string
}
