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

package llmagent

// minOutputReserve is the floor for the answer-room reservation.
const minOutputReserve = 4096

// contextBudget derives every compaction line from the model's context window.
//
//	usable    = window - reserve   // reserve leaves room for the answer
//	trigger   = 0.8 x usable       // WHEN to compact (default)
//	target    = 0.6 x usable       // HOW FAR to reduce (default)
//	hardLimit = usable             // never knowingly exceed
//
// hardLimit is a safety boundary, not a compaction goal: a request that
// cannot reach target but fits under hardLimit is sent, not rejected.
type contextBudget struct {
	window    int
	reserve   int
	usable    int
	trigger   int
	target    int
	hardLimit int
}

// newContextBudget derives the context budget from the model's constraints and the resolved
// compaction config. window must be > 0 (validated at construction).
func newContextBudget(window, maxOutput int, cfg CompactionConfig) contextBudget {
	reserve := cfg.OutputReserve
	if reserve == 0 {
		reserve = max(minOutputReserve, window/10)
		if maxOutput > 0 {
			reserve = min(maxOutput, reserve)
		}
	}

	usable := window - reserve

	trigger := cfg.TriggerFraction
	if trigger == 0 {
		trigger = DefaultTriggerFraction
	}

	target := cfg.TargetFraction
	if target == 0 {
		target = DefaultTargetFraction
	}

	return contextBudget{
		window:    window,
		reserve:   reserve,
		usable:    usable,
		trigger:   int(trigger * float64(usable)),
		target:    int(target * float64(usable)),
		hardLimit: usable,
	}
}
