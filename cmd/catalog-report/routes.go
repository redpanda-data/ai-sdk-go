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

package main

import (
	"cmp"
	"fmt"
	"maps"
	"slices"
	"strings"

	"github.com/redpanda-data/ai-sdk-go/internal/reconcile"
)

// routeKey is one bare Bedrock name called from one AWS region.
type routeKey struct{ region, name string }

// Routes maps each region and bare name to the model ID the SDK sends it to,
// or "-" where it refuses the pair: cmd/catalog-routes' output.
type Routes map[routeKey]string

// ParseRoutes reads cmd/catalog-routes' output strictly: after a patch it
// comes from code the agent wrote, so any line that isn't a listed region and
// two safe values fails the run.
func ParseRoutes(data []byte) (Routes, error) {
	routes := Routes{}

	for i, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		fields := strings.Fields(line)
		if len(fields) != 3 || !slices.Contains(reconcile.AWSRegions, fields[0]) || !safeValue(fields[1]) || !safeValue(fields[2]) {
			return nil, fmt.Errorf("route line %d: %q", i+1, line)
		}

		key := routeKey{fields[0], fields[1]}
		if _, seen := routes[key]; seen {
			return nil, fmt.Errorf("route line %d: %s from %s listed twice", i+1, key.name, key.region)
		}

		routes[key] = fields[2]
	}

	return routes, nil
}

// routingRows names each bare name whose calls from some regions now go to
// another copy, once per name and move, with the regions, the input price
// change and where the new copy processes data. A pair the SDK refuses, or
// the patch removes, reads as a call that fails.
func (base Base) routingRows() []string {
	type move struct{ name, from, to string }

	after := func(key routeKey) string { return cmp.Or(base.RoutesAfter[key], "-") }
	keys := slices.DeleteFunc(slices.SortedFunc(maps.Keys(base.RoutesBefore), func(a, b routeKey) int {
		return cmp.Or(cmp.Compare(a.name, b.name), cmp.Compare(a.region, b.region))
	}), func(key routeKey) bool { return after(key) == base.RoutesBefore[key] })

	moves, regions := groupBy(keys, func(key routeKey) (move, string) {
		return move{key.name, base.RoutesBefore[key], after(key)}, key.region
	})
	rows := make([]string, 0, len(moves))

	for _, m := range moves {
		rows = append(rows, fmt.Sprintf("| %s | %s | %s | %s | %s | %s |\n",
			code(m.name), valueList(regions[m]), routed(m.from), routed(m.to), base.routeInputChange(m.from, m.to), processedIn(m.to)))
	}

	return rows
}

func routed(id string) string {
	if id == "-" {
		return "call fails (no copy)"
	}

	return code(id)
}

// processedIn names where a copy processes data, from its profile prefix, or
// "—" for no copy or a prefix with no name.
func processedIn(id string) string {
	prefix, _, _ := strings.Cut(id, ".")
	return cmp.Or(reconcile.PrefixNames[prefix], "—")
}

// routeInputChange is the input price change between two Bedrock copies,
// from the catalogue after the patch, or "—" without one.
func (base Base) routeInputChange(from, to string) string {
	if base.After == nil {
		return "—"
	}

	source, _ := base.After.Offering("bedrock", from)
	target, _ := base.After.Offering("bedrock", to)

	return cmp.Or(strings.TrimPrefix(inputChange(source, target), "input "), "—")
}
