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
	"regexp"
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

// routeMove is one bare name whose calls from some regions now go to
// another copy: the copy before and after ("-" for none), and the regions.
type routeMove struct {
	name, from, to string
	regions        []string
}

// routeMoves names each bare name whose calls from some regions now go to
// another copy, once per name and move.
func (base Base) routeMoves() []routeMove {
	type move struct{ name, from, to string }

	after := func(key routeKey) string { return cmp.Or(base.RoutesAfter[key], "-") }
	keys := slices.DeleteFunc(slices.SortedFunc(maps.Keys(base.RoutesBefore), func(a, b routeKey) int {
		return cmp.Or(cmp.Compare(a.name, b.name), cmp.Compare(a.region, b.region))
	}), func(key routeKey) bool { return after(key) == base.RoutesBefore[key] })

	order, regions := groupBy(keys, func(key routeKey) (move, string) {
		return move{key.name, base.RoutesBefore[key], after(key)}, key.region
	})
	moves := make([]routeMove, 0, len(order))

	for _, m := range order {
		moves = append(moves, routeMove{m.name, m.from, m.to, regions[m]})
	}

	return moves
}

// mermaidID is an ID the routing diagram can show as it is. The IDs come
// from catalog-routes, which runs the patched code, so anything else leaves
// the diagram out.
var mermaidID = regexp.MustCompile(`^[A-Za-z0-9.:_/-]{1,100}$`)

// routingDiagram draws each move as its bare name and callers, the copy it
// went to before, and the copy it goes to after, with the input price change
// and where that copy processes data; or nothing when an ID isn't safe to
// draw.
func (base Base) routingDiagram(moves []routeMove) string {
	var diagram strings.Builder

	diagram.WriteString("```mermaid\nflowchart LR\n")

	label := func(id string) string {
		if id == "-" {
			return "no copy: the call fails"
		}

		return id + "<br>" + processedIn(id)
	}

	for i, m := range moves {
		for _, id := range []string{m.name, m.from, m.to} {
			if id != "-" && !mermaidID.MatchString(id) {
				return ""
			}
		}

		fmt.Fprintf(&diagram, "  n%[1]d[\"%[2]s<br>from %[3]s\"]\n  n%[1]d -. before .-> b%[1]d[\"%[4]s\"]\n  n%[1]d == \"after, input %[5]s\" ==> a%[1]d[\"%[6]s\"]\n",
			i+1, m.name, strings.Join(m.regions, ", "), label(m.from), base.routeInputChange(m.from, m.to), label(m.to))
	}

	return diagram.String() + "```\n\n"
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

// routeInputChange is the input price change a moved caller sees: the copy
// it left as priced before the patch, against the copy it goes to after it,
// or "—" without both catalogues.
func (base Base) routeInputChange(from, to string) string {
	if base.Before == nil || base.After == nil {
		return "—"
	}

	source, _ := base.Before.Offering("bedrock", from)
	target, _ := base.After.Offering("bedrock", to)

	return cmp.Or(strings.TrimPrefix(inputChange(source, target), "input "), "—")
}
