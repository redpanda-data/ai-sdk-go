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
	"net/url"
	"regexp"
	"slices"
	"strings"

	"github.com/redpanda-data/ai-sdk-go/internal/reconcile"
)

// The agent's report types live in internal/reconcile, shared with
// cmd/catalog-gate; these names keep this package's code and tests short.
type (
	Report     = reconcile.Report
	Change     = reconcile.Change
	NeedsHuman = reconcile.NeedsHuman
	Skipped    = reconcile.Skipped
)

var (
	fields       = []string{"new_model", "new_copy", "price", "limit", "lifecycle", "replacement", "knowledge_cutoff", "release_date", "capability", "location"}
	reasons      = []string{"not_on_named_sources", "out_of_scope", "limited_availability", "vendor_excluded", regionNotOffered, "copy_unlisted", "fixed_fact_differs"}
	valuePattern = regexp.MustCompile(`^[A-Za-z0-9 ._:/+@-]{0,64}$`)
	// sourcePattern allows only characters that can't end a markdown link,
	// end a table cell or start HTML, so a URL always renders as one link.
	sourcePattern = regexp.MustCompile(`^https://[A-Za-z0-9.-]+(/[A-Za-z0-9._~/%#?=&+:-]*)?$`)
	// codeFilePattern keeps listed file paths to plain path characters, so a
	// path can't break out of its code span in the PR body.
	codeFilePattern = regexp.MustCompile(`^[A-Za-z0-9_./-]{1,200}$`)
)

// safeValue reports whether a value can go into the PR body as it is. "//"
// keeps a bare URL out of the tables: honest values (prices, dates, IDs,
// "us/eu …") never contain it.
func safeValue(v string) bool {
	return valuePattern.MatchString(v) && !strings.Contains(v, "//")
}

// A copy in a region its host doesn't offer is skipped with regionNotOffered,
// under RULES.md's regionRule.
const (
	regionNotOffered = "region_not_offered"
	regionRule       = "region-not-offered"
)

// regionOf splits a <package>:<prefix>.<model ID> offering into its package and
// region prefix, and is false without a package. Bedrock IDs contain ":", so
// only the first one separates.
func regionOf(offering string) (string, string, bool) {
	host, id, named := strings.Cut(offering, ":")
	prefix, _, _ := strings.Cut(id, ".")

	return host, prefix, named
}

// ValidateCodeFiles lists every path that would be unsafe to render.
func ValidateCodeFiles(files []string) []string {
	var problems []string

	for _, file := range files {
		if !codeFilePattern.MatchString(file) {
			problems = append(problems, "bad code file path "+fmt.Sprintf("%q", file))
		}
	}

	return problems
}

// Validate lists every problem that would make the report unsafe to render,
// including a row that names a rule RULES.md doesn't have.
func Validate(report Report, hosts []string, rules reconcile.Rules) []string {
	var problems []string
	// A row's rule must exist and produce the row's outcome (RULES.md, Report values).
	checkRule := func(where, id, outcome string) bool {
		if _, ok := rules.Lines[id]; !ok {
			problems = append(problems, where+": unknown rule "+fmt.Sprintf("%q", id))
			return false
		}

		if !rules.Fits(id, outcome) {
			problems = append(problems, where+": rule "+id+" doesn't produce "+outcome)
			return false
		}

		return true
	}
	checkOffering := func(where, offering string) {
		if !reconcile.OfferingPattern.MatchString(offering) {
			problems = append(problems, where+": bad offering "+fmt.Sprintf("%q", offering))
		}
	}
	checkValue := func(where, value string) {
		if !safeValue(value) {
			problems = append(problems, where+": bad value "+fmt.Sprintf("%q", value))
		}
	}

	checkSource := func(where, raw string) {
		parsed, err := url.Parse(raw)
		if err != nil || !sourcePattern.MatchString(raw) || parsed.Scheme != "https" || !slices.Contains(hosts, parsed.Hostname()) {
			problems = append(problems, where+": source not on a named-source host: "+fmt.Sprintf("%q", raw))
		}
	}
	for index, change := range report.Changes {
		where := fmt.Sprintf("changes[%d]", index)
		checkOffering(where, change.Offering)

		if !slices.Contains(fields, change.Field) {
			problems = append(problems, where+": unknown field "+fmt.Sprintf("%q", change.Field))
		}

		checkValue(where, change.Old)
		checkValue(where, change.New)
		checkSource(where, change.SourceURL)

		// A Bedrock family's source regions route its bare names, so only a
		// new copy moves them (RULES.md, location-follow).
		if host, _, _ := regionOf(change.Offering); change.Field == "location" && host == "bedrock" {
			problems = append(problems, where+": a location change on bedrock re-routes callers, so it goes to a person")
		}

		// A change rests on a Default for its field, never on judgment, so
		// rows of one kind always name the same rule.
		if change.Rule == reconcile.JudgmentRule {
			problems = append(problems, where+": a change needs a Default, not "+reconcile.JudgmentRule)
			continue
		}

		if covered := rules.RuleFields[change.Rule]; checkRule(where, change.Rule, "change") && !slices.Contains(covered, change.Field) {
			problems = append(problems, where+": rule "+change.Rule+" covers "+strings.Join(covered, ", ")+", not "+change.Field)
		}
	}

	for index, item := range report.NeedsHuman {
		where := fmt.Sprintf("needs_human[%d]", index)
		checkOffering(where, item.Offering)

		if !slices.Contains(fields, item.Field) {
			problems = append(problems, where+": unknown field "+fmt.Sprintf("%q", item.Field))
		}

		for _, value := range rowValues(item) {
			checkValue(where, value)
		}

		if len(item.SourceURLs) == 0 {
			problems = append(problems, where+": needs at least one source")
		}

		for _, source := range item.SourceURLs {
			checkSource(where, source)
		}

		checkRule(where, item.Rule, "needs_human")
	}

	for index, item := range report.Skipped {
		where := fmt.Sprintf("skipped[%d]", index)
		checkOffering(where, item.Offering)

		if !slices.Contains(reasons, item.Reason) {
			problems = append(problems, where+": unknown reason "+fmt.Sprintf("%q", item.Reason))
		}

		checkRule(where, item.Rule, "skip:"+code(item.Reason))

		// Only a host's own Offered row can show a region isn't offered.
		if item.Reason == regionNotOffered {
			host, prefix, named := regionOf(item.Offering)

			switch offered, listed := rules.Offered[host]; {
			case !named:
				problems = append(problems, where+": "+item.Offering+" names no package")
			case !listed:
				problems = append(problems, where+": "+host+" has no Offered regions row")
			case slices.Contains(offered, prefix):
				problems = append(problems, where+": region "+prefix+" is offered on "+host)
			}
		}
	}

	return problems
}

// ValidateAdded lists each new_model or new_copy row whose field the
// catalogue contradicts: a row adding an offering of a model the catalogue
// before the patch has facts for is a new_copy, and any other a new_model
// (RULES.md, Report values). Without both catalogues it checks nothing.
func (base Base) ValidateAdded(report Report, provider string) []string {
	if base.Before == nil || base.After == nil {
		return nil
	}

	var problems []string

	for index, change := range report.Changes {
		if change.Field != reconcile.FieldNewModel && change.Field != reconcile.FieldNewCopy {
			continue
		}

		pkg, id := reconcile.PackageOf(change.Offering, provider)

		offering, found := base.After.Offering(pkg, id)
		if !found {
			continue
		}

		model, _ := offering["model"].(string)

		want, carries := reconcile.FieldNewModel, "doesn't carry"
		if _, ok := base.Before.Facts[model]; ok {
			want, carries = reconcile.FieldNewCopy, "already carries"
		}

		if change.Field != want {
			problems = append(problems, fmt.Sprintf("changes[%d]: the catalogue %s %q, so the field is %s, not %s", index, carries, model, want, change.Field))
		}
	}

	return problems
}

func code(value string) string { return "`" + value + "`" }

// value renders a report value as code, or "none" when it's empty (e.g. the
// old value of a new model), since an empty code span shows as stray backticks.
func value(v string) string {
	if v == "" {
		return "none"
	}

	return code(v)
}

func link(raw string) string {
	parsed, err := url.Parse(raw)
	if err != nil {
		return code(raw)
	}

	return "[" + parsed.Hostname() + "](" + raw + ")"
}

// valueList renders report values as comma-separated code spans.
// rowValues is every value a needs_human row carries: the catalogue and page
// sides, when it has them, then the rest.
func rowValues(item NeedsHuman) []string {
	return slices.DeleteFunc(append([]string{item.Catalogue, item.Page}, item.Values...), func(v string) bool { return v == "" })
}

func valueList(values []string) string {
	rendered := make([]string, 0, len(values))
	for _, v := range values {
		rendered = append(rendered, value(v))
	}

	return strings.Join(rendered, ", ")
}

// Base is what Render reads besides the reports: a link prefix for files at
// the base commit (empty for no links), that commit's rules and decisions,
// today as YYYY-MM-DD, and the catalogue before and after the patch. On the
// Base passed to Render that is the whole PR's, for the routing changes'
// prices; inside a vendor's section it is that vendor's, for each change's
// effect. Without both catalogues the Changes table has no Effect column.
type Base struct {
	URL       string
	Rules     reconcile.Rules
	Decisions []reconcile.Decision
	Today     string

	Before, After *reconcile.Snapshot
	// RoutesBefore and RoutesAfter are where bare Bedrock names go before
	// and after the patch, for the Routing changes section.
	RoutesBefore, RoutesAfter Routes
	// Branch is the branch each run rebuilds the bot branch from, named in
	// the body.
	Branch string
}

// fileLink links text to a repo file at the base commit, at line when it's
// above 0. Without a base URL it's a code span. GitHub renders markdown
// files, so a line link to one needs ?plain=1.
func (base Base) fileLink(text, path string, line int) string {
	if base.URL == "" {
		return code(text)
	}

	target := base.URL + "/" + path
	if line > 0 {
		if strings.HasSuffix(path, ".md") {
			target += "?plain=1"
		}

		target += fmt.Sprintf("#L%d", line)
	}

	return "[" + text + "](" + target + ")"
}

// ruleLink links a rule ID to its row in RULES.md.
func (base Base) ruleLink(id string) string {
	return base.fileLink(id, reconcile.RulesPath, base.Rules.Lines[id])
}

// ruleCell names a row's rule, flagged when no rule covers the case.
func (base Base) ruleCell(id string) string {
	if id == reconcile.JudgmentRule {
		return "⚠ " + base.ruleLink(id)
	}

	return base.ruleLink(id)
}

// groupBy collects one value per item under the item's key, keys in the
// order first seen.
func groupBy[T any, K comparable, V any](items []T, key func(T) (K, V)) ([]K, map[K][]V) {
	var keys []K

	groups := map[K][]V{}

	for _, item := range items {
		k, value := key(item)
		if _, seen := groups[k]; !seen {
			keys = append(keys, k)
		}

		groups[k] = append(groups[k], value)
	}

	return keys, groups
}

// skippedRows groups skipped offerings by the rule and reason that left them out.
func (base Base) skippedRows(skipped []Skipped) []string {
	type group struct{ rule, reason string }

	groups, offerings := groupBy(skipped, func(item Skipped) (group, string) { return group{item.Rule, item.Reason}, item.Offering })
	rows := make([]string, 0, len(groups))

	for _, key := range groups {
		why := code(key.reason)
		if meaning := base.Rules.Reasons[key.reason]; meaning != "" {
			why = meaning + " (" + why + ")"
		}

		rows = append(rows, fmt.Sprintf("| %s | %s | %s |\n", base.ruleCell(key.rule), why, valueList(offerings[key])))
	}

	return rows
}

// region is a host package and one of its region prefixes.
type region struct{ host, prefix string }

// byRegion groups region_not_offered rows by host and region, in the order
// first seen.
func byRegion(skipped []Skipped) ([]region, map[region][]string) {
	return groupBy(skipped, func(item Skipped) (region, string) {
		host, prefix, _ := regionOf(item.Offering)
		return region{host, prefix}, item.Offering
	})
}

// answer reads a region's note in RULES.md's Not offered column: "asked" or
// "declined" and its date, or empty strings when the column doesn't list it.
func (base Base) answer(key region) (string, string) {
	answer, date, _ := strings.Cut(base.Rules.NotOffered[key.host][key.prefix], " ")
	return answer, date
}

// waitingLines names each host and region a waiting row is in, once, with
// since when it has been asked.
func (base Base) waitingLines(skipped []Skipped) []string {
	regions, offerings := byRegion(skipped)
	lines := make([]string, 0, len(regions))

	for _, key := range regions {
		copies := fmt.Sprintf("%d copies", len(offerings[key]))
		if len(offerings[key]) == 1 {
			copies = "1 copy"
		}

		status := "not in the table yet"
		if answer, date := base.answer(key); answer == "asked" {
			status = "asked since " + date
		}

		lines = append(lines, fmt.Sprintf("- Offer %s on %s (%s)? %s: %s\n", code(key.prefix), key.host, status, copies, valueList(offerings[key])))
	}

	return lines
}

// question is the needs_human rows that ask one thing, on one offering or
// more. row is the first of them, with every row's sources.
type question struct {
	row       NeedsHuman
	offerings []string
}

// questions groups needs_human rows that differ only by offering and sources,
// in the order first seen, so one cause is one question.
func questions(needs []NeedsHuman) []question {
	type group struct{ rule, field, catalogue, page, values string }

	keys, rows := groupBy(needs, func(item NeedsHuman) (group, NeedsHuman) {
		return group{item.Rule, item.Field, item.Catalogue, item.Page, fmt.Sprintf("%q", item.Values)}, item
	})
	grouped := make([]question, 0, len(keys))

	for _, key := range keys {
		q := question{row: rows[key][0]}
		q.row.SourceURLs = nil

		for _, item := range rows[key] {
			q.offerings = append(q.offerings, item.Offering)

			for _, source := range item.SourceURLs {
				if !slices.Contains(q.row.SourceURLs, source) {
					q.row.SourceURLs = append(q.row.SourceURLs, source)
				}
			}
		}

		grouped = append(grouped, q)
	}

	return grouped
}

// resolveAlert is the alert at the top for the reviewer: how many "Resolve
// before merging" questions block the merge, naming the first five, with
// their vendor when the body has more than one. It shows only when there is
// one.
func resolveAlert(parts []vendorPart) string {
	var named []string

	asks := 0

	for _, part := range parts {
		for _, q := range part.asks {
			if asks++; asks > 5 {
				continue
			}

			name := code(q.row.Offering) + " " + q.row.Field
			if len(q.offerings) > 1 {
				name = fmt.Sprintf("%s on %d offerings", q.row.Field, len(q.offerings))
			}

			if len(parts) > 1 {
				name = part.Name + " " + name
			}

			named = append(named, name)
		}
	}

	if asks == 0 {
		return ""
	}

	list := strings.Join(named, ", ")
	if more := asks - len(named); more > 0 {
		list += fmt.Sprintf(" and %d more", more)
	}

	count := "1 question"
	if asks > 1 {
		count = fmt.Sprintf("%d questions", asks)
	}

	return "> [!IMPORTANT]\n> **Resolve " + count + " before merging:** " + list + ". See Resolve before merging below.\n\n"
}

// waitingAlert is the alert at the top for product: each waiting host and
// region, with where it processes data and since when it has been asked. It
// shows only when there is one.
func (base Base) waitingAlert(waiting []Skipped) string {
	regions, _ := byRegion(waiting)
	if len(regions) == 0 {
		return ""
	}

	asked := make([]string, 0, len(regions))

	for _, key := range regions {
		status := "new"
		if answer, date := base.answer(key); answer == "asked" {
			status = "asked since " + date
		}

		if name := reconcile.PrefixNames[key.prefix]; name != "" {
			status = name + ", " + status
		}

		asked = append(asked, "offer "+code(key.prefix)+" on "+key.host+" ("+status+")?")
	}

	decisions, they := "1 product decision", "it doesn't"
	if len(regions) > 1 {
		decisions, they = fmt.Sprintf("%d product decisions", len(regions)), "they don't"
	}

	return "> [!NOTE]\n> **" + decisions + " waiting** (" + they + " block this PR): " + strings.Join(asked, " ") + "\n\n"
}

// needsHumanBlock renders one question: its offerings, its rule, the rule's
// fixed question, what the catalogue and the page say, anything else the bot
// found, and the sources. A no-rule question with both sides asks instead,
// after them, to keep one or take the other.
func (base Base) needsHumanBlock(number int, q question) string {
	item := q.row

	why := "rule " + base.ruleLink(item.Rule)
	if item.Rule == reconcile.JudgmentRule {
		why = "⚠ no rule covers this (" + base.ruleLink(item.Rule) + ")"
	}

	sources := make([]string, 0, len(item.SourceURLs))
	for _, source := range item.SourceURLs {
		sources = append(sources, link(source))
	}

	subject, listed := code(item.Offering), ""
	if len(q.offerings) > 1 {
		subject, listed = fmt.Sprintf("%d offerings", len(q.offerings)), "- **Offerings:** "+valueList(q.offerings)+"\n"
	}

	block := fmt.Sprintf("**%d. %s · %s · %s**\n%s", number, subject, item.Field, why, listed)

	decide := item.Rule == reconcile.JudgmentRule && item.Catalogue != "" && item.Page != ""
	if fixed := base.Rules.Questions[item.Rule]; fixed != "" && !decide {
		block += "- **Question:** " + fixed + "\n"
	}

	if item.Catalogue != "" {
		block += "- **Catalogue:** " + code(item.Catalogue) + "\n"
	}

	if item.Page != "" {
		block += "- **Page:** " + code(item.Page) + "\n"
	}

	if len(item.Values) > 0 {
		block += "- **Also found:** " + valueList(item.Values) + "\n"
	}

	if decide {
		block += "- **Decide:** keep " + code(item.Catalogue) + ", or change to " + code(item.Page) + ". If this can come back, add a rule.\n"
	}

	return block + "- **Sources:** " + strings.Join(sources, ", ") + "\n\n"
}

// sameOffering reports whether a report offering names a decision's. Report
// offerings in the run's own package have no prefix (RULES.md, Report values); Bedrock
// IDs contain ":", so the prefix can't be told apart by looking for one.
func sameOffering(reported, provider, decided string) bool {
	return decided == reported || decided == provider+":"+reported
}

// hold takes the needs_human and skipped rows a decision already answers out
// of the report, and returns the rest with the held table's rows.
func (base Base) hold(report Report, provider string) (Report, []string) {
	find := func(offering string, wanted ...string) (reconcile.Decision, bool) {
		for _, d := range base.Decisions {
			if slices.Contains(wanted, d.Field) && sameOffering(offering, provider, d.Offering) {
				return d, true
			}
		}

		return reconcile.Decision{}, false
	}
	row := func(d reconcile.Decision, offering, saw string) string {
		return fmt.Sprintf("| %s | %s | %s | %s | %s |\n", base.fileLink(d.ID, reconcile.DecisionsPath, d.Line),
			code(offering), d.Field, saw, strings.ReplaceAll(d.Reason, "|", `\|`))
	}

	var held []string

	// Fresh slices: RenderCapped renders the same report more than once.
	var needs []NeedsHuman

	for _, item := range report.NeedsHuman {
		if d, ok := find(item.Offering, item.Field); ok {
			held = append(held, row(d, item.Offering, valueList(rowValues(item))))
		} else {
			needs = append(needs, item)
		}
	}

	var skipped []Skipped

	for _, item := range report.Skipped {
		if d, ok := find(item.Offering, reconcile.FieldNewModel, reconcile.FieldNewCopy); ok {
			held = append(held, row(d, item.Offering, item.Reason))
		} else {
			skipped = append(skipped, item)
		}
	}

	report.NeedsHuman, report.Skipped = needs, skipped

	return report, held
}

// byRule groups change rows by rule, in RULES.md's order so every run lists
// them alike, each group sorted by package and model.
func (base Base) byRule(changes []Change, provider string) ([]string, map[string][]Change) {
	key := func(change Change) string {
		pkg, id := reconcile.PackageOf(change.Offering, provider)

		var offering map[string]any
		if base.After != nil {
			offering, _ = base.After.Offering(pkg, id)
		}

		return pkg + " " + cmp.Or(safeText(offering["model"]), id) + " " + id
	}

	sorted := slices.Clone(changes)
	slices.SortStableFunc(sorted, func(a, b Change) int {
		return cmp.Or(cmp.Compare(base.Rules.Lines[a.Rule], base.Rules.Lines[b.Rule]), cmp.Compare(a.Rule, b.Rule), cmp.Compare(key(a), key(b)))
	})

	return groupBy(sorted, func(change Change) (string, Change) { return change.Rule, change })
}

// changesSection renders the Changes table, one small table per rule. A note
// describes an offering's field, so rows that share one show it once per rule.
func (base Base) changesSection(changes []Change, provider string) string {
	effects := base.Before != nil && base.After != nil

	header := []string{"Offering", "Field", "Old", "New", "Source"}
	if effects {
		header = slices.Insert(header, 4, "Effect")
	}

	table := fmt.Sprintf("| %s |\n|%s\n", strings.Join(header, " | "), strings.Repeat("---|", len(header)))
	noted := map[[3]string]bool{}

	var section strings.Builder

	fmt.Fprintf(&section, "### Changes (%d)\n\n", len(changes))

	rules, groups := base.byRule(changes, provider)
	for _, rule := range rules {
		fmt.Fprintf(&section, "**%s** (%d)\n\n%s", base.ruleLink(rule), len(groups[rule]), table)

		for _, change := range groups[rule] {
			cells := []string{code(change.Offering), change.Field, value(change.Old), value(change.New)}
			if key := [3]string{rule, change.Offering, change.Field}; effects {
				note := ""
				if !noted[key] {
					note, noted[key] = base.effect(change, provider), true
				}

				cells = append(cells, note)
			}

			fmt.Fprintf(&section, "| %s |\n", strings.Join(append(cells, link(change.SourceURL)), " | "))
		}

		section.WriteString("\n")
	}

	return section.String()
}

// suggestions drafts a Defaults row for each field that two or more no-rule
// rows share this run, with the values all of them found, so a person can
// turn a repeated case into a rule. Code writes it, so it never varies.
func suggestions(needs []NeedsHuman) []string {
	type group struct{ rule, field string }

	keys, found := groupBy(needs, func(item NeedsHuman) (group, []string) { return group{item.Rule, item.Field}, rowValues(item) })
	lines := make([]string, 0, len(keys))

	for _, key := range keys {
		if key.rule != reconcile.JudgmentRule || len(found[key]) < 2 {
			continue
		}

		shared := slices.Clone(found[key][0])
		for _, values := range found[key][1:] {
			shared = slices.DeleteFunc(shared, func(v string) bool { return !slices.Contains(values, v) })
		}

		when := fmt.Sprintf("%d rows this run", len(found[key]))
		if len(shared) > 0 {
			when += ", all with " + valueList(shared)
		}

		lines = append(lines, fmt.Sprintf("| new-default | %s | %s | ? | ? | ? |\n", code(key.field), when))
	}

	return lines
}

// splitWaiting takes the region_not_offered rows out of the skipped ones,
// except those in a region RULES.md records as declined.
func (base Base) splitWaiting(report Report) (Report, []Skipped) {
	// Fresh slices, as in hold.
	var waiting, skipped []Skipped

	for _, item := range report.Skipped {
		host, prefix, _ := regionOf(item.Offering)
		if answer, _ := base.answer(region{host, prefix}); item.Reason == regionNotOffered && answer != "declined" {
			waiting = append(waiting, item)
		} else {
			skipped = append(skipped, item)
		}
	}

	report.Skipped = skipped

	return report, waiting
}

// summary is one line about decisions.yaml itself: how many decisions it
// holds, and which are past their review date. It shows on every run.
func (base Base) summary() string {
	if len(base.Decisions) == 0 {
		return ""
	}

	var overdue []string

	for _, d := range base.Decisions {
		if d.ReviewBy < base.Today {
			overdue = append(overdue, d.ID)
		}
	}

	due := "none."
	if len(overdue) > 0 {
		due = strings.Join(overdue, ", ") + " (still enforced)."
	}

	return fmt.Sprintf("Recorded decisions: %d, in %s. Past their review date: %s\n\n",
		len(base.Decisions), base.fileLink("decisions.yaml", reconcile.DecisionsPath, 0), due)
}

// Vendor is one vendor's part of the PR: its report, and the catalogue
// before and after its commit, for effect notes and "Changed with no row".
// Failed is a vendor whose run failed, so the PR has nothing from it.
// Replayed is set when the vendor's change was replayed from an earlier run.
type Vendor struct {
	Name          string
	Report        Report
	Before, After *reconcile.Snapshot
	Failed        bool
	Replayed      Replay
}

// Replay is the agent run a replayed vendor's change and report came from:
// its run ID and its day. Both are empty when the agent ran in this run.
type Replay struct {
	Run, Date string
}

// vendorPart is one vendor ready to render: base with the vendor's own
// catalogue, the rows its section shows once decisions and waiting regions
// are taken out, its questions grouped, and the offerings its patch changes
// with no row.
type vendorPart struct {
	Vendor

	base       Base
	shown      Report
	held       []string
	waiting    []Skipped
	asks       []question
	unreported []string
}

func (base Base) part(vendor Vendor) vendorPart {
	part := vendorPart{Vendor: vendor, base: base}
	part.base.Before, part.base.After = vendor.Before, vendor.After
	part.shown, part.held = part.base.hold(vendor.Report, vendor.Name)
	part.shown, part.waiting = part.base.splitWaiting(part.shown)
	part.asks = questions(part.shown.NeedsHuman)
	// Held and waiting rows still count as rows here.
	part.unreported = part.base.unreported(vendor.Report, vendor.Name)

	return part
}

// rows counts the rows the vendor's section shows.
func (part vendorPart) rows() int {
	return len(part.shown.Changes) + len(part.shown.NeedsHuman) + len(part.shown.Skipped) + len(part.waiting) + len(part.held)
}

// Render writes the PR body: what applies to the whole PR once, then one
// section per vendor, then the snapshot diff. Call it only on reports
// Validate accepted and code files ValidateCodeFiles accepted. codeFiles are
// the files whose change isn't data only, listed first so a reviewer reads
// them as code.
func Render(vendors []Vendor, snapshotDiff string, codeFiles []string, base Base) string {
	top, sections := base.render(vendors, codeFiles, snapshotDiff != "")
	return top + strings.Join(sections, "") + diffBlock(snapshotDiff)
}

// render writes the top of the body and each vendor's section.
func (base Base) render(vendors []Vendor, codeFiles []string, hasDiff bool) (string, []string) {
	var body strings.Builder

	names := make([]string, 0, len(vendors))
	parts := make([]vendorPart, 0, len(vendors))
	rows, failed := 0, false

	var waiting []Skipped

	for _, vendor := range vendors {
		part := base.part(vendor)
		names, parts = append(names, vendor.Name), append(parts, part)
		waiting = append(waiting, part.waiting...)
		rows += part.rows()
		failed = failed || vendor.Failed
	}

	body.WriteString("## Catalogue drift: " + strings.Join(names, ", ") + "\n\n")

	if rows == 0 && !failed && !hasDiff {
		body.WriteString("No drift found.\n\n" + base.summary())
		return body.String(), nil
	}

	body.WriteString(summaryLine(parts) + base.checksLine(codeFiles))

	fmt.Fprintf(&body, "Each run rebuilds this branch from %[1]s, leaving out a clean \"Update branch\" merge. A commit you push here is carried onto each rebuild while it still applies exactly; if it doesn't, or it's another merge commit or changes `.github/`, runs stop updating this PR and leave a comment saying why. To settle a row, merge a fix on %[1]s or record a decision in `.github/reconcile-models/decisions.yaml`. Later runs don't read PR comments.\n\n", base.Branch)

	body.WriteString(resolveAlert(parts) + base.waitingAlert(waiting))
	body.WriteString(base.summary())

	if len(codeFiles) > namedCodeFiles {
		fmt.Fprintf(&body, "### Code changed (%d)\n\n", len(codeFiles))

		for _, file := range codeFiles {
			body.WriteString("- " + code(file) + "\n")
		}

		body.WriteString("\n")
	}

	if moves := base.routingRows(); len(moves) > 0 {
		fmt.Fprintf(&body, "### Routing changes (%d)\n\n⚑ Bare-name calls from these regions now go to another copy. Check the price and where the data is processed.\n\n"+
			"| Model | Callers in | Today | After | Input price | Data processed in |\n|---|---|---|---|---|---|\n", len(moves))
		body.WriteString(strings.Join(moves, "") + "\n")
	}

	sections := make([]string, 0, len(parts))
	for _, part := range parts {
		sections = append(sections, part.section())
	}

	return body.String(), sections
}

// kindNouns names each report field as the summary line counts it, one and
// many. TestRulesFile holds its keys to fields.
var kindNouns = map[string][2]string{
	"new_model": {"new model", "new models"}, "new_copy": {"new copy", "new copies"}, "price": {"price", "prices"},
	"limit": {"token limit", "token limits"}, "lifecycle": {"lifecycle date", "lifecycle dates"}, "replacement": {"replacement", "replacements"},
	"knowledge_cutoff": {"knowledge cutoff", "knowledge cutoffs"}, "release_date": {"release date", "release dates"},
	"capability": {"capability", "capabilities"}, "location": {"location", "locations"},
}

// namedCodeFiles is how many code files the checks line names; above it,
// they get their own section.
const namedCodeFiles = 3

// count is n of something, in the singular when n is 1.
func count(n int, one, many string) string {
	if n == 1 {
		return "1 " + one
	}

	return fmt.Sprintf("%d %s", n, many)
}

// summaryLine counts the PR's changes by kind, in the order they first
// appear, and the rows that leave the catalogue as it is.
func summaryLine(parts []vendorPart) string {
	var changes []Change

	unreported, asks, held, waiting, skipped := 0, 0, 0, 0, 0

	for _, part := range parts {
		changes = append(changes, part.shown.Changes...)
		unreported += len(part.unreported)
		asks += len(part.asks)
		held += len(part.held)
		waiting += len(part.waiting)
		skipped += len(part.shown.Skipped)
	}

	line := "**No changes** to the catalogue."

	if total := len(changes) + unreported; total > 0 {
		order, byField := groupBy(changes, func(change Change) (string, Change) { return change.Field, change })
		kinds := make([]string, 0, len(order)+1)

		for _, field := range order {
			noun := kindNouns[field]
			kinds = append(kinds, count(len(byField[field]), noun[0], noun[1]))
		}

		if unreported > 0 {
			kinds = append(kinds, count(unreported, "offering changed with no row", "offerings changed with no row"))
		}

		line = "**" + count(total, "change", "changes") + "** to the catalogue: " + strings.Join(kinds, ", ") + "."
	}

	var same []string

	for _, rows := range []struct {
		n         int
		one, many string
	}{
		{asks, "open question", "open questions"},
		{held, "held by a decision", "held by decisions"},
		{waiting, "copy waiting on a product decision", "copies waiting on a product decision"},
		{skipped, "skipped offering", "skipped offerings"},
	} {
		if rows.n > 0 {
			same = append(same, count(rows.n, rows.one, rows.many))
		}
	}

	if len(same) > 0 {
		line += " Not changed: " + strings.Join(same, ", ") + "."
	}

	return line + "\n\n"
}

// checksLine is what a reviewer must do before marking the PR ready, then
// what the bot checked: pr-job.sh's steps, which only the bot's own commits
// go through.
func (base Base) checksLine(codeFiles []string) string {
	line := "**Before you mark it ready:** check each value in Changes against its source"

	switch n := len(codeFiles); {
	case n == 0:
		line += "."
	case n <= namedCodeFiles:
		list, them := code(codeFiles[n-1]), "them"
		if n == 1 {
			them = "it"
		} else {
			list = valueList(codeFiles[:n-1]) + " and " + list
		}

		line += ", and read " + list + " as code; marking ready runs " + them + " with secrets."
	default:
		line += fmt.Sprintf(", and read the %d files under Code changed as code; marking ready runs them with secrets.", n)
	}

	return line + " Only the agent compared the values with their sources. For its own commits, the bot checked the report's format and the rule on each row, " +
		"that sources are on named hosts and that only allowed files change, and ran unit tests and lint without secrets. " +
		base.fileLink("REVIEW.md", "REVIEW.md#bot-catalog-prs", 0) + " has the full list.\n\n"
}

// section renders one vendor's rows under its own heading.
func (part vendorPart) section() string {
	var body strings.Builder

	base, report := part.base, part.shown
	body.WriteString("## " + part.Name + "\n\n")

	if part.Replayed.Run != "" {
		fmt.Fprintf(&body, "> Replayed with no model: this is the change the agent made in run %s on %s, checked to be exactly the same on this base.\n\n", part.Replayed.Run, part.Replayed.Date)
	}

	switch {
	case part.Failed:
		body.WriteString("This vendor's run failed, so this PR has nothing from it. The workflow run shows why.\n\n")
		return body.String()
	case part.rows() == 0:
		body.WriteString("The run found nothing to change.\n\n")
		return body.String()
	}

	if len(report.Changes) > 0 {
		body.WriteString(base.changesSection(report.Changes, part.Name))
	}

	if names := part.unreported; len(names) > 0 {
		fmt.Fprintf(&body, "### Changed with no row (%d)\n\nThe patch changes these offerings, but the report has no row for them. Check them in the snapshot diff.\n\n- %s\n\n",
			len(names), valueList(names))
	}

	if len(part.asks) > 0 {
		judged := 0

		for _, q := range part.asks {
			if q.row.Rule == reconcile.JudgmentRule {
				judged++
			}
		}

		fmt.Fprintf(&body, "### Resolve before merging (%d): %d by a rule, %d with no rule\n\n", len(part.asks), len(part.asks)-judged, judged)

		for index, q := range part.asks {
			body.WriteString(base.needsHumanBlock(index+1, q))
		}
	}

	if lines := suggestions(report.NeedsHuman); len(lines) > 0 {
		fmt.Fprintf(&body, "### Suggested rules (%d)\n\nSome rows under Resolve before merging went to a person under no-rule and share a field. If the case can come back, add a Default to RULES.md: give it a slug, write When as the condition, and replace each ?.\n\n| Default | Field | When | Bot does | Why | Outcome |\n|---|---|---|---|---|---|\n",
			len(lines))
		body.WriteString(strings.Join(lines, "") + "\n")
	}

	if lines := base.waitingLines(part.waiting); len(lines) > 0 {
		fmt.Fprintf(&body, "### Waiting on a product decision (%d)\n\nModel cards list copies in regions this catalogue doesn't offer, so the bot left them out (rule %s). Nothing changes until someone decides, and they don't block the PR. "+
			"To answer, either offer the region (move its prefix to Offered in %s, with its code, in a reviewed PR) or decline it (write it under Not offered as `` `<prefix>` (declined YYYY-MM-DD) ``, and its copies show under Skipped from then on).\n\n",
			len(lines), base.ruleLink(regionRule), base.fileLink("RULES.md", reconcile.RulesPath, 0))
		body.WriteString(strings.Join(lines, "") + "\n")
	}

	if len(part.held) > 0 {
		fmt.Fprintf(&body, "### Held by decisions (%d)\n\nQuestions a recorded decision already answers. Nothing here changes.\n\n| Decision | Offering | Field | What the bot saw | Why it's settled |\n|---|---|---|---|---|\n", len(part.held))
		body.WriteString(strings.Join(part.held, "") + "\n")
	}

	if len(report.Skipped) > 0 {
		fmt.Fprintf(&body, "### Skipped (%d)\n\n| Rule | Why | Offerings |\n|---|---|---|\n", len(report.Skipped))
		body.WriteString(strings.Join(base.skippedRows(report.Skipped), "") + "\n")
	}

	return body.String()
}

// diffBlock is the snapshot diff under its own heading, so it doesn't read as
// part of the last vendor's section, folded; or nothing without one.
func diffBlock(snapshotDiff string) string {
	if snapshotDiff == "" {
		return ""
	}

	return "## Snapshot diff\n\n<details><summary>Snapshot diff</summary>\n\n```diff\n" + snapshotDiff + "\n```\n</details>\n"
}

// truncatedNote ends a snapshot diff that RenderCapped shortened.
const truncatedNote = "\n… (truncated; see the workflow artifact)"

// cutNote ends a body whose tables alone don't fit.
const cutNote = "\n\n… (the rest is cut to fit GitHub's size limit; the full report is in the workflow artifacts)\n"

// RenderCapped is Render, kept within limit bytes so GitHub accepts the PR
// body. The snapshot diff is shortened first; then each vendor's section
// gets an equal share of the room left, so one big vendor can't push the
// others out. Both are cut only between lines.
func RenderCapped(vendors []Vendor, snapshotDiff string, codeFiles []string, base Base, limit int) string {
	top, sections := base.render(vendors, codeFiles, snapshotDiff != "")

	body, diff := top+strings.Join(sections, ""), diffBlock(snapshotDiff)
	if len(body)+len(diff) <= limit {
		return body + diff
	}

	if room := limit - len(body) - len(diffBlock(truncatedNote)); room > 0 {
		return body + diffBlock(cutOnLine(snapshotDiff, room)+truncatedNote)
	}

	// The tables alone don't fit: drop the diff, and cut so no table row or
	// fence is left half written.
	if body = top + strings.Join(fit(sections, limit-len(top)), ""); len(body) <= limit {
		return body
	}

	return cutOnLine(body, limit-len(cutNote)) + cutNote
}

// fit cuts sections to room bytes in all: each gets an equal share, a
// section shorter than its share passes the rest on, and a longer one is cut
// on a line.
func fit(sections []string, room int) []string {
	fitted := slices.Clone(sections)
	order := make([]int, len(fitted))

	for i := range order {
		order[i] = i
	}

	slices.SortStableFunc(order, func(a, b int) int { return cmp.Compare(len(fitted[a]), len(fitted[b])) })

	for done, i := range order {
		if share := room / (len(order) - done); len(fitted[i]) > share {
			fitted[i] = cutOnLine(fitted[i], share-len(cutNote)-1) + cutNote + "\n"
		}

		room -= len(fitted[i])
	}

	return fitted
}

// cutOnLine is the longest start of text, at most limit bytes, that ends
// before a line break.
func cutOnLine(text string, limit int) string {
	text = text[:max(min(limit, len(text)), 0)]
	return text[:max(strings.LastIndexByte(text, '\n'), 0)]
}
