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
	"path"
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
	// sentencePattern is the agent's own words, shown as plain text: no
	// character that starts markup, a link, a mention, an emoji or math.
	sentencePattern = regexp.MustCompile(`^[A-Za-z0-9 .,;()'?%-]{0,300}$`)
	// autolinked is what GitHub links in plain text with those characters.
	autolinked = regexp.MustCompile(`(?i)www\.|\bgh-[0-9]`)
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

// safeSentence reports whether the agent's words can go into the PR body as
// plain text.
func safeSentence(s string) bool {
	return sentencePattern.MatchString(s) && !autolinked.MatchString(s)
}

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

		for _, sentence := range []string{item.Why, item.RuleWhen, item.RuleDoes} {
			if !safeSentence(sentence) {
				problems = append(problems, where+": bad sentence "+fmt.Sprintf("%q", sentence))
			}
		}

		if (item.RuleWhen == "") != (item.RuleDoes == "") {
			problems = append(problems, where+": a rule draft needs both rule_when and rule_does")
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

// link names a source by its page, so several links on one host can be told
// apart: the last path segment without "model-card-" and a .md or .html
// ending, its folder for an overview or index page, or the host when there
// is none or it isn't safe to print. A version's dot ("gpt-6.1") stays.
func link(raw string) string {
	parsed, err := url.Parse(raw)
	if err != nil {
		return code(raw)
	}

	page := parsed.Path
	for _, ending := range []string{".md", ".html", ".htm"} {
		page = strings.TrimSuffix(page, ending)
	}

	name := path.Base(page)
	if name == "overview" || name == "index" {
		name = path.Base(path.Dir(page))
	}

	name = strings.TrimPrefix(name, "model-card-")
	if name == "" || name == "." || name == "/" || !safeValue(name) {
		name = parsed.Hostname()
	}

	return "[" + name + "](" + raw + ")"
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
	// Carried is each commit people pushed that the carry job carried onto
	// the vendors' commits.
	Carried []Commit
}

// Commit is a commit people pushed: its full SHA and its author's name, as
// git log gives them.
type Commit struct {
	SHA, Author string
}

// printable is an author or subject that can go into the body in a code
// span: printable ASCII with no backtick, so it can't leave the span, and
// inside it nothing is a link, a mention or markup.
var printable = regexp.MustCompile("^[ -_a-~]{1,100}$")

// carriedLine names the carried commits, so a reviewer knows the PR holds
// commits people pushed. Their files are under Code changed like the bot's.
// It shows only when something was carried.
func (base Base) carriedLine() string {
	if len(base.Carried) == 0 {
		return ""
	}

	commits := make([]string, 0, len(base.Carried))

	for _, commit := range base.Carried {
		line := code(commit.SHA[:7])
		if printable.MatchString(commit.Author) {
			line += " by " + code(commit.Author)
		}

		commits = append(commits, line)
	}

	return "**Not from the bot:** " + count(len(commits), "commit", "commits") + ": " + strings.Join(commits, "; ") + ".\n\n"
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

		rows = append(rows, fmt.Sprintf("| %s | %s | %s |\n", base.ruleLink(key.rule), why, valueList(offerings[key])))
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

// numbered is a question with its vendor, its number across the PR, and the
// name the box and Open questions both show. A region question has a place
// and the copies waiting on it instead of a question.
type numbered struct {
	question

	vendor, name string
	n            int
	place        region
	copies       []string
}

// numberQuestions numbers every vendor's questions in one run, 1 to N, and
// names each so no two look alike: its vendor when the body has more than
// one, the region of a new copy, its first offering and how many more, and
// the field. Then each host and region the waiting copies are in is one
// more question, whether to offer it.
func (base Base) numberQuestions(parts []vendorPart, waiting []Skipped) []numbered {
	var all []numbered

	for _, part := range parts {
		for _, q := range part.asks {
			row := q.row

			name := code(row.Offering)
			if more := len(q.offerings) - 1; more > 0 {
				name += fmt.Sprintf(" and %d more", more)
			}

			if _, prefix, _ := regionOf(row.Offering); row.Field == reconcile.FieldNewCopy && reconcile.PrefixNames[prefix] != "" {
				name = prefix + plural(len(q.offerings), " copy: ", " copies: ") + name
			}

			if len(parts) > 1 {
				name = part.Name + " · " + name
			}

			name += " · " + kindNouns[row.Field][0]
			all = append(all, numbered{question: q, vendor: part.Name, name: name, n: len(all) + 1})
		}
	}

	places, copies := byRegion(waiting)
	for _, place := range places {
		all = append(all, numbered{place: place, copies: copies[place], name: base.regionName(place, len(copies[place])), n: len(all) + 1})
	}

	return all
}

// regionName names a region question: the host, the region and its name,
// how many copies wait on it, and since when it has been asked.
func (base Base) regionName(place region, copies int) string {
	where := code(place.prefix)
	if name := reconcile.PrefixNames[place.prefix]; name != "" {
		where += " (" + name + ")"
	}

	status := "new"
	if answer, date := base.answer(place); answer == "asked" {
		status = "asked since " + date
	}

	return place.host + " · offer region " + where + "? · " + count(copies, "copy", "copies") + " · " + status
}

// addModelSkill lists every place in the code that offering a new region
// takes.
const addModelSkill = ".claude/skills/add-model/SKILL.md"

// regionBlock renders a region question: the copies waiting on it, why a
// person decides (RULES.md, Offered regions), and how to offer or decline
// the region, with this region and today in the decline line.
func (base Base) regionBlock(q numbered) string {
	these, them := plural(len(q.copies), "this copy", "these copies"), plural(len(q.copies), "it", "them")

	where := "region " + code(q.place.prefix)
	if name := reconcile.PrefixNames[q.place.prefix]; name != "" {
		where = name
	}

	return fmt.Sprintf("**%d. %s** · rule %s\n", q.n, q.name, base.ruleLink(regionRule)) +
		"- **Copies waiting:** " + valueList(q.copies) + "\n" +
		"- **Why the bot asks:** The vendor's pages list " + these + " in " + where + ", a region the catalogue doesn't offer, so the bot left " + them + " out. " +
		"A new region changes routing, price and where data is processed for callers there, so offering one is a product call.\n" +
		"- **Decide:** offer it, or decline it? To offer it, follow the " + base.fileLink("add-model skill", addModelSkill, 0) + "'s steps for a new region: the provider's code, the copies and " +
		base.fileLink("RULES.md", reconcile.RulesPath, 0) + "'s Offered list, in one PR. " +
		"To decline it, list it under Not offered in RULES.md as `` " + code(q.place.prefix) + " (declined " + cmp.Or(base.Today, "YYYY-MM-DD") + ") ``.\n\n"
}

// boxQuestions is how many questions the box names. The top of the body is
// never cut, so the rest are only under Open questions.
const boxQuestions = 10

// questionsBox is the alert at the top, with the first open questions
// numbered and named as in Open questions and the note that none of them
// blocks merging, then how to answer one. It shows only when there is one.
func (base Base) questionsBox(all []numbered) string {
	if len(all) == 0 {
		return ""
	}

	var box strings.Builder

	fmt.Fprintf(&box, "> [!IMPORTANT]\n> **%s,** listed under Open questions below. %s block merging: merging changes nothing %s about, and a later run asks again.\n>\n",
		count(len(all), "open question", "open questions"), plural(len(all), "It doesn't", "They don't"), plural(len(all), "it asks", "they ask"))

	for _, item := range all[:min(len(all), boxQuestions)] {
		fmt.Fprintf(&box, "> %d. %s\n", item.n, item.name)
	}

	if more := len(all) - boxQuestions; more > 0 {
		fmt.Fprintf(&box, "> …and %d more, under Open questions.\n", more)
	}

	fmt.Fprintf(&box, "\n**To answer a question,** open a PR into %s that does what its Decide line says. If the same case can come up again, add a rule to %s instead.\n\n",
		base.Branch, base.fileLink("RULES.md", reconcile.RulesPath, 0))

	return box.String()
}

// questionsSection is Open questions: each question in full, with the box's
// number and name.
func (base Base) questionsSection(all []numbered) string {
	if len(all) == 0 {
		return ""
	}

	var section strings.Builder

	fmt.Fprintf(&section, "### Open questions (%d)\n\n", len(all))

	for _, item := range all {
		if item.place.prefix != "" {
			section.WriteString(base.regionBlock(item))
		} else {
			section.WriteString(base.questionBlock(item))
		}
	}

	return section.String()
}

// folded puts content behind a <details> fold with summary as its label.
func folded(summary, content string) string {
	return "<details><summary>" + summary + "</summary>\n\n" + content + "</details>\n\n"
}

// questionBlock renders one question: its offerings, its rule, the two
// sides and the agent's notes, the Decide line, and the sources.
func (base Base) questionBlock(q numbered) string {
	item := q.row

	why := "rule " + base.ruleLink(item.Rule)
	if item.Rule == reconcile.JudgmentRule {
		why = "no rule covers this (" + base.ruleLink(item.Rule) + ")"
	}

	sources := make([]string, 0, len(item.SourceURLs))
	for _, source := range item.SourceURLs {
		sources = append(sources, link(source))
	}

	block := fmt.Sprintf("**%d. %s** · %s\n", q.n, q.name, why)
	if len(q.offerings) > 1 {
		block += "- **Offerings:** " + valueList(q.offerings) + "\n"
	}

	if item.Catalogue != "" {
		block += "- **We have:** " + code(item.Catalogue) + "\n"
	}

	if item.Page != "" {
		block += "- **Vendor's page:** " + code(item.Page) + "\n"
	}

	// Validate kept the agent's words to characters that can't make markup.
	if item.Why != "" {
		block += "- **Why the bot asks:** " + item.Why + "\n"
	}

	if len(item.Values) > 0 {
		block += "- **Agent's notes:** " + valueList(item.Values) + "\n"
	}

	if decide := base.decide(q); decide != "" {
		block += "- **Decide:** " + decide + "\n"
	}

	return block + "- **Sources:** " + strings.Join(sources, ", ") + "\n\n"
}

// decide is a question's Decide line: the rule's fixed question, or for a
// no-rule row add or leave out a new model or copy, and keep or change a
// value both sides give, each with the decision that stops the question.
func (base Base) decide(q numbered) string {
	item, n := q.row, len(q.offerings)
	decisions := base.fileLink("decisions.yaml", reconcile.DecisionsPath, 0)

	switch {
	case item.Rule != reconcile.JudgmentRule:
	case item.Field == reconcile.FieldNewCopy || item.Field == reconcile.FieldNewModel:
		what := plural(n, "this copy", "these copies")
		if item.Field == reconcile.FieldNewModel {
			what = plural(n, "this model", "these models")
		}

		them := plural(n, "it", "them")

		return "add " + what + ", or leave " + them + " out? To leave " + them + " out and stop the question coming back, add a `" + reconcile.KindHold + "` for " + plural(n, "it", "each") + " to " + decisions + "."
	case item.Catalogue != "" && item.Page != "":
		return "keep what we have, or change it to match the vendor's page? To keep it and stop the question coming back, add a `" + reconcile.KindKeep + "` for " + plural(n, "it", "each") + " to " + decisions + "."
	}

	return base.Rules.Questions[item.Rule]
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
					note, noted[key] = base.rowEffect(change, provider), true
				}

				cells = append(cells, note)
			}

			fmt.Fprintf(&section, "| %s |\n", strings.Join(append(cells, link(change.SourceURL)), " | "))
		}

		section.WriteString("\n")
	}

	return section.String()
}

// ruleDrafts lists each rule draft the agent gave two or more no-rule
// offerings across the PR, with the questions it would settle, so a person
// can turn a case that keeps coming into a rule. A draft on one offering
// shows none.
func ruleDrafts(asks []numbered, needs []NeedsHuman) string {
	type draft struct{ when, does string }

	tidy := func(s string) string { return strings.TrimRight(strings.Join(strings.Fields(s), " "), ".") }
	of := func(item NeedsHuman) draft { return draft{tidy(item.RuleWhen), tidy(item.RuleDoes)} }

	drafted := slices.DeleteFunc(slices.Clone(needs), func(item NeedsHuman) bool { return item.Rule != reconcile.JudgmentRule || item.RuleWhen == "" })
	keys, offerings := groupBy(drafted, func(item NeedsHuman) (draft, string) { return of(item), item.Offering })

	var lines []string

	for _, key := range keys {
		if len(offerings[key]) < 2 {
			continue
		}

		var settles []int

		for _, q := range asks {
			if q.place.prefix == "" && q.row.Rule == reconcile.JudgmentRule && of(q.row) == key {
				settles = append(settles, q.n)
			}
		}

		which := count(len(offerings[key]), "offering", "offerings")
		if named := questionNumbers(settles); named != "" {
			which = named + " (" + which + ")"
		}

		lines = append(lines, "- "+which+": When "+key.when+", the bot "+key.does+".\n")
	}

	if len(lines) == 0 {
		return ""
	}

	return "**Rules that would settle these** (the agent's drafts, for cases on two or more offerings; to use one, add it to RULES.md in a reviewed PR):\n\n" +
		strings.Join(lines, "") + "\n"
}

// questionNumbers names questions by number: "Question 2", "Questions 2-5"
// for three or more in a row, else "Questions 1 and 3".
func questionNumbers(ns []int) string {
	switch {
	case len(ns) == 0:
		return ""
	case len(ns) == 1:
		return fmt.Sprintf("Question %d", ns[0])
	case len(ns) > 2 && ns[len(ns)-1]-ns[0] == len(ns)-1:
		return fmt.Sprintf("Questions %d-%d", ns[0], ns[len(ns)-1])
	}

	words := make([]string, 0, len(ns))
	for _, n := range ns {
		words = append(words, fmt.Sprint(n))
	}

	return "Questions " + strings.Join(words[:len(words)-1], ", ") + " and " + words[len(words)-1]
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
	// noRow is how many offerings and models unreported covers.
	noRow int
}

func (base Base) part(vendor Vendor) vendorPart {
	part := vendorPart{Vendor: vendor, base: base}
	part.base.Before, part.base.After = vendor.Before, vendor.After
	part.shown, part.held = part.base.hold(vendor.Report, vendor.Name)
	part.shown, part.waiting = part.base.splitWaiting(part.shown)
	part.asks = questions(part.shown.NeedsHuman)
	// Held and waiting rows still count as rows here.
	part.unreported, part.noRow = part.base.unreported(vendor.Report, vendor.Name)

	return part
}

// rows counts the rows the vendor's section shows.
func (part vendorPart) rows() int {
	return len(part.shown.Changes) + len(part.shown.NeedsHuman) + len(part.shown.Skipped) + len(part.waiting) + len(part.held)
}

// Render writes the PR body: what applies to the whole PR once, then one
// section per vendor. Call it only on reports Validate accepted and code
// files ValidateCodeFiles accepted. codeFiles are the files whose change isn't
// data only, listed first so a reviewer reads them as code. snapshotDiff only
// tells a changed catalogue from no drift: the diff is under Files changed.
func Render(vendors []Vendor, snapshotDiff string, codeFiles []string, base Base) string {
	top, sections := base.render(vendors, codeFiles, snapshotDiff != "")
	return top + strings.Join(sections, "")
}

// render writes the top of the body and each vendor's section.
func (base Base) render(vendors []Vendor, codeFiles []string, hasDiff bool) (string, []string) {
	var body strings.Builder

	names := make([]string, 0, len(vendors))
	parts := make([]vendorPart, 0, len(vendors))
	rows, failed := 0, false

	var (
		waiting []Skipped
		needs   []NeedsHuman
	)

	for _, vendor := range vendors {
		part := base.part(vendor)
		names, parts = append(names, vendor.Name), append(parts, part)
		waiting, needs = append(waiting, part.waiting...), append(needs, part.shown.NeedsHuman...)
		rows += part.rows()
		failed = failed || vendor.Failed
	}

	body.WriteString("## Catalogue drift: " + strings.Join(names, ", ") + "\n\n")

	if rows == 0 && !failed && !hasDiff {
		body.WriteString("No drift found.\n\n" + base.summary())
		return body.String(), nil
	}

	asks := base.numberQuestions(parts, waiting)

	body.WriteString(base.contextLine() + summaryLine(parts) + base.checksLine(codeFiles) + base.carriedLine() + base.questionsBox(asks))

	if len(codeFiles) > 0 {
		fmt.Fprintf(&body, "### Code changed (%d)\n\n", len(codeFiles))

		for _, file := range codeFiles {
			body.WriteString("- " + code(file) + "\n")
		}

		body.WriteString("\n")
	}

	body.WriteString(base.routingSection())

	body.WriteString(impactTable(parts))

	// Pushing, and the recorded decisions, come after what needs a decision.
	// A run carries a pushed commit onto the rebuilt branch, or stops and
	// says why (plan.sh, carry.sh); REVIEW.md has the details.
	body.WriteString("You can push commits here: the bot never removes them. It doesn't read comments.\n\n")

	body.WriteString(base.summary())

	// What the PR asks of people takes a share of the room like a vendor's
	// section, so many questions can't push every Changes table out of a
	// capped body.
	sections := make([]string, 0, len(parts)+1)
	if shared := base.questionsSection(asks) + ruleDrafts(asks, needs); shared != "" {
		sections = append(sections, shared)
	}

	for _, part := range parts {
		sections = append(sections, part.section())
	}

	return body.String(), sections
}

// routingSection is Routing changes: a diagram of each move, then a table row
// per move with the regions, the input price change and where the new copy
// processes data. A pair the SDK refuses, or the patch removes, reads as a
// call that fails.
func (base Base) routingSection() string {
	moves := base.routeMoves()
	if len(moves) == 0 {
		return ""
	}

	var section strings.Builder

	fmt.Fprintf(&section, "### Routing changes (%d)\n\n⚑ Bare-name calls from these regions now go to another copy. Check the price and where the data is processed.\n\n%s"+
		"| Model | Callers in | Today | After | Input price | Data processed in |\n|---|---|---|---|---|---|\n", len(moves), base.routingDiagram(moves))

	for _, m := range moves {
		fmt.Fprintf(&section, "| %s | %s | %s | %s | %s | %s |\n",
			code(m.name), valueList(m.regions), routed(m.from), routed(m.to), base.routeInputChange(m.from, m.to), processedIn(m.to))
	}

	return section.String() + "\n"
}

// impactTable is "What changes for customers": one row per kind of change,
// with how many offerings it touches and what it means for SDK users and for
// ADP. A row without the vendor's catalogue has no kind, and isn't counted.
func impactTable(parts []vendorPart) string {
	counts := map[string]int{}

	for _, part := range parts {
		// rowKinds reads the whole offering, so two rows of one offering
		// name the same kinds; each offering counts once per kind.
		counted := map[[2]string]bool{}

		for _, change := range part.shown.Changes {
			for _, kind := range part.base.rowKinds(change, part.Name) {
				if key := [2]string{change.Offering, kind}; !counted[key] {
					counts[kind]++
					counted[key] = true
				}
			}
		}
	}

	if len(counts) == 0 {
		return ""
	}

	var table strings.Builder

	table.WriteString("### What changes for customers\n\n| Change | Offerings | SDK users, once they upgrade | ADP, once it moves to this SDK and redeploys |\n|---|---|---|---|\n")

	for _, kind := range impacts {
		if n := counts[kind.key]; n > 0 {
			fmt.Fprintf(&table, "| %s | %d | %s | %s |\n", kind.name, n, kind.sdk, kind.adp)
		}
	}

	return table.String() + "\n"
}

// kindNouns names each report field as the summary line counts it, one and
// many. TestRulesFile holds its keys to fields.
var kindNouns = map[string][2]string{
	"new_model": {"new model", "new models"}, "new_copy": {"new copy", "new copies"}, "price": {"price", "prices"},
	"limit": {"token limit", "token limits"}, "lifecycle": {"lifecycle date", "lifecycle dates"}, "replacement": {"replacement", "replacements"},
	"knowledge_cutoff": {"knowledge cutoff", "knowledge cutoffs"}, "release_date": {"release date", "release dates"},
	"capability": {"capability", "capabilities"}, "location": {"location", "locations"},
}

// plural is one when n is 1, else many.
func plural(n int, one, many string) string {
	if n == 1 {
		return one
	}

	return many
}

// count is n of something, in the singular when n is 1.
func count(n int, one, many string) string {
	return fmt.Sprintf("%d %s", n, plural(n, one, many))
}

// summaryLine counts the PR's changes by kind, in the order they first
// appear, and the rows that leave the catalogue as it is.
func summaryLine(parts []vendorPart) string {
	var (
		changes []Change
		waiting []Skipped
	)

	unreported, asks, held, skipped := 0, 0, 0, 0

	for _, part := range parts {
		changes = append(changes, part.shown.Changes...)
		waiting = append(waiting, part.waiting...)
		unreported += part.noRow
		asks += len(part.asks)
		held += len(part.held)
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
			kinds = append(kinds, count(unreported, "change with no row", "changes with no row"))
		}

		line = "**" + count(total, "change", "changes") + "** to the catalogue: " + strings.Join(kinds, ", ") + "."
	}

	var same []string

	regions, _ := byRegion(waiting)

	for _, rows := range []struct {
		n         int
		one, many string
	}{
		{asks + len(regions), "open question", "open questions"},
		{held, "held by a decision", "held by decisions"},
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

// checksLine is what a reviewer must do before marking the PR ready: check
// the values, since pr-job.sh checks only a report's form, and read the code
// files, which run with secrets once the PR is ready.
func (base Base) checksLine(codeFiles []string) string {
	line := "**Before you mark it ready:** check each value in Changes against its source, because no automated check does."

	switch n := len(codeFiles); {
	case n == 1:
		line += " Also read the file under Code changed as code: marking ready runs it with secrets."
	case n > 1:
		line += fmt.Sprintf(" Also read the %d files under Code changed as code: marking ready runs them with secrets.", n)
	}

	return line + "\n\n"
}

// contextLine says what the bot is, for a reviewer who hasn't met it, and
// links the review guide.
func (base Base) contextLine() string {
	return "**Context:** an AI agent per vendor checks the SDK's model catalog against the vendor's docs, and this PR holds the fixes it proposes. " +
		base.fileLink("REVIEW.md", "REVIEW.md#bot-catalog-prs", 0) + " says how to review a bot PR.\n\n"
}

// section renders one vendor's rows under its own heading.
func (part vendorPart) section() string {
	var body strings.Builder

	base, report := part.base, part.shown
	body.WriteString("## " + part.Name + "\n\n")

	if part.Replayed.Run != "" {
		// replay.sh checked it makes exactly the same change on this base.
		run := "run on " + part.Replayed.Date + " (run " + part.Replayed.Run + ")"
		if repo, _, found := strings.Cut(base.URL, "/blob/"); found {
			run = "[run on " + part.Replayed.Date + "](" + repo + "/actions/runs/" + part.Replayed.Run + ")"
		}

		fmt.Fprintf(&body, "Reused from the agent's %s; no new agent run this time.\n\n", run)
	}

	switch {
	case part.Failed:
		body.WriteString("This vendor's run failed, so this PR has nothing from it. The workflow run shows why.\n\n")
		return body.String()
	case part.rows() == 0 && len(part.unreported) == 0:
		body.WriteString("The run found nothing to change.\n\n")
		return body.String()
	}

	head := body.Len()

	if len(report.Changes) > 0 {
		body.WriteString(base.changesSection(report.Changes, part.Name))
	}

	if names := part.unreported; len(names) > 0 {
		fmt.Fprintf(&body, "### Changed with no row (%d)\n\nThe patch changes these offerings, but the report has no row for them. Check them in `catalog/snapshot.json` under Files changed.\n\n- %s\n\n",
			len(names), valueList(names))
	}

	if len(part.held) > 0 {
		fmt.Fprintf(&body, "### Held by decisions (%d)\n\nQuestions a recorded decision already answers. Nothing here changes.\n\n| Decision | Offering | Field | What the bot saw | Why it's settled |\n|---|---|---|---|---|\n", len(part.held))
		body.WriteString(strings.Join(part.held, "") + "\n")
	}

	if len(report.Skipped) > 0 {
		body.WriteString(folded(fmt.Sprintf("Skipped (%d): left out by a rule", len(report.Skipped)),
			"| Rule | Why | Offerings |\n|---|---|---|\n"+strings.Join(base.skippedRows(report.Skipped), "")+"\n"))
	}

	// Its questions, its waiting copies' regions among them, show once for
	// the PR, above.
	if body.Len() == head {
		body.WriteString("The run changed nothing; its rows are under Open questions above.\n\n")
	}

	return body.String()
}

// cutNote ends a body whose tables alone don't fit.
const cutNote = "\n\n… (the rest is cut to fit GitHub's size limit; the full report is in the workflow artifacts)\n"

// RenderCapped is Render, kept within limit bytes so GitHub accepts the PR
// body: each vendor's section gets an equal share of the room the top
// leaves, so one big vendor can't push the others out, cut only between
// lines.
func RenderCapped(vendors []Vendor, snapshotDiff string, codeFiles []string, base Base, limit int) string {
	top, sections := base.render(vendors, codeFiles, snapshotDiff != "")

	body := top + strings.Join(sections, "")
	if len(body) <= limit {
		return body
	}

	// Cut so no table row or fence is left half written.
	if body = top + strings.Join(fit(sections, limit-len(top)), ""); len(body) <= limit {
		return body
	}

	return closeOpen(cutOnLine(body, limit-len(cutNote)-len(closings))) + cutNote
}

// closings is the most closeOpen adds: a code fence and a fold.
const closings = "\n```\n</details>\n"

// closeOpen closes a code fence and a <details> fold that a cut left open,
// so the cut note and what follows don't read as part of them.
func closeOpen(text string) string {
	if strings.Count(text, "```")%2 == 1 {
		text += "\n```"
	}

	if strings.Count(text, "<details>") > strings.Count(text, "</details>") {
		text += "\n</details>"
	}

	return text
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
			fitted[i] = closeOpen(cutOnLine(fitted[i], share-len(cutNote)-len(closings)-1)) + cutNote + "\n"
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
