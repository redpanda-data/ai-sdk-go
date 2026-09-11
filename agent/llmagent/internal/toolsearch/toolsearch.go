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

// Package toolsearch ranks tool definitions against a model's query. It
// implements the query grammar of the tool_search tool (exact "select:" names,
// a "+term" name filter, free keywords) and BM25 ranking over each tool's name,
// group, description and schema. Callers need only Parse and Rank; everything
// below them is the search engine.
package toolsearch

import (
	"cmp"
	"encoding/json"
	"math"
	"slices"
	"strings"
	"unicode"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

type Query struct {
	// Selected is non-nil only for the select: form, and then holds the exact
	// names requested.
	Selected []string

	// Required, when set, must appear in a tool's name for it to rank.
	Required string

	// Terms are the ranking terms.
	Terms []string
}

// Parse accepts select:name,..., a +name filter, or keywords.
func Parse(raw string) Query {
	trimmed := strings.TrimSpace(raw)

	if rest, ok := cutPrefixFold(trimmed, "select:"); ok {
		// Non-nil even when empty: the caller distinguishes "asked for exact
		// names and none resolved" from "asked for keywords".
		selected := []string{}

		for part := range strings.SplitSeq(rest, ",") {
			name := strings.TrimSpace(part)
			if name != "" {
				selected = append(selected, name)
			}
		}

		return Query{Selected: selected}
	}

	var query Query

	for word := range strings.FieldsSeq(trimmed) {
		if after, ok := strings.CutPrefix(word, "+"); ok && query.Required == "" {
			required := strings.ToLower(strings.TrimSpace(after))
			if required != "" {
				query.Required = required
				continue
			}
		}

		query.Terms = append(query.Terms, tokenize(word)...)
	}

	// Count each query term once when computing document frequencies.
	query.Terms = dedupe(query.Terms)

	return query
}

// dedupe removes repeats while preserving first-seen order.
func dedupe(values []string) []string {
	if len(values) < 2 {
		return values
	}

	seen := make(map[string]bool, len(values))
	unique := make([]string, 0, len(values))

	for _, value := range values {
		if seen[value] {
			continue
		}

		seen[value] = true
		unique = append(unique, value)
	}

	return unique
}

func cutPrefixFold(s, prefix string) (string, bool) {
	if len(s) < len(prefix) || !strings.EqualFold(s[:len(prefix)], prefix) {
		return s, false
	}

	return s[len(prefix):], true
}

// searchDoc is one deferred tool prepared for ranking.
type searchDoc struct {
	name   string
	tokens []string
}

// BM25 parameters. The standard defaults; length normalization matters here
// because MCP tool descriptions range from a few words to several paragraphs.
const (
	bm25K1 = 1.2
	bm25B  = 0.75

	// Field weight as token repetition, which keeps the scorer single-field.
	// A name match should beat a description match: the model is choosing
	// between names it can already read.
	nameWeight  = 3
	groupWeight = 2
)

// Rank ranks names, groups, descriptions, and schema text using BM25.
// A non-positive limit returns all matches.
func Rank(deferred []llm.ToolDefinition, query Query, limit int) []string {
	if len(query.Terms) == 0 {
		// A bare "+term" with nothing to rank by is a name filter, in
		// registry order.
		if query.Required == "" {
			return nil
		}

		var names []string

		for _, def := range deferred {
			if strings.Contains(strings.ToLower(def.Name), query.Required) {
				names = append(names, def.Name)
			}
		}

		return truncate(names, limit)
	}

	docs := make([]searchDoc, 0, len(deferred))

	for _, def := range deferred {
		if query.Required != "" && !strings.Contains(strings.ToLower(def.Name), query.Required) {
			continue
		}

		docs = append(docs, searchDoc{name: def.Name, tokens: documentTokens(def)})
	}

	if len(docs) == 0 {
		return nil
	}

	scores := bm25(docs, query.Terms)

	ranked := make([]string, 0, len(scores))
	for _, scored := range scores {
		ranked = append(ranked, scored.name)
	}

	return truncate(ranked, limit)
}

type scoredDoc struct {
	name  string
	score float64
}

func bm25(docs []searchDoc, terms []string) []scoredDoc {
	docFreq := make(map[string]int, len(terms))
	totalLen := 0

	for _, doc := range docs {
		totalLen += len(doc.tokens)

		present := make(map[string]bool, len(doc.tokens))
		for _, token := range doc.tokens {
			present[token] = true
		}

		for _, term := range terms {
			if present[term] {
				docFreq[term]++
			}
		}
	}

	avgLen := float64(totalLen) / float64(len(docs))
	if avgLen == 0 {
		return nil
	}

	n := float64(len(docs))
	scored := make([]scoredDoc, 0, len(docs))

	for _, doc := range docs {
		termFreq := make(map[string]int, len(doc.tokens))
		for _, token := range doc.tokens {
			termFreq[token]++
		}

		var score float64

		for _, term := range terms {
			freq := float64(termFreq[term])
			if freq == 0 {
				continue
			}

			df := float64(docFreq[term])
			idf := math.Log(1 + (n-df+0.5)/(df+0.5))
			norm := bm25K1 * (1 - bm25B + bm25B*float64(len(doc.tokens))/avgLen)

			score += idf * (freq * (bm25K1 + 1)) / (freq + norm)
		}

		if score > 0 {
			scored = append(scored, scoredDoc{name: doc.name, score: score})
		}
	}

	// Score descending, then name ascending: a deterministic total order, so an
	// unchanged registry and query always produce the same result.
	slices.SortFunc(scored, func(a, b scoredDoc) int {
		if c := cmp.Compare(b.score, a.score); c != 0 {
			return c
		}

		return cmp.Compare(a.name, b.name)
	})

	return scored
}

func truncate(names []string, limit int) []string {
	if limit > 0 && len(names) > limit {
		return names[:limit]
	}

	return names
}

// documentTokens builds the searchable token bag for one tool: name and group
// name weighted, then the group's description, the tool's description and its
// schema's argument names and descriptions.
func documentTokens(def llm.ToolDefinition) []string {
	tokens := make([]string, 0, 64)

	nameTokens := tokenize(def.Name)
	for range nameWeight {
		tokens = append(tokens, nameTokens...)
	}

	groupTokens := tokenize(def.Group.Name)
	for range groupWeight {
		tokens = append(tokens, groupTokens...)
	}

	tokens = append(tokens, tokenize(def.Group.Description)...)
	tokens = append(tokens, tokenize(def.Description)...)
	tokens = append(tokens, schemaTokens(def.Parameters)...)

	return tokens
}

// maxSchemaDepth bounds the searchable nesting depth; $ref values are not resolved.
const maxSchemaDepth = 6

// schemaTokens collects argument names and descriptions from a JSON Schema.
func schemaTokens(raw json.RawMessage) []string {
	if len(raw) == 0 {
		return nil
	}

	var decoded any

	err := json.Unmarshal(raw, &decoded)
	if err != nil {
		return nil
	}

	var tokens []string

	walkSchema(decoded, 0, &tokens)

	return tokens
}

func walkSchema(node any, depth int, tokens *[]string) {
	if depth > maxSchemaDepth {
		return
	}

	switch typed := node.(type) {
	case map[string]any:
		for key, value := range typed {
			switch key {
			case "properties":
				properties, ok := value.(map[string]any)
				if !ok {
					continue
				}

				for name, schema := range properties {
					*tokens = append(*tokens, tokenize(name)...)
					walkSchema(schema, depth+1, tokens)
				}
			case "description", "title":
				text, ok := value.(string)
				if ok {
					*tokens = append(*tokens, tokenize(text)...)
				}
			case "enum":
				// Enum values name real-world entities ("incident",
				// "change_request") and are often the most searchable text in
				// an otherwise undescribed schema.
				values, ok := value.([]any)
				if !ok {
					continue
				}

				for _, entry := range values {
					text, ok := entry.(string)
					if ok {
						*tokens = append(*tokens, tokenize(text)...)
					}
				}
			default:
				walkSchema(value, depth+1, tokens)
			}
		}
	case []any:
		for _, entry := range typed {
			walkSchema(entry, depth+1, tokens)
		}
	}
}

// tokenize normalizes case, word separators, camelCase, and regular plurals.
func tokenize(text string) []string {
	if text == "" {
		return nil
	}

	var (
		tokens  []string
		current strings.Builder
	)

	flush := func() {
		if current.Len() > 0 {
			tokens = append(tokens, singularize(current.String()))
			current.Reset()
		}
	}

	runes := []rune(text)
	for i, r := range runes {
		switch {
		case unicode.IsUpper(r):
			// A capital starts a new token at a lower-to-upper or digit-to-upper
			// boundary ("createIncident", "oauth2Token") but not after another
			// capital, so "ID" and "HTTPServer" do not shatter into single
			// letters.
			if i > 0 && (unicode.IsLower(runes[i-1]) || unicode.IsDigit(runes[i-1])) {
				flush()
			}

			current.WriteRune(unicode.ToLower(r))
		case unicode.IsLetter(r) || unicode.IsDigit(r):
			current.WriteRune(unicode.ToLower(r))
		default:
			flush()
		}
	}

	flush()

	return tokens
}

// singularize folds regular English plurals for both documents and queries.
func singularize(token string) string {
	if len(token) < 4 {
		return token
	}

	if strings.HasSuffix(token, "ies") && len(token) >= 5 {
		return token[:len(token)-3] + "y"
	}

	if !strings.HasSuffix(token, "s") {
		return token
	}

	// "status", "class", "analysis": a final s that is part of the stem.
	for _, keep := range [...]string{"ss", "us", "is"} {
		if strings.HasSuffix(token, keep) {
			return token
		}
	}

	return token[:len(token)-1]
}
