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
	"bytes"
	"cmp"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"path"
	"regexp"
	"slices"
	"strings"

	"golang.org/x/net/html"
	"golang.org/x/net/html/atom"
)

// The page types the gate can fingerprint.
const (
	typeMarkdown  = "markdown"
	typeHTML      = "html"
	typePriceList = "pricelist"
)

// defaultMinLines is the content floor's line count, unless a page sets its
// own. A page below its floor is an error, so an error page or a stub never
// hashes as "unchanged".
const defaultMinLines = 10

var (
	markdownImage = regexp.MustCompile(`!\[[^\]]*\]\(([^)\s]+)[^)]*\)`)
	invisible     = strings.NewReplacer("\u200b", "", "\u200c", "", "\u200d", "", "\u2060", "", "\ufeff", "", "\u00ad", "")
	// usagePrefix is the region code at the start of an AWS usage type.
	usagePrefix = regexp.MustCompile(`^[A-Z0-9]+-MP:[A-Z0-9]+_`)
	batch       = regexp.MustCompile(`(?i)batch`)
	// The tags whose text a page's reader never sees as content.
	skipped = []atom.Atom{
		atom.Head, atom.Script, atom.Style, atom.Noscript, atom.Template, atom.Svg, atom.Iframe,
		atom.Object, atom.Canvas, atom.Nav, atom.Header, atom.Footer,
	}
	inline = []atom.Atom{
		atom.A, atom.Abbr, atom.B, atom.Bdi, atom.Bdo, atom.Cite, atom.Code, atom.Data, atom.Dfn, atom.Em,
		atom.I, atom.Kbd, atom.Mark, atom.Q, atom.S, atom.Samp, atom.Small, atom.Span, atom.Strong,
		atom.Sub, atom.Sup, atom.Time, atom.U, atom.Var, atom.Wbr,
	}
)

// normalize turns a fetched body into the lines the gate hashes: the text a
// reader sees, never the raw bytes, which change on every fetch of some pages.
// A page below its content floor is an error.
func normalize(p page, body []byte) ([]string, error) {
	var (
		text []string
		err  error
	)

	switch p.Type {
	case typeMarkdown:
		text = normalizeMarkdown(body)
	case typeHTML:
		text, err = normalizeHTML(body, p.Fragment)
	case typePriceList:
		text, err = normalizePriceList(body, p.Match)
	default:
		err = fmt.Errorf("unknown page type %q", p.Type)
	}

	switch minLines := cmp.Or(p.MinLines, defaultMinLines); {
	case err != nil:
		return nil, err
	case !strings.Contains(strings.Join(text, "\n"), p.MustContain):
		return nil, fmt.Errorf("the page doesn't contain %q", p.MustContain)
	case len(text) < minLines:
		return nil, fmt.Errorf("the page has %d lines, fewer than %d", len(text), minLines)
	}

	return text, nil
}

// hash is the sha256 of the lines, each ending in a newline.
func hash(lines []string) string {
	sum := sha256.New()
	for _, line := range lines {
		sum.Write([]byte(line + "\n"))
	}

	return hex.EncodeToString(sum.Sum(nil))
}

// cleanLines drops invisible characters, collapses whitespace and drops
// empty lines.
func cleanLines(text string) []string {
	var lines []string

	for line := range strings.SplitSeq(invisible.Replace(text), "\n") {
		if line = strings.Join(strings.Fields(line), " "); line != "" {
			lines = append(lines, line)
		}
	}

	return lines
}

// imageToken stands for an image by its file name. Support tables on AWS
// pages are icons, and their alt text is generated and changes by itself.
func imageToken(src string) string { return "[img:" + path.Base(src) + "]" }

func normalizeMarkdown(body []byte) []string {
	return cleanLines(markdownImage.ReplaceAllStringFunc(string(body), func(image string) string {
		return imageToken(markdownImage.FindStringSubmatch(image)[1])
	}))
}

// normalizeHTML keeps the text of the page's one content root: the element
// whose id is fragment, or else the element with the class
// devsite-article-body. There is no fallback to <body>, which the parser adds
// to any input, so a stub or an error page fails instead of matching.
func normalizeHTML(body []byte, fragment string) ([]string, error) {
	doc, err := html.Parse(bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	isRoot := func(node *html.Node) bool {
		for _, attr := range node.Attr {
			if fragment != "" && attr.Key == "id" && attr.Val == fragment ||
				fragment == "" && attr.Key == "class" && slices.Contains(strings.Fields(attr.Val), "devsite-article-body") {
				return true
			}
		}

		return false
	}

	var roots []*html.Node

	var find func(*html.Node)

	find = func(node *html.Node) {
		if node.Type == html.ElementNode && isRoot(node) {
			roots = append(roots, node)
			return
		}

		for child := node.FirstChild; child != nil; child = child.NextSibling {
			find(child)
		}
	}
	find(doc)

	if len(roots) != 1 {
		return nil, fmt.Errorf("the page has %d content roots, want 1", len(roots))
	}

	var text strings.Builder

	writeText(&text, roots[0])

	return cleanLines(text.String()), nil
}

// writeText writes a node's visible text: a line per block and one per table
// row, with " | " between cells. Line breaks in the source are only layout.
func writeText(text *strings.Builder, node *html.Node) {
	switch {
	case node.Type == html.TextNode:
		text.WriteString(strings.ReplaceAll(node.Data, "\n", " "))
		return
	case node.Type != html.ElementNode || slices.Contains(skipped, node.DataAtom):
		return
	case node.DataAtom == atom.Img:
		for _, attr := range node.Attr {
			if attr.Key == "src" {
				text.WriteString(" " + imageToken(attr.Val) + " ")
			}
		}

		return
	case node.DataAtom == atom.Br:
		text.WriteString("\n")
		return
	case node.DataAtom == atom.Tr:
		var cells []string

		for child := node.FirstChild; child != nil; child = child.NextSibling {
			if child.DataAtom == atom.Td || child.DataAtom == atom.Th {
				var cell strings.Builder

				writeText(&cell, child)
				cells = append(cells, strings.Join(strings.Fields(cell.String()), " "))
			}
		}

		text.WriteString("\n" + strings.Join(cells, " | ") + "\n")

		return
	}

	block := !slices.Contains(inline, node.DataAtom)
	if block {
		text.WriteString("\n")
	}

	for child := node.FirstChild; child != nil; child = child.NextSibling {
		writeText(text, child)
	}

	if block {
		text.WriteString("\n")
	}
}

// normalizePriceList keeps one line per on-demand per-token rate of the
// services that match, outside GovCloud and batch, with the region dropped:
// regions with the same price collapse into one line, and AWS republishing
// the file for other services changes nothing.
func normalizePriceList(body []byte, match string) ([]string, error) {
	service, err := regexp.Compile(match)
	if err != nil {
		return nil, fmt.Errorf("bad match: %w", err)
	}

	//nolint:tagliatelle // AWS's field names
	var list struct {
		Products map[string]struct {
			Attributes struct {
				Servicename string `json:"servicename"`
				RegionCode  string `json:"regionCode"`
				Usagetype   string `json:"usagetype"`
			} `json:"attributes"`
		} `json:"products"`
		Terms struct {
			OnDemand map[string]map[string]struct {
				PriceDimensions map[string]struct {
					Unit         string            `json:"unit"`
					BeginRange   string            `json:"beginRange"`
					EndRange     string            `json:"endRange"`
					PricePerUnit map[string]string `json:"pricePerUnit"`
				} `json:"priceDimensions"`
			} `json:"OnDemand"`
		} `json:"terms"`
	}

	if err := json.Unmarshal(body, &list); err != nil {
		return nil, fmt.Errorf("not a price list: %w", err)
	}

	var rows []string

	for sku, product := range list.Products {
		attrs := product.Attributes
		if !service.MatchString(attrs.Servicename) || strings.HasPrefix(attrs.RegionCode, "us-gov-") || batch.MatchString(attrs.Usagetype) {
			continue
		}

		for _, term := range list.Terms.OnDemand[sku] {
			for _, dim := range term.PriceDimensions {
				if dim.Unit == "1M tokens" {
					rows = append(rows, strings.Join([]string{attrs.Servicename, usagePrefix.ReplaceAllString(attrs.Usagetype, ""), dim.BeginRange, dim.EndRange, dim.PricePerUnit["USD"]}, "\t"))
				}
			}
		}
	}

	slices.Sort(rows)

	return slices.Compact(rows), nil
}
