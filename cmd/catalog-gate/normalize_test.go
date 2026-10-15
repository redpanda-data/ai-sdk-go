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
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// lines repeats a row so a fixture clears the content floor.
func lines(row string) string { return strings.Repeat(row+"\n", 12) }

func TestNormalizeMarkdown(t *testing.T) {
	t.Parallel()

	base := "# Claude pricing\n\n| Model | Input |\n|---|---|\n| Claude Opus 5.5 | $5 |\n![Supported](https://example.test/icons/icon-yes.png)\n" + lines("| Claude row | $1 |")
	plain := page{ID: "p", Type: typeMarkdown, MustContain: "Claude"}

	tests := []struct {
		name    string
		body    string
		same    bool
		wantErr string
	}{
		{"extra whitespace and zero-width characters", strings.ReplaceAll(base, "| $5 |", "|   $5\u200b   |") + "\n\n\n", true, ""},
		{"an image's alt text", strings.Replace(base, "![Supported]", "![Green check]", 1), true, ""},
		{"an icon flip", strings.Replace(base, "icon-yes.png", "icon-no.png", 1), false, ""},
		{"a price edit", strings.Replace(base, "$5", "$6", 1), false, ""},
		{"no Claude on the page", strings.ReplaceAll(base, "Claude", "Model"), false, `doesn't contain "Claude"`},
		{"too few lines", "# Claude\n", false, "lines"},
	}
	want, err := normalize(plain, []byte(base))
	require.NoError(t, err)
	assert.Contains(t, want, "[img:icon-yes.png]")

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got, err := normalize(plain, []byte(tt.body))
			if tt.wantErr != "" {
				assert.ErrorContains(t, err, tt.wantErr)
				return
			}

			require.NoError(t, err)
			assert.Equal(t, tt.same, hash(got) == hash(want), "\n%s", strings.Join(got, "\n"))
		})
	}
}

func TestNormalizeHTML(t *testing.T) {
	t.Parallel()

	wrap := func(nonce, root string) string {
		return `<html><head><script nonce="` + nonce + `">var x=1</script><style>p{}</style></head><body>` +
			`<nav>menu</nav><header>top</header><div class="devsite-article-body clearfix">` + root +
			`</div><footer>bottom</footer><script nonce="` + nonce + `">track()</script></body></html>`
	}
	// Pretty-printed like Google's pages: line breaks between cells are layout.
	root := "<h2>Claude on Vertex</h2><table>\n<tr>\n<th>Model</th>\n<th>Price</th>\n</tr>\n<tr>\n<td>Claude <b>Opus</b></td>\n<td>$5<br>per 1M</td>\n</tr>\n</table>" +
		`<p>Supported <img src="/icons/icon-yes.png" alt="yes"></p>` + strings.Repeat("<p>Claude line</p>", 12)
	plain := page{ID: "p", Type: typeHTML, MustContain: "Claude"}

	tests := []struct {
		name    string
		body    string
		same    bool
		wantErr string
	}{
		{"new nonces, scripts and nav", strings.ReplaceAll(wrap("n2", root), "menu", "other menu"), true, ""},
		{"attribute order", strings.Replace(wrap("n1", root), `src="/icons/icon-yes.png" alt="yes"`, `alt="yes" src="/icons/icon-yes.png"`, 1), true, ""},
		{"source reflowed", wrap("n1", strings.ReplaceAll(root, "\n", "")), true, ""},
		{"a price edit in the root", wrap("n1", strings.Replace(root, "$5", "$6", 1)), false, ""},
		{"an icon flip", wrap("n1", strings.Replace(root, "icon-yes", "icon-no", 1)), false, ""},
		{"no root", "<html><body>" + root + "</body></html>", false, "content roots, want 1"},
		{"two roots", wrap("n1", root) + `<div class="devsite-article-body">x</div>`, false, "content roots, want 1"},
		{"the 2-byte stub Google serves some clients", "OK", false, "content root"},
	}
	want, err := normalize(plain, []byte(wrap("n1", root)))
	require.NoError(t, err)
	assert.Contains(t, want, "Model | Price")
	assert.Contains(t, want, "Claude Opus | $5 per 1M", "one line per table row")
	assert.Contains(t, want, "Supported [img:icon-yes.png]")

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got, err := normalize(plain, []byte(tt.body))
			if tt.wantErr != "" {
				assert.ErrorContains(t, err, tt.wantErr)
				return
			}

			require.NoError(t, err)
			assert.Equal(t, tt.same, hash(got) == hash(want), "\n%s", strings.Join(got, "\n"))
		})
	}

	fragment := page{ID: "p", Type: typeHTML, MustContain: "Claude", Fragment: "claude"}
	got, err := normalize(fragment, []byte(wrap("n1", `<section id="claude">`+root+`</section><p>Gemini $1</p>`)))
	require.NoError(t, err)
	assert.NotContains(t, strings.Join(got, "\n"), "Gemini", "a fragment leaves out the rest of the page")
}

// priceList builds an AWS price list: 60 Claude rates in each of three
// regions, plus rows the gate ignores. A rate whose region or usage type
// contains bump costs 1 more.
func priceList(version, bump string) string {
	var products, terms []string

	add := func(service, region, usage, unit string, price float64) {
		sku := fmt.Sprintf("SKU%d", len(products))
		products = append(products, fmt.Sprintf(`"%s": {"attributes": {"servicename": %q, "regionCode": %q, "usagetype": %q}}`, sku, service, region, usage))
		terms = append(terms, fmt.Sprintf(`"%s": {"%s.T": {"effectiveDate": "%s", "priceDimensions": {"%s.T.D": {"unit": %q, "beginRange": "0", "endRange": "Inf", "pricePerUnit": {"USD": "%.10f"}}}}}`, sku, sku, version, sku, unit, price))
	}

	for region, code := range map[string]string{"us-east-1": "USE1", "eu-west-1": "EU", "us-gov-west-1": "UGW1"} {
		for i := range 60 {
			usage := fmt.Sprintf("%s-MP:%s_input_tokens_tier%d-Units", code, code, i)
			price := 5 + float64(i)

			if bump != "" && (strings.Contains(region, bump) || strings.Contains(usage, bump)) {
				price++
			}

			add("Claude Opus 5.5 (Amazon Bedrock Edition)", region, usage, "1M tokens", price)
		}

		add("Claude Opus 5.5 (Amazon Bedrock Edition)", region, code+"-MP:"+code+"_InputTokenCount_Batch-Units", "1M tokens", 2.5)
		add("Claude Opus 5.5 (Amazon Bedrock Edition)", region, code+"-MP:"+code+"_input_tokens_batch-Units", "1M tokens", 2.5)
		add("Claude Opus 5.5 (Amazon Bedrock Edition)", region, code+"-MP:"+code+"_ProvisionedThroughput-Units", "1M TPM Hour", 400)
		add("Nova Pro (Amazon Bedrock Edition)", region, code+"-MP:"+code+"_input_tokens-Units", "1M tokens", 0.8)
	}

	return fmt.Sprintf(`{"version": %q, "publicationDate": "%s", "products": {%s}, "terms": {"OnDemand": {%s}}}`,
		version, version, strings.Join(products, ","), strings.Join(terms, ","))
}

func TestNormalizePriceList(t *testing.T) {
	t.Parallel()

	list := page{ID: "p", Type: typePriceList, Match: `^Claude .*\(Amazon Bedrock Edition\)$`, MinLines: 50}
	ignored := strings.NewReplacer(`"USD": "2.5000000000"`, `"USD": "9.0000000000"`, `"USD": "400.0000000000"`, `"USD": "401.0000000000"`, `"USD": "0.8000000000"`, `"USD": "0.9000000000"`)

	tests := []struct {
		name    string
		body    string
		same    bool
		wantErr string
	}{
		{"a new version and dates only", priceList("2026-10-14", ""), true, ""},
		{"batch, provisioned, GovCloud and other models move", ignored.Replace(priceList("2026-10-14", "gov")), true, ""},
		{"an on-demand Claude price", priceList("2026-10-14", "tier3-"), false, ""},
		{"one region's price", priceList("2026-10-14", "eu-west-1"), false, ""},
		{"a renamed service matches nothing", strings.ReplaceAll(priceList("2026-10-14", ""), "(Amazon Bedrock Edition)", "(Bedrock)"), false, "lines"},
		{"not JSON", "<html>", false, "price list"},
	}
	want, err := normalize(list, []byte(priceList("2026-10-01", "")))
	require.NoError(t, err)
	assert.Len(t, want, 60, "one row per usage type: regions with the same price collapse, the rest are ignored")

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got, err := normalize(list, []byte(tt.body))
			if tt.wantErr != "" {
				assert.ErrorContains(t, err, tt.wantErr)
				return
			}

			require.NoError(t, err)
			assert.Equal(t, tt.same, hash(got) == hash(want))
		})
	}
}
