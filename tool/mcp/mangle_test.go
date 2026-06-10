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

package mcp

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMangleHeadIfTooLong(t *testing.T) {
	t.Parallel()

	t.Run("short names pass through unchanged", func(t *testing.T) {
		t.Parallel()

		assert.Equal(t, "github__create-issue", mangleHeadIfTooLong("github__create-issue", 64))
	})

	t.Run("exact length passes through", func(t *testing.T) {
		t.Parallel()

		name := strings.Repeat("a", 64)
		assert.Equal(t, name, mangleHeadIfTooLong(name, 64))
	})

	t.Run("one over gets mangled", func(t *testing.T) {
		t.Parallel()

		name := strings.Repeat("a", 65)
		result := mangleHeadIfTooLong(name, 64)
		assert.Len(t, result, 64)
		assert.NotEqual(t, name, result)
	})

	t.Run("mangled output never exceeds maxLen", func(t *testing.T) {
		t.Parallel()

		names := []string{
			"servicenow-catalog__delete_api_sn_sc_servicecatalog_cart_by_sys_id_empty",
			"servicenow-catalog__post_api_sn_sc_servicecatalog_items_by_sys_id_versioning_checkout",
			"long-server-id__very_long_tool_name_that_exceeds_the_sixty_four_character_limit_by_quite_a_bit",
		}
		for _, name := range names {
			result := mangleHeadIfTooLong(name, 64)
			assert.LessOrEqual(t, len(result), 64, "name %q mangled to %q (%d chars)", name, result, len(result))
		}
	})

	t.Run("deterministic", func(t *testing.T) {
		t.Parallel()

		name := "servicenow-catalog__delete_api_sn_sc_servicecatalog_cart_by_sys_id_empty"
		a := mangleHeadIfTooLong(name, 64)
		b := mangleHeadIfTooLong(name, 64)
		assert.Equal(t, a, b)
	})

	t.Run("preserves tail", func(t *testing.T) {
		t.Parallel()

		name := "servicenow-catalog__delete_api_sn_sc_servicecatalog_cart_by_sys_id_empty"
		result := mangleHeadIfTooLong(name, 64)
		assert.True(t, strings.HasSuffix(result, "sys_id_empty"), "result %q should end with tail of original", result)
	})

	t.Run("different inputs produce different outputs", func(t *testing.T) {
		t.Parallel()

		a := mangleHeadIfTooLong("servicenow-catalog__delete_api_sn_sc_servicecatalog_cart_by_sys_id_empty", 64)
		b := mangleHeadIfTooLong("servicenow-catalog__post_api_sn_sc_servicecatalog_items_by_sys_id_versioning_checkout", 64)
		assert.NotEqual(t, a, b)
	})

	t.Run("zero maxLen", func(t *testing.T) {
		t.Parallel()

		assert.Empty(t, mangleHeadIfTooLong("anything", 0))
	})

	t.Run("tiny maxLen", func(t *testing.T) {
		t.Parallel()

		result := mangleHeadIfTooLong(strings.Repeat("x", 100), 5)
		assert.LessOrEqual(t, len(result), 5)
	})
}

func TestNamespaceToolMangling(t *testing.T) {
	t.Parallel()

	t.Run("short names are prefixed normally", func(t *testing.T) {
		t.Parallel()

		c := &clientImpl{serverID: "github"}
		result := c.namespaceTool("create-issue")
		assert.Equal(t, "github__create-issue", result)
		assert.LessOrEqual(t, len(result), maxToolNameLen)
	})

	t.Run("long names get mangled to fit", func(t *testing.T) {
		t.Parallel()

		c := &clientImpl{serverID: "servicenow-catalog"}
		name := "delete_api_sn_sc_servicecatalog_cart_by_sys_id_empty"

		full := "servicenow-catalog__" + name
		require.Greater(t, len(full), maxToolNameLen, "test setup: full name should exceed limit")

		result := c.namespaceTool(name)
		assert.LessOrEqual(t, len(result), maxToolNameLen)
		assert.NotEqual(t, full, result, "should be mangled")
	})

	t.Run("mangling is deterministic", func(t *testing.T) {
		t.Parallel()

		c := &clientImpl{serverID: "servicenow-catalog"}
		a := c.namespaceTool("post_api_sn_sc_servicecatalog_items_by_sys_id_versioning_checkout")
		b := c.namespaceTool("post_api_sn_sc_servicecatalog_items_by_sys_id_versioning_checkout")
		assert.Equal(t, a, b)
	})
}

// Mangled names must be valid tool names on every provider. Gemini is
// the strictest: the name must start with a letter or underscore. A
// base-36 hash prefix starts with a digit ~88% of the time — without
// the leading-digit remap, the exact name below was sent to Gemini and
// 400-failed every generation of the financial-advisor agent
// (2026-06-10).
func TestMangleHeadIfTooLong_GeminiSafeLeadingChar(t *testing.T) {
	const live = "morningstar-portfolio__nnhz8bnl3a_rningstarPortfolioService_CalculatePortfolioRiskScore"
	got := mangleHeadIfTooLong(live, maxToolNameLen)
	assert.Len(t, got, maxToolNameLen)
	// The raw hash for this name is "1hqljtt9c2..." — '1' maps to 'h'.
	assert.Equal(t, "hhqljtt9c2_rningstarPortfolioService_CalculatePortfolioRiskScore", got)
	assert.Regexp(t, `^[a-zA-Z_]`, got)

	// Deterministic, and letter-leading for any input.
	assert.Equal(t, got, mangleHeadIfTooLong(live, maxToolNameLen))
	inputs := []string{
		"0" + strings.Repeat("y", 100),
		strings.Repeat("x", 100),
		"server__" + strings.Repeat("a", 80),
	}
	for _, in := range inputs {
		m := mangleHeadIfTooLong(in, maxToolNameLen)
		assert.Regexp(t, `^[a-zA-Z_]`, m, "input %q", in)
	}
}
