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

package agent_test

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

// tampered is written into snapshots to prove they are copies.
const tampered = "tampered"

func newInv(opts ...agent.InvocationOption) *agent.InvocationMetadata {
	return agent.NewInvocationMetadata(
		&session.State{ID: "sess-1"}, agent.Info{Name: "a"}, opts...)
}

func TestInvocationAttributes_SetAndRead(t *testing.T) {
	t.Parallel()

	inv := newInv(agent.WithAttributes(map[string]string{
		agent.AttrUserID: "alice@example.test",
		"user.tier":      "premium",
	}))

	assert.Equal(t, "alice@example.test", inv.Attribute(agent.AttrUserID))
	assert.Equal(t, "premium", inv.Attribute("user.tier"))
	assert.Equal(t, map[string]string{
		agent.AttrUserID: "alice@example.test",
		"user.tier":      "premium",
	}, inv.Attributes())
}

func TestInvocationAttributes_EmptyIsNotAnAttribute(t *testing.T) {
	t.Parallel()

	// Knowing nothing must leave no attribute, not a blank one.
	inv := newInv(agent.WithAttributes(map[string]string{
		"blank-value": "",
		"":            "blank-key",
		"kept":        "v",
	}))

	assert.Equal(t, map[string]string{"kept": "v"}, inv.Attributes())
	assert.Empty(t, inv.Attribute("blank-value"))
}

func TestInvocationAttributes_LaterOptionWins(t *testing.T) {
	t.Parallel()

	inv := newInv(
		agent.WithAttributes(map[string]string{agent.AttrUserID: "first", "tenant.id": "acme"}),
		agent.WithAttributes(map[string]string{agent.AttrUserID: "second", "ignored": ""}),
	)

	assert.Equal(t, map[string]string{
		agent.AttrUserID: "second",
		"tenant.id":      "acme",
	}, inv.Attributes())
}

func TestInvocationAttributes_NoneByDefault(t *testing.T) {
	t.Parallel()

	assert.Empty(t, newInv().Attributes())
	assert.Empty(t, newInv(agent.WithAttributes(nil)).Attributes())
}

func TestInvocationAttributes_SnapshotIsACopy(t *testing.T) {
	t.Parallel()

	inv := newInv(agent.WithAttributes(map[string]string{"a": "1"}))

	got := inv.Attributes()
	got["a"] = tampered
	got["b"] = "added"

	assert.Equal(t, map[string]string{"a": "1"}, inv.Attributes())
}

func TestInvocationAttributes_SeparateFromMetadata(t *testing.T) {
	t.Parallel()

	inv := newInv(agent.WithAttributes(map[string]string{"k": "attribute"}))
	inv.SetMetadata("k", "metadata")

	assert.Equal(t, "attribute", inv.Attribute("k"))
	assert.Equal(t, "metadata", inv.GetMetadata("k"))
	assert.Equal(t, map[string]string{"k": "attribute"}, inv.Attributes())
}
