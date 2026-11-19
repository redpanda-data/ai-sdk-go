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
	"context"
	"errors"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
)

type registryTransport struct {
	*sdkmcp.StreamableClientTransport

	entry *registryEntry
}

func (t *registryTransport) Connect(ctx context.Context) (sdkmcp.Connection, error) {
	// Return the concrete SDK connection unchanged: its unexported optional
	// interfaces perform protocol negotiation and HTTP session initialization.
	//nolint:contextcheck // Inherits cancellation; background values belong only to the latest validated partition caller.
	return t.StreamableClientTransport.Connect(registryIdentityContext{Context: ctx, entry: t.entry})
}

//nolint:containedctx // Stores validated partition caller values without retaining operation cancellation.
type registryCallerContext struct {
	context context.Context
}

func (e *registryEntry) rememberIdentity(ctx context.Context) {
	e.identity.Store(&registryCallerContext{context: context.WithoutCancel(ctx)})
}

//nolint:containedctx // SDK owns cancellation; background credentials follow the latest validated caller.
type registryIdentityContext struct {
	context.Context

	entry *registryEntry
}

type registryBackgroundIdentityKey struct{}

func (c registryIdentityContext) Value(key any) any {
	if _, ok := key.(registryBackgroundIdentityKey); ok {
		return c.entry
	}

	return nil
}

func registryCredentialContext(ctx context.Context) (context.Context, error) {
	entry, ok := ctx.Value(registryBackgroundIdentityKey{}).(*registryEntry)
	if !ok {
		return ctx, nil
	}

	caller := entry.identity.Load()
	if caller == nil {
		return nil, errors.New("mcp: missing background credential context")
	}

	return registryRequestIdentityContext{Context: ctx, caller: caller}, nil
}

//nolint:containedctx // One HTTP credential resolution retains SDK cancellation and one immutable caller context.
type registryRequestIdentityContext struct {
	context.Context

	caller *registryCallerContext
}

func (c registryRequestIdentityContext) Value(key any) any {
	if _, ok := key.(registryBackgroundIdentityKey); ok {
		return nil
	}

	return c.caller.context.Value(key)
}
