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
	"time"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
)

func (r *Registry) newSDKClient(identity context.Context, entry *registryEntry) *sdkmcp.Client {
	client := sdkmcp.NewClient(&sdkmcp.Implementation{Name: "redpanda-ai-agent-sdk", Version: "v1.0.0"}, &sdkmcp.ClientOptions{
		ToolListChangedHandler: func(context.Context, *sdkmcp.ToolListChangedRequest) {
			entry.generation.Add(1)
		},
	})
	identity = context.WithoutCancel(identity)

	client.AddSendingMiddleware(func(next sdkmcp.MethodHandler) sdkmcp.MethodHandler {
		//nolint:contextcheck // The bridge inherits cancellation and all request values, adding only trusted identity values.
		return func(ctx context.Context, method string, req sdkmcp.Request) (sdkmcp.Result, error) {
			if method == "subscriptions/listen" {
				// The SDK roots subscriptions in Background; retain trusted identity
				// values without borrowing the discovery operation's cancellation.
				ctx = registryIdentityContext{Context: ctx, identity: identity}
			}

			result, err := next(ctx, method, req)
			if err != nil {
				return nil, err
			}

			if listed, ok := result.(*sdkmcp.ListToolsResult); ok {
				if listed.TTLMs <= 0 {
					entry.fetchTTL = 0
				} else if int64(listed.TTLMs) < int64(entry.fetchTTL/time.Millisecond) {
					entry.fetchTTL = time.Duration(listed.TTLMs) * time.Millisecond
				}

				// SDK cache insertion races its notification invalidation. Keep it
				// disabled; the partition cache below publishes with a generation.
				listed.TTLMs = 0
			}

			return result, nil
		}
	})

	return client
}

func (r *Registry) catalog(ctx context.Context, entry *registryEntry) ([]*sdkmcp.Tool, error) {
	select {
	case entry.catalogGate <- struct{}{}:
		defer func() { <-entry.catalogGate }()
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		generation := entry.generation.Load()
		started := r.config.now()

		if entry.catalogGeneration == generation && started.Before(entry.catalogExpires) {
			return entry.catalog, nil
		}

		entry.fetchTTL = r.config.maxToolCacheTTL
		var fetched []*sdkmcp.Tool

		for remote, err := range entry.session.Tools(ctx, nil) {
			if err != nil {
				r.retireClosed(entry, err)

				return nil, err
			}

			fetched = append(fetched, remote)
		}

		if generation != entry.generation.Load() {
			continue
		}

		entry.catalog = fetched
		entry.catalogGeneration = generation
		entry.catalogExpires = started.Add(entry.fetchTTL)

		return fetched, nil
	}
}

//nolint:containedctx // Bridges SDK-owned subscription cancellation and caller identity values.
type registryIdentityContext struct {
	context.Context

	identity context.Context
}

func (c registryIdentityContext) Value(key any) any {
	if value := c.Context.Value(key); value != nil {
		return value
	}

	return c.identity.Value(key)
}
