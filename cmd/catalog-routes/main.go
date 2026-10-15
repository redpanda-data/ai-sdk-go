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

// Command catalog-routes prints where this SDK sends each bare Bedrock model
// name from each AWS region: one "region name model-ID" line per pair,
// sorted, with "-" where the SDK refuses the pair. The reconcile PR job runs
// it on the code before and after a patch and compares the two, so a
// reviewer sees which callers a change moves. It sends nothing: a recorder
// catches each request and reads the model ID from it.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"slices"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"

	"github.com/redpanda-data/ai-sdk-go/internal/reconcile"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/bedrock"
)

func main() {
	lines, err := routes(reconcile.AWSRegions, bareNames())
	if err != nil {
		fmt.Fprintln(os.Stderr, "catalog-routes:", err)
		os.Exit(1)
	}

	if _, err := fmt.Fprintln(os.Stdout, strings.Join(lines, "\n")); err != nil {
		fmt.Fprintln(os.Stderr, "catalog-routes:", err)
		os.Exit(1)
	}
}

// bareNames lists the names a caller can pass without a region prefix: every
// Bedrock offering ID with its profile prefix removed, once, sorted.
func bareNames() []string {
	offerings := bedrock.Catalog().All()
	names := make([]string, 0, len(offerings))

	for _, o := range offerings {
		name := o.ID
		if geo := o.Attributes[bedrock.ModelMetadataInferenceGeo]; geo != "" {
			name = strings.TrimPrefix(name, geo+".")
		}

		names = append(names, name)
	}

	slices.Sort(names)

	return slices.Compact(names)
}

// routes resolves every name from every region through the SDK itself.
func routes(regions, names []string) ([]string, error) {
	var lines []string

	for _, region := range regions {
		recorder := &recorder{}

		// An explicit config reads no environment and no ~/.aws, so the routes
		// are the same on every machine. The placeholder keys only sign
		// requests the recorder catches. (Anonymous credentials make the AWS
		// SDK pick bearer auth for Converse, and it panics without a token.)
		provider, err := bedrock.NewProvider(context.Background(),
			bedrock.WithAWSConfig(aws.Config{
				Region: region, Credentials: placeholderKeys,
				// The recorder answers 200, so nothing retries; this keeps it so.
				Retryer: func() aws.Retryer { return aws.NopRetryer{} },
			}),
			bedrock.WithHTTPClient(&http.Client{Transport: recorder}))
		if err != nil {
			return nil, fmt.Errorf("%s: %w", region, err)
		}

		for _, name := range names {
			id, err := route(provider, recorder, name)
			if err != nil {
				return nil, fmt.Errorf("%s %s: %w", region, name, err)
			}

			lines = append(lines, region+" "+name+" "+id)
		}
	}

	slices.Sort(lines)

	return lines, nil
}

var placeholderKeys = aws.CredentialsProviderFunc(func(context.Context) (aws.Credentials, error) {
	return aws.Credentials{AccessKeyID: "catalog-routes", SecretAccessKey: "catalog-routes", Source: "catalog-routes"}, nil
})

// route is the model ID one request for name goes to, or "-" when the SDK
// refuses the name. A model that sends nothing the recorder can read is an
// error, so a changed request format can't pass as a refusal.
func route(provider *bedrock.Provider, recorder *recorder, name string) (string, error) {
	model, err := provider.NewModel(name)
	if err != nil {
		return "-", nil //nolint:nilerr // a refusal is a route, not a failure
	}

	recorder.model = ""
	// The recorder's canned reply is no valid response; only the request matters.
	_, _ = model.Generate(context.Background(), &llm.Request{Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi"))}})

	if recorder.model == "" {
		return "", errors.New("no model ID in the request")
	}

	return recorder.model, nil
}

// recorder answers every request with an empty body and keeps the model ID it
// was for: from a Converse path (/model/<ID>/converse), or a mantle body's
// "model".
type recorder struct{ model string }

func (r *recorder) RoundTrip(req *http.Request) (*http.Response, error) {
	if id, ok := strings.CutPrefix(req.URL.Path, "/model/"); ok {
		r.model, _, _ = strings.Cut(id, "/")
	} else if req.Body != nil {
		var body struct {
			Model string `json:"model"`
		}

		if raw, err := io.ReadAll(req.Body); err == nil && json.Unmarshal(raw, &body) == nil {
			r.model = body.Model
		}
	}

	return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader("{}")), Header: make(http.Header), Request: req}, nil
}
