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

package bedrock

import (
	"bytes"
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"
	awsretry "github.com/aws/aws-sdk-go-v2/aws/retry"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/internal/shouldretry/shouldretrytest"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

// testAWSConfig points the provider at baseURL with fake credentials and a
// retryer that backs off for a millisecond.
func testAWSConfig(baseURL string) aws.Config {
	return aws.Config{
		Region:       "us-east-1",
		BaseEndpoint: aws.String(baseURL),
		Credentials: aws.CredentialsProviderFunc(func(context.Context) (aws.Credentials, error) {
			return aws.Credentials{AccessKeyID: "test", SecretAccessKey: "test"}, nil
		}),
		Retryer: func() aws.Retryer {
			return awsretry.NewStandard(func(o *awsretry.StandardOptions) {
				o.Backoff = awsretry.BackoffDelayerFunc(func(int, error) (time.Duration, error) { return time.Millisecond, nil })
			})
		},
	}
}

func newTestModel(t *testing.T, model string, opts ...ProviderOption) llm.Model {
	t.Helper()

	p, err := NewProvider(context.Background(), opts...)
	require.NoError(t, err)

	m, err := p.NewModel(model)
	require.NoError(t, err)

	return m
}

func TestShouldRetryHeader(t *testing.T) {
	t.Parallel()

	shouldretrytest.Run(t, func(t *testing.T, baseURL string) llm.Model {
		t.Helper()

		return newTestModel(t, ModelClaudeSonnet55Global, WithAWSConfig(testAWSConfig(baseURL)))
	}, converseFailures(t)...)
}

// The provider wraps a caller's HTTP client too. This fails if that wrap
// moves before the option that installs the caller's client.
func TestShouldRetryHeader_WithHTTPClient(t *testing.T) {
	t.Parallel()

	shouldretrytest.Run(t, func(t *testing.T, baseURL string) llm.Model {
		t.Helper()

		return newTestModel(t, ModelClaudeSonnet55Global, WithAWSConfig(testAWSConfig(baseURL)), WithHTTPClient(&http.Client{}))
	}, converseFailures(t)...)
}

// Mantle models go through the openai provider over a SigV4 transport, not
// through the Converse client.
func TestShouldRetryHeader_Mantle(t *testing.T) {
	t.Parallel()

	shouldretrytest.Run(t, func(t *testing.T, baseURL string) llm.Model {
		t.Helper()

		return newTestModel(t, ModelGemma431B, WithAWSConfig(testAWSConfig(baseURL)))
	},
		shouldretrytest.Failure{
			Name:    "HTTP 503",
			Status:  http.StatusServiceUnavailable,
			Headers: map[string]string{"Content-Type": "application/json"},
			Body:    []byte(`{"error":{"type":"server_error","code":"server_error","message":"every member failed"}}`),
		},
		shouldretrytest.Failure{
			Name:       "error event on a 200 stream",
			Status:     http.StatusOK,
			Headers:    map[string]string{"Content-Type": "text/event-stream"},
			Body:       []byte("data: {\"error\":{\"type\":\"server_error\",\"code\":\"server_error\",\"message\":\"every member failed\"}}\n\n"),
			StreamOnly: true,
		},
	)
}

func converseFailures(t *testing.T) []shouldretrytest.Failure {
	t.Helper()

	return []shouldretrytest.Failure{
		{
			Name:   "HTTP 503",
			Status: http.StatusServiceUnavailable,
			Headers: map[string]string{
				"Content-Type":     "application/json",
				"X-Amzn-Errortype": "ServiceUnavailableException",
			},
			Body: []byte(`{"message":"every member failed"}`),
		},
		{
			Name:       "exception event on a 200 stream",
			Status:     http.StatusOK,
			Headers:    map[string]string{"Content-Type": "application/vnd.amazon.eventstream"},
			Body:       exceptionEvent(t, "serviceUnavailableException", `{"message":"every member failed"}`),
			StreamOnly: true,
		},
	}
}

func exceptionEvent(t *testing.T, exceptionType, payload string) []byte {
	t.Helper()

	headers := eventstream.Headers{}
	headers.Set(":message-type", eventstream.StringValue("exception"))
	headers.Set(":exception-type", eventstream.StringValue(exceptionType))
	headers.Set(":content-type", eventstream.StringValue("application/json"))

	var buf bytes.Buffer
	require.NoError(t, eventstream.NewEncoder().Encode(&buf, eventstream.Message{Headers: headers, Payload: []byte(payload)}))

	return buf.Bytes()
}
