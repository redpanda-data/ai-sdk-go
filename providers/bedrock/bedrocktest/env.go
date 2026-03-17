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

package bedrocktest

import (
	"context"
	"os"
	"testing"

	awsconfig "github.com/aws/aws-sdk-go-v2/config"
)

// SkipUnlessAWSCredentials skips the test if AWS credentials are not available.
// This checks the full AWS credential chain (env vars, profiles, instance roles, SSO, etc.)
// rather than individual environment variables.
func SkipUnlessAWSCredentials(t *testing.T) {
	t.Helper()

	region := os.Getenv("AWS_REGION")
	if region == "" {
		region = TestRegion
	}

	cfg, err := awsconfig.LoadDefaultConfig(context.Background(), awsconfig.WithRegion(region))
	if err != nil {
		t.Skipf("skipping test: unable to load AWS config: %v", err)
	}

	creds, err := cfg.Credentials.Retrieve(context.Background())
	if err != nil || !creds.HasKeys() {
		t.Skip("skipping test: no AWS credentials available")
	}
}
