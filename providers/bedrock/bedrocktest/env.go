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
