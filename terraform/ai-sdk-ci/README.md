# ai-sdk-ci

Terraform configuration for AWS resources needed to run Bedrock conformance tests in CI.

## What it creates

- **GitHub Actions OIDC provider** — allows GitHub Actions to assume an IAM role without static credentials
- **IAM role** (`ai-sdk-go-ci-bedrock`) — assumed by the `redpanda-data/ai-sdk-go` repo via OIDC
- **IAM policy** — grants Bedrock Converse API access (invoke, list models, inference profiles)

## Architecture

```
GitHub Actions workflow
  │
  ├─ requests OIDC token from GitHub
  │
  ├─ aws-actions/configure-aws-credentials
  │    └─ exchanges OIDC token for temporary AWS credentials via sts:AssumeRoleWithWebIdentity
  │
  └─ go test ./...
       └─ Bedrock conformance tests use AWS credentials from the environment
```

## Bootstrap

These steps were performed once to set up the infrastructure. They are documented here for reference.

### 1. Create the S3 state bucket (manual, not managed by Terraform)

```bash
aws s3api create-bucket \
  --bucket ai-sdk-ci-tfstate \
  --region us-east-2 \
  --create-bucket-configuration LocationConstraint=us-east-2

aws s3api put-bucket-versioning \
  --bucket ai-sdk-ci-tfstate \
  --versioning-configuration Status=Enabled

aws s3api put-public-access-block \
  --bucket ai-sdk-ci-tfstate \
  --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

aws s3api put-bucket-encryption \
  --bucket ai-sdk-ci-tfstate \
  --server-side-encryption-configuration \
    '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"aws:kms"},"BucketKeyEnabled":true}]}'
```

### 2. Apply Terraform

The S3 backend is already configured in `main.tf`, so init connects directly to the bucket:

```bash
terraform init
terraform plan -var="create_oidc_provider=false"   # if the GitHub OIDC provider already exists
terraform apply -var="create_oidc_provider=false"
```

### 3. Enable Bedrock model access (manual)

Models must be enabled individually via the AWS CLI. Get the offer token, then create the agreement:

```bash
OFFER_TOKEN=$(aws bedrock list-foundation-model-agreement-offers \
  --model-id anthropic.claude-sonnet-4-5-20250929-v1:0 \
  --region us-east-1 \
  --query 'offers[0].offerToken' --output text)

aws bedrock create-foundation-model-agreement \
  --model-id anthropic.claude-sonnet-4-5-20250929-v1:0 \
  --offer-token "$OFFER_TOKEN" \
  --region us-east-1
```

Repeat for each model needed. The following models are currently enabled:

- `anthropic.claude-sonnet-4-5-20250929-v1:0`
- `anthropic.claude-sonnet-4-6`
- `anthropic.claude-haiku-4-5-20251001-v1:0`
- `anthropic.claude-opus-4-5-20251101-v1:0`
- `anthropic.claude-opus-4-6-v1`
- `anthropic.claude-opus-4-7`

### 4. Update the GitHub Actions workflow

Update the role ARN in `.github/workflows/test.yaml` with the `role_arn` output from the apply.

## Day-to-day usage

```bash
cd terraform/ai-sdk-ci
terraform plan
terraform apply
```

### If the GitHub OIDC provider already exists in another account

Set `create_oidc_provider = false`, or import the existing one:

```bash
terraform import aws_iam_openid_connect_provider.github \
  arn:aws:iam::<ACCOUNT_ID>:oidc-provider/token.actions.githubusercontent.com
```

## Outputs

| Name | Description |
|------|-------------|
| `role_arn` | IAM role ARN — use as `role-to-assume` in the GitHub Actions workflow |
| `role_name` | IAM role name |
