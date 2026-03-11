terraform {
  required_version = ">= 1.10"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket       = "ai-sdk-ci-tfstate"
    key          = "ai-sdk-ci/terraform.tfstate"
    region       = "us-east-2"
    use_lockfile = true
    encrypt      = true
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

# --------------------------------------------------------------------------
# GitHub Actions OIDC provider
# --------------------------------------------------------------------------
# If your account already has the GitHub OIDC provider, import it:
#   terraform import aws_iam_openid_connect_provider.github \
#     arn:aws:iam::<ACCOUNT_ID>:oidc-provider/token.actions.githubusercontent.com
#
# Or set var.create_oidc_provider = false to reference the existing one.
# --------------------------------------------------------------------------

resource "aws_iam_openid_connect_provider" "github" {
  count = var.create_oidc_provider ? 1 : 0

  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  # AWS no longer validates thumbprints for GitHub's OIDC provider (it uses
  # a trusted CA library instead). The field is still required by the API,
  # so the all-f placeholder is the standard convention.
  thumbprint_list = ["ffffffffffffffffffffffffffffffffffffffff"]
}

data "aws_iam_openid_connect_provider" "github" {
  count = var.create_oidc_provider ? 0 : 1
  url   = "https://token.actions.githubusercontent.com"
}

locals {
  oidc_provider_arn = (
    var.create_oidc_provider
    ? aws_iam_openid_connect_provider.github[0].arn
    : data.aws_iam_openid_connect_provider.github[0].arn
  )
  account_id = data.aws_caller_identity.current.account_id
  partition  = data.aws_partition.current.partition
}

# --------------------------------------------------------------------------
# IAM role assumed by GitHub Actions via OIDC
# --------------------------------------------------------------------------

data "aws_iam_policy_document" "assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [local.oidc_provider_arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = [for r in var.github_repos : "repo:${r}:*"]
    }
  }
}

resource "aws_iam_role" "ci_bedrock" {
  name               = var.role_name
  assume_role_policy = data.aws_iam_policy_document.assume_role.json
}

# --------------------------------------------------------------------------
# Bedrock permissions
# --------------------------------------------------------------------------
# The provider uses the Converse API with cross-region inference profiles
# (e.g. "us.anthropic.claude-sonnet-4-6"). These require permissions on
# both foundation model and inference profile ARNs.
# --------------------------------------------------------------------------

data "aws_iam_policy_document" "bedrock" {
  # Converse / ConverseStream (used for all model calls)
  statement {
    effect = "Allow"
    actions = [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream",
    ]
    resources = [
      # Foundation models
      "arn:${local.partition}:bedrock:*::foundation-model/*",
      # Cross-region inference profiles
      "arn:${local.partition}:bedrock:*:${local.account_id}:inference-profile/*",
      # System-defined cross-region inference profiles
      "arn:${local.partition}:bedrock:*:*:inference-profile/*",
    ]
  }

  # ListFoundationModels / GetFoundationModel (used for model discovery)
  statement {
    effect = "Allow"
    actions = [
      "bedrock:ListFoundationModels",
      "bedrock:GetFoundationModel",
    ]
    resources = ["*"]
  }

  # ListInferenceProfiles (may be needed for profile resolution)
  statement {
    effect = "Allow"
    actions = [
      "bedrock:ListInferenceProfiles",
      "bedrock:GetInferenceProfile",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "bedrock" {
  name   = "${var.role_name}-bedrock"
  policy = data.aws_iam_policy_document.bedrock.json
}

resource "aws_iam_role_policy_attachment" "bedrock" {
  role       = aws_iam_role.ci_bedrock.name
  policy_arn = aws_iam_policy.bedrock.arn
}
