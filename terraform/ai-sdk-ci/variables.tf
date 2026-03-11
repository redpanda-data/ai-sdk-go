variable "aws_region" {
  description = "AWS region for the provider and Bedrock access."
  type        = string
  default     = "us-east-1"
}

variable "role_name" {
  description = "Name for the IAM role assumed by GitHub Actions."
  type        = string
  default     = "ai-sdk-go-ci-bedrock"
}

variable "github_repos" {
  description = "GitHub repos allowed to assume the role (org/repo format)."
  type        = list(string)
  default     = ["redpanda-data/ai-sdk-go"]
}

variable "create_oidc_provider" {
  description = "Set to false if the GitHub OIDC provider already exists in your account."
  type        = bool
  default     = true
}

