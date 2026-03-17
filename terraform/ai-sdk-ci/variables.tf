# Copyright 2026 Redpanda Data, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

