output "role_arn" {
  description = "ARN of the IAM role for GitHub Actions. Use this in the workflow's role-to-assume parameter."
  value       = aws_iam_role.ci_bedrock.arn
}

output "role_name" {
  description = "Name of the IAM role."
  value       = aws_iam_role.ci_bedrock.name
}