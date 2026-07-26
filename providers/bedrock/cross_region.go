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

import "strings"

// IsModelAllowedFromRegion reports whether invoking modelID from awsRegion
// crosses a Bedrock inference-profile geography boundary.
//
// Bedrock geo inference profiles (us./eu./au./jp.) accept only the source
// regions listed on each model card. The "global." profile is unrestricted
// outside GovCloud, and bare model IDs (no profile prefix) defer to AWS
// in-region availability.
//
// Unknown regions and regions that AWS lists as global-only (no Geo profile
// at all, e.g. me-central-1, sa-east-1) cause any prefixed call other than
// "global." to be rejected — the SDK has no way to know whether AWS would
// route the call, so we err on the side of failing fast rather than letting
// the request reach AWS just to be rejected there.
//
// Examples:
//
//	IsModelAllowedFromRegion("eu.anthropic.claude-sonnet-4-6", "us-east-1") → false
//	IsModelAllowedFromRegion("us.anthropic.claude-sonnet-4-6", "us-east-1") → true
//	IsModelAllowedFromRegion("us.anthropic.claude-sonnet-4-6", "ca-central-1") → true
//	IsModelAllowedFromRegion("jp.anthropic.claude-sonnet-4-5-…", "ap-northeast-1") → true
//	IsModelAllowedFromRegion("global.anthropic.claude-opus-4-6-v1", "me-central-1") → true
//	IsModelAllowedFromRegion("anthropic.claude-sonnet-4-6", "us-east-1") → true (bare)
func IsModelAllowedFromRegion(modelID, awsRegion string) bool {
	if !hasRegionPrefix(modelID) {
		return true
	}

	prefix, _, _ := strings.Cut(modelID, ".")

	// The Sonnet 4.5 US profile is the only current Claude profile published
	// for GovCloud source regions. Global routing is not available there.
	if strings.HasPrefix(awsRegion, "us-gov-") {
		return modelID == ModelClaudeSonnet45US
	}

	if prefix == "global" {
		return true
	}

	// Opus 4.7 and 4.8 narrow the AU profile to Sydney and Melbourne.
	// Their model cards list New Zealand as global-only.
	switch modelID {
	case ModelClaudeOpus47AU, ModelClaudeOpus48AU:
		return awsRegion == "ap-southeast-2" || awsRegion == "ap-southeast-4"
	case ModelClaudeHaiku45US, ModelClaudeOpus45US:
		return awsRegion != "ca-west-1" && prefix == sourceRegionGeoPrefix(awsRegion)
	}

	return prefix == sourceRegionGeoPrefix(awsRegion)
}

// sourceRegionGeoPrefix returns the geo-inference-profile prefix that AWS
// accepts when calling from awsRegion (e.g. "us-east-1" → "us",
// "ap-northeast-1" → "jp", "ap-southeast-2" → "au"). It returns "global"
// for known global-only regions and "" for regions the SDK does not recognize.
//
// This is the default profile family used for bare-ID resolution. Individual
// model cards may narrow availability within a geography:
//   - https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-4-6.html
//   - https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-4-7.html
//   - https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-4-8.html
//   - https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-haiku-4-5.html
//   - https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-4-5.html
//   - https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-4-5.html
func sourceRegionGeoPrefix(awsRegion string) string {
	// Regions where the geo doesn't match the AWS region prefix —
	// Canada is part of the US Geo, ap-northeast-* maps to JP, and a
	// subset of ap-southeast-* maps to AU.
	switch awsRegion {
	case "ca-central-1", "ca-west-1":
		return "us"
	case "ap-northeast-1", "ap-northeast-3":
		return "jp"
	case "ap-southeast-2", "ap-southeast-4", "ap-southeast-6":
		return "au"
	}

	// US (including GovCloud) and EU regions line up with their region prefix.
	idx := strings.IndexByte(awsRegion, '-')
	if idx <= 0 {
		return ""
	}

	switch awsRegion[:idx] {
	case "us", "eu":
		return awsRegion[:idx]
	case "ap", "il", "me", "af", "sa", "mx":
		// Remaining listed regions have no default Geo assignment.
		return "global"
	}

	// Unknown region.
	return ""
}
