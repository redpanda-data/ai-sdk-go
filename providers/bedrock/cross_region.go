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

const (
	awsRegionCalgary   = "ca-west-1"
	awsRegionMelbourne = "ap-southeast-4"
	awsRegionSydney    = "ap-southeast-2"
)

// IsModelAllowedFromRegion reports whether invoking modelID from awsRegion
// crosses a Bedrock inference-profile geography boundary.
//
// Bedrock geo inference profiles (us./eu./au./jp.) accept only the source
// regions listed on each model card. The "global." profile is unrestricted
// outside GovCloud and China, and bare model IDs (no profile prefix) defer to AWS
// in-region availability.
//
// Unknown regions and regions that AWS lists as global-only (no Geo profile at
// all, e.g. me-central-1, sa-east-1) reject geo-prefixed calls but permit
// "global.". China rejects every profile because Bedrock does not publish
// Claude there. The SDK errs on the side of failing before an AWS API call.
//
// Examples:
//
//	IsModelAllowedFromRegion("eu.anthropic.claude-sonnet-4-6", "us-east-1") → false
//	IsModelAllowedFromRegion("us.anthropic.claude-sonnet-4-6", "us-east-1") → true
//	IsModelAllowedFromRegion("us.anthropic.claude-sonnet-4-6", "ca-central-1") → true
//	IsModelAllowedFromRegion("jp.anthropic.claude-sonnet-4-5-…", "ap-northeast-1") → true
//	IsModelAllowedFromRegion("global.anthropic.claude-opus-4-6-v1", "me-central-1") → true
//	IsModelAllowedFromRegion("global.anthropic.claude-sonnet-4-6", "cn-north-1") → false
//	IsModelAllowedFromRegion("anthropic.claude-sonnet-4-6", "us-east-1") → true (bare)
func IsModelAllowedFromRegion(modelID, awsRegion string) bool {
	if !hasRegionPrefix(modelID) {
		return true
	}

	prefix, _, _ := strings.Cut(modelID, ".")

	// Sonnet 4.5 US is the only current catalog profile whose source-region
	// table publishes GovCloud. The Opus 4.8 card's summary claims GovCloud
	// support but its source table omits both regions, so use the stricter
	// source-table interpretation. Global is unavailable.
	if strings.HasPrefix(awsRegion, "us-gov-") {
		return modelID == ModelClaudeSonnet45US
	}

	if strings.HasPrefix(awsRegion, "cn-") {
		return false
	}

	if allowed, constrained := isNova2LiteSourceRegion(modelID, awsRegion); constrained {
		return allowed
	}

	if prefix == "global" {
		return true
	}

	// Opus 4.7 and 4.8 narrow the AU profile to Sydney and Melbourne.
	// Their model cards list New Zealand as global-only.
	switch modelID {
	case ModelClaudeOpus47AU, ModelClaudeOpus48AU:
		return awsRegion == awsRegionSydney || awsRegion == awsRegionMelbourne
	case ModelClaudeHaiku45US, ModelClaudeOpus45US, ModelClaudeSonnet45US:
		// These cards mark Calgary Geo unsupported and omit it from Geo: US.
		return awsRegion != awsRegionCalgary && prefix == sourceRegionGeoPrefix(awsRegion)
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
//   - https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-2-lite.html
func sourceRegionGeoPrefix(awsRegion string) string {
	// Regions where the geo doesn't match the AWS region prefix —
	// Canada is part of the US Geo, ap-northeast-* maps to JP, and a
	// subset of ap-southeast-* maps to AU.
	switch awsRegion {
	case "ca-central-1", awsRegionCalgary:
		return "us"
	case "ap-northeast-1", "ap-northeast-3":
		return "jp"
	case awsRegionSydney, awsRegionMelbourne, "ap-southeast-6":
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

// isNova2LiteSourceRegion applies Nova 2 Lite's narrower published source
// tables. It returns constrained=false for every other model.
func isNova2LiteSourceRegion(modelID, awsRegion string) (bool, bool) {
	switch modelID {
	case ModelNova2LiteEU:
		switch awsRegion {
		case "eu-central-1", "eu-north-1", "eu-south-1", "eu-south-2", "eu-west-1", "eu-west-3":
			return true, true
		default:
			return false, true
		}
	case ModelNova2LiteJP:
		return awsRegion == "ap-northeast-1", true
	case ModelNova2LiteGlobal:
		switch awsRegion {
		case "us-east-1", "us-east-2", "us-west-1", "us-west-2",
			"ca-central-1", awsRegionCalgary,
			"eu-central-1", "eu-north-1", "eu-south-1", "eu-south-2",
			"eu-west-1", "eu-west-2", "eu-west-3",
			"il-central-1", "me-central-1",
			"ap-east-2", "ap-northeast-1", "ap-northeast-2", "ap-south-1",
			"ap-southeast-1", awsRegionSydney, "ap-southeast-3",
			awsRegionMelbourne, "ap-southeast-5", "ap-southeast-6", "ap-southeast-7":
			return true, true
		default:
			return false, true
		}
	default:
		return false, false
	}
}
