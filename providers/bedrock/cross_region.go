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

const globalProfileRegion = "global"

// IsModelAllowedFromRegion reports whether invoking modelID from awsRegion
// crosses a Bedrock inference-profile geography boundary.
//
// Bedrock geo inference profiles (us./eu./au./jp.) only accept source regions
// in the same geography. The "global." profile is unrestricted, and bare
// model IDs (no profile prefix) defer to AWS in-region availability. This
// function returns false only for the cross-geography case — e.g. "eu.*"
// invoked from us-east-1, or "jp.*" invoked from eu-west-1.
//
// Unknown regions and regions that AWS lists as global-only (no Geo profile
// at all, e.g. me-central-1, sa-east-1) cause any prefixed call other than
// "global." to be rejected — the SDK has no way to know whether AWS would
// route the call, so we err on the side of failing fast rather than letting
// the request reach AWS just to be rejected there.
//
// Models registered with a per-family profile resolver use their published
// availability table instead of the shared source-region mapping.
//
// Examples:
//
//	IsModelAllowedFromRegion("eu.anthropic.claude-sonnet-4-6", "us-east-1") → false
//	IsModelAllowedFromRegion("us.anthropic.claude-sonnet-4-6", "us-east-1") → true
//	IsModelAllowedFromRegion("us.anthropic.claude-sonnet-4-6", "ca-central-1") → true
//	IsModelAllowedFromRegion("jp.anthropic.claude-sonnet-4-5-…", "ap-northeast-1") → true
//	IsModelAllowedFromRegion("au.anthropic.claude-opus-5", "ap-southeast-6") → false
//	IsModelAllowedFromRegion("global.anthropic.claude-opus-4-6-v1", "me-central-1") → true
//	IsModelAllowedFromRegion("anthropic.claude-sonnet-4-6", "us-east-1") → true (bare)
func IsModelAllowedFromRegion(modelID, awsRegion string) bool {
	if !hasRegionPrefix(modelID) {
		return true
	}

	prefix, _, _ := strings.Cut(modelID, ".")
	if prefix == globalProfileRegion {
		return true
	}

	if resolver, ok := profileRegionResolverFor(modelID); ok {
		profile, known := resolver(awsRegion)

		return known && prefix == profile
	}

	return prefix == sourceRegionGeoPrefix(awsRegion)
}

// claudeOpus5ProfileRegions maps every published source region to its
// preferred profile. Opus 5 publishes US, EU, AU, and global profiles, with
// AU limited to Sydney and Melbourne. Other published commercial regions use
// global.
//
// Source: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-5.html
var claudeOpus5ProfileRegions = map[string]string{
	"us-east-1":    "us",
	"us-east-2":    "us",
	"us-west-1":    "us",
	"us-west-2":    "us",
	"ca-central-1": "us",
	"ca-west-1":    "us",

	"eu-central-1": "eu",
	"eu-central-2": "eu",
	"eu-north-1":   "eu",
	"eu-south-1":   "eu",
	"eu-south-2":   "eu",
	"eu-west-1":    "eu",
	"eu-west-2":    "eu",
	"eu-west-3":    "eu",

	"ap-southeast-2": "au",
	"ap-southeast-4": "au",

	"ap-east-2":      globalProfileRegion,
	"ap-northeast-1": globalProfileRegion,
	"ap-northeast-2": globalProfileRegion,
	"ap-northeast-3": globalProfileRegion,
	"ap-south-1":     globalProfileRegion,
	"ap-south-2":     globalProfileRegion,
	"ap-southeast-1": globalProfileRegion,
	"ap-southeast-3": globalProfileRegion,
	"ap-southeast-5": globalProfileRegion,
	"ap-southeast-6": globalProfileRegion,
	"ap-southeast-7": globalProfileRegion,
	"il-central-1":   globalProfileRegion,
	"me-central-1":   globalProfileRegion,
	"me-south-1":     globalProfileRegion,
	"af-south-1":     globalProfileRegion,
	"sa-east-1":      globalProfileRegion,
	"mx-central-1":   globalProfileRegion,
}

func claudeOpus5ProfileRegion(awsRegion string) (string, bool) {
	profileRegion, ok := claudeOpus5ProfileRegions[awsRegion]

	return profileRegion, ok
}

// sourceRegionGeoPrefix returns the geo-inference-profile prefix that AWS
// accepts when calling from awsRegion (e.g. "us-east-1" → "us",
// "ap-northeast-1" → "jp", "ap-southeast-2" → "au"). Returns "" for regions
// that have no Geo profile assignment (global-only) or that the SDK does not
// recognise.
//
// Source: per-model regional availability tables on the Anthropic Claude
// model cards in the Bedrock user guide. As of 2026-05 every Claude model
// uses the same source-region → geo assignments, so a single table is enough.
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

	// us-* (incl. us-gov-*) and eu-* line up with their region prefix.
	idx := strings.IndexByte(awsRegion, '-')
	if idx <= 0 {
		return ""
	}

	switch awsRegion[:idx] {
	case "us", "eu":
		return awsRegion[:idx]
	}

	// Everything else is global-only as far as Claude on Bedrock is
	// concerned (ap-east-2, ap-northeast-2, ap-south-*, ap-southeast-1/3/5/7,
	// il-*, me-*, af-*, sa-*, mx-*).
	return ""
}
