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
	if prefix == "global" {
		return true
	}

	return prefix == sourceRegionGeoPrefix(awsRegion)
}

// sourceRegionGeoPrefix returns the geo-inference-profile prefix that AWS
// accepts when calling from awsRegion (e.g. "us-east-1" → "us",
// "ap-northeast-1" → "jp", "ap-southeast-2" → "au"). It returns "global"
// for known global-only regions and "" for regions the SDK does not recognize.
//
// This is the default profile family used for bare-ID resolution. Individual
// model cards may narrow availability within a geography.
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
	case "ap", "il", "me", "af", "sa", "mx":
		// Remaining listed regions have no default Geo assignment.
		return "global"
	}

	// Unknown region.
	return ""
}
