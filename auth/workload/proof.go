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

package workload

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"sync"
	"time"
)

// processKey is the one DPoP key of this process. It is made on the first
// enabled NewIssuer and never leaves memory.
var processKey = sync.OnceValues(func() (*ecdsa.PrivateKey, error) {
	return ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
})

// proofSigner signs DPoP proofs with one P-256 key.
type proofSigner struct {
	key    *ecdsa.PrivateKey
	header string // base64url JOSE header, the same for every proof
}

type publicJWK struct {
	Crv string `json:"crv"`
	Kty string `json:"kty"`
	X   string `json:"x"`
	Y   string `json:"y"`
}

func newProofSigner(key *ecdsa.PrivateKey) (*proofSigner, error) {
	jwk, err := publicKeyJWK(&key.PublicKey)
	if err != nil {
		return nil, err
	}

	header, err := json.Marshal(struct {
		Typ string    `json:"typ"`
		Alg string    `json:"alg"`
		JWK publicJWK `json:"jwk"`
	}{Typ: "dpop+jwt", Alg: "ES256", JWK: jwk})
	if err != nil {
		return nil, fmt.Errorf("workload: encode DPoP header: %w", err)
	}

	return &proofSigner{key: key, header: base64.RawURLEncoding.EncodeToString(header)}, nil
}

func publicKeyJWK(key *ecdsa.PublicKey) (publicJWK, error) {
	point, err := key.Bytes()
	if err != nil {
		return publicJWK{}, fmt.Errorf("workload: encode DPoP public key: %w", err)
	}

	// An uncompressed P-256 point is 0x04 || X (32 bytes) || Y (32 bytes).
	if len(point) != 65 || point[0] != 4 {
		return publicJWK{}, errors.New("workload: DPoP key is not an uncompressed P-256 point")
	}

	return publicJWK{
		Crv: "P-256",
		Kty: "EC",
		X:   base64.RawURLEncoding.EncodeToString(point[1:33]),
		Y:   base64.RawURLEncoding.EncodeToString(point[33:]),
	}, nil
}

// proofClaims are the RFC 9449 section 4.2 claims.
type proofClaims struct {
	JTI   string `json:"jti"`
	HTM   string `json:"htm"`
	HTU   string `json:"htu"`
	IAT   int64  `json:"iat"`
	Nonce string `json:"nonce,omitempty"`
	ATH   string `json:"ath,omitempty"`
}

// proof returns a DPoP proof for one request. accessToken is empty for a
// token request, and nonce is empty when the server has not sent one.
func (s *proofSigner) proof(method string, target *url.URL, now time.Time, nonce, accessToken string) (string, error) {
	jti := make([]byte, 16)
	if _, err := rand.Read(jti); err != nil {
		return "", fmt.Errorf("workload: DPoP proof ID: %w", err)
	}

	claims := proofClaims{
		JTI:   base64.RawURLEncoding.EncodeToString(jti),
		HTM:   method,
		HTU:   proofTarget(target),
		IAT:   now.Unix(),
		Nonce: nonce,
	}
	if accessToken != "" {
		claims.ATH = accessTokenHash(accessToken)
	}

	payload, err := json.Marshal(claims)
	if err != nil {
		return "", fmt.Errorf("workload: encode DPoP claims: %w", err)
	}

	signingInput := s.header + "." + base64.RawURLEncoding.EncodeToString(payload)
	digest := sha256.Sum256([]byte(signingInput))

	r, sig, err := ecdsa.Sign(rand.Reader, s.key, digest[:])
	if err != nil {
		return "", fmt.Errorf("workload: sign DPoP proof: %w", err)
	}

	// JWS ES256 signatures are R || S, each left-padded to 32 bytes (RFC 7518 section 3.4).
	signature := make([]byte, 64)
	r.FillBytes(signature[:32])
	sig.FillBytes(signature[32:])

	return signingInput + "." + base64.RawURLEncoding.EncodeToString(signature), nil
}

// proofTarget is the htu claim: the request URL without query or fragment.
func proofTarget(target *url.URL) string {
	htu := *target
	htu.User = nil
	htu.RawQuery = ""
	htu.ForceQuery = false
	htu.Fragment = ""
	htu.RawFragment = ""

	return htu.String()
}

// accessTokenHash is the ath claim: base64url(SHA-256(access token)).
func accessTokenHash(accessToken string) string {
	sum := sha256.Sum256([]byte(accessToken))

	return base64.RawURLEncoding.EncodeToString(sum[:])
}
