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

package webfetch

import (
	"bytes"
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"
)

// resp represents the internal response structure for Execute to marshal.
type resp struct {
	URL         string
	FinalURL    string
	StatusCode  int
	Status      string
	MediaType   string
	Encoding    string
	RetrievedAt time.Time
	Body        []byte
	Truncated   bool
	Redirected  bool
}

func doRequest(ctx context.Context, cfg Config, method, rawURL string) (*resp, error) {
	u, err := url.Parse(rawURL)
	if err != nil {
		return nil, fmt.Errorf("invalid URL: %w", err)
	}

	if err := validateURLPreflight(cfg, u); err != nil {
		return nil, err
	}

	// Build transport with safe dialers and security hardening
	transport := &http.Transport{
		Proxy:                 nil, // CRITICAL: Disable proxy to prevent bypass via ProxyFromEnvironment
		DialContext:           safeDialContext(cfg),
		DialTLSContext:        safeTLSDialContext(cfg),                                                               // Handle TLS SNI properly when dialing by IP
		TLSClientConfig:       &tls.Config{MinVersion: tls.VersionTLS12, InsecureSkipVerify: cfg.InsecureSkipVerify}, //nolint:gosec // InsecureSkipVerify is only for testing
		ResponseHeaderTimeout: 5 * time.Second,
		TLSHandshakeTimeout:   5 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
		DisableKeepAlives:     true,
		IdleConnTimeout:       0,
		MaxIdleConns:          0,
		MaxIdleConnsPerHost:   0,
	}

	redirected := false
	client := &http.Client{
		Transport: transport,
		Timeout:   cfg.Timeout,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= cfg.MaxRedirects {
				return errors.New("too many redirects")
			}

			// CRITICAL: Re-validate the redirect URL with full security checks
			// This ensures redirects cannot bypass scheme/port/IP validation
			err := validateURLPreflight(cfg, req.URL)
			if err != nil {
				return fmt.Errorf("redirect validation failed: %w", err)
			}

			// Check if this is a cross-origin redirect and strip sensitive headers
			if len(via) > 0 {
				prev := via[len(via)-1]
				if req.URL.Host != prev.URL.Host || req.URL.Scheme != prev.URL.Scheme {
					// Cross-origin redirect: strip potentially sensitive headers
					req.Header.Del("Authorization")
					req.Header.Del("Cookie")
					req.Header.Del("X-Api-Key")
				}
			}

			redirected = true

			return nil
		},
	}

	// Create request
	req, err := http.NewRequestWithContext(ctx, method, u.String(), nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set safe headers
	req.Header.Set("User-Agent", "Redpanda AI-Agent-SDK WebFetch/1.0")
	req.Header.Set("Accept", "text/*,application/json,application/xml,application/xhtml+xml")
	req.Header.Set("Accept-Language", "en-US,en;q=0.9")

	// Perform request
	res, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}

	defer func() { _ = res.Body.Close() }()

	// Build response
	response := &resp{
		URL:         rawURL,
		FinalURL:    res.Request.URL.String(),
		StatusCode:  res.StatusCode,
		Status:      res.Status,
		RetrievedAt: time.Now(),
		Redirected:  redirected,
	}

	// Handle body only for GET requests with 2xx status
	if method == http.MethodGet && res.StatusCode >= 200 && res.StatusCode < 300 {
		// Determine media type
		contentType := res.Header.Get("Content-Type")
		media := strings.ToLower(strings.TrimSpace(strings.Split(contentType, ";")[0]))

		var bodyReader io.Reader = res.Body

		// If media type is missing, sniff first 512 bytes and rebuild stream
		if media == "" {
			peek := make([]byte, 512)

			n, _ := io.ReadFull(io.LimitReader(res.Body, 512), peek)
			if n > 0 {
				sniff := http.DetectContentType(peek[:n])
				media = strings.ToLower(strings.Split(sniff, ";")[0])
				bodyReader = io.MultiReader(bytes.NewReader(peek[:n]), res.Body)
			}
		}

		// Check media type against allowlist using proper MIME parsing
		if !allowedMedia(cfg.AllowedMedia, contentType) {
			return nil, fmt.Errorf("content type %q not allowed", contentType)
		}

		// Read max+1 bytes to detect truncation
		maxBytes := cfg.MaxResponseBytes

		buf, err := io.ReadAll(io.LimitReader(bodyReader, maxBytes+1))
		if err != nil {
			return nil, fmt.Errorf("failed to read response body: %w", err)
		}

		truncated := int64(len(buf)) > maxBytes
		if truncated {
			buf = buf[:maxBytes]
		}

		// Normalize charset for text-like content
		if shouldNormalize(media) {
			data, encoding, err := toUTF8(buf, contentType)
			if err != nil {
				return nil, fmt.Errorf("charset conversion failed: %w", err)
			}

			response.Body = data
			response.Encoding = encoding
		} else {
			response.Body = buf
			response.Encoding = ""
		}

		response.MediaType = media
		response.Truncated = truncated
	} else {
		// No body for HEAD requests or non-2xx responses
		contentType := res.Header.Get("Content-Type")
		response.MediaType = strings.ToLower(strings.TrimSpace(strings.Split(contentType, ";")[0]))
		response.Encoding = ""
	}

	return response, nil
}

// validateURLPreflight performs comprehensive URL validation before making the request.
func validateURLPreflight(cfg Config, u *url.URL) error {
	// Check absolute URL length to prevent DoS via memory exhaustion
	urlStr := u.String()
	if len(urlStr) > 8192 {
		return fmt.Errorf("URL too long (%d bytes, max 8192)", len(urlStr))
	}

	// Check scheme allowlist
	scheme := strings.ToLower(u.Scheme)
	if !slices.Contains(cfg.AllowedSchemes, scheme) {
		return fmt.Errorf("scheme %q not allowed", scheme)
	}

	// Only allow http and https schemes (block file:, ftp:, gopher:, etc.)
	if scheme != "http" && scheme != "https" {
		return fmt.Errorf("scheme %q not supported", scheme)
	}

	// Check port allowlist (if configured)
	port := u.Port()
	if port == "" {
		switch scheme {
		case "https":
			port = "443"
		case "http":
			port = "80"
		}
	}

	portNum, err := net.LookupPort("tcp", port)
	if err != nil {
		return fmt.Errorf("invalid port %s: %w", port, err)
	}
	// If AllowedPorts is empty/nil, allow all ports (useful for testing)
	if len(cfg.AllowedPorts) > 0 && !slices.Contains(cfg.AllowedPorts, portNum) {
		return fmt.Errorf("port %s not allowed", port)
	}

	// Reject URLs with credentials (user:pass@host)
	if u.User != nil {
		return errors.New("URLs with credentials not allowed")
	}

	// Require hostname
	hostname := u.Hostname()
	if hostname == "" {
		return errors.New("hostname is required")
	}

	// Reject IPv6 URLs with zone identifiers (e.g., [fe80::1%eth0])
	if strings.Contains(hostname, "%") {
		return errors.New("IPv6 zone identifiers not allowed in URLs")
	}

	return nil
}
