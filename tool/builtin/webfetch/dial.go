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
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"net"
	"net/netip"
	"time"
)

// extraBlocked contains additional IP ranges commonly blocked for SSRF hardening.
// These are beyond what netip.Addr built-in methods cover (IsPrivate, IsLoopback, etc.)
var extraBlocked = []netip.Prefix{
	// IPv4 special-purpose ranges
	mustPrefix("0.0.0.0/8"),          // "this network" (RFC 1122)
	mustPrefix("100.64.0.0/10"),      // Carrier-Grade NAT (RFC 6598)
	mustPrefix("192.0.0.0/24"),       // IETF Protocol Assignments (RFC 6890)
	mustPrefix("192.0.2.0/24"),       // Documentation (TEST-NET-1)
	mustPrefix("192.88.99.0/24"),     // 6to4 Relay Anycast (RFC 3068)
	mustPrefix("198.18.0.0/15"),      // Benchmarking (RFC 2544)
	mustPrefix("198.51.100.0/24"),    // Documentation (TEST-NET-2)
	mustPrefix("203.0.113.0/24"),     // Documentation (TEST-NET-3)
	mustPrefix("240.0.0.0/4"),        // Reserved for Future Use (RFC 1112)
	mustPrefix("255.255.255.255/32"), // Limited Broadcast (RFC 0919)

	// IPv6 special-purpose ranges
	mustPrefix("2001:db8::/32"), // Documentation (RFC 3849)
	mustPrefix("2001:10::/28"),  // ORCHID (RFC 4843)
	mustPrefix("2001:20::/28"),  // ORCHIDv2 (RFC 7343)
	mustPrefix("64:ff9b::/96"),  // IPv4-IPv6 Translation (RFC 6052)
	mustPrefix("fec0::/10"),     // Site-Local (deprecated, RFC 3879)
}

// mustPrefix parses a CIDR prefix and panics on error (for compile-time constants).
func mustPrefix(s string) netip.Prefix {
	p, err := netip.ParsePrefix(s)
	if err != nil {
		panic(fmt.Sprintf("invalid prefix %q: %v", s, err))
	}

	return p
}

// isPrivateOrReserved returns true if addr should be blocked for SSRF purposes.
func isPrivateOrReserved(addr netip.Addr) bool {
	if !addr.IsValid() {
		return true // treat unknown as unsafe
	}

	// If it's an IPv4-mapped IPv6 (e.g. ::ffff:192.168.1.2), unmap to v4 first.
	if addr.Is4In6() {
		addr = addr.Unmap()
	}

	// Built-in quick checks
	if addr.IsPrivate() || addr.IsLoopback() || addr.IsLinkLocalUnicast() ||
		addr.IsMulticast() || addr.IsUnspecified() {
		return true
	}

	// Extra IPv4 ranges commonly blocked for SSRF hardening
	for _, p := range extraBlocked {
		if p.Contains(addr) {
			return true
		}
	}

	return false
}

// hostnameKey is used to pass the original hostname through context for TLS SNI.
type hostnameKey struct{}

// safeDialContext creates a context-aware dialer that blocks private IP addresses.
// This prevents SSRF attacks by resolving the hostname, validating all IPs, and
// dialing the first safe IP directly (not the hostname) to prevent DNS rebinding.
func safeDialContext(cfg Config) func(ctx context.Context, network, addr string) (net.Conn, error) {
	return func(ctx context.Context, network, addr string) (net.Conn, error) {
		// If private IP blocking is disabled, use default dialer
		if !cfg.DenyPrivateIPs {
			dialer := &net.Dialer{
				Timeout:   3 * time.Second,
				KeepAlive: 30 * time.Second,
			}

			return dialer.DialContext(ctx, network, addr)
		}

		// Extract hostname and port from address
		host, port, err := net.SplitHostPort(addr)
		if err != nil {
			return nil, err
		}

		// If it's already an IP literal, validate it directly
		if ip, err := netip.ParseAddr(host); err == nil {
			if isPrivateOrReserved(ip) {
				return nil, errors.New("connection to private/reserved IP blocked")
			}
			// Safe to dial IP directly
			dialer := &net.Dialer{
				Timeout:   3 * time.Second,
				KeepAlive: 30 * time.Second,
			}

			return dialer.DialContext(ctx, network, addr)
		}

		// Resolve hostname to check all IPs
		ips, err := net.DefaultResolver.LookupIPAddr(ctx, host)
		if err != nil {
			return nil, err
		}

		if len(ips) == 0 {
			return nil, errors.New("hostname resolved to no IP addresses")
		}

		// Find the first safe IP to dial
		var safeIP netip.Addr

		for _, ipAddr := range ips {
			addr, ok := netip.AddrFromSlice(ipAddr.IP)
			if !ok || isPrivateOrReserved(addr) {
				continue // Skip private/reserved IPs
			}

			safeIP = addr

			break
		}

		// If no safe IP found, block the request
		if !safeIP.IsValid() {
			return nil, errors.New("all resolved IPs are private/reserved - connection blocked")
		}

		// CRITICAL: Dial the validated IP directly, not the hostname
		// This prevents DNS rebinding attacks where DNS changes between validation and connect
		safeAddr := net.JoinHostPort(safeIP.String(), port)
		dialer := &net.Dialer{
			Timeout:   3 * time.Second,
			KeepAlive: 30 * time.Second,
		}
		// Store original hostname in context for TLS SNI
		ctx = context.WithValue(ctx, hostnameKey{}, host)

		return dialer.DialContext(ctx, network, safeAddr)
	}
}

// safeTLSDialContext creates a TLS dialer that properly handles SNI when dialing by IP.
func safeTLSDialContext(cfg Config) func(ctx context.Context, network, addr string) (net.Conn, error) {
	return func(ctx context.Context, network, addr string) (net.Conn, error) {
		// First establish the TCP connection using safeDialContext
		conn, err := safeDialContext(cfg)(ctx, network, addr)
		if err != nil {
			return nil, err
		}

		// Extract original hostname from context for SNI
		hostname, ok := ctx.Value(hostnameKey{}).(string)
		if !ok || hostname == "" {
			// Fallback: extract from addr (this shouldn't happen with our setup)
			hostname, _, _ = net.SplitHostPort(addr)
		}

		// Create TLS config with proper ServerName for certificate validation
		tlsConfig := &tls.Config{
			MinVersion:         tls.VersionTLS12,
			ServerName:         hostname,               // Use original hostname for SNI, not the IP
			InsecureSkipVerify: cfg.InsecureSkipVerify, //nolint:gosec // InsecureSkipVerify is only for testing
		}

		// Wrap connection with TLS
		tlsConn := tls.Client(conn, tlsConfig)
		if err := tlsConn.Handshake(); err != nil {
			_ = conn.Close()
			return nil, err
		}

		return tlsConn, nil
	}
}
