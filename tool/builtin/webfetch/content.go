package webfetch

import (
	"bytes"
	"io"
	"mime"
	"strings"

	"golang.org/x/net/html/charset"
)

const (
	charsetUTF8 = "utf-8"
)

// allowedMedia checks if a media type is allowed by the configuration.
// Uses proper MIME parsing to handle parameters and supports wildcards like "text/*".
func allowedMedia(allowed []string, contentTypeHeader string) bool {
	if contentTypeHeader == "" {
		return false
	}

	// Parse the content type to extract just the media type (ignore parameters)
	mediaType, _, err := mime.ParseMediaType(contentTypeHeader)
	if err != nil {
		// If parsing fails, fall back to simple extraction
		mediaType = strings.ToLower(strings.TrimSpace(strings.Split(contentTypeHeader, ";")[0]))
	} else {
		mediaType = strings.ToLower(mediaType)
	}

	if mediaType == "" {
		return false
	}

	for _, allowedType := range allowed {
		allowedType = strings.ToLower(allowedType)

		// Support wildcards like "text/*"
		if before, ok := strings.CutSuffix(allowedType, "/*"); ok {
			prefix := before
			if strings.HasPrefix(mediaType, prefix+"/") {
				return true
			}
		}

		// Exact match
		if mediaType == allowedType {
			return true
		}
	}

	return false
}

// shouldNormalize determines if content should be normalized to UTF-8.
// Returns true for text-based content types that benefit from charset conversion.
func shouldNormalize(media string) bool {
	if strings.HasPrefix(media, "text/") {
		return true
	}

	switch media {
	case "application/json",
		"application/xml",
		"application/xhtml+xml":
		return true
	default:
		return false
	}
}

// toUTF8 converts content to UTF-8 encoding using charset detection.
// It handles both explicit charset parameters and heuristic detection.
func toUTF8(content []byte, contentType string) ([]byte, string, error) {
	// Parse content type to extract charset parameter
	_, params, err := mime.ParseMediaType(contentType)

	var declaredCharset string
	if err == nil {
		declaredCharset = strings.ToLower(params["charset"])
	}

	// Use charset detection to determine the actual encoding
	// This handles cases where the declared charset is wrong or missing
	encoding, name, _ := charset.DetermineEncoding(content, contentType)
	if name != "" {
		declaredCharset = strings.ToLower(name)
	}

	// If already UTF-8, return as-is
	if declaredCharset == charsetUTF8 || declaredCharset == "" {
		return content, charsetUTF8, nil
	}

	// Convert to UTF-8
	reader := encoding.NewDecoder().Reader(bytes.NewReader(content))

	converted, err := io.ReadAll(reader)
	if err != nil {
		return nil, "", err
	}

	return converted, charsetUTF8, nil
}
