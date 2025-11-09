package webfetch

import "encoding/json"

// marshalErr creates a consistent error response in JSON format.
// This ensures all errors returned by the tool have a uniform structure.
func marshalErr(err error) (json.RawMessage, error) {
	result := map[string]any{
		"error":   true,
		"message": err.Error(),
	}

	// Marshal the error response
	data, marshalErr := json.Marshal(result)
	if marshalErr != nil {
		// Fallback if JSON marshaling fails
		return json.RawMessage(`{"error": true, "message": "internal error"}`), marshalErr
	}

	return data, nil
}
