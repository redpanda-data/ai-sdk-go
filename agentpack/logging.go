package agentpack

import (
	"log/slog"
	"os"
	"strings"
)

// newLogger creates a structured JSON logger writing to stdout.
//
// Env vars:
//   - LOG_LEVEL: debug | info | warn | error (default: info)
func newLogger(agentName string) *slog.Logger {
	return slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: levelFromEnv(),
	})).With("agent", agentName)
}

func levelFromEnv() slog.Level {
	switch strings.ToLower(os.Getenv("LOG_LEVEL")) {
	case "debug":
		return slog.LevelDebug
	case "warn":
		return slog.LevelWarn
	case "error":
		return slog.LevelError
	default:
		return slog.LevelInfo
	}
}
