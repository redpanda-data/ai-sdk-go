module github.com/redpanda-data/ai-sdk-go/examples/vercelaisdk_chat

go 1.26

require github.com/redpanda-data/ai-sdk-go v0.0.0

require (
	github.com/openai/openai-go/v3 v3.42.0 // indirect
	github.com/rs/xid v1.6.0 // indirect
	github.com/tidwall/gjson v1.18.0 // indirect
	github.com/tidwall/match v1.2.0 // indirect
	github.com/tidwall/pretty v1.2.1 // indirect
	github.com/tidwall/sjson v1.2.5 // indirect
	github.com/twmb/go-cache v1.3.0 // indirect
	golang.org/x/sync v0.21.0 // indirect
)

replace github.com/redpanda-data/ai-sdk-go => ../..
