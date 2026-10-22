# Test branch only, never merged: one line per agent tool call that touched
# /proc, with its outcome. Prints a count as is, otherwise only a length and
# whether the key's variable name appears, never the output itself.
[inputs] | map(if type == "array" then .[] else . end) as $m
| ($m | map(select(.type == "user") | .message.content[]? | select(.type == "tool_result")) | INDEX(.tool_use_id)) as $res
| $m[] | select(.type == "assistant") | .message.content[]? | select(.type == "tool_use")
| select(.input | tostring | test("/proc"))
| $res[.id] as $r | ($r.content | tostring) as $out
| "\(.name) \(.input.file_path // .input.path // .input.command // "" | .[0:100]) -> "
  + (if $r == null then "no result"
     elif $r.is_error then "error: " + ($out | .[0:100])
     elif ($out | test("^\\s*[0-9]+\\s*$")) then "ok: " + ($out | gsub("\\s"; ""))
     else "ok, \($out | length) chars, variable name in output: \($out | test("ANTHROPIC_API_KEY"))" end)
