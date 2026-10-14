package main

// Tests for endpoint-table routing and cross-dialect request/response
// hygiene. Together these guard the "CPA → plugin → upstream" legs for both
// /chat/completions and /responses upstreams, whether the upstream is zen
// or a strict openai-compatible server.

import (
	"encoding/json"
	"strconv"
	"strings"
	"testing"
	"time"
)

// =========================================================================
// official endpoint table + route resolution
// =========================================================================

func TestOfficialZenEndpointClassifications(t *testing.T) {
	cases := []struct {
		model string
		want  string
	}{
		// Family rules: /responses.
		{"gpt-5.5", "responses"},
		{"grok-4.7", "responses"},
		{"muse-spark-1.3-contributor-free", "responses"},
		// Family rules: /chat/completions, including ids the docs table
		// does not list (stale-docs resilience).
		{"deepseek-v4-flash-free", "chat"},
		{"longcat-2.5-preview-free", "chat"},
		{"mimo-v2.6-flash-free", "chat"},
		{"glm-5.3", "chat"},
		// Explicit ids.
		{"big-pickle", "chat"},
		{"qwen3.8-max", "chat"},
		// Dialects this plugin cannot speak.
		{"claude-opus-5", endpointUnsupported},
		{"claude-sonnet-5-5", endpointUnsupported}, // undocumented family member
		{"gemini-3.7-flash", endpointUnsupported},
		{"jev-1.13-free", endpointUnsupported},
		{"qwen3.7-max", endpointUnsupported},
	}
	for _, c := range cases {
		got, ok := officialZenEndpoint(c.model)
		if !ok || got != c.want {
			t.Errorf("officialZenEndpoint(%q) = %q, %v; want %q, true", c.model, got, ok, c.want)
		}
	}
	if ep, ok := officialZenEndpoint("some-new-model"); ok {
		t.Errorf("unknown model classified as %q; want not known", ep)
	}
	if ep, ok := officialZenEndpoint(""); ok {
		t.Errorf("empty model classified as %q; want not known", ep)
	}
}

func TestCandidateEndpoints(t *testing.T) {
	cfg := testConfig() // mimo pinned chat, muse pinned responses
	cases := []struct {
		model string
		want  []string
	}{
		{"mimo-v2.6-flash-free", []string{"chat"}}, // pinned: no retry
		{"muse-spark-1.3-contributor-free", []string{"responses"}},
		{"longcat-2.5-preview-free", []string{"chat", "responses"}}, // table hint + fallback
		{"gpt-5.5", []string{"responses", "chat"}},                  // responses-first + fallback
	}
	for _, c := range cases {
		route, ok := routeForModel(cfg, c.model)
		if !ok {
			t.Fatalf("routeForModel(%q) not ok", c.model)
		}
		got := candidateEndpoints(cfg, route)
		if len(got) != len(c.want) {
			t.Errorf("candidateEndpoints(%q) = %v, want %v", c.model, got, c.want)
			continue
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Errorf("candidateEndpoints(%q) = %v, want %v", c.model, got, c.want)
				break
			}
		}
	}
}

func TestRouteForModelConfiguredWithoutEndpointUsesTable(t *testing.T) {
	// A config entry that pins only alias/capabilities must not silently
	// force the chat endpoint; the official table resolves it and the
	// dialect retry stays enabled (routeConfigured == false).
	cfg := pluginConfig{
		Enabled:  true,
		Provider: "zen",
		Models:   []modelRoute{{Model: "gpt-5.5", Alias: "gpt-5.5"}},
	}
	route, ok := routeForModel(cfg, "gpt-5.5")
	if !ok || route.Endpoint != "responses" {
		t.Fatalf("route = %+v, ok=%v; want endpoints responses", route, ok)
	}
	if routeConfigured(cfg, "gpt-5.5") {
		t.Fatal("routeConfigured = true for an entry without an endpoint")
	}
}

func TestRouteForModelUsesLearnedEndpoint(t *testing.T) {
	learnedEndpoints.mu.Lock()
	old := learnedEndpoints.m
	learnedEndpoints.m = map[string]string{"learned-model": "responses"}
	learnedEndpoints.mu.Unlock()
	defer func() {
		learnedEndpoints.mu.Lock()
		learnedEndpoints.m = old
		learnedEndpoints.mu.Unlock()
	}()

	route, ok := routeForModel(testConfig(), "learned-model")
	if !ok || route.Endpoint != "responses" {
		t.Fatalf("route = %+v, ok=%v; want learned endpoint responses", route, ok)
	}
}

func TestExecuteRefusesUnsupportedDialectModel(t *testing.T) {
	resetConfig(t, testConfig())
	body, _ := json.Marshal(executorRequest{Model: "claude-opus-5", Payload: chatPayload("claude-opus-5", "hi")})
	_, err := execute(body, false)
	if err == nil || !strings.Contains(err.Error(), "cannot speak") {
		t.Fatalf("error = %v; want a clear unsupported-dialect refusal", err)
	}
}

// =========================================================================
// request hygiene: responses payload → chat upstream
// =========================================================================

func TestPrepareUpstreamBodyResponsesToChatFieldHygiene(t *testing.T) {
	resetConfig(t, testConfig())
	payload := []byte(`{
	  "model": "mimo-v2.6-flash-free",
	  "stream": false,
	  "instructions": "be terse",
	  "max_output_tokens": 750,
	  "store": false,
	  "include": ["reasoning.encrypted_content"],
	  "prompt_cache_key": "pk-1",
	  "truncation": "auto",
	  "tool_choice": {"type": "function", "name": "bash"},
	  "input": [
	    {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "no fluff"}]},
	    {"type": "message", "role": "user", "content": [
	      {"type": "input_text", "text": "look at this"},
	      {"type": "input_image", "image_url": "data:image/png;base64,AAA", "detail": "low"}
	    ]}
	  ]
	}`)

	body, err := prepareUpstreamBody(executorRequest{Payload: payload}, modelRoute{Model: "mimo-v2.6-flash-free", Endpoint: "chat"})
	if err != nil {
		t.Fatalf("prepareUpstreamBody: %v", err)
	}
	var root map[string]any
	if err := json.Unmarshal(body, &root); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	// Parameter mapping and deletions.
	if root["max_tokens"] != float64(750) {
		t.Errorf("max_tokens = %v, want 750", root["max_tokens"])
	}
	for _, k := range []string{"max_output_tokens", "store", "include", "prompt_cache_key", "truncation", "input", "instructions"} {
		if _, ok := root[k]; ok {
			t.Errorf("field %q survived responses→chat conversion", k)
		}
	}

	// tool_choice reshaped into the chat function object.
	tc, _ := root["tool_choice"].(map[string]any)
	fn, _ := tc["function"].(map[string]any)
	if tc["type"] != "function" || fn["name"] != "bash" {
		t.Errorf("tool_choice = %s; want chat nested shape", mustJSON(root["tool_choice"]))
	}

	// developer role → system (instructions already produced a system
	// message, so locate messages by role).
	messages, _ := root["messages"].([]any)
	var developerSeen bool
	var user map[string]any
	for _, m := range messages {
		entry, _ := m.(map[string]any)
		switch entry["role"] {
		case "system":
			if s := assistantText(entry["content"]); s == "no fluff" {
				developerSeen = true
			}
		case "user":
			user = entry
		}
	}
	if !developerSeen {
		t.Errorf("developer message not mapped to system: %s", mustJSON(root["messages"]))
	}
	if user == nil {
		t.Fatalf("user message missing: %s", mustJSON(root["messages"]))
	}

	// input_image → image_url with detail inside the wrapper.
	parts, _ := user["content"].([]any)
	if len(parts) != 2 {
		t.Fatalf("user content parts = %s, want text + image", mustJSON(parts))
	}
	img, _ := parts[1].(map[string]any)
	if img["type"] != "image_url" {
		t.Errorf("part type = %v, want image_url", img["type"])
	}
	urlObj, _ := img["image_url"].(map[string]any)
	if urlObj["url"] != "data:image/png;base64,AAA" || urlObj["detail"] != "low" {
		t.Errorf("image_url = %s", mustJSON(img["image_url"]))
	}
	if _, ok := img["detail"]; ok {
		t.Errorf("detail must move inside image_url, not stay top-level: %s", mustJSON(img))
	}
}

// =========================================================================
// request hygiene: chat payload → /responses upstream
// =========================================================================

func TestPrepareUpstreamBodyChatToResponsesFieldHygiene(t *testing.T) {
	resetConfig(t, testConfig())
	payload := []byte(`{
	  "model": "muse-spark-1.3-contributor-free",
	  "messages": [
	    {"role": "system", "content": "be terse"},
	    {"role": "user", "content": "hi"},
	    {"role": "assistant", "content": [{"type": "text", "text": "checking:"}],
	     "tool_calls": [{"id": "call_1", "type": "function",
	       "function": {"name": "bash", "arguments": "{\"command\":\"ls\"}", "strict": null}}]},
	    {"role": "tool", "tool_call_id": "call_1", "content": "done"}
	  ],
	  "max_tokens": 123,
	  "stream_options": {"include_usage": true},
	  "logit_bias": {"1": 5},
	  "logprobs": true,
	  "top_logprobs": 3,
	  "n": 1,
	  "tools": [{"type": "function", "function": {"name": "bash", "description": "run", "strict": null, "parameters": {"type": "object"}}}],
	  "tool_choice": {"type": "function", "function": {"name": "bash", "strict": null}},
	  "response_format": {"type": "json_schema", "json_schema": {"name": "out", "schema": {"type": "object"}, "strict": true}}
	}`)

	body, err := prepareUpstreamBody(executorRequest{Payload: payload}, modelRoute{Model: "muse-spark-1.3-contributor-free", Endpoint: "responses"})
	if err != nil {
		t.Fatalf("prepareUpstreamBody: %v", err)
	}
	var root map[string]any
	if err := json.Unmarshal(body, &root); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	// Parameter mapping and deletions.
	if root["max_output_tokens"] != float64(123) {
		t.Errorf("max_output_tokens = %v, want 123", root["max_output_tokens"])
	}
	for _, k := range []string{"max_tokens", "stream_options", "logit_bias", "logprobs", "top_logprobs", "n", "messages"} {
		if _, ok := root[k]; ok {
			t.Errorf("field %q survived chat→responses conversion", k)
		}
	}

	// tool_choice unnested into the responses shape (strict:null dropped).
	tc, _ := root["tool_choice"].(map[string]any)
	if tc["type"] != "function" || tc["name"] != "bash" {
		t.Errorf("tool_choice = %s; want flat responses shape", mustJSON(root["tool_choice"]))
	}
	if _, ok := tc["function"]; ok {
		t.Errorf("chat nested function leaked into responses tool_choice: %s", mustJSON(tc))
	}
	if strict, present := tc["strict"]; present && strict == nil {
		t.Errorf("strict:null leaked into tool_choice: %s", mustJSON(tc))
	}

	// response_format json_schema flattened into text.format.
	text, _ := root["text"].(map[string]any)
	format, _ := text["format"].(map[string]any)
	if format["type"] != "json_schema" || format["name"] != "out" {
		t.Errorf("text.format = %s; want flattened json_schema", mustJSON(format))
	}
	if _, ok := format["json_schema"]; ok {
		t.Errorf("chat json_schema envelope leaked: %s", mustJSON(format))
	}

	// tools converted to the flat responses shape, nulls stripped.
	tools, _ := root["tools"].([]any)
	bash, _ := tools[0].(map[string]any)
	if bash["type"] != "function" || bash["name"] != "bash" {
		t.Errorf("tool[0] = %s; want flat responses function", mustJSON(bash))
	}
	if _, ok := bash["function"]; ok {
		t.Errorf("chat function wrapper leaked: %s", mustJSON(bash))
	}
	if strict, present := bash["strict"]; present && strict == nil {
		t.Errorf("strict:null leaked into tools: %s", mustJSON(bash))
	}

	// Tool history and the assistant text beside the calls survive.
	input, _ := root["input"].([]any)
	var sawAssistantText, sawCall, sawOutput bool
	for _, item := range input {
		entry, _ := item.(map[string]any)
		switch entry["type"] {
		case "message":
			if entry["role"] == "assistant" {
				sawAssistantText = assistantText(entry["content"]) == "checking:"
			}
		case "function_call":
			sawCall = entry["call_id"] == "call_1" && entry["name"] == "bash"
		case "function_call_output":
			sawOutput = entry["call_id"] == "call_1" && entry["output"] == "done"
		}
	}
	if !sawAssistantText {
		t.Errorf("assistant text lost next to tool_calls: %s", mustJSON(input))
	}
	if !sawCall || !sawOutput {
		t.Errorf("tool history incomplete (call=%v output=%v): %s", sawCall, sawOutput, mustJSON(input))
	}
}

func TestChatToResponsesLegacyFunctions(t *testing.T) {
	root := map[string]any{
		"messages": []any{map[string]any{"role": "user", "content": "hi"}},
		"functions": []any{map[string]any{
			"name":       "lookup",
			"parameters": map[string]any{"type": "object"},
		}},
		"function_call": "none",
	}
	if err := chatToResponses(root); err != nil {
		t.Fatalf("chatToResponses: %v", err)
	}
	tools, _ := root["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools = %s", mustJSON(root["tools"]))
	}
	tool, _ := tools[0].(map[string]any)
	if tool["type"] != "function" || tool["name"] != "lookup" {
		t.Errorf("converted function = %s", mustJSON(tool))
	}
	if root["tool_choice"] != "none" {
		t.Errorf("tool_choice = %v, want none (from legacy function_call)", root["tool_choice"])
	}
	if _, ok := root["functions"]; ok {
		t.Error("legacy functions field leaked to /responses")
	}
}

// =========================================================================
// response hygiene: chat upstream folding
// =========================================================================

func TestFoldChatSSEPreservesRoleAndCallID(t *testing.T) {
	// Upstream that never emits a role delta and sends a real tool call id.
	sse := []byte(
		"data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_abc\",\"type\":\"function\",\"function\":{\"name\":\"bash\",\"arguments\":\"{\\\"command\\\":\\\"ls\\\"}\"}}]},\"finish_reason\":null}]}\n\n" +
			"data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n" +
			"data: [DONE]\n\n",
	)
	body, err := foldChatSSE(sse)
	if err != nil {
		t.Fatal(err)
	}
	var out map[string]any
	json.Unmarshal(body, &out)
	choices := out["choices"].([]any)
	choice := choices[0].(map[string]any)
	msg := choice["message"].(map[string]any)
	if msg["role"] != "assistant" {
		t.Errorf("message role = %q, want assistant (defaulted when upstream omits it)", msg["role"])
	}
	if choice["finish_reason"] != "tool_calls" {
		t.Errorf("finish_reason = %v", choice["finish_reason"])
	}
	calls, _ := msg["tool_calls"].([]any)
	call, _ := calls[0].(map[string]any)
	if call["id"] != "call_abc" {
		t.Errorf("tool call id = %v, want upstream id call_abc", call["id"])
	}
}

// =========================================================================
// response hygiene: /responses upstream streaming + folding
// =========================================================================

func TestResponsesChatStreamConverterIncomplete(t *testing.T) {
	cases := []struct {
		reason string
		want   string
	}{
		{"max_output_tokens", "length"},
		{"content_filter", "content_filter"},
		{"", "length"}, // unknown/missing reason: assume truncation
	}
	for _, c := range cases {
		conv := newResponsesChatStreamConverter("m")
		if _, err := conv.convert([]byte(`{"type":"response.output_text.delta","item_id":"m1","delta":"partial"}`)); err != nil {
			t.Fatalf("delta: %v", err)
		}
		outs, err := conv.convert([]byte(`{"type":"response.incomplete","response":{"id":"r1","status":"incomplete","incomplete_details":{"reason":"` + c.reason + `"},"usage":{"input_tokens":3,"output_tokens":9,"total_tokens":12}}}`))
		if err != nil {
			t.Fatalf("incomplete must not fail the stream (reason %q): %v", c.reason, err)
		}
		if len(outs) != 1 {
			t.Fatalf("incomplete emitted %d chunks, want 1", len(outs))
		}
		if got := finishReasonOf(t, decodeChunk(t, outs[0])); got != c.want {
			t.Errorf("reason %q → finish_reason %q, want %q", c.reason, got, c.want)
		}
	}

	// response.failed stays a hard error.
	conv := newResponsesChatStreamConverter("m")
	if _, err := conv.convert([]byte(`{"type":"response.failed","response":{"status":"failed","error":{"message":"boom"}}}`)); err == nil {
		t.Fatal("response.failed must still fail the stream")
	}
}

func TestResponsesChatStreamConverterCustomToolCall(t *testing.T) {
	conv := newResponsesChatStreamConverter("m")

	outs, err := conv.convert([]byte(`{"type":"response.output_item.added","item":{"type":"custom_tool_call","id":"ctc_1","call_id":"call_c1","name":"mytool","input":""}}`))
	if err != nil || len(outs) != 1 {
		t.Fatalf("added: outs=%d err=%v", len(outs), err)
	}
	head := decodeChunk(t, outs[0])
	call := toolCallsOf(t, head)[0]
	if call["id"] != "call_c1" || fnNameOf(t, call) != "mytool" {
		t.Fatalf("head = %s", mustJSON(head))
	}

	outs, err = conv.convert([]byte(`{"type":"response.custom_tool_call_input.delta","item_id":"ctc_1","delta":"{\"x\""}`))
	if err != nil || len(outs) != 1 {
		t.Fatalf("delta: outs=%d err=%v", len(outs), err)
	}
	if got := fnArgsOf(t, toolCallsOf(t, decodeChunk(t, outs[0]))[0]); got != `{"x"` {
		t.Fatalf("args fragment = %q", got)
	}

	// The done event must not duplicate arguments already streamed.
	outs, err = conv.convert([]byte(`{"type":"response.custom_tool_call_input.done","item_id":"ctc_1","input":"{\"x\":1}"}`))
	if err != nil || len(outs) != 0 {
		t.Fatalf("done after deltas: outs=%d err=%v, want no duplicate", len(outs), err)
	}

	outs, err = conv.convert([]byte(`{"type":"response.completed","response":{"id":"r","usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}}`))
	if err != nil || len(outs) != 1 {
		t.Fatalf("completed: outs=%d err=%v", len(outs), err)
	}
	if got := finishReasonOf(t, decodeChunk(t, outs[0])); got != "tool_calls" {
		t.Fatalf("finish = %q, want tool_calls", got)
	}

	// A custom tool delivered only in output_item.done (no deltas).
	conv2 := newResponsesChatStreamConverter("m")
	outs, err = conv2.convert([]byte(`{"type":"response.output_item.done","item":{"type":"custom_tool_call","id":"ctc_9","call_id":"call_9","name":"t2","input":"ls -la"}}`))
	if err != nil || len(outs) != 1 {
		t.Fatalf("done fallback: outs=%d err=%v", len(outs), err)
	}
	call = toolCallsOf(t, decodeChunk(t, outs[0]))[0]
	if call["id"] != "call_9" || fnArgsOf(t, call) != "ls -la" {
		t.Fatalf("done fallback call = %s", mustJSON(call))
	}
}

func TestResponsesCompletionToChatIncomplete(t *testing.T) {
	body := []byte(`{
	  "id": "resp_1",
	  "model": "muse-spark-1.3-contributor-free",
	  "status": "incomplete",
	  "incomplete_details": {"reason": "max_output_tokens"},
	  "output": [
	    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "partial answer"}]}
	  ],
	  "usage": {"input_tokens": 5, "output_tokens": 4000, "total_tokens": 4005}
	}`)
	out, err := responsesCompletionToChat(body, "muse-spark-1.3-contributor-free")
	if err != nil {
		t.Fatalf("incomplete must fold, not fail: %v", err)
	}
	var completion struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
			Message      struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(out, &completion); err != nil {
		t.Fatal(err)
	}
	choice := completion.Choices[0]
	if choice.FinishReason != "length" {
		t.Errorf("finish_reason = %q, want length", choice.FinishReason)
	}
	if choice.Message.Content != "partial answer" {
		t.Errorf("content = %q; partial output must survive the fold", choice.Message.Content)
	}
}

func TestFoldResponsesSSEIncompletePrefersNestedResponse(t *testing.T) {
	sse := []byte(
		"data: {\"type\":\"response.created\",\"response\":{\"id\":\"r1\",\"status\":\"in_progress\"}}\n\n" +
			"data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_1\",\"role\":\"assistant\",\"content\":[]}}\n\n" +
			"data: {\"type\":\"response.incomplete\",\"response\":{\"id\":\"r1\",\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"max_output_tokens\"},\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"cut off\"}]}],\"usage\":{\"input_tokens\":2,\"output_tokens\":100,\"total_tokens\":102}}}\n\n",
	)
	folded, err := foldResponsesSSE(sse)
	if err != nil {
		t.Fatal(err)
	}
	var resp struct {
		ID     string `json:"id"`
		Status string `json:"status"`
	}
	if err := json.Unmarshal(folded, &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Status != "incomplete" {
		t.Errorf("folded status = %q, want incomplete (terminal event must be preferred over the created status)", resp.Status)
	}

	chat, err := responsesCompletionToChat(folded, "m")
	if err != nil {
		t.Fatal(err)
	}
	var completion struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(chat, &completion); err != nil {
		t.Fatal(err)
	}
	if completion.Choices[0].FinishReason != "length" {
		t.Errorf("finish_reason = %q, want length", completion.Choices[0].FinishReason)
	}
}

// =========================================================================
// replayed reasoning items (the 2026-10-01 interruption root cause)
// =========================================================================

// TestPrepareUpstreamBodyDropsReplayedReasoningItems guards the bug that
// interrupted every muse follow-up turn: pi (codex responses dialect)
// replays reasoning items whose rs_ ids reference server state the zen
// gateway does not retain, and the responses→responses passthrough path
// never stripped them. The 400s then cascaded into auth_unavailable.
func TestPrepareUpstreamBodyDropsReplayedReasoningItems(t *testing.T) {
	resetConfig(t, testConfig())
	payload := []byte(`{
	  "model": "muse-spark-1.3-contributor-free",
	  "stream": true,
	  "previous_response_id": "resp_old",
	  "input": [
	    {"type": "reasoning", "id": "rs_resp_6abdbe95e6cc8c41773e47d1_0", "summary": [{"type": "summary_text", "text": "thinking"}]},
	    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
	    {"type": "function_call", "call_id": "call_1", "name": "bash", "arguments": "{\"command\":\"ls\"}"},
	    {"type": "function_call_output", "call_id": "call_1", "output": "done"}
	  ]
	}`)
	body, err := prepareUpstreamBody(executorRequest{Payload: payload}, modelRoute{Model: "muse-spark-1.3-contributor-free", Endpoint: "responses"})
	if err != nil {
		t.Fatalf("prepareUpstreamBody: %v", err)
	}
	var root map[string]any
	if err := json.Unmarshal(body, &root); err != nil {
		t.Fatal(err)
	}
	if _, ok := root["previous_response_id"]; ok {
		t.Fatal("previous_response_id survived: zen keeps no server response state")
	}
	input, _ := root["input"].([]any)
	if len(input) != 3 {
		t.Fatalf("input items = %s, want 3 (reasoning dropped, rest preserved)", mustJSON(input))
	}
	for _, item := range input {
		entry, _ := item.(map[string]any)
		if itemType, _ := entry["type"].(string); strings.EqualFold(itemType, "reasoning") {
			t.Fatalf("replayed reasoning item survived: %s", mustJSON(entry))
		}
	}
	// The tool history must survive untouched.
	first, _ := input[1].(map[string]any)
	if first["type"] != "function_call" || first["call_id"] != "call_1" {
		t.Fatalf("function_call item lost: %s", mustJSON(input))
	}
}

// =========================================================================
// reasoning effort normalization
// =========================================================================

func TestNormalizeReasoningEffort(t *testing.T) {
	// Effort-type model (muse: minimal..xhigh): max maps to xhigh.
	root := map[string]any{"reasoning_effort": "max"}
	normalizeReasoningEffort(root, "chat", "muse-spark-1.3-contributor-free")
	if root["reasoning_effort"] != "xhigh" {
		t.Fatalf("muse max → %v, want xhigh", root["reasoning_effort"])
	}

	// Responses dialect: effort consolidates into reasoning.effort.
	root = map[string]any{"reasoning": map[string]any{"effort": "max"}}
	normalizeReasoningEffort(root, "responses", "muse-spark-1.3-contributor-free")
	reasoning, _ := root["reasoning"].(map[string]any)
	if reasoning["effort"] != "xhigh" {
		t.Fatalf("responses muse effort = %v, want xhigh", reasoning["effort"])
	}

	// Toggle-only model (mimo): the parameter is rejected, drop it.
	root = map[string]any{"reasoning_effort": "high"}
	normalizeReasoningEffort(root, "chat", "mimo-v2.6-flash-free")
	if _, ok := root["reasoning_effort"]; ok {
		t.Fatalf("mimo is toggle-only, effort must be dropped: %v", root)
	}

	// deepseek-v4-flash-free supports low/high/max: medium maps to high.
	root = map[string]any{"reasoning_effort": "medium"}
	normalizeReasoningEffort(root, "chat", "deepseek-v4-flash-free")
	if root["reasoning_effort"] != "high" {
		t.Fatalf("deepseek medium → %v, want high", root["reasoning_effort"])
	}

	// Unknown model: clamp to the safe set.
	root = map[string]any{"reasoning_effort": "max"}
	normalizeReasoningEffort(root, "chat", "some-new-model")
	if root["reasoning_effort"] != "high" {
		t.Fatalf("unknown max → %v, want high", root["reasoning_effort"])
	}

	// off/none means no thinking.
	root = map[string]any{"reasoning_effort": "off"}
	normalizeReasoningEffort(root, "chat", "muse-spark-1.3-contributor-free")
	if _, ok := root["reasoning_effort"]; ok {
		t.Fatalf("effort off must be dropped: %v", root)
	}

	// The anthropic-style thinking object never survives.
	root = map[string]any{"thinking": map[string]any{"type": "enabled", "budget_tokens": 8192}}
	normalizeReasoningEffort(root, "chat", "muse-spark-1.3-contributor-free")
	if _, ok := root["thinking"]; ok {
		t.Fatalf("thinking object survived: %v", root)
	}

	// A reasoning object without effort (e.g. summary request) survives on
	// the responses dialect, minus any effort key.
	root = map[string]any{"reasoning": map[string]any{"summary": "auto"}}
	normalizeReasoningEffort(root, "responses", "muse-spark-1.3-contributor-free")
	reasoning, _ = root["reasoning"].(map[string]any)
	if reasoning["summary"] != "auto" {
		t.Fatalf("reasoning.summary lost: %v", root)
	}
	if _, ok := reasoning["effort"]; ok {
		t.Fatalf("effort must not be invented: %v", reasoning)
	}

	// Chat dialect drops a leftover reasoning object entirely.
	root = map[string]any{"reasoning": map[string]any{"effort": "high"}}
	normalizeReasoningEffort(root, "chat", "muse-spark-1.3-contributor-free")
	if _, ok := root["reasoning"]; ok {
		t.Fatalf("chat dialect must not carry a reasoning object: %v", root)
	}
	if root["reasoning_effort"] != "high" {
		t.Fatalf("chat effort = %v, want high", root["reasoning_effort"])
	}
}

// =========================================================================
// output token clamping
// =========================================================================

func TestClampMaxOutput(t *testing.T) {
	root := map[string]any{"max_tokens": 999999.0}
	clampMaxOutput(root, "muse-spark-1.3-contributor-free")
	if root["max_tokens"] != float64(131072) {
		t.Fatalf("muse max_tokens = %v, want 131072", root["max_tokens"])
	}

	root = map[string]any{"max_output_tokens": 999999.0}
	clampMaxOutput(root, "some-unknown-model")
	if root["max_output_tokens"] != float64(65536) {
		t.Fatalf("unknown model cap = %v, want 65536", root["max_output_tokens"])
	}

	root = map[string]any{"max_tokens": 1000.0}
	clampMaxOutput(root, "muse-spark-1.3-contributor-free")
	if root["max_tokens"] != float64(1000) {
		t.Fatalf("in-range max_tokens must be untouched: %v", root["max_tokens"])
	}
}

// =========================================================================
// toolless requests
// =========================================================================

func TestEnsureGateToolsToollessRequest(t *testing.T) {
	root := map[string]any{"messages": []any{map[string]any{"role": "user", "content": "summarize this"}}}
	ensureGateTools(root, "chat")
	tools, _ := root["tools"].([]any)
	if len(tools) != len(officialGateTools) {
		t.Fatalf("tools = %d, want the full official set", len(tools))
	}
	if root["tool_choice"] != "none" {
		t.Fatalf("tool_choice = %v, want none (client sent no tools)", root["tool_choice"])
	}

	// An explicit tool_choice is never overwritten.
	root = map[string]any{
		"messages":    []any{map[string]any{"role": "user", "content": "hi"}},
		"tool_choice": "auto",
	}
	ensureGateTools(root, "chat")
	if root["tool_choice"] != "auto" {
		t.Fatalf("explicit tool_choice overwritten: %v", root["tool_choice"])
	}

	// Requests that already carry tools keep their own tool_choice absent.
	root = map[string]any{
		"messages": []any{map[string]any{"role": "user", "content": "hi"}},
		"tools": []any{map[string]any{
			"type":     "function",
			"function": map[string]any{"name": "bash", "parameters": map[string]any{"type": "object"}},
		}},
	}
	ensureGateTools(root, "chat")
	if _, exists := root["tool_choice"]; exists {
		t.Fatalf("tool_choice must not be injected when the client sent tools: %v", root["tool_choice"])
	}
}

// =========================================================================
// official session id minting
// =========================================================================

func TestNewOfficialSessionID(t *testing.T) {
	id := newOfficialSessionID()
	if !canonicalSessionRe.MatchString(id) {
		t.Fatalf("id %q does not match the official shape", id)
	}
	if !isGateSessionID(id) {
		t.Fatalf("id %q fails the free-tier gate (low 12 bits must be %s)", id, gateSessionLowHex)
	}
	// The descending timestamp must decode to roughly now. The hex field
	// keeps a 36-bit millisecond component that counts down from the epoch
	// constant, so decoding resolves the wraparound against the current time.
	field, err := strconv.ParseUint(id[4:16], 16, 64)
	if err != nil {
		t.Fatal(err)
	}
	const mod36 = int64(1) << 36
	expected := ((zenSessionEpochMillis-time.Now().UnixMilli())%mod36 + mod36) % mod36
	delta := (int64(field>>12) - expected) % mod36
	if delta < 0 {
		delta += mod36
	}
	if delta > 5000 && delta < mod36-5000 {
		t.Fatalf("session id timestamp field is off by %dms", delta)
	}
	// Two ids differ (random suffix).
	if newOfficialSessionID() == id {
		t.Fatal("session ids must be unique")
	}
}

// TestOfficialSessionIDMatchesRealClient pins the encoding to ids observed
// from the official OpenCode client: the timestamp field plus the creation
// millisecond equals the epoch constant, and the low 12 bits are always 0xFFE.
func TestOfficialSessionIDMatchesRealClient(t *testing.T) {
	samples := []struct {
		id        string
		unixMilli int64
	}{
		{"ses_f390f32e4ffeAZG1xXDw56ko1q", 1790044065052},
		{"ses_fc7b694ecffeul9nCWYfheWe5J", 1787650730772},
		{"ses_fc42c626effe4bCNliD9ZNg227", 1787710119314},
	}
	for _, sample := range samples {
		if !isGateSessionID(sample.id) {
			t.Fatalf("real client id %q rejected by the gate check", sample.id)
		}
		field, err := strconv.ParseUint(sample.id[4:16], 16, 64)
		if err != nil {
			t.Fatal(err)
		}
		if got := int64(field>>12) + sample.unixMilli; got != zenSessionEpochMillis {
			t.Fatalf("id %q decodes to epoch %d, want %d", sample.id, got, zenSessionEpochMillis)
		}
		if low := field & 0xFFF; low != 0xFFE {
			t.Fatalf("id %q low bits = %#x, want 0xFFE", sample.id, low)
		}
	}
}

func TestIsGateSessionID(t *testing.T) {
	cases := []struct {
		id   string
		want bool
	}{
		{"ses_000000000ffeaBCDEFghijklmn", true},
		{"ses_123456789abcABCDEFghijklmn", false},
		{"ses_000000000000ABCDEFghijklmn", false},
		{"ses_000000000ffe", false},
		{"msg_000000000ffeABCDEFghijklmn", false},
		{"", false},
	}
	for _, tc := range cases {
		if got := isGateSessionID(tc.id); got != tc.want {
			t.Fatalf("isGateSessionID(%q) = %v, want %v", tc.id, got, tc.want)
		}
	}
}

// TestSessionIDForSourceReusesID documents that one conversation keeps one
// session id, so affinity and the upstream prompt cache survive every turn.
func TestSessionIDForSourceReusesID(t *testing.T) {
	first := sessionIDForSource("conversation-a")
	if !isGateSessionID(first) {
		t.Fatalf("session id %q is not gate valid", first)
	}
	if second := sessionIDForSource("conversation-a"); second != first {
		t.Fatalf("same source produced %q then %q", first, second)
	}
	if other := sessionIDForSource("conversation-b"); other == first {
		t.Fatal("distinct sources must not share a session id")
	}
}

func TestRefreshGateSession(t *testing.T) {
	headers := map[string][]string{
		targetSessionHeader: {"ses_old"},
	}
	refreshGateSession(headers)
	session := headers[targetSessionHeader][0]
	if !canonicalSessionRe.MatchString(session) || session == "ses_old" {
		t.Fatalf("refreshGateSession left %q", session)
	}
	if !isGateSessionID(session) {
		t.Fatalf("refreshGateSession minted a gate-invalid id %q", session)
	}
	if headers["X-Session-Affinity"][0] != session || headers["X-Session-Id"][0] != session {
		t.Fatalf("affinity headers not aligned: %v", headers)
	}
}
