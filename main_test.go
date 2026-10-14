package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

var canonicalShapeRe = regexp.MustCompile(`^(ses|msg)_[0-9a-f]{12}[0-9A-Za-z]{14}$`)

func resetConfig(t *testing.T, cfg pluginConfig) {
	t.Helper()
	storeConfig(cfg)
	t.Cleanup(func() { storeConfig(defaultPluginConfig()) })
}

func TestNormalizeSSEFramePreservesEventLine(t *testing.T) {
	res := normalizeSSEFrame(false, []byte("event: response.completed\n"))
	if string(res) != "event: response.completed\n\n" {
		t.Fatalf("event line = %q", string(res))
	}
}

func testConfig() pluginConfig {
	return pluginConfig{
		Enabled:  true,
		Provider: "zen",
		BaseURL:  "https://opencode.ai/zen/v1",
		APIKeys:  []string{"sk-zen-test-1", "sk-zen-test-2"},
		Client:   "cli",
		Project:  "global",
		Models: []modelRoute{
			{Model: "mimo-v2.6-flash-free", Alias: "mimo-v2.6-flash-free", Endpoint: "chat"},
			{Model: "muse-spark-1.3-contributor-free", Alias: "muse-spark-1.3-contributor-free", Endpoint: "responses"},
		},
	}
}

func chatPayload(model, content string) []byte {
	raw, _ := json.Marshal(map[string]any{
		"model":    model,
		"messages": []any{map[string]any{"role": "user", "content": content}},
		"stream":   true,
	})
	return raw
}

func responsesPayload(model, input string) []byte {
	raw, _ := json.Marshal(map[string]any{
		"model":  model,
		"input":  []any{map[string]any{"role": "user", "content": input}},
		"stream": true,
	})
	return raw
}

// =========================================================================
// registration & config
// =========================================================================

func TestRegister(t *testing.T) {
	out, err := handleMethod("plugin.register", nil)
	if err != nil {
		t.Fatal(err)
	}
	var env envelope
	if err := json.Unmarshal(out, &env); err != nil {
		t.Fatal(err)
	}
	if !env.OK {
		t.Fatalf("register not ok: %s", out)
	}
	var reg registration
	if err := json.Unmarshal(env.Result, &reg); err != nil {
		t.Fatal(err)
	}
	if reg.SchemaVersion != abiVersion {
		t.Fatalf("schema version = %d, want %d", reg.SchemaVersion, abiVersion)
	}
	if !reg.Capabilities.Executor {
		t.Fatal("want executor capability")
	}
	if !reg.Capabilities.ModelRegistrar {
		t.Fatal("want model_registrar capability")
	}
	if !reg.Capabilities.AuthProvider {
		t.Fatal("want auth_provider capability")
	}
	if reg.Metadata.Name != pluginID {
		t.Fatalf("metadata name = %q", reg.Metadata.Name)
	}
}

func TestConfigureParsesModels(t *testing.T) {
	// The host marshals config_yaml as []byte, which JSON-encodes as base64.
	raw, _ := json.Marshal(map[string]any{
		"config_yaml": []byte("provider: zen\napi-keys:\n  - sk-key1\n  - sk-key2\nmodels:\n  - model: mimo-v2.6-flash-free\n    endpoint: chat\n  - model: muse-spark-1.3-contributor-free\n    endpoint: responses\nclient: opencode\n"),
	})
	if err := configure(raw); err != nil {
		t.Fatal(err)
	}
	cfg := loadedConfig()
	if len(cfg.Models) != 2 {
		t.Fatalf("models = %v", cfg.Models)
	}
	if cfg.Client != "opencode" {
		t.Fatalf("client = %q", cfg.Client)
	}
	if cfg.APIKeys[0] != "sk-key1" || cfg.APIKeys[1] != "sk-key2" {
		t.Fatalf("api keys = %v", cfg.APIKeys)
	}
	if cfg.Models[0].Endpoint != "chat" || cfg.Models[1].Endpoint != "responses" {
		t.Fatalf("endpoints: %q %q", cfg.Models[0].Endpoint, cfg.Models[1].Endpoint)
	}
}

func TestModelRegistration(t *testing.T) {
	resetConfig(t, testConfig())
	out, err := modelRegistration()
	if err != nil {
		t.Fatal(err)
	}
	// modelRegistration returns the full envelope; unwrap it first.
	var env envelope
	if err := json.Unmarshal(out, &env); err != nil {
		t.Fatal(err)
	}
	if !env.OK {
		t.Fatalf("registration envelope not ok: %s", out)
	}
	var resp modelRegistrationResponse
	if err := json.Unmarshal(env.Result, &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Provider != "zen" || len(resp.Models) != 2 {
		t.Fatalf("registration = %+v", resp)
	}
	if resp.Models[0].ID != "mimo-v2.6-flash-free" || resp.Models[1].ID != "muse-spark-1.3-contributor-free" {
		t.Fatalf("model IDs = %q %q", resp.Models[0].ID, resp.Models[1].ID)
	}
}

func TestModelRegistrationIncludesCapabilities(t *testing.T) {
	resetConfig(t, testConfig())
	out, err := modelRegistration()
	if err != nil {
		t.Fatal(err)
	}
	var env envelope
	json.Unmarshal(out, &env)
	var resp modelRegistrationResponse
	json.Unmarshal(env.Result, &resp)
	var mimo *modelInfo
	for i := range resp.Models {
		if resp.Models[i].ID == "mimo-v2.6-flash-free" {
			mimo = &resp.Models[i]
		}
	}
	if mimo == nil {
		t.Fatal("mimo model missing from registration")
	}
	// Aligned with the OpenCode client's own catalog (models.dev, "opencode"
	// provider): mimo free tier is 200K context, not 1M.
	if mimo.ContextLength != 200000 {
		t.Fatalf("mimo ContextLength = %d, want 200000", mimo.ContextLength)
	}
	if mimo.MaxCompletionTokens != 32000 {
		t.Fatalf("mimo MaxCompletionTokens = %d, want 32000", mimo.MaxCompletionTokens)
	}
	var muse *modelInfo
	for i := range resp.Models {
		if resp.Models[i].ID == "muse-spark-1.3-contributor-free" {
			muse = &resp.Models[i]
		}
	}
	if muse == nil {
		t.Fatal("muse model missing from registration")
	}
	// muse is 1M context / 128K output; without the declaration CPA's codex
	// catalog falls back to the gpt-5.5 template (272000) and clients show
	// a wrong context window.
	if muse.ContextLength != 1048576 {
		t.Fatalf("muse ContextLength = %d, want 1048576", muse.ContextLength)
	}
	if muse.MaxCompletionTokens != 131072 {
		t.Fatalf("muse MaxCompletionTokens = %d, want 131072", muse.MaxCompletionTokens)
	}
	if muse.Thinking == nil || len(muse.Thinking.Levels) == 0 {
		t.Fatal("muse must announce its reasoning effort levels")
	}
}

func TestConfigureParsesModelCapabilities(t *testing.T) {
	raw, _ := json.Marshal(map[string]any{
		"config_yaml": []byte("models:\n  - model: custom-model\n    endpoint: chat\n    context-length: 2000000\n    max-completion-tokens: 32768\n    input-modalities:\n      - text\n      - image\n    reasoning-levels: none, low, high\n"),
	})
	if err := configure(raw); err != nil {
		t.Fatal(err)
	}
	cfg := loadedConfig()
	if len(cfg.Models) != 1 {
		t.Fatalf("models = %v", cfg.Models)
	}
	m := cfg.Models[0]
	if m.ContextLength != 2000000 || m.MaxCompletionTokens != 32768 {
		t.Fatalf("capability values = %+v", m)
	}
	if len(m.InputModalities) != 2 || m.InputModalities[0] != "text" || m.InputModalities[1] != "image" {
		t.Fatalf("input modalities = %v", m.InputModalities)
	}
	if len(m.ReasoningLevels) != 3 || m.ReasoningLevels[0] != "none" {
		t.Fatalf("reasoning levels = %v", m.ReasoningLevels)
	}
	caps := resolveModelCapabilities(m)
	if caps.Thinking == nil || len(caps.Thinking.Levels) != 3 {
		t.Fatalf("thinking support = %+v", caps.Thinking)
	}
}

func TestIdentifierMethod(t *testing.T) {
	resetConfig(t, testConfig())
	out, err := handleMethod("executor.identifier", nil)
	if err != nil {
		t.Fatal(err)
	}
	var env envelope
	if err := json.Unmarshal(out, &env); err != nil {
		t.Fatal(err)
	}
	if !env.OK {
		t.Fatalf("identifier not ok: %s", out)
	}
	var id identifierResponse
	if err := json.Unmarshal(env.Result, &id); err != nil {
		t.Fatal(err)
	}
	if id.Identifier != "zen" {
		t.Fatalf("identifier = %q", id.Identifier)
	}
}

func TestCountTokens(t *testing.T) {
	out, err := handleMethod("executor.count_tokens", nil)
	if err != nil {
		t.Fatal(err)
	}
	var env envelope
	json.Unmarshal(out, &env)
	if !env.OK {
		t.Fatalf("count_tokens not ok: %s", out)
	}
}

func TestAuthParseReturnsRecords(t *testing.T) {
	resetConfig(t, testConfig())
	out, err := handleMethod("auth.parse", nil)
	if err != nil {
		t.Fatal(err)
	}
	var env envelope
	if err := json.Unmarshal(out, &env); err != nil {
		t.Fatal(err)
	}
	if !env.OK {
		t.Fatalf("auth.parse not ok: %s", out)
	}
	var resp authParseResponse
	if err := json.Unmarshal(env.Result, &resp); err != nil {
		t.Fatal(err)
	}
	if !resp.Handled {
		t.Fatalf("auth.parse should be handled: %s", out)
	}
	if len(resp.Auths) != 2 {
		t.Fatalf("auths = %d, want 2: %s", len(resp.Auths), out)
	}
	for _, a := range resp.Auths {
		if a.Provider != "zen" {
			t.Fatalf("auth provider = %q", a.Provider)
		}
		if len(a.StorageJSON) == 0 {
			t.Fatalf("auth storage json empty for %q", a.ID)
		}
		if a.ID == "" || a.FileName == "" {
			t.Fatalf("auth id/file empty: %+v", a)
		}
	}
}

func TestAuthIdentifierMethod(t *testing.T) {
	resetConfig(t, testConfig())
	out, err := handleMethod("auth.identifier", nil)
	if err != nil {
		t.Fatal(err)
	}
	var env envelope
	if err := json.Unmarshal(out, &env); err != nil {
		t.Fatal(err)
	}
	if !env.OK {
		t.Fatalf("auth.identifier not ok: %s", out)
	}
	var id identifierResponse
	if err := json.Unmarshal(env.Result, &id); err != nil {
		t.Fatal(err)
	}
	if id.Identifier != "zen" {
		t.Fatalf("identifier = %q", id.Identifier)
	}
}

func TestModelRegistrationDefaultFallback(t *testing.T) {
	resetConfig(t, pluginConfig{Enabled: true, Provider: "zen"})
	out, err := modelRegistration()
	if err != nil {
		t.Fatal(err)
	}
	var env envelope
	if err := json.Unmarshal(out, &env); err != nil {
		t.Fatal(err)
	}
	var resp modelRegistrationResponse
	if err := json.Unmarshal(env.Result, &resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Models) == 0 {
		t.Fatal("expected default models when none configured")
	}
}

func TestBaseURLAndAPIKeyFromStorageJSON(t *testing.T) {
	cfg := pluginConfig{Enabled: true, Provider: "zen", BaseURL: "https://default.url"}
	req := executorRequest{
		StorageJSON: []byte(`{"base_url": "https://custom.url/v1", "api_key": "sk-custom-key"}`),
	}
	u := baseURLForRequest(req, cfg)
	if u != "https://custom.url/v1" {
		t.Fatalf("baseURL = %q, want https://custom.url/v1", u)
	}
	k := apiKeyForRequest(req, cfg)
	if k != "sk-custom-key" {
		t.Fatalf("apiKey = %q, want sk-custom-key", k)
	}
}

func TestAuthParseFromFilePayload(t *testing.T) {
	resetConfig(t, pluginConfig{Enabled: true, Provider: "zen"})

	// Scenario 1: User creates a zen auth file with provider="zen"
	payload, _ := json.Marshal(map[string]any{
		"Provider":    "zen",
		"StorageJSON": json.RawMessage(`{"provider":"zen","api_key":"sk-opencode-secret-123","base_url":"https://opencode.ai/zen/v1"}`),
		"FileName":    "zen-test.json",
	})
	out, err := handleMethod("auth.parse", payload)
	if err != nil {
		t.Fatal(err)
	}
	var env envelope
	if err := json.Unmarshal(out, &env); err != nil {
		t.Fatal(err)
	}
	if !env.OK {
		t.Fatalf("auth.parse not ok: %s", out)
	}
	var resp authParseResponse
	if err := json.Unmarshal(env.Result, &resp); err != nil {
		t.Fatal(err)
	}
	if !resp.Handled || resp.Auth.Provider != "zen" || resp.Auth.FileName != "zen-test.json" {
		t.Fatalf("unexpected auth response: %+v", resp)
	}

	// Scenario 2: User creates an auth file where type="zen"
	payload2, _ := json.Marshal(map[string]any{
		"StorageJSON": json.RawMessage(`{"type":"zen","key":"sk-opencode-secret-456"}`),
		"FileName":    "my-zen-key.json",
	})
	out2, err := handleMethod("auth.parse", payload2)
	if err != nil {
		t.Fatal(err)
	}
	var env2 envelope
	json.Unmarshal(out2, &env2)
	var resp2 authParseResponse
	json.Unmarshal(env2.Result, &resp2)
	if !resp2.Handled || resp2.Auth.Provider != "zen" {
		t.Fatalf("unexpected auth response 2: %+v", resp2)
	}

	// Scenario 3: CPA v8 host sends the auth file payload as RawJSON
	payload3, _ := json.Marshal(map[string]any{
		"Provider": "zen",
		"RawJSON":  json.RawMessage(`{"type":"zen","provider":"zen","api_key":"sk-opencode-secret-789","base_url":"https://opencode.ai/zen/v1"}`),
		"FileName": "zen-key.json",
	})
	out3, err := handleMethod("auth.parse", payload3)
	if err != nil {
		t.Fatal(err)
	}
	var env3 envelope
	json.Unmarshal(out3, &env3)
	var resp3 authParseResponse
	json.Unmarshal(env3.Result, &resp3)
	if !resp3.Handled || resp3.Auth.Provider != "zen" {
		t.Fatalf("unexpected auth response 3: %+v", resp3)
	}
	if len(resp3.Auth.StorageJSON) == 0 || !strings.Contains(string(resp3.Auth.StorageJSON), "sk-opencode-secret-789") {
		t.Fatalf("auth storage json should carry the raw key: %s", resp3.Auth.StorageJSON)
	}
}

// TestAuthParseRawJSONVirtualKeys verifies that virtual auths from plugin
// config api-keys are still returned when the host sends no file payload.
func TestAuthParseRawJSONVirtualKeys(t *testing.T) {
	resetConfig(t, pluginConfig{Enabled: true, Provider: "zen", APIKeys: []string{"sk-opencode-raw-1", "sk-opencode-raw-2"}})
	out, err := handleMethod("auth.parse", nil)
	if err != nil {
		t.Fatal(err)
	}
	var env envelope
	json.Unmarshal(out, &env)
	var resp authParseResponse
	json.Unmarshal(env.Result, &resp)
	if !resp.Handled || len(resp.Auths) != 2 {
		t.Fatalf("auths = %d, want 2: %s", len(resp.Auths), out)
	}
}

// =========================================================================
// route resolution
// =========================================================================

func TestRouteForModelExactMatch(t *testing.T) {
	cfg := testConfig()
	route, ok := routeForModel(cfg, "mimo-v2.6-flash-free")
	if !ok || route.Endpoint != "chat" {
		t.Fatalf("route = %+v", route)
	}
}

func TestRouteForModelAliasMatch(t *testing.T) {
	cfg := testConfig()
	route, ok := routeForModel(cfg, "muse-spark-1.3-contributor-free")
	if !ok || route.Endpoint != "responses" {
		t.Fatalf("route = %+v", route)
	}
}

func TestRouteForModelUnknownInference(t *testing.T) {
	cfg := testConfig()
	// Unknown model without "muse" or "responses" defaults to "chat"
	route, ok := routeForModel(cfg, "deepseek-v3")
	if !ok || route.Endpoint != "chat" {
		t.Fatalf("expected inferred chat route, got: %+v, ok=%v", route, ok)
	}

	// Unknown model with "muse" defaults to "responses"
	routeMuse, ok := routeForModel(cfg, "muse-spark-custom")
	if !ok || routeMuse.Endpoint != "responses" {
		t.Fatalf("expected inferred responses route, got: %+v, ok=%v", routeMuse, ok)
	}
}

// =========================================================================
// gate: canonical IDs
// =========================================================================

func TestCanonicalIDChatcmplShape(t *testing.T) {
	id := canonicalID("chatcmpl", "test-source")
	shape := regexp.MustCompile(`^chatcmpl_[0-9a-f]{12}[0-9A-Za-z]{14}$`)
	if !shape.MatchString(id) {
		t.Fatalf("id = %q, want chatcmpl_<12hex><14alnum>", id)
	}
}

func TestCanonicalIDMsgShape(t *testing.T) {
	id := canonicalID("msg", "test-source")
	if !canonicalShapeRe.MatchString(id) || !strings.HasPrefix(id, "msg_") {
		t.Fatalf("id = %q, want msg_<12hex><14alnum>", id)
	}
	if len(id) != 30 {
		t.Fatalf("id length = %d, want 30", len(id))
	}
}

func TestCanonicalIDDeterministic(t *testing.T) {
	a := canonicalID("msg", "source")
	b := canonicalID("msg", "source")
	if a != b {
		t.Fatal("canonicalID is not deterministic")
	}
}

func TestCanonicalIDDistinctSources(t *testing.T) {
	a := canonicalID("msg", "source-a")
	b := canonicalID("msg", "source-b")
	if a == b {
		t.Fatal("distinct sources must produce distinct ids")
	}
}

// TestCanonicalIDSessionPrefixIsNotGateValid documents why sessions must not
// be minted by canonicalID: the hash-derived hex field does not carry the
// 0xFFE low bits the free-tier gate reads.
func TestCanonicalIDSessionPrefixIsNotGateValid(t *testing.T) {
	if isGateSessionID(canonicalID("ses", "real-client")) {
		t.Fatal("hash-derived session id unexpectedly passed the gate check")
	}
}

func TestResolveSessionFromClientHeader(t *testing.T) {
	resetConfig(t, testConfig())
	req := &executorRequest{
		Model:   "mimo-v2.6-flash-free",
		Payload: chatPayload("mimo-v2.6-flash-free", "hello"),
		Headers: map[string][]string{"Session-Id": {"codex-session-1"}},
	}
	session, ok := resolveSession(req)
	if !ok || !canonicalShapeRe.MatchString(session) || !strings.HasPrefix(session, "ses_") {
		t.Fatalf("session = %q, ok=%v", session, ok)
	}
}

func TestResolveSessionPassesGateValidClientValue(t *testing.T) {
	resetConfig(t, testConfig())
	existing := newOfficialSessionID()
	req := &executorRequest{
		Model:   "mimo-v2.6-flash-free",
		Payload: chatPayload("mimo-v2.6-flash-free", "hello"),
		Headers: map[string][]string{targetSessionHeader: {existing}},
	}
	session, ok := resolveSession(req)
	if !ok || session != existing {
		t.Fatalf("session = %q, want %q", session, existing)
	}
}

// TestResolveSessionRemintsUngatedClientValue covers the regression that made
// every free-tier request fail with "403 FreeTierError": a client value that
// only looks canonical (hash-derived hex field) must be replaced by a gate
// valid id, and the replacement must stay stable for that conversation.
func TestResolveSessionRemintsUngatedClientValue(t *testing.T) {
	resetConfig(t, testConfig())
	existing := canonicalID("ses", "real-client")
	if isGateSessionID(existing) {
		t.Fatalf("fixture %q unexpectedly passes the gate check", existing)
	}
	req := &executorRequest{
		Model:   "mimo-v2.6-flash-free",
		Payload: chatPayload("mimo-v2.6-flash-free", "hello"),
		Headers: map[string][]string{targetSessionHeader: {existing}},
	}
	session, ok := resolveSession(req)
	if !ok || session == existing {
		t.Fatalf("session = %q, want a re-minted id", session)
	}
	if !isGateSessionID(session) {
		t.Fatalf("re-minted session %q is not gate valid", session)
	}
	again, _ := resolveSession(req)
	if again != session {
		t.Fatalf("re-minted session id is not stable: %q then %q", session, again)
	}
}

// TestResolveSessionAlwaysGateValid walks every resolution path: the returned
// id must satisfy the gate invariant the upstream enforces.
func TestResolveSessionAlwaysGateValid(t *testing.T) {
	resetConfig(t, testConfig())
	requests := []*executorRequest{
		{
			Model:   "mimo-v2.6-flash-free",
			Payload: chatPayload("mimo-v2.6-flash-free", "hello"),
			Headers: map[string][]string{"Session-Id": {"codex-session-1"}},
		},
		{
			Model:    "mimo-v2.6-flash-free",
			Payload:  chatPayload("mimo-v2.6-flash-free", "hello"),
			Headers:  map[string][]string{},
			Metadata: map[string]any{"canonical_session_id": "claude:probe-1"},
		},
		{
			Model:    "mimo-v2.6-flash-free",
			Payload:  chatPayload("mimo-v2.6-flash-free", "unique-opening"),
			Headers:  map[string][]string{},
			Metadata: map[string]any{},
		},
	}
	for i, req := range requests {
		session, ok := resolveSession(req)
		if !ok {
			t.Fatalf("request %d did not resolve a session", i)
		}
		if !isGateSessionID(session) {
			t.Fatalf("request %d resolved gate-invalid session %q", i, session)
		}
	}
}

func TestResolveSessionFromMetadataFallback(t *testing.T) {
	resetConfig(t, testConfig())
	req := &executorRequest{
		Model:    "mimo-v2.6-flash-free",
		Payload:  chatPayload("mimo-v2.6-flash-free", "hello"),
		Headers:  map[string][]string{},
		Metadata: map[string]any{"canonical_session_id": "claude:probe-1"},
	}
	session, ok := resolveSession(req)
	if !ok || !canonicalShapeRe.MatchString(session) || !strings.HasPrefix(session, "ses_") {
		t.Fatalf("session = %q, ok=%v", session, ok)
	}
}

func TestResolveSessionFromContentFallback(t *testing.T) {
	resetConfig(t, testConfig())
	req := &executorRequest{
		Model:    "mimo-v2.6-flash-free",
		Payload:  chatPayload("mimo-v2.6-flash-free", "unique-opening"),
		Headers:  map[string][]string{},
		Metadata: map[string]any{},
	}
	session, ok := resolveSession(req)
	if !ok || !canonicalShapeRe.MatchString(session) || !strings.HasPrefix(session, "ses_") {
		t.Fatalf("session = %q, ok=%v", session, ok)
	}
	// Same content → same session id (stable).
	session2, _ := resolveSession(req)
	if session != session2 {
		t.Fatal("content-derived session id is not stable")
	}
}

func TestResolveSessionConflictingHeadersFailsClosed(t *testing.T) {
	resetConfig(t, testConfig())
	req := &executorRequest{
		Model:   "mimo-v2.6-flash-free",
		Payload: chatPayload("mimo-v2.6-flash-free", "hello"),
		Headers: map[string][]string{"Session-Id": {"a", "b"}},
	}
	_, ok := resolveSession(req)
	if ok {
		t.Fatal("conflicting session headers must fail closed")
	}
}

func TestResolveSessionInvalidCharsFailsClosed(t *testing.T) {
	resetConfig(t, testConfig())
	req := &executorRequest{
		Model:   "mimo-v2.6-flash-free",
		Payload: chatPayload("mimo-v2.6-flash-free", "hello"),
		Headers: map[string][]string{"Session-Id": {"bad\x01value"}},
	}
	_, ok := resolveSession(req)
	if ok {
		t.Fatal("invalid session header must fail closed")
	}
}

// =========================================================================
// gate: headers
// =========================================================================

func TestGateHeadersIncludesRequiredFields(t *testing.T) {
	resetConfig(t, testConfig())
	req := executorRequest{
		Model:    "mimo-v2.6-flash-free",
		Payload:  chatPayload("mimo-v2.6-flash-free", "hello"),
		Headers:  map[string][]string{"Session-Id": {"s1"}},
		Metadata: map[string]any{"request_id": "exec-1"},
	}
	h := gateHeaders(req)
	if h["User-Agent"][0] != defaultUserAgent {
		t.Fatalf("User-Agent = %q", h["User-Agent"][0])
	}
	if h["Accept"][0] != "*/*" {
		t.Fatalf("Accept = %q", h["Accept"][0])
	}
	if h[targetClientHeader][0] != "cli" {
		t.Fatalf("X-Opencode-Client = %q", h[targetClientHeader][0])
	}
	if h[targetProjectHeader][0] != "global" {
		t.Fatalf("X-Opencode-Project = %q", h[targetProjectHeader][0])
	}
	if _, ok := h[targetSessionHeader]; !ok {
		t.Fatal("X-Opencode-Session header missing")
	}
	if _, ok := h[targetRequestHeader]; !ok {
		t.Fatal("X-Opencode-Request header missing")
	}
}

func TestGateHeadersPreservesClientIdentity(t *testing.T) {
	resetConfig(t, testConfig())
	req := executorRequest{
		Model:   "mimo-v2.6-flash-free",
		Payload: chatPayload("mimo-v2.6-flash-free", "hello"),
		Headers: map[string][]string{
			"Session-Id":        {"s1"},
			"X-Opencode-Client": {"opencode"},
		},
	}
	h := gateHeaders(req)
	if h[targetClientHeader][0] != "opencode" {
		t.Fatalf("X-Opencode-Client = %q, want opencode", h[targetClientHeader][0])
	}
}

// =========================================================================
// gate: tools
// =========================================================================

func TestEnsureGateToolsChatDialect(t *testing.T) {
	root, _ := json.Marshal(map[string]any{
		"messages": []any{map[string]any{"role": "user", "content": "hi"}},
	})
	var body map[string]any
	json.Unmarshal(root, &body)
	ensureGateTools(body, "chat")
	tools, _ := body["tools"].([]any)
	names := map[string]bool{}
	for _, item := range tools {
		entry := item.(map[string]any)
		fn := entry["function"].(map[string]any)
		names[fn["name"].(string)] = true
	}
	if !names["bash"] || !names["read"] {
		t.Fatalf("gate tools missing: %v", names)
	}
}

func TestEnsureGateToolsResponsesDialect(t *testing.T) {
	root, _ := json.Marshal(map[string]any{
		"input": "hello",
	})
	var body map[string]any
	json.Unmarshal(root, &body)
	ensureGateTools(body, "responses")
	tools, _ := body["tools"].([]any)
	names := map[string]bool{}
	for _, item := range tools {
		entry := item.(map[string]any)
		names[entry["name"].(string)] = true
	}
	if !names["bash"] || !names["read"] {
		t.Fatalf("gate tools missing: %v", names)
	}
}

func TestEnsureGateToolsIdempotent(t *testing.T) {
	root, _ := json.Marshal(map[string]any{
		"messages": []any{map[string]any{"role": "user", "content": "hi"}},
		"tools": []any{map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        "bash",
				"parameters":  map[string]any{"type": "object"},
				"description": "existing",
			},
		}},
	})
	var body map[string]any
	json.Unmarshal(root, &body)
	ensureGateTools(body, "chat")
	tools := body["tools"].([]any)
	if len(tools) != len(officialGateTools) {
		t.Fatalf("tools count = %d, want %d (existing bash + missing official tools)", len(tools), len(officialGateTools))
	}
	// Second call does not duplicate.
	ensureGateTools(body, "chat")
	tools = body["tools"].([]any)
	if len(tools) != len(officialGateTools) {
		t.Fatalf("tools count = %d after second call, want %d", len(tools), len(officialGateTools))
	}
	// Tools are sorted by function name, official-client style.
	for i := 1; i < len(tools); i++ {
		prev, _ := tools[i-1].(map[string]any)
		cur, _ := tools[i].(map[string]any)
		if toolEntryName(prev, "chat") > toolEntryName(cur, "chat") {
			t.Fatal("tools are not sorted by function name")
		}
	}
}

// =========================================================================
// gate: stream forced true
// =========================================================================

func TestPrepareUpstreamBodyForcesStream(t *testing.T) {
	resetConfig(t, testConfig())
	req := executorRequest{
		Model:   "mimo-v2.6-flash-free",
		Payload: chatPayload("mimo-v2.6-flash-free", "hello"),
	}
	route, _ := routeForModel(testConfig(), "mimo-v2.6-flash-free")
	body, err := prepareUpstreamBody(req, route)
	if err != nil {
		t.Fatal(err)
	}
	var out map[string]any
	json.Unmarshal(body, &out)
	if v, ok := out["stream"]; !ok || v != true {
		t.Fatalf("stream = %v", v)
	}
}

func TestPrepareUpstreamBodyInjectsTools(t *testing.T) {
	resetConfig(t, testConfig())
	body, _ := json.Marshal(map[string]any{
		"model":    "mimo-v2.6-flash-free",
		"messages": []any{map[string]any{"role": "user", "content": "hi"}},
		"stream":   true,
	})
	req := executorRequest{Model: "mimo-v2.6-flash-free", Payload: body}
	route, _ := routeForModel(testConfig(), "mimo-v2.6-flash-free")
	out, err := prepareUpstreamBody(req, route)
	if err != nil {
		t.Fatal(err)
	}
	var root map[string]any
	json.Unmarshal(out, &root)
	tools, _ := root["tools"].([]any)
	if len(tools) < 2 {
		t.Fatalf("tools = %v, want >= 2", tools)
	}
}

func TestPrepareUpstreamBodyConvertsResponsesToChat(t *testing.T) {
	resetConfig(t, testConfig())
	body, _ := json.Marshal(map[string]any{
		"model":        "mimo-v2.6-flash-free",
		"instructions": "You are helpful.",
		"input": []any{map[string]any{
			"role": "user",
			"content": []any{map[string]any{
				"type": "input_text",
				"text": "hello",
			}},
		}},
		"tools": []any{map[string]any{
			"type":        "function",
			"name":        "bash",
			"description": "Runs a shell.",
			"parameters":  map[string]any{"type": "object"},
		}},
	})
	req := executorRequest{Model: "mimo-v2.6-flash-free", Payload: body}
	route, _ := routeForModel(testConfig(), "mimo-v2.6-flash-free")
	out, err := prepareUpstreamBody(req, route)
	if err != nil {
		t.Fatal(err)
	}
	var root map[string]any
	json.Unmarshal(out, &root)
	if _, exists := root["input"]; exists {
		t.Fatal("input was not converted to messages")
	}
	messages, _ := root["messages"].([]any)
	if len(messages) != 2 {
		t.Fatalf("messages = %v", messages)
	}
	first := messages[0].(map[string]any)
	if first["role"] != "system" || first["content"] != "You are helpful." {
		t.Fatalf("system message = %v", first)
	}
	second := messages[1].(map[string]any)
	parts, _ := second["content"].([]any)
	part, _ := parts[0].(map[string]any)
	if part["type"] != "text" || part["text"] != "hello" {
		t.Fatalf("converted content = %v", second["content"])
	}
	tools, _ := root["tools"].([]any)
	if len(tools) != len(officialGateTools) {
		t.Fatalf("tools = %v, want the full official tool set", tools)
	}
	bash, _ := tools[0].(map[string]any)
	fn, _ := bash["function"].(map[string]any)
	if fn["name"] != "bash" {
		t.Fatalf("bash tool was not converted to chat format: %v", bash)
	}
}

func TestPrepareUpstreamBodyConvertsChatToResponses(t *testing.T) {
	resetConfig(t, testConfig())
	body, _ := json.Marshal(map[string]any{
		"model": "muse-spark-1.3-contributor-free",
		"messages": []any{map[string]any{
			"role": "user",
			"content": []any{map[string]any{
				"type": "text",
				"text": "hello",
			}},
		}},
		"tools": []any{map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        "bash",
				"description": "Runs a shell.",
				"parameters":  map[string]any{"type": "object"},
			},
		}},
	})
	req := executorRequest{Model: "muse-spark-1.3-contributor-free", Payload: body}
	route, _ := routeForModel(testConfig(), "muse-spark-1.3-contributor-free")
	out, err := prepareUpstreamBody(req, route)
	if err != nil {
		t.Fatal(err)
	}
	var root map[string]any
	json.Unmarshal(out, &root)
	if _, exists := root["messages"]; exists {
		t.Fatal("messages were not converted to input")
	}
	input, _ := root["input"].([]any)
	message, _ := input[0].(map[string]any)
	parts, _ := message["content"].([]any)
	part, _ := parts[0].(map[string]any)
	if part["type"] != "input_text" || part["text"] != "hello" {
		t.Fatalf("converted content = %v", message["content"])
	}
	tools, _ := root["tools"].([]any)
	if len(tools) != len(officialGateTools) {
		t.Fatalf("tools = %v, want the full official tool set", tools)
	}
	bash, _ := tools[0].(map[string]any)
	if bash["name"] != "bash" {
		t.Fatalf("bash tool was not converted to responses format: %v", bash)
	}
}

// =========================================================================
// endpoint path
// =========================================================================

func TestEndpointPathChat(t *testing.T) {
	m := modelRoute{Endpoint: "chat"}
	if m.EndpointPath() != "/chat/completions" {
		t.Fatalf("EndpointPath = %q", m.EndpointPath())
	}
}

func TestEndpointPathResponses(t *testing.T) {
	m := modelRoute{Endpoint: "responses"}
	if m.EndpointPath() != "/responses" {
		t.Fatalf("EndpointPath = %q", m.EndpointPath())
	}
}

// =========================================================================
// header helpers
// =========================================================================

func TestHeaderValueSimple(t *testing.T) {
	h := map[string][]string{"Session-Id": {"abc"}}
	v, ok := headerValue(h, "Session-Id")
	if !ok || v != "abc" {
		t.Fatalf("headerValue = %q, %v", v, ok)
	}
}

func TestHeaderValueCaseInsensitive(t *testing.T) {
	h := map[string][]string{"session-id": {"abc"}}
	v, ok := headerValue(h, "Session-Id")
	if !ok || v != "abc" {
		t.Fatalf("headerValue = %q, %v", v, ok)
	}
}

func TestHeaderValueMissing(t *testing.T) {
	h := map[string][]string{}
	_, ok := headerValue(h, "Nonexistent")
	if ok {
		t.Fatal("expected missing header to return false")
	}
}

func TestHeaderValueConflicting(t *testing.T) {
	h := map[string][]string{"Session-Id": {"a", "b"}}
	v, ok := headerValue(h, "Session-Id")
	if !ok || v != "" {
		t.Fatalf("headerValue = %q, %v, want empty/present for conflict", v, ok)
	}
}

func TestNormalizeSessionIDValid(t *testing.T) {
	v, ok := normalizeSessionID("   abc-123   ")
	if !ok || v != "abc-123" {
		t.Fatalf("normalize = %q, %v", v, ok)
	}
}

func TestNormalizeSessionIDEmpty(t *testing.T) {
	_, ok := normalizeSessionID("   ")
	if ok {
		t.Fatal("empty session id must return false")
	}
}

func TestNormalizeSessionIDControlChars(t *testing.T) {
	_, ok := normalizeSessionID("abc\x00def")
	if ok {
		t.Fatal("session id with control chars must return false")
	}
}

// =========================================================================
// SSE folding
// =========================================================================

func TestFoldChatSSEBasic(t *testing.T) {
	sse := []byte(
		"data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"mimo\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n" +
			"data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"mimo\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"},\"finish_reason\":null}]}\n\n" +
			"data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"mimo\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}\n\n" +
			"data: [DONE]\n\n",
	)
	body, err := foldChatSSE(sse)
	if err != nil {
		t.Fatal(err)
	}
	var out map[string]any
	json.Unmarshal(body, &out)
	if out["object"] != "chat.completion" {
		t.Fatalf("object = %q", out["object"])
	}
	choices := out["choices"].([]any)
	msg := choices[0].(map[string]any)["message"].(map[string]any)
	if msg["content"] != "Hello world" {
		t.Fatalf("content = %q", msg["content"])
	}
}

func TestFoldChatSSEToolCalls(t *testing.T) {
	sse := []byte(
		"data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"mimo\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_0\",\"type\":\"function\",\"function\":{\"name\":\"bash\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n" +
			"data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"mimo\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"ls\"}}]},\"finish_reason\":null}]}\n\n" +
			"data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"mimo\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n" +
			"data: [DONE]\n\n",
	)
	body, err := foldChatSSE(sse)
	if err != nil {
		t.Fatal(err)
	}
	var out map[string]any
	json.Unmarshal(body, &out)
	choices := out["choices"].([]any)
	msg := choices[0].(map[string]any)["message"].(map[string]any)
	toolCalls := msg["tool_calls"].([]any)
	if len(toolCalls) != 1 {
		t.Fatalf("tool_calls = %v", toolCalls)
	}
	tc := toolCalls[0].(map[string]any)
	fn := tc["function"].(map[string]any)
	if fn["name"] != "bash" || fn["arguments"] != "ls" {
		t.Fatalf("tool call = %+v", tc)
	}
}

func TestFoldResponsesSSEBasic(t *testing.T) {
	sse := []byte(
		"data: {\"type\":\"response.created\",\"id\":\"resp_1\",\"model\":\"muse\",\"status\":\"in_progress\"}\n\n" +
			"data: {\"type\":\"response.output_item.added\",\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"pong\"}],\"status\":\"completed\"}}\n\n" +
			"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"status\":\"completed\",\"model\":\"muse\",\"output\":[{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"pong\"}],\"status\":\"completed\"}],\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\n" +
			"data: [DONE]\n\n",
	)
	body, err := foldResponsesSSE(sse)
	if err != nil {
		t.Fatal(err)
	}
	var out map[string]any
	json.Unmarshal(body, &out)
	if out["object"] != "response" || out["status"] != "completed" {
		t.Fatalf("response = %v", out)
	}
}

func TestSSEFrames(t *testing.T) {
	sse := []byte("data: {\"type\":\"response.created\"}\n\ndata: [DONE]\n\n")
	frames := sseFrames(sse)
	if len(frames) != 2 {
		t.Fatalf("frames = %v", frames)
	}
	if frames[1] != "[DONE]" {
		t.Fatalf("last frame = %q", frames[1])
	}
}

// =========================================================================
// dispatch & error handling
// =========================================================================

func TestUnknownMethod(t *testing.T) {
	raw, err := handleMethod("bogus.method", nil)
	if err != nil {
		t.Fatal(err)
	}
	var env envelope
	if err := json.Unmarshal(raw, &env); err != nil {
		t.Fatal(err)
	}
	if env.OK || env.Error.Code != "unknown_method" {
		t.Fatalf("envelope = %+v", env)
	}
}

func TestProcessPluginCallInvalidJSON(t *testing.T) {
	raw, status := processPluginCall("executor.execute", []byte(`{invalid`))
	if status != 1 {
		t.Fatalf("status = %d, want 1", status)
	}
	var env envelope
	if err := json.Unmarshal(raw, &env); err != nil {
		t.Fatal(err)
	}
	if env.OK || env.Error.Code != "plugin_error" {
		t.Fatalf("envelope = %+v", env)
	}
}

func TestExecuteUnknownModel(t *testing.T) {
	resetConfig(t, testConfig())
	body, _ := json.Marshal(executorRequest{Model: "", Payload: chatPayload("nonexistent-model", "hi")})
	_, err := execute(body, false)
	if err == nil || !strings.Contains(err.Error(), "no model") {
		t.Fatalf("error = %v", err)
	}
}

func TestExecuteStreamRequiresStreamID(t *testing.T) {
	resetConfig(t, testConfig())
	cfg := loadedConfig()
	route, _ := routeForModel(cfg, "mimo-v2.6-flash-free")
	body, _ := json.Marshal(executorRequest{Model: "mimo-v2.6-flash-free", Payload: chatPayload("mimo-v2.6-flash-free", "hi")})
	_, err := execute(body, true)
	if err == nil || !strings.Contains(err.Error(), "stream_id") {
		t.Fatalf("error = %v", err)
	}
	_ = route
}

func TestNormalizeSSEFrameNonDataLines(t *testing.T) {
	// Keep-alive comments are dropped.
	if res := normalizeSSEFrame(false, []byte(": keep-alive\n")); len(res) != 0 {
		t.Fatalf("expected keep-alive to be dropped, got %q", string(res))
	}
	// Event lines carry no data payload in raw mode.
	if res := normalizeSSEFrame(true, []byte("event: response.completed\n")); len(res) != 0 {
		t.Fatalf("raw event line = %q", string(res))
	}
	// A bare JSON line is treated as an unframed data payload.
	bare := []byte(`{"choices":[]}`)
	if res := normalizeSSEFrame(true, bare); string(res) != `{"choices":[]}` {
		t.Fatalf("raw bare JSON = %q", string(res))
	}
	if res := normalizeSSEFrame(false, bare); string(res) != "data: {\"choices\":[]}\n\n" {
		t.Fatalf("framed bare JSON = %q", string(res))
	}
}

func TestSanitizeChatMessages(t *testing.T) {
	root := map[string]any{
		"messages": []any{
			// roleless artifact (translated reasoning item): dropped
			map[string]any{"role": nil},
			// text part without a usable "text" field: dropped, content becomes ""
			map[string]any{"role": "assistant", "content": []any{map[string]any{"type": "text"}}},
			// null content normalized to ""
			map[string]any{"role": "assistant", "content": nil},
			// valid parts are kept, output_text normalized, annotations dropped
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "output_text", "text": "hello", "annotations": []any{}},
			}},
			// string content untouched
			map[string]any{"role": "user", "content": "hi"},
		},
	}
	sanitizeChatMessages(root)
	messages, _ := root["messages"].([]any)
	if len(messages) != 4 {
		t.Fatalf("messages = %v", messages)
	}
	first := messages[0].(map[string]any)
	if first["content"] != "" {
		t.Fatalf("textless part should leave empty content, got %v", first["content"])
	}
	second := messages[1].(map[string]any)
	if second["content"] != "" {
		t.Fatalf("null content should become empty string, got %v", second["content"])
	}
	third := messages[2].(map[string]any)
	parts, _ := third["content"].([]any)
	part, _ := parts[0].(map[string]any)
	if part["type"] != "text" || part["text"] != "hello" {
		t.Fatalf("normalized part = %v", part)
	}
	if _, exists := part["annotations"]; exists {
		t.Fatal("annotations should be dropped")
	}
	if messages[3].(map[string]any)["content"] != "hi" {
		t.Fatalf("string content should be untouched, got %v", messages[3])
	}
}

func TestPrepareUpstreamBodyDropsReasoningItems(t *testing.T) {
	resetConfig(t, testConfig())
	body, _ := json.Marshal(map[string]any{
		"model": "mimo-v2.6-flash-free",
		"input": []any{
			map[string]any{"role": "user", "content": []any{map[string]any{"type": "input_text", "text": "hello"}}},
			map[string]any{
				"id":                "rs_gen-test",
				"type":              "reasoning",
				"encrypted_content": "",
				"summary":           []any{map[string]any{"type": "summary_text", "text": "thinking"}},
			},
			map[string]any{"type": "message", "role": "assistant", "content": []any{map[string]any{"type": "output_text", "text": "hi there", "annotations": []any{}}}},
			map[string]any{"role": "user", "content": []any{map[string]any{"type": "input_text", "text": "reply one word"}}},
		},
	})
	req := executorRequest{Model: "mimo-v2.6-flash-free", Payload: body}
	route, _ := routeForModel(testConfig(), "mimo-v2.6-flash-free")
	out, err := prepareUpstreamBody(req, route)
	if err != nil {
		t.Fatal(err)
	}
	var root map[string]any
	json.Unmarshal(out, &root)
	messages, _ := root["messages"].([]any)
	// user + assistant + user: no roleless reasoning artifact may survive
	if len(messages) != 3 {
		t.Fatalf("messages = %v", messages)
	}
	for _, m := range messages {
		msg, _ := m.(map[string]any)
		if role, _ := msg["role"].(string); strings.TrimSpace(role) == "" {
			t.Fatalf("roleless message survived: %v", msg)
		}
		content, _ := msg["content"].([]any)
		for _, p := range content {
			part, _ := p.(map[string]any)
			if part["type"] == "text" {
				if text, _ := part["text"].(string); strings.TrimSpace(text) == "" {
					t.Fatalf("textless part survived: %v", part)
				}
			}
		}
	}
}

func TestAuthParseExtractsModelsList(t *testing.T) {
	resetConfig(t, pluginConfig{Enabled: true, Provider: "zen"})
	payload, _ := json.Marshal(map[string]any{
		"Provider":    "zen",
		"StorageJSON": json.RawMessage(`{"provider":"zen","api_key":"sk-test","models":[{"name":"custom-model"}]}`),
		"FileName":    "zen-custom.json",
	})
	out, err := handleMethod("auth.parse", payload)
	if err != nil {
		t.Fatal(err)
	}
	var env envelope
	json.Unmarshal(out, &env)
	var resp authParseResponse
	json.Unmarshal(env.Result, &resp)
	if !resp.Handled || len(resp.Auth.Models) != 1 {
		t.Fatalf("expected models in authData, got: %+v", resp)
	}
}

func TestLineBufferReassembly(t *testing.T) {
	// Simulating broken incoming chunks:
	// Chunk 1: "data: {\"id\":\"123\","
	// Chunk 2: "\"content\":\"hello\"}\n: keep-alive\ndata: [DONE]\n"
	part1 := []byte("data: {\"id\":\"123\",")
	part2 := []byte("\"content\":\"hello\"}\n: keep-alive\ndata: [DONE]\n")

	var emitted [][]byte
	r := &sseReassembler{
		rawData: false,
		emit:    func(frame []byte) error { emitted = append(emitted, frame); return nil },
	}

	if err := r.write(part1); err != nil {
		t.Fatal(err)
	}
	if len(emitted) != 0 {
		t.Fatalf("expected 0 emitted lines while line is incomplete, got %d", len(emitted))
	}

	if err := r.write(part2); err != nil {
		t.Fatal(err)
	}
	// Stream finished: flush the pending event (no trailing blank
	// line in this input, mirroring a done marker arriving with the last
	// payload bytes).
	if err := r.flush(); err != nil {
		t.Fatal(err)
	}
	// The reassembled JSON is the only emitted frame: the [DONE] terminator
	// is dropped here because the host appends its own done tail after the
	// plugin stream closes, and the keep-alive comment is ignored.
	if len(emitted) != 1 {
		t.Fatalf("expected 1 emitted frame, got %d: %v", len(emitted), emitted)
	}

	if string(emitted[0]) != "data: {\"id\":\"123\",\"content\":\"hello\"}\n\n" {
		t.Fatalf("unexpected line 0: %q", string(emitted[0]))
	}
}

func TestSSEReassemblerJoinsMultiLineData(t *testing.T) {
	var emitted [][]byte
	r := &sseReassembler{
		rawData: false,
		emit:    func(frame []byte) error { emitted = append(emitted, frame); return nil },
	}
	// One event whose JSON is split across two data lines (SSE-legal):
	// the payloads must be joined with "\n" into a single frame.
	if err := r.write([]byte("data: {\"id\":\"1\",\n")); err != nil {
		t.Fatal(err)
	}
	if len(emitted) != 0 {
		t.Fatalf("expected 0 frames while event is incomplete, got %d", len(emitted))
	}
	if err := r.write([]byte("data: \"content\":\"hi\"}\n\n")); err != nil {
		t.Fatal(err)
	}
	if len(emitted) != 1 {
		t.Fatalf("expected 1 joined frame, got %d: %v", len(emitted), emitted)
	}
	want := "data: {\"id\":\"1\",\n\"content\":\"hi\"}\n\n"
	if string(emitted[0]) != want {
		t.Fatalf("unexpected frame: %q, want %q", string(emitted[0]), want)
	}
}

func TestSSEReassemblerRawModeEmitsBarePayloads(t *testing.T) {
	var emitted [][]byte
	r := &sseReassembler{
		rawData: true,
		emit:    func(frame []byte) error { emitted = append(emitted, frame); return nil },
	}
	input := "data: {\"choices\":[]}\n\ndata: [DONE]\n\n"
	if err := r.write([]byte(input)); err != nil {
		t.Fatal(err)
	}
	if err := r.flush(); err != nil {
		t.Fatal(err)
	}
	// Only the JSON payload is emitted; [DONE] is dropped because the host
	// appends its own done tail after the plugin stream closes.
	if len(emitted) != 1 {
		t.Fatalf("expected 1 frame, got %d: %v", len(emitted), emitted)
	}
	if string(emitted[0]) != "{\"choices\":[]}" {
		t.Fatalf("unexpected frame 0: %q", string(emitted[0]))
	}
}

func TestSSEReassemblerFlushKeepsTailWithoutTrailingNewline(t *testing.T) {
	// Simulates the host bridge delivering the final body bytes together
	// with the done marker: the stream ends without a trailing blank line
	// and the last event must still reach the client.
	var emitted [][]byte
	r := &sseReassembler{
		rawData: true,
		emit:    func(frame []byte) error { emitted = append(emitted, frame); return nil },
	}
	if err := r.write([]byte("data: {\"a\":1}\n\ndata: {\"b\":2}")); err != nil {
		t.Fatal(err)
	}
	if len(emitted) != 1 {
		t.Fatalf("expected 1 frame before flush, got %d", len(emitted))
	}
	if err := r.flush(); err != nil {
		t.Fatal(err)
	}
	if len(emitted) != 2 {
		t.Fatalf("expected 2 frames after flush, got %d: %v", len(emitted), emitted)
	}
	if string(emitted[1]) != "{\"b\":2}" {
		t.Fatalf("unexpected tail frame: %q", string(emitted[1]))
	}
}

func TestEmitStreamFrameDropsDoneTerminator(t *testing.T) {
	// The host appends its own done tail after the plugin stream closes;
	// forwarding the upstream terminator would duplicate it.
	done := []string{"[DONE]", "data: [DONE]", "data: [DONE]\n\n", " [DONE] \t"}
	for _, s := range done {
		if !isDoneTerminator(s) {
			t.Fatalf("%q should be recognized as a done terminator", s)
		}
	}
	notDone := []string{"data: {}", "data: {\"a\":1}", "", "[DONEX]"}
	for _, s := range notDone {
		if isDoneTerminator(s) {
			t.Fatalf("%q should NOT be recognized as a done terminator", s)
		}
	}
}

func TestSSEReassemblerCRLFAndKeepAlive(t *testing.T) {
	var emitted [][]byte
	r := &sseReassembler{
		rawData: false,
		emit:    func(frame []byte) error { emitted = append(emitted, frame); return nil },
	}
	input := "data: {\"x\":1}\r\n\r\n: keep-alive\r\n\r\ndata: [DONE]\r\n\r\n"
	if err := r.write([]byte(input)); err != nil {
		t.Fatal(err)
	}
	if err := r.flush(); err != nil {
		t.Fatal(err)
	}
	// [DONE] is dropped (host appends its own tail), so only the JSON frame.
	if len(emitted) != 1 {
		t.Fatalf("expected 1 frame, got %d: %v", len(emitted), emitted)
	}
	if string(emitted[0]) != "data: {\"x\":1}\n\n" {
		t.Fatalf("unexpected frame 0: %q", string(emitted[0]))
	}
}

func TestResponsesChatStreamConverter(t *testing.T) {
	c := newResponsesChatStreamConverter("muse-spark-1.3-contributor-free")
	var out [][]byte
	feed := func(payload string) {
		outs, err := c.convert([]byte(payload))
		if err != nil {
			t.Fatalf("convert(%s): %v", payload, err)
		}
		out = append(out, outs...)
	}
	feed(`{"type":"response.created","response":{"id":"resp_abc","created_at":1700000000}}`)
	feed(`{"type":"response.in_progress","response":{"id":"resp_abc"}}`)
	feed(`{"type":"response.output_item.added","output_index":0,"item":{"id":"rs_1","type":"reasoning"}}`)
	feed(`{"type":"response.output_item.done","output_index":0,"item":{"id":"rs_1","type":"reasoning"}}`)
	feed(`{"type":"response.output_item.added","output_index":1,"item":{"id":"msg_1","type":"message"}}`)
	feed(`{"type":"response.output_text.delta","item_id":"msg_1","delta":"1\n2\n"}`)
	feed(`{"type":"response.output_text.delta","item_id":"msg_1","delta":"3"}`)
	feed(`{"type":"response.completed","response":{"id":"resp_abc","status":"completed","usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}`)

	if len(out) != 4 {
		t.Fatalf("expected 4 chunks, got %d: %v", len(out), out)
	}
	var chunks []map[string]any
	for _, raw := range out {
		var m map[string]any
		if err := json.Unmarshal(raw, &m); err != nil {
			t.Fatalf("chunk not JSON: %v", err)
		}
		chunks = append(chunks, m)
	}
	if chunks[0]["id"] != "resp_abc" || chunks[0]["object"] != "chat.completion.chunk" {
		t.Fatalf("unexpected chunk 0: %v", chunks[0])
	}
	role := chunks[0]["choices"].([]any)[0].(map[string]any)["delta"].(map[string]any)
	if role["role"] != "assistant" {
		t.Fatalf("chunk 0 delta should carry assistant role, got %v", role)
	}
	if content := chunks[1]["choices"].([]any)[0].(map[string]any)["delta"].(map[string]any)["content"]; content != "1\n2\n" {
		t.Fatalf("chunk 1 content: %v", content)
	}
	if content := chunks[2]["choices"].([]any)[0].(map[string]any)["delta"].(map[string]any)["content"]; content != "3" {
		t.Fatalf("chunk 2 content: %v", content)
	}
	last := chunks[3]["choices"].([]any)[0].(map[string]any)
	if last["finish_reason"] != "stop" {
		t.Fatalf("final finish_reason: %v", last["finish_reason"])
	}
	usage := chunks[3]["usage"].(map[string]any)
	if usage["prompt_tokens"] != float64(10) || usage["completion_tokens"] != float64(5) || usage["total_tokens"] != float64(15) {
		t.Fatalf("final usage: %v", usage)
	}
}

func TestResponsesChatStreamConverterItemDoneFallback(t *testing.T) {
	c := newResponsesChatStreamStreamConverterFallbackHelper(t)
	// No deltas streamed: the completed message item must be emitted once.
	outs, err := c.convert([]byte(`{"type":"response.output_item.done","item":{"id":"msg_9","type":"message","content":[{"type":"output_text","text":"hello world"}]}}`))
	if err != nil {
		t.Fatal(err)
	}
	if len(outs) != 1 {
		t.Fatalf("expected 1 fallback chunk, got %d", len(outs))
	}
	var m map[string]any
	if err := json.Unmarshal(outs[0], &m); err != nil {
		t.Fatal(err)
	}
	delta := m["choices"].([]any)[0].(map[string]any)["delta"].(map[string]any)
	if delta["content"] != "hello world" {
		t.Fatalf("fallback content: %v", delta)
	}
	// The same item done again must not re-emit (delta already covered).
	outs, err = c.convert([]byte(`{"type":"response.output_text.delta","item_id":"msg_9","delta":"x"}`))
	if err != nil {
		t.Fatal(err)
	}
	if len(outs) != 1 {
		t.Fatalf("delta chunk expected, got %d", len(outs))
	}
	outs, err = c.convert([]byte(`{"type":"response.output_item.done","item":{"id":"msg_9","type":"message","content":[{"type":"output_text","text":"hello world"}]}}`))
	if err != nil {
		t.Fatal(err)
	}
	if len(outs) != 0 {
		t.Fatalf("item done after deltas should not re-emit, got %d", len(outs))
	}
}

func newResponsesChatStreamStreamConverterFallbackHelper(t *testing.T) *responsesChatStreamConverter {
	t.Helper()
	return newResponsesChatStreamConverter("muse-spark")
}

func TestResponsesChatStreamConverterFailed(t *testing.T) {
	c := newResponsesChatStreamConverter("muse-spark")
	_, err := c.convert([]byte(`{"type":"response.failed","response":{"status":"failed","error":{"message":"rate limited"}}}`))
	if err == nil || !strings.Contains(err.Error(), "rate limited") {
		t.Fatalf("expected failure error, got %v", err)
	}
}

func TestResponsesChatStreamConverterNonJSONDropped(t *testing.T) {
	c := newResponsesChatStreamConverter("muse-spark")
	outs, err := c.convert([]byte(": keep-alive"))
	if err != nil || len(outs) != 0 {
		t.Fatalf("non-JSON payload should be dropped, got %v %v", outs, err)
	}
}

func TestResponsesCompletionToChat(t *testing.T) {
	folded := `{
		"id":"resp_abc","object":"response","status":"completed","model":"muse-spark-1.3-contributor-free","created_at":1700000000,
		"output":[
			{"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"thinking"}]},
			{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"天空是蓝色的。"}]}
		],
		"usage":{"input_tokens":12,"output_tokens":7,"total_tokens":19}
	}`
	out, err := responsesCompletionToChat([]byte(folded), "fallback-model")
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	if m["object"] != "chat.completion" || m["id"] != "resp_abc" || m["model"] != "muse-spark-1.3-contributor-free" {
		t.Fatalf("header fields wrong: %v", m)
	}
	choice := m["choices"].([]any)[0].(map[string]any)
	if choice["finish_reason"] != "stop" {
		t.Fatalf("finish_reason: %v", choice["finish_reason"])
	}
	msg := choice["message"].(map[string]any)
	if msg["role"] != "assistant" || msg["content"] != "天空是蓝色的。" {
		t.Fatalf("message wrong: %v", msg)
	}
	if msg["reasoning_content"] != "thinking" {
		t.Fatalf("reasoning_content: %v", msg["reasoning_content"])
	}
	usage := m["usage"].(map[string]any)
	if usage["prompt_tokens"] != float64(12) || usage["completion_tokens"] != float64(7) || usage["total_tokens"] != float64(19) {
		t.Fatalf("usage: %v", usage)
	}
}

func TestResponsesCompletionToChatFailed(t *testing.T) {
	_, err := responsesCompletionToChat([]byte(`{"id":"r","status":"failed","error":{"message":"boom"}}`), "m")
	if err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("expected failure error, got %v", err)
	}
}

func TestStripSSEDataFraming(t *testing.T) {
	cases := map[string]string{
		"data: {\"a\":1}\n\n": "{\"a\":1}",
		"{\"a\":1}":           "{\"a\":1}",
		"data: [DONE]":        "[DONE]",
	}
	for in, want := range cases {
		if got := stripSSEDataFraming([]byte(in)); got != want {
			t.Fatalf("stripSSEDataFraming(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestResponsesToolsToChatStripsNullFields(t *testing.T) {
	tools := []any{
		map[string]any{"type": "function", "name": "read", "description": "Read a file",
			"parameters": map[string]any{"type": "object"}, "strict": nil},
		map[string]any{"type": "function", "function": map[string]any{
			"name": "write", "strict": nil, "description": "d"}},
	}
	out := responsesToolsToChat(tools)
	for _, item := range out {
		tool := item.(map[string]any)
		fn, _ := tool["function"].(map[string]any)
		if fn == nil {
			t.Fatalf("missing function wrapper: %v", tool)
		}
		if _, ok := fn["strict"]; ok {
			t.Fatalf("strict survived conversion: %v", fn)
		}
	}
}

func TestFetchZenModelsFiltersFreeTier(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models" {
			http.NotFound(w, r)
			return
		}
		if got := r.Header.Get("Authorization"); got != "Bearer k1" {
			t.Errorf("auth header = %q", got)
		}
		fmt.Fprint(w, `{"object":"list","data":[{"id":"mimo-v2.6-flash-free"},{"id":"claude-opus-5"},{"id":"muse-spark-1.3-contributor-free"}]}`)
	}))
	defer srv.Close()
	ids, err := fetchZenModels(srv.URL, "k1")
	if err != nil {
		t.Fatalf("fetchZenModels: %v", err)
	}
	if len(ids) != 2 || ids[0] != "mimo-v2.6-flash-free" || ids[1] != "muse-spark-1.3-contributor-free" {
		t.Fatalf("ids = %v", ids)
	}
}

func TestModelRegistrationMergesDiscoveredModels(t *testing.T) {
	discoveredModels.mu.Lock()
	discoveredModels.ids = []string{"mimo-v2.6-flash-free", "jev-1.13-free"}
	discoveredModels.fetchedAt = time.Now()
	discoveredModels.mu.Unlock()
	defer func() {
		discoveredModels.mu.Lock()
		discoveredModels.ids, discoveredModels.fetchedAt = nil, time.Time{}
		discoveredModels.mu.Unlock()
	}()

	out, err := modelRegistration()
	if err != nil {
		t.Fatalf("modelRegistration: %v", err)
	}
	var resp struct {
		Result struct {
			Models []struct {
				ID string `json:"id"`
			} `json:"models"`
		} `json:"result"`
	}
	if err := json.Unmarshal(out, &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	seen := map[string]bool{}
	for _, m := range resp.Result.Models {
		seen[m.ID] = true
	}
	// jev-1.13-free is served on /systemone (a structured state-evaluation
	// API, not chat/responses), so announcing it would only produce a model
	// that can never answer. The official endpoint table keeps it hidden.
	if seen["jev-1.13-free"] {
		t.Fatalf("unsupported-dialect model announced: %v", seen)
	}
	if len(resp.Result.Models) != len(defaultZenModels) {
		t.Fatalf("model count = %d, want %d (defaults only; mimo is a dup, jev unsupported)", len(resp.Result.Models), len(defaultZenModels))
	}
}

func TestModelRegistrationExcludesConfiguredModels(t *testing.T) {
	discoveredModels.mu.Lock()
	discoveredModels.ids = []string{"mimo-v2.6-flash-free", "jev-1.13-free", "space-bunny-free"}
	discoveredModels.fetchedAt = time.Now()
	discoveredModels.mu.Unlock()
	defer func() {
		discoveredModels.mu.Lock()
		discoveredModels.ids, discoveredModels.fetchedAt = nil, time.Time{}
		discoveredModels.mu.Unlock()
	}()
	resetConfig(t, pluginConfig{Enabled: true, Provider: "zen", ExcludeModels: []string{"jev-1.13-free", "SPACE-BUNNY-FREE"}})

	out, err := modelRegistration()
	if err != nil {
		t.Fatalf("modelRegistration: %v", err)
	}
	var resp struct {
		Result struct {
			Models []struct {
				ID string `json:"id"`
			} `json:"models"`
		} `json:"result"`
	}
	if err := json.Unmarshal(out, &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	for _, m := range resp.Result.Models {
		if m.ID == "jev-1.13-free" || m.ID == "space-bunny-free" {
			t.Fatalf("excluded model announced: %s", m.ID)
		}
	}
}

func TestConfigureParsesExcludeModelsCommaString(t *testing.T) {
	payload, err := json.Marshal(struct {
		ConfigYAML []byte `json:"config_yaml"`
	}{[]byte("exclude-models: \"jev-1.13-free, space-bunny-free\"\napi-keys: k1,k2\n")})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := configure(payload); err != nil {
		t.Fatalf("configure: %v", err)
	}
	t.Cleanup(func() { storeConfig(defaultPluginConfig()) })
	cfg := loadedConfig()
	if len(cfg.ExcludeModels) != 2 || cfg.ExcludeModels[0] != "jev-1.13-free" || cfg.ExcludeModels[1] != "space-bunny-free" {
		t.Fatalf("ExcludeModels = %v", cfg.ExcludeModels)
	}
	if len(cfg.APIKeys) != 2 || cfg.APIKeys[0] != "k1" || cfg.APIKeys[1] != "k2" {
		t.Fatalf("APIKeys = %v", cfg.APIKeys)
	}
}

func TestConfigureParsesSemicolonSeparatedKeys(t *testing.T) {
	// Regression: management-center users entered "key1;key2" and the plugin
	// treated the whole string as one key, producing an invalid credential.
	payload, err := json.Marshal(struct {
		ConfigYAML []byte `json:"config_yaml"`
	}{[]byte("api-keys: \"oc_sk_aaa;oc_sk_bbb, oc_sk_ccc\"\n")})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := configure(payload); err != nil {
		t.Fatalf("configure: %v", err)
	}
	t.Cleanup(func() { storeConfig(defaultPluginConfig()) })
	cfg := loadedConfig()
	want := []string{"oc_sk_aaa", "oc_sk_bbb", "oc_sk_ccc"}
	if len(cfg.APIKeys) != len(want) {
		t.Fatalf("APIKeys = %v, want %v", cfg.APIKeys, want)
	}
	for i, k := range want {
		if cfg.APIKeys[i] != k {
			t.Fatalf("APIKeys = %v, want %v", cfg.APIKeys, want)
		}
	}
}

// ---------------------------------------------------------------------------
// stream resume (automatic reconnect of a broken upstream stream)
// ---------------------------------------------------------------------------

func TestStreamResumeMonitorChatFrames(t *testing.T) {
	m := &streamResumeMonitor{}
	m.observeChatFrame([]byte(`data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}` + "\n\n"))
	if m.sawFinish {
		t.Fatal("role chunk must not mark the stream terminal")
	}
	m.observeChatFrame([]byte(`data: {"choices":[{"delta":{"content":"你"}}]}` + "\n\n"))
	m.observeChatFrame([]byte(`data: {"choices":[{"delta":{"content":"好"}}]}` + "\n\n"))
	m.observeChatFrame([]byte(`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}` + "\n\n"))
	if !m.sawFinish {
		t.Fatal("finish_reason chunk must mark the stream terminal")
	}
	if got := m.partialText(); got != "你好" {
		t.Fatalf("partialText = %q, want %q", got, "你好")
	}
	if m.sawToolCall {
		t.Fatal("no tool calls were observed")
	}
}

func TestStreamResumeMonitorIgnoresHeartbeatShape(t *testing.T) {
	m := &streamResumeMonitor{}
	m.observeChatFrame([]byte(`{"id":"chatcmpl_x","choices":[{"index":0,"delta":{},"finish_reason":null}]}`))
	if m.sawFinish || m.partialText() != "" || m.sawToolCall {
		t.Fatal("heartbeat-shaped chunk must not affect the monitor")
	}
}

func TestStreamResumeMonitorToolCall(t *testing.T) {
	m := &streamResumeMonitor{}
	m.observeChatFrame([]byte(`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"bash","arguments":""}}]}}]}` + "\n\n"))
	if !m.sawToolCall {
		t.Fatal("tool_calls delta must flag the monitor")
	}
}

func TestStreamResumeMonitorBoundedTail(t *testing.T) {
	m := &streamResumeMonitor{}
	big := strings.Repeat("a", 3*streamResumeMaxPartialBytes)
	m.observeChatFrame([]byte(`data: {"choices":[{"delta":{"content":"` + big + `"}}]}` + "\n\n"))
	if got := m.partialText(); !strings.HasPrefix(got, streamResumeElidedMarker) {
		t.Fatalf("oversized partial must be tail-capped, len=%d", len(got))
	}
}

func TestAppendContinuationPayloadChat(t *testing.T) {
	base := chatPayload("deepseek-v4.1-flash-free", "写个东西")
	next, err := appendContinuationPayload(base, "已经写到一半")
	if err != nil {
		t.Fatal(err)
	}
	var root struct {
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(next, &root); err != nil {
		t.Fatal(err)
	}
	if len(root.Messages) != 3 {
		t.Fatalf("messages = %d, want 3", len(root.Messages))
	}
	if root.Messages[1].Role != "assistant" || root.Messages[1].Content != "已经写到一半" {
		t.Fatalf("assistant partial = %+v", root.Messages[1])
	}
	if root.Messages[2].Role != "user" || !strings.Contains(root.Messages[2].Content, "Continue the unfinished task") {
		t.Fatalf("resume instruction = %+v", root.Messages[2])
	}
}

func TestAppendContinuationPayloadResponsesInput(t *testing.T) {
	base := responsesPayload("muse-spark-1.3-contributor-free", "hello")
	next, err := appendContinuationPayload(base, "partial answer")
	if err != nil {
		t.Fatal(err)
	}
	var root struct {
		Input []struct {
			Type    string          `json:"type"`
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"input"`
	}
	if err := json.Unmarshal(next, &root); err != nil {
		t.Fatal(err)
	}
	if len(root.Input) != 3 {
		t.Fatalf("input items = %d, want 3", len(root.Input))
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(root.Input[1].Content, &parts); err != nil {
		t.Fatal(err)
	}
	if root.Input[1].Role != "assistant" || parts[0].Type != "output_text" || parts[0].Text != "partial answer" {
		t.Fatalf("assistant item = %+v %+v", root.Input[1], parts)
	}
	if err := json.Unmarshal(root.Input[2].Content, &parts); err != nil {
		t.Fatal(err)
	}
	if root.Input[2].Role != "user" || parts[0].Type != "input_text" || !strings.Contains(parts[0].Text, "Continue the unfinished task") {
		t.Fatalf("user resume item = %+v %+v", root.Input[2], parts)
	}
}

func TestAppendContinuationPayloadEmptyPartialReplays(t *testing.T) {
	base := chatPayload("deepseek-v4.1-flash-free", "hi")
	next, err := appendContinuationPayload(base, "")
	if err != nil {
		t.Fatal(err)
	}
	if string(next) != string(base) {
		t.Fatal("empty partial must return the payload unchanged")
	}
}

func TestAppendContinuationPayloadStringInput(t *testing.T) {
	base := []byte(`{"model":"m","input":"original question"}`)
	next, err := appendContinuationPayload(base, "partial")
	if err != nil {
		t.Fatal(err)
	}
	var root struct {
		Input string `json:"input"`
	}
	if err := json.Unmarshal(next, &root); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(root.Input, "original question") || !strings.Contains(root.Input, "partial") || !strings.Contains(root.Input, "Continue the unfinished task") {
		t.Fatalf("string input not extended: %q", root.Input)
	}
}

func TestAppendContinuationPayloadUnsupportedShape(t *testing.T) {
	if _, err := appendContinuationPayload([]byte("not json"), "partial"); err == nil {
		t.Fatal("non-JSON payload must fail")
	}
	if _, err := appendContinuationPayload([]byte(`{"model":"m"}`), "partial"); err == nil {
		t.Fatal("payload without messages/input must fail")
	}
}

func TestSSEReassemblerSawDone(t *testing.T) {
	r := &sseReassembler{
		rawData: false,
		emit:    func(frame []byte) error { return nil },
	}
	if err := r.write([]byte("data: {\"x\":1}\n\n")); err != nil {
		t.Fatal(err)
	}
	if r.sawDone {
		t.Fatal("no [DONE] yet")
	}
	if err := r.write([]byte("data: [DONE]\n\n")); err != nil {
		t.Fatal(err)
	}
	if !r.sawDone {
		t.Fatal("[DONE] terminator must set sawDone")
	}
}

func TestStreamRetryDelayCapped(t *testing.T) {
	if got := streamRetryDelay(1); got != time.Second {
		t.Fatalf("delay(1) = %v", got)
	}
	if got := streamRetryDelay(2); got != 2*time.Second {
		t.Fatalf("delay(2) = %v", got)
	}
	if got := streamRetryDelay(3); got != 4*time.Second {
		t.Fatalf("delay(3) = %v", got)
	}
	if got := streamRetryDelay(9); got != 8*time.Second {
		t.Fatalf("delay(9) = %v, want capped 8s", got)
	}
}

func TestStreamRetryableStatus(t *testing.T) {
	for _, s := range []int{500, 502, 503, 504, 408} {
		if !streamRetryableStatus(s) {
			t.Fatalf("status %d should be retryable", s)
		}
	}
	for _, s := range []int{200, 400, 401, 403, 404, 429} {
		if streamRetryableStatus(s) {
			t.Fatalf("status %d should NOT be retryable", s)
		}
	}
}

func TestConfigureParsesStreamRetries(t *testing.T) {
	payload, err := json.Marshal(struct {
		ConfigYAML []byte `json:"config_yaml"`
	}{[]byte("stream-retries: 5\n")})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := configure(payload); err != nil {
		t.Fatalf("configure: %v", err)
	}
	t.Cleanup(func() { storeConfig(defaultPluginConfig()) })
	if got := loadedConfig().StreamRetries; got != 5 {
		t.Fatalf("StreamRetries = %d, want 5", got)
	}

	payload, _ = json.Marshal(struct {
		ConfigYAML []byte `json:"config_yaml"`
	}{[]byte("stream-retries: 99\n")})
	if err := configure(payload); err != nil {
		t.Fatalf("configure: %v", err)
	}
	if got := loadedConfig().StreamRetries; got != 10 {
		t.Fatalf("StreamRetries clamp = %d, want 10", got)
	}
}

func TestResumeTailStringUTF8Boundary(t *testing.T) {
	s := strings.Repeat("汉", streamResumeMaxPartialBytes/3+10) // 3 bytes each
	tail := resumeTailString(s, 30)
	if !utf8.ValidString(tail) {
		t.Fatal("tail cut a rune in half")
	}
	if got := resumeTailString("short", 30); got != "short" {
		t.Fatalf("short string changed: %q", got)
	}
}

func TestTrimJSONEnvelopeStripsBOM(t *testing.T) {
	bom := []byte{0xEF, 0xBB, 0xBF}

	if got := string(trimJSONEnvelope(append(bom, []byte(`{"model":"m"}`)...))); got != `{"model":"m"}` {
		t.Fatalf("BOM not stripped: %q", got)
	}
	if got := string(trimJSONEnvelope([]byte("  \n{\"a\":1}\n "))); got != `{"a":1}` {
		t.Fatalf("whitespace not trimmed: %q", got)
	}
}

func TestStreamResumeMonitorDanglingLeadIn(t *testing.T) {
	frm := func(text string) *streamResumeMonitor {
		m := &streamResumeMonitor{}
		raw, _ := json.Marshal(map[string]any{"choices": []any{
			map[string]any{"delta": map[string]any{"content": text}},
		}})
		m.observeChatFrame([]byte("data: " + string(raw) + "\n\n"))
		return m
	}
	if !frm("编辑破坏了顺序，直接重写脚本：").danglingLeadIn() {
		t.Fatal("text ending with a full-width colon must be dangling")
	}
	if !frm("接下来：").danglingLeadIn() {
		t.Fatal("colon must be dangling")
	}
	if !frm("```python\nprint(1)").danglingLeadIn() {
		t.Fatal("unclosed code fence must be dangling")
	}
	if frm("已经写完了。").danglingLeadIn() {
		t.Fatal("complete sentence must not be dangling")
	}
	if (&streamResumeMonitor{}).danglingLeadIn() {
		t.Fatal("empty text must not be dangling")
	}
}

func TestPrepareUpstreamBodyAcceptsBOMPayload(t *testing.T) {
	bom := []byte{0xEF, 0xBB, 0xBF}
	req := executorRequest{
		Model:   "mimo-v2.6-flash-free",
		Payload: append(bom, chatPayload("mimo-v2.6-flash-free", "hi")...),
	}
	if _, err := prepareUpstreamBody(req, modelRoute{Model: "mimo-v2.6-flash-free", Endpoint: "chat"}); err != nil {
		t.Fatalf("BOM payload rejected: %v", err)
	}
}

func TestStreamResumeMonitorAnyOutput(t *testing.T) {
	m := &streamResumeMonitor{}
	if m.anyOutput() {
		t.Fatal("fresh monitor must report no output")
	}
	m.observeChatFrame([]byte(`data: {"choices":[{"delta":{"content":"x"}}]}` + "\n\n"))
	if !m.anyOutput() {
		t.Fatal("text must count as output")
	}
	tc := &streamResumeMonitor{}
	tc.observeChatFrame([]byte(`data: {"choices":[{"delta":{"tool_calls":[{"index":0}]}}]}` + "\n\n"))
	if !tc.anyOutput() {
		t.Fatal("tool call must count as output")
	}
}

func TestEmptyCompletionDetection(t *testing.T) {
	empty := []byte(`{"choices":[{"index":0,"message":{"role":"assistant","content":""}}]}`)
	if !isEmptyCompletion(empty) {
		t.Fatal("empty content must be detected")
	}
	hasText := []byte(`{"choices":[{"index":0,"message":{"role":"assistant","content":"hi"}}]}`)
	if isEmptyCompletion(hasText) {
		t.Fatal("content must not be reported empty")
	}
	hasTool := []byte(`{"choices":[{"index":0,"message":{"role":"assistant","content":"","tool_calls":[{"id":"c1"}]}}]}`)
	if isEmptyCompletion(hasTool) {
		t.Fatal("tool call must not be reported empty")
	}
}

func TestConverterRecoversToolCallFromCompletedPayload(t *testing.T) {
	// Some gateways stream a short prefix and then deliver the authoritative
	// items only inside response.completed; the tool call must not be lost.
	c := newResponsesChatStreamConverter("mimo-v2.6-flash-free")
	payload := []byte(`{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[{"id":"fc_1","type":"function_call","call_id":"call_1","name":"bash","arguments":"{\"command\":\"ls\"}"}]}}`)
	outs, err := c.convert(payload)
	if err != nil {
		t.Fatal(err)
	}
	var sawTool, sawFinish bool
	for _, out := range outs {
		var chunk struct {
			Choices []struct {
				Delta struct {
					ToolCalls []struct {
						ID       string `json:"id"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
		}
		if err := json.Unmarshal(out, &chunk); err != nil {
			t.Fatal(err)
		}
		for _, ch := range chunk.Choices {
			if len(ch.Delta.ToolCalls) > 0 {
				sawTool = true
				call := ch.Delta.ToolCalls[0]
				if call.ID != "call_1" || call.Function.Name != "bash" || call.Function.Arguments != `{"command":"ls"}` {
					t.Fatalf("tool call = %+v", call)
				}
			}
			if ch.FinishReason != nil && *ch.FinishReason != "" {
				sawFinish = true
				if *ch.FinishReason != "tool_calls" {
					t.Fatalf("finish = %q, want tool_calls", *ch.FinishReason)
				}
			}
		}
	}
	if !sawTool || !sawFinish {
		t.Fatalf("sawTool=%v sawFinish=%v", sawTool, sawFinish)
	}
}

func TestConverterRecoversTextFromCompletedPayload(t *testing.T) {
	c := newResponsesChatStreamConverter("mimo-v2.6-flash-free")
	payload := []byte(`{"type":"response.completed","response":{"id":"resp_2","status":"completed","output":[{"id":"msg_1","type":"message","content":[{"type":"output_text","text":"最终答复"}]}]}}`)
	outs, err := c.convert(payload)
	if err != nil {
		t.Fatal(err)
	}
	var sawText bool
	for _, out := range outs {
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal(out, &chunk); err != nil {
			t.Fatal(err)
		}
		for _, ch := range chunk.Choices {
			if ch.Delta.Content == "最终答复" {
				sawText = true
			}
		}
	}
	if !sawText {
		t.Fatal("text delivered only in response.completed must be recovered")
	}
}

func TestConverterDoesNotDuplicateStreamedItems(t *testing.T) {
	// When the items were already streamed, the completion payload must not
	// re-emit them.
	c := newResponsesChatStreamConverter("mimo-v2.6-flash-free")
	if _, err := c.convert([]byte(`{"type":"response.output_text.delta","item_id":"msg_1","delta":"hi"}`)); err != nil {
		t.Fatal(err)
	}
	outs, err := c.convert([]byte(`{"type":"response.completed","response":{"id":"resp_3","status":"completed","output":[{"id":"msg_1","type":"message","content":[{"type":"output_text","text":"hi"}]}]}}`))
	if err != nil {
		t.Fatal(err)
	}
	for _, out := range outs {
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal(out, &chunk); err != nil {
			t.Fatal(err)
		}
		for _, ch := range chunk.Choices {
			if ch.Delta.Content != "" {
				t.Fatalf("streamed text re-emitted: %q", ch.Delta.Content)
			}
		}
	}
}
