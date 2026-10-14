// SPDX-License-Identifier: MIT
package main

/*
#include <stdint.h>
#include <stdlib.h>

typedef struct {
	void* ptr;
	size_t len;
} cliproxy_buffer;

typedef struct {
	uint32_t abi_version;
	void* host_ctx;
	int (*call)(void*, const char*, const uint8_t*, size_t, cliproxy_buffer*);
	void (*free_buffer)(void*, size_t);
} cliproxy_host_api;

typedef int (*cliproxy_plugin_call_fn)(char*, uint8_t*, size_t, cliproxy_buffer*);
typedef void (*cliproxy_plugin_free_fn)(void*, size_t);
typedef void (*cliproxy_plugin_shutdown_fn)(void);

typedef struct {
	uint32_t abi_version;
	cliproxy_plugin_call_fn call;
	cliproxy_plugin_free_fn free_buffer;
	cliproxy_plugin_shutdown_fn shutdown;
} cliproxy_plugin_api;

static cliproxy_host_api* stored_host;

static void store_host_api(cliproxy_host_api* host) {
	stored_host = host;
}

static int call_host_api(const char* method, const uint8_t* request, size_t request_len, cliproxy_buffer* response) {
	if (stored_host == NULL || stored_host->call == NULL) {
		return 1;
	}
	return stored_host->call(stored_host->host_ctx, method, request, request_len, response);
}

static void free_host_buffer(void* ptr, size_t len) {
	if (stored_host != NULL && stored_host->free_buffer != NULL && ptr != NULL) {
		stored_host->free_buffer(ptr, len);
	}
}

extern int cliproxyPluginCall(char*, uint8_t*, size_t, cliproxy_buffer*);
extern void cliproxyPluginFree(void*, size_t);
extern void cliproxyPluginShutdown(void);
*/
import "C"

import (
	"bytes"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode"
	"unicode/utf8"
	"unsafe"

	"gopkg.in/yaml.v3"
)

const abiVersion uint32 = 1

const (
	pluginID       = "zen"
	pluginProvider = "zen"

	zenBaseURL = "https://opencode.ai/zen/v1"

	maxSessionIDBytes = 1024

	targetSessionHeader = "X-Opencode-Session"
	targetRequestHeader = "X-Opencode-Request"
	targetClientHeader  = "X-Opencode-Client"
	targetProjectHeader = "X-Opencode-Project"

	defaultUserAgent = "opencode/1.18.31 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14"
)

var pluginVersion = "0.8.3"

var (
	canonicalSessionRe = regexp.MustCompile(`^ses_[0-9a-f]{12}[0-9A-Za-z]{14}$`)
)

// sessionSources lists downstream client headers that carry a stable
// per-conversation session id, in priority order (same semantics as
// opencode-session-mapper).
var sessionSources = []string{
	"Session-Id",
	"Session_id",
	"Thread-Id",
	"Thread_id",
	"X-Claude-Code-Session-Id",
	"X-DeepSeek-Harness-Session-Id",
	"X-Session-Affinity",
	"X-Session-Id",
	"X-Client-Request-Id",
}

// ---------------------------------------------------------------------------
// plugin configuration
// ---------------------------------------------------------------------------

type modelRoute struct {
	Model    string
	Endpoint string // "chat" (default) or "responses"
	Alias    string

	// Optional capability metadata announced to the host model registry.
	// Explicit values override the built-in defaults for the model.
	ContextLength       int64    // yaml: context-length
	MaxCompletionTokens int64    // yaml: max-completion-tokens
	InputModalities     []string // yaml: input-modalities
	ReasoningLevels     []string // yaml: reasoning-levels
}

// EndpointPath returns the upstream path suffix for this route.
func (m modelRoute) EndpointPath() string {
	if m.Endpoint == "responses" {
		return "/responses"
	}
	return "/chat/completions"
}

type pluginConfig struct {
	Enabled       bool
	Provider      string
	BaseURL       string
	APIKeys       []string
	ExcludeModels []string
	Client        string
	Project       string
	Models        []modelRoute

	// StreamRetries bounds the automatic reconnects of a broken upstream
	// stream (silent truncation, transport EOF, 5xx). yaml: stream-retries.
	StreamRetries int
}

func defaultPluginConfig() pluginConfig {
	return pluginConfig{
		Enabled:       true,
		Provider:      pluginProvider,
		BaseURL:       zenBaseURL,
		Client:        "cli",
		Project:       "global",
		StreamRetries: 3,
	}
}

var (
	configMu sync.RWMutex
	current  = defaultPluginConfig()
)

func loadedConfig() pluginConfig {
	configMu.RLock()
	defer configMu.RUnlock()
	return current
}

func storeConfig(cfg pluginConfig) {
	configMu.Lock()
	defer configMu.Unlock()
	current = cfg
}

func configure(raw []byte) error {
	cfg := defaultPluginConfig()
	if len(raw) > 0 {
		var req struct {
			ConfigYAML []byte `json:"config_yaml"`
		}
		if err := json.Unmarshal(raw, &req); err != nil {
			return err
		}
		if len(req.ConfigYAML) > 0 {
			var node map[string]any
			if err := yaml.Unmarshal(req.ConfigYAML, &node); err != nil {
				return err
			}
			applyConfigNode(&cfg, node)
		}
	}
	cfg.Provider = strings.ToLower(strings.TrimSpace(cfg.Provider))
	if cfg.Provider == "" {
		cfg.Provider = pluginProvider
	}
	cfg.BaseURL = strings.TrimRight(strings.TrimSpace(cfg.BaseURL), "/")
	if cfg.BaseURL == "" {
		cfg.BaseURL = zenBaseURL
	}
	if cfg.Client = strings.TrimSpace(cfg.Client); cfg.Client == "" {
		cfg.Client = "cli"
	}
	if cfg.Project = strings.TrimSpace(cfg.Project); cfg.Project == "" {
		cfg.Project = "global"
	}
	if cfg.StreamRetries < 0 {
		cfg.StreamRetries = 0
	}
	if cfg.StreamRetries > 10 {
		cfg.StreamRetries = 10
	}
	cfg.APIKeys = trimNonEmpty(cfg.APIKeys)
	cfg.ExcludeModels = trimNonEmpty(cfg.ExcludeModels)
	for i := range cfg.Models {
		m := &cfg.Models[i]
		m.Model = strings.TrimSpace(m.Model)
		m.Alias = strings.TrimSpace(m.Alias)
		m.Endpoint = strings.ToLower(strings.TrimSpace(m.Endpoint))
		if m.Endpoint != "responses" {
			m.Endpoint = "chat"
		}
		if m.Alias == "" {
			m.Alias = m.Model
		}
	}
	storeConfig(cfg)
	persistVirtualAuths(cfg)
	return nil
}

// persistVirtualAuths materializes plugin-config api-keys as real credential
// files in the host auth directory via the host.auth.save callback. CPA v8
// only creates runtime auths from files discovered in the auth dir (the
// auth.parse payload carries RawJSON, not StorageJSON), so virtual keys
// configured through the management center must be written to disk to
// become selectable credentials.
func persistVirtualAuths(cfg pluginConfig) {
	if !cfg.Enabled || len(cfg.APIKeys) == 0 {
		return
	}
	for _, key := range cfg.APIKeys {
		id := sha256Prefix(key)
		name := fmt.Sprintf("%s-%s.json", cfg.Provider, id)
		payload, _ := json.Marshal(map[string]string{
			"type":     cfg.Provider,
			"provider": cfg.Provider,
			"api_key":  key,
			"base_url": cfg.BaseURL,
		})
		// Best effort: failures are logged by the host; the auth.parse
		// fallback still returns virtual auths for hosts that support it.
		_, _ = callHost("host.auth.save", map[string]any{
			"Name": name,
			"JSON": json.RawMessage(payload),
		})
	}
}

func trimNonEmpty(in []string) []string {
	out := make([]string, 0, len(in))
	for _, s := range in {
		if s = strings.TrimSpace(s); s != "" {
			out = append(out, s)
		}
	}
	return out
}

func applyConfigNode(cfg *pluginConfig, node map[string]any) {
	if node == nil {
		return
	}
	if v, ok := node["enabled"]; ok {
		cfg.Enabled = boolValue(v)
	}
	if v, ok := node["provider"].(string); ok {
		cfg.Provider = v
	}
	if v, ok := node["base-url"].(string); ok {
		cfg.BaseURL = v
	}
	if v, ok := node["client"].(string); ok {
		cfg.Client = v
	}
	if _, ok := node["stream-retries"]; ok {
		cfg.StreamRetries = int(intValue(node["stream-retries"], node["stream_retries"]))
	}
	if v, ok := node["project"].(string); ok {
		cfg.Project = v
	}
	switch t := node["api-keys"].(type) {
	case []any:
		for _, item := range t {
			if s, ok := item.(string); ok {
				cfg.APIKeys = append(cfg.APIKeys, s)
			}
		}
	case string:
		// Accept comma, semicolon, or whitespace separated keys; management
		// center users have used ";" and gotten one invalid concatenated key.
		for _, part := range strings.FieldsFunc(t, func(r rune) bool {
			return r == ',' || r == ';' || r == ' ' || r == '\t' || r == '\n' || r == '\r'
		}) {
			cfg.APIKeys = append(cfg.APIKeys, part)
		}
	}
	switch t := node["exclude-models"].(type) {
	case []any:
		for _, item := range t {
			if s, ok := item.(string); ok {
				cfg.ExcludeModels = append(cfg.ExcludeModels, s)
			}
		}
	case string:
		for _, part := range strings.FieldsFunc(t, func(r rune) bool {
			return r == ',' || r == ';' || r == ' ' || r == '\t' || r == '\n' || r == '\r'
		}) {
			cfg.ExcludeModels = append(cfg.ExcludeModels, part)
		}
	}
	if rawModels, ok := node["models"].([]any); ok {
		for _, item := range rawModels {
			entry, ok := item.(map[string]any)
			if !ok {
				continue
			}
			m := modelRoute{}
			if s, ok := entry["model"].(string); ok {
				m.Model = s
			}
			if s, ok := entry["name"].(string); ok && m.Model == "" {
				m.Model = s
			}
			if s, ok := entry["endpoint"].(string); ok {
				m.Endpoint = s
			}
			if s, ok := entry["alias"].(string); ok {
				m.Alias = s
			}
			m.ContextLength = intValue(entry["context-length"], entry["context_length"])
			m.MaxCompletionTokens = intValue(entry["max-completion-tokens"], entry["max_completion_tokens"])
			m.InputModalities = stringSliceValue(entry["input-modalities"], entry["input_modalities"])
			m.ReasoningLevels = stringSliceValue(entry["reasoning-levels"], entry["reasoning_levels"])
			if m.Model != "" {
				cfg.Models = append(cfg.Models, m)
			}
		}
	}
}

func boolValue(v any) bool {
	switch t := v.(type) {
	case bool:
		return t
	case string:
		return strings.EqualFold(strings.TrimSpace(t), "true")
	default:
		return false
	}
}

// intValue returns the first numeric value among candidates as int64.
func intValue(values ...any) int64 {
	for _, v := range values {
		switch t := v.(type) {
		case int:
			return int64(t)
		case int64:
			return t
		case float64:
			return int64(t)
		}
	}
	return 0
}

// stringSliceValue returns the first non-empty string slice among candidates.
// Accepts a YAML list or a comma-separated string.
func stringSliceValue(values ...any) []string {
	for _, v := range values {
		switch t := v.(type) {
		case []any:
			out := make([]string, 0, len(t))
			for _, item := range t {
				if s, ok := item.(string); ok {
					if s = strings.TrimSpace(s); s != "" {
						out = append(out, s)
					}
				}
			}
			if len(out) > 0 {
				return out
			}
		case string:
			var out []string
			for _, part := range strings.Split(t, ",") {
				if part = strings.TrimSpace(part); part != "" {
					out = append(out, part)
				}
			}
			if len(out) > 0 {
				return out
			}
		}
	}
	return nil
}

// inferEndpoint resolves the upstream endpoint ("chat" or "responses") for a model.
// Any model containing "muse" or "responses" defaults to "responses"; others to "chat".
func inferEndpoint(model string) string {
	lower := strings.ToLower(model)
	if strings.Contains(lower, "muse") || strings.Contains(lower, "responses") {
		return "responses"
	}
	return "chat"
}

// learnedEndpoints records upstream endpoints discovered by retrying a
// failed chat request against /responses (models zen only serves there).
var learnedEndpoints struct {
	mu sync.RWMutex
	m  map[string]string
}

func rememberEndpoint(model, endpoint string) {
	learnedEndpoints.mu.Lock()
	defer learnedEndpoints.mu.Unlock()
	if learnedEndpoints.m == nil {
		learnedEndpoints.m = map[string]string{}
	}
	learnedEndpoints.m[strings.ToLower(strings.TrimSpace(model))] = endpoint
}

func learnedEndpoint(model string) (string, bool) {
	learnedEndpoints.mu.RLock()
	defer learnedEndpoints.mu.RUnlock()
	ep, ok := learnedEndpoints.m[strings.ToLower(strings.TrimSpace(model))]
	return ep, ok
}

// endpointUnsupported marks models whose upstream endpoint speaks a dialect
// this plugin cannot translate (Anthropic Messages, Jev's /systemone, the
// AI-SDK Gemini path). Requests to such models are refused up front instead
// of being posted to a wrong path.
const endpointUnsupported = "unsupported"

// officialZenEndpoint classifies a model by its upstream endpoint per the
// official OpenCode Zen docs endpoint table
// (https://opencode.ai/docs/zh-cn/zen/#端点), snapshotted 2026-10-01.
//
// The table is a routing *hint*, never a source of truth:
//   - model existence still comes from live /models discovery; the /models
//     payload carries no endpoint field, so the docs are the only
//     machine-usable source of endpoint data;
//   - classification prefers family prefixes over per-model rows so
//     undocumented ids (deepseek-v4-flash-free, muse-spark-1.2-...) and
//     docs churn still classify correctly;
//   - execute()/runStream() retry the opposite dialect on a path-level
//     failure (400/404/405) unless the route is explicitly configured, so
//     a stale entry self-corrects at runtime;
//   - an explicit plugins.configs.zen models entry with an endpoint always
//     wins, which is the escape hatch for any misclassification.
func officialZenEndpoint(model string) (string, bool) {
	lower := strings.ToLower(strings.TrimSpace(model))
	if lower == "" {
		return "", false
	}
	if ep, ok := officialZenEndpointModels[lower]; ok {
		return ep, true
	}
	for _, rule := range officialZenEndpointPrefixes {
		if strings.HasPrefix(lower, rule.prefix) {
			return rule.endpoint, true
		}
	}
	return "", false
}

// officialZenEndpointPrefixes maps model-id families to endpoints. Family
// rules are used where the docs show a consistent family-wide endpoint.
var officialZenEndpointPrefixes = []struct {
	prefix   string
	endpoint string
}{
	// Served on /responses (OpenAI-style).
	{"gpt-", "responses"},
	{"grok-", "responses"},
	{"muse-", "responses"},
	// Non-OpenAI dialects this plugin cannot speak.
	{"claude-", endpointUnsupported}, // Anthropic /messages
	{"gemini-", endpointUnsupported}, // AI-SDK Google path
	{"jev-", endpointUnsupported},    // /systemone
	// Served on /chat/completions (OpenAI-compatible).
	{"deepseek-", "chat"},
	{"glm-", "chat"},
	{"kimi-", "chat"},
	{"ling-", "chat"},
	{"longcat-", "chat"},
	{"minimax-", "chat"},
	{"mimo-", "chat"},
	{"nemotron-", "chat"},
}

// officialZenEndpointModels covers ids whose endpoint cannot be derived
// from a family prefix.
var officialZenEndpointModels = map[string]string{
	"big-pickle":       "chat",
	"space-bunny-free": "chat",
	"qwen3.8-max":      "chat",
	// Qwen models on the Anthropic Messages endpoint.
	"qwen3.8-flash": endpointUnsupported,
	"qwen3.7-max":   endpointUnsupported,
	"qwen3.7-plus":  endpointUnsupported,
	"qwen3.6-plus":  endpointUnsupported,
	"qwen3.5-plus":  endpointUnsupported,
}

// candidateEndpoints returns the endpoints to attempt for a request, in
// order. Explicit configuration pins a single endpoint; otherwise the
// opposite dialect is tried once after a path-level failure so endpoint
// hints that drift from reality self-correct.
func candidateEndpoints(cfg pluginConfig, route modelRoute) []string {
	first := "chat"
	if route.Endpoint == "responses" {
		first = "responses"
	}
	if routeConfigured(cfg, route.Model) {
		return []string{first}
	}
	second := "chat"
	if first == "chat" {
		second = "responses"
	}
	return []string{first, second}
}

// isEndpointMismatch reports whether an upstream status looks like "wrong
// path for this model" rather than a request we should never repeat.
func isEndpointMismatch(status int) bool {
	return status == 400 || status == 404 || status == 405
}

// routeConfigured reports whether the model has an explicit entry in plugin
// config *with a pinned endpoint*; only then is the endpoint authoritative
// (learned endpoints and dialect retries never override it). An entry
// without an endpoint (e.g. configured just for alias/capabilities) still
// gets inference and retries.
func routeConfigured(cfg pluginConfig, model string) bool {
	for _, m := range cfg.Models {
		if strings.EqualFold(m.Model, model) || strings.EqualFold(m.Alias, model) {
			if strings.TrimSpace(m.Endpoint) != "" {
				return true
			}
		}
	}
	return false
}

// routeForModel resolves the route for a requested model id.
// Endpoint resolution order: explicit config endpoint > endpoint learned
// from a live retry > official endpoint table > name inference.
func routeForModel(cfg pluginConfig, model string) (modelRoute, bool) {
	model = strings.TrimSpace(model)
	if model == "" {
		return modelRoute{}, false
	}
	var configured *modelRoute
	for i := range cfg.Models {
		m := &cfg.Models[i]
		if strings.EqualFold(m.Model, model) || strings.EqualFold(m.Alias, model) {
			if strings.TrimSpace(m.Endpoint) != "" {
				return *m, true
			}
			if configured == nil {
				configured = m
			}
		}
	}
	if configured != nil {
		// Entry pins the upstream name/alias but not the endpoint: resolve
		// the endpoint as if the model were unconfigured.
		r := *configured
		r.Endpoint = resolveEndpoint(model)
		return r, true
	}
	return modelRoute{Model: model, Alias: model, Endpoint: resolveEndpoint(model)}, true
}

// resolveEndpoint picks the best endpoint hint for an unconfigured model.
func resolveEndpoint(model string) string {
	if ep, ok := learnedEndpoint(model); ok {
		return ep
	}
	if ep, known := officialZenEndpoint(model); known && ep != endpointUnsupported {
		return ep
	}
	return inferEndpoint(model)
}

// ---------------------------------------------------------------------------
// RPC envelope types
// ---------------------------------------------------------------------------

type envelope struct {
	OK     bool            `json:"ok"`
	Result json.RawMessage `json:"result,omitempty"`
	Error  *envelopeError  `json:"error,omitempty"`
}

type envelopeError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type registration struct {
	SchemaVersion uint32       `json:"schema_version"`
	Metadata      metadata     `json:"metadata"`
	Capabilities  capabilities `json:"capabilities"`
}

type metadata struct {
	Name             string `json:"Name"`
	Version          string `json:"Version"`
	Author           string `json:"Author"`
	GitHubRepository string `json:"GitHubRepository"`
	Logo             string `json:"Logo"`
	ConfigFields     []any  `json:"ConfigFields"`
}

type capabilities struct {
	Executor              bool     `json:"executor"`
	ExecutorModelScope    string   `json:"executor_model_scope"`
	ExecutorInputFormats  []string `json:"executor_input_formats"`
	ExecutorOutputFormats []string `json:"executor_output_formats"`
	ModelRegistrar        bool     `json:"model_registrar"`
	AuthProvider          bool     `json:"auth_provider"`
}

type modelInfo struct {
	ID          string `json:"ID"`
	Object      string `json:"Object"`
	Created     int64  `json:"Created"`
	OwnedBy     string `json:"OwnedBy"`
	Type        string `json:"Type"`
	DisplayName string `json:"DisplayName"`

	// Capability metadata consumed by the host model registry and forwarded
	// to clients as context_window / max_tokens / input_modalities /
	// supported_reasoning_levels. Zero values are omitted.
	ContextLength            int64            `json:"ContextLength,omitempty"`
	MaxCompletionTokens      int64            `json:"MaxCompletionTokens,omitempty"`
	SupportedInputModalities []string         `json:"SupportedInputModalities,omitempty"`
	Thinking                 *thinkingSupport `json:"Thinking,omitempty"`
}

// thinkingSupport mirrors the host plugin SDK ThinkingSupport shape.
type thinkingSupport struct {
	Min            int      `json:"Min,omitempty"`
	Max            int      `json:"Max,omitempty"`
	ZeroAllowed    bool     `json:"ZeroAllowed,omitempty"`
	DynamicAllowed bool     `json:"DynamicAllowed,omitempty"`
	Levels         []string `json:"Levels,omitempty"`
}

// modelCapabilities is the resolved capability metadata for one model.
type modelCapabilities struct {
	ContextLength       int64
	MaxCompletionTokens int64
	InputModalities     []string
	ReasoningLevels     []string
	DisplayName         string
	Thinking            *thinkingSupport
}

type modelRegistrationResponse struct {
	Provider string      `json:"Provider"`
	Models   []modelInfo `json:"Models"`
}

type identifierResponse struct {
	Identifier string `json:"identifier"`
}

type executorRequest struct {
	AuthID          string              `json:"AuthID"`
	AuthProvider    string              `json:"AuthProvider"`
	Model           string              `json:"Model"`
	Format          string              `json:"Format"`
	Stream          bool                `json:"Stream"`
	Alt             string              `json:"Alt"`
	Headers         map[string][]string `json:"Headers"`
	OriginalRequest []byte              `json:"OriginalRequest"`
	SourceFormat    string              `json:"SourceFormat"`
	Payload         []byte              `json:"Payload"`
	StorageJSON     []byte              `json:"StorageJSON"`
	AuthMetadata    map[string]any      `json:"AuthMetadata"`
	AuthAttributes  map[string]any      `json:"AuthAttributes"`
	Metadata        map[string]any      `json:"Metadata"`
	StreamID        string              `json:"stream_id,omitempty"`
	HostCallbackID  string              `json:"host_callback_id,omitempty"`
}

type executorResponse struct {
	Payload []byte              `json:"Payload"`
	Headers map[string][]string `json:"Headers,omitempty"`
}

type executorStreamResponse struct {
	Headers map[string][]string `json:"headers,omitempty"`
}

type httpStreamChunk struct {
	Payload []byte `json:"payload,omitempty"`
	Error   string `json:"error,omitempty"`
	Done    bool   `json:"done,omitempty"`
}

// ---------------------------------------------------------------------------
// exported C ABI
// ---------------------------------------------------------------------------

func main() {}

//export cliproxy_plugin_init
func cliproxy_plugin_init(host *C.cliproxy_host_api, plugin *C.cliproxy_plugin_api) C.int {
	if plugin == nil {
		return 1
	}
	C.store_host_api(host)
	plugin.abi_version = C.uint32_t(abiVersion)
	plugin.call = C.cliproxy_plugin_call_fn(C.cliproxyPluginCall)
	plugin.free_buffer = C.cliproxy_plugin_free_fn(C.cliproxyPluginFree)
	plugin.shutdown = C.cliproxy_plugin_shutdown_fn(C.cliproxyPluginShutdown)
	return 0
}

//export cliproxyPluginCall
func cliproxyPluginCall(method *C.char, request *C.uint8_t, requestLen C.size_t, response *C.cliproxy_buffer) (status C.int) {
	status = 1
	if response == nil {
		return 1
	}
	response.ptr = nil
	response.len = 0
	defer func() {
		if recover() != nil {
			writeResponse(response, errorEnvelope("plugin_panic", "plugin call failed"))
			status = 1
		}
	}()
	if method == nil {
		writeResponse(response, errorEnvelope("invalid_method", "method is required"))
		return 1
	}
	var payload []byte
	if requestLen > 0 {
		if request == nil {
			writeResponse(response, errorEnvelope("invalid_request", "request buffer is required"))
			return 1
		}
		payload = C.GoBytes(unsafe.Pointer(request), C.int(requestLen))
	}
	raw, callStatus := processPluginCall(C.GoString(method), payload)
	writeResponse(response, raw)
	return C.int(callStatus)
}

//export cliproxyPluginFree
func cliproxyPluginFree(ptr unsafe.Pointer, length C.size_t) {
	if ptr != nil {
		C.free(ptr)
	}
	_ = length
}

//export cliproxyPluginShutdown
func cliproxyPluginShutdown() {}

// ---------------------------------------------------------------------------
// method dispatch
// ---------------------------------------------------------------------------

func processPluginCall(method string, payload []byte) ([]byte, int) {
	raw, errHandle := handleMethod(method, payload)
	if errHandle != nil {
		return errorEnvelope("plugin_error", errHandle.Error()), 1
	}
	return raw, 0
}

func handleMethod(method string, payload []byte) ([]byte, error) {
	switch method {
	case "plugin.register", "plugin.reconfigure":
		if err := configure(payload); err != nil {
			return nil, err
		}
		return okEnvelopeJSON(registration{
			SchemaVersion: abiVersion,
			Metadata: metadata{
				Name:             pluginID,
				Version:          pluginVersion,
				Author:           "Victor9578",
				GitHubRepository: "https://github.com/Victor9578/cpa-plugin-opencodezen",
				Logo:             "https://raw.githubusercontent.com/Victor9578/cpa-plugin-opencodezen/main/logo.svg",
				ConfigFields: []any{
					map[string]any{"Name": "enabled", "Type": "boolean", "Description": "Enable the zen provider plugin."},
					map[string]any{"Name": "api-keys", "Type": "string", "Description": "OpenCode zen API keys (comma-separated). Each key becomes a virtual zen credential; alternatively drop key files into the auth directory."},
					map[string]any{"Name": "exclude-models", "Type": "string", "Description": "Model IDs (comma-separated) to exclude from the models auto-discovered from the zen /models endpoint."},
				},
			},
			Capabilities: capabilities{
				Executor:              true,
				ExecutorModelScope:    "both",
				ExecutorInputFormats:  []string{"chat-completions", "responses"},
				ExecutorOutputFormats: []string{"chat-completions"},
				ModelRegistrar:        true,
				AuthProvider:          true,
			},
		})
	case "executor.identifier", "auth.identifier":
		return okEnvelopeJSON(identifierResponse{Identifier: loadedConfig().Provider})
	case "auth.parse":
		return authParse(payload)
	case "model.register":
		return modelRegistration()
	case "executor.execute":
		return execute(payload, false)
	case "executor.execute_stream":
		return execute(payload, true)
	case "executor.count_tokens":
		return okEnvelopeJSON(executorResponse{Payload: []byte(`{"total_tokens":0}`)})
	default:
		return errorEnvelope("unknown_method", "unknown method: "+method), nil
	}
}

// ---------------------------------------------------------------------------
// auth provider
// ---------------------------------------------------------------------------

// authParse inspects credential files discovered in the auth directory.
// It recognizes:
//  1. Files where provider is "zen" or type is "zen"
//  2. Any file containing an "api_key" or "key" field with "sk-" prefix
//  3. Files matching configured keys in plugins.configs.zen
//
// The host sends the auth file payload as RawJSON (AuthParseRequest.RawJSON
// in CPA v8); StorageJSON is accepted as well for compatibility with hosts
// that deliver it under that name.
func authParse(payload []byte) ([]byte, error) {
	cfg := loadedConfig()
	if len(payload) > 0 {
		var req struct {
			Provider    string          `json:"Provider"`
			StorageJSON json.RawMessage `json:"StorageJSON"`
			RawJSON     json.RawMessage `json:"RawJSON"`
			FileName    string          `json:"FileName"`
			ID          string          `json:"ID"`
		}
		if err := json.Unmarshal(payload, &req); err == nil {
			raw := req.StorageJSON
			if len(raw) == 0 {
				raw = req.RawJSON
			}
			var stored map[string]any
			if len(raw) > 0 {
				_ = json.Unmarshal(raw, &stored)
			}
			if len(stored) > 0 {
				key := extractAPIKey(stored)
				pName := extractProvider(stored, req.Provider)

				// If the file explicitly mentions zen, or if the host identifies it as zen,
				// or if it matches any configured api-key. Filename/ID only need the
				// provider prefix so zen.json, zen1.json, zen-abc.json all match.
				isZen := strings.EqualFold(pName, cfg.Provider) ||
					strings.HasPrefix(strings.ToLower(req.FileName), cfg.Provider) ||
					strings.HasPrefix(strings.ToLower(req.ID), cfg.Provider)

				if !isZen && key != "" {
					for _, k := range cfg.APIKeys {
						if k == key {
							isZen = true
							break
						}
					}
				}

				if isZen && key != "" {
					// Remember the key so model discovery can query /models, and
					// refresh the discovered free-tier list in the background.
					rememberAuthKey(key)
					go refreshDiscoveredModels(authBaseURL(stored, cfg), key)
					id := sha256Prefix(key)
					fileName := req.FileName
					if fileName == "" {
						fileName = fmt.Sprintf("%s-%s.json", cfg.Provider, id)
					}
					authID := req.ID
					if authID == "" {
						authID = fmt.Sprintf("%s-%s", cfg.Provider, id)
					}
					modelsList := extractAuthModels(stored)
					return okEnvelopeJSON(authParseResponse{
						Handled: true,
						Auth: authData{
							Provider:    cfg.Provider,
							ID:          authID,
							FileName:    fileName,
							Label:       authLabel(cfg.Provider, id),
							StorageJSON: raw,
							Models:      modelsList,
							Metadata:    map[string]any{"type": cfg.Provider},
						},
					})
				}
			}
		}
	}

	// Fallback for virtual auths from plugin config (if any configured)
	if len(cfg.APIKeys) == 0 {
		return okEnvelopeJSON(authParseResponse{Handled: false})
	}
	auths := make([]authData, 0, len(cfg.APIKeys))
	for _, key := range cfg.APIKeys {
		if key == "" {
			continue
		}
		storageJSON, _ := json.Marshal(map[string]string{
			"provider": cfg.Provider,
			"type":     cfg.Provider,
			"api_key":  key,
		})
		id := sha256Prefix(key)
		auths = append(auths, authData{
			Provider:    cfg.Provider,
			ID:          fmt.Sprintf("%s-%s", cfg.Provider, id),
			FileName:    fmt.Sprintf("%s-%s.json", cfg.Provider, id),
			Label:       authLabel(cfg.Provider, id),
			StorageJSON: storageJSON,
			Metadata:    map[string]any{"type": cfg.Provider},
		})
	}
	return okEnvelopeJSON(authParseResponse{Handled: true, Auths: auths})
}

func extractAPIKey(stored map[string]any) string {
	for _, field := range []string{"api_key", "api-key", "key", "token", "access_token"} {
		if v, ok := stored[field].(string); ok && strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

func extractProvider(stored map[string]any, hostProvider string) string {
	if v, ok := stored["provider"].(string); ok && strings.TrimSpace(v) != "" {
		return strings.TrimSpace(v)
	}
	if v, ok := stored["type"].(string); ok && strings.TrimSpace(v) != "" {
		return strings.TrimSpace(v)
	}
	return strings.TrimSpace(hostProvider)
}

func extractAuthModels(stored map[string]any) []any {
	raw, ok := stored["models"]
	if !ok {
		return nil
	}
	switch t := raw.(type) {
	case []any:
		return t
	default:
		return nil
	}
}

func authLabel(provider, id string) string {
	if provider == "zen" {
		return fmt.Sprintf("OpenCode Zen (%s…)", id)
	}
	return fmt.Sprintf("%s (%s…)", provider, id)
}

type authParseResponse struct {
	Handled bool       `json:"Handled"`
	Auth    authData   `json:"Auth,omitempty"`
	Auths   []authData `json:"Auths,omitempty"`
}

type authData struct {
	Provider         string          `json:"Provider"`
	ID               string          `json:"ID"`
	FileName         string          `json:"FileName"`
	Label            string          `json:"Label"`
	Prefix           string          `json:"Prefix"`
	ProxyURL         string          `json:"ProxyURL"`
	Disabled         bool            `json:"Disabled"`
	StorageJSON      json.RawMessage `json:"StorageJSON"`
	Models           []any           `json:"Models,omitempty"`
	Metadata         map[string]any  `json:"Metadata"`
	Attributes       map[string]any  `json:"Attributes"`
	NextRefreshAfter string          `json:"NextRefreshAfter"`
}

func sha256Prefix(key string) string {
	sum := sha256.Sum256([]byte(key))
	return hex.EncodeToString(sum[:4])
}

// ---------------------------------------------------------------------------
// model registration
// ---------------------------------------------------------------------------

// defaultZenModels announces the free-tier Zen models supported out of the box.
var defaultZenModels = []modelRoute{
	{Model: "mimo-v2.6-flash-free", Alias: "mimo-v2.6-flash-free", Endpoint: "chat"},
	{Model: "mimo-v2.5-free", Alias: "mimo-v2.5-free", Endpoint: "chat"},
	{Model: "ling-3.0-flash-fin-free", Alias: "ling-3.0-flash-fin-free", Endpoint: "chat"},
	{Model: "nemotron-3-ultra-free", Alias: "nemotron-3-ultra-free", Endpoint: "chat"},
	{Model: "muse-spark-1.3-contributor-free", Alias: "muse-spark-1.3-contributor-free", Endpoint: "responses"},
}

// defaultModelCapabilities announces built-in capability metadata for the
// Zen free-tier models. Values are aligned with the OpenCode client's own
// model catalog (models.dev, "opencode" provider — the catalog OpenCode
// itself consumes), snapshotted 2026-10-01:
//   - context window and max output tokens from limit.context / limit.output;
//   - reasoning effort levels from reasoning_options (effort-type models);
//     models with toggle-only thinking carry no levels;
//   - input modalities capped at text+image (audio/video parts are not
//     translated by this plugin);
//   - display names from the catalog.
//
// Values can be overridden per model through plugins.configs.zen models
// entries (context-length, max-completion-tokens, input-modalities,
// reasoning-levels).
var defaultModelCapabilities = map[string]modelCapabilities{
	"mimo-v2.6-flash-free":            {ContextLength: 200000, MaxCompletionTokens: 32000, InputModalities: []string{"text", "image"}, DisplayName: "MiMo-V2.6-Flash Free"},
	"mimo-v2.5-free":                  {ContextLength: 200000, MaxCompletionTokens: 32000, InputModalities: []string{"text", "image"}, DisplayName: "MiMo V2.5 Free"},
	"ling-3.0-flash-fin-free":         {ContextLength: 262144, MaxCompletionTokens: 32768, DisplayName: "Ling 3.0 Flash Fin Free"},
	"nemotron-3-ultra-free":           {ContextLength: 1000000, MaxCompletionTokens: 128000, DisplayName: "Nemotron 3 Ultra Free"},
	"nemotron-3.5-lightning-free":     {ContextLength: 262144, MaxCompletionTokens: 262144, DisplayName: "Nemotron 3.5 Lightning Free"},
	"muse-spark-1.3-contributor-free": {ContextLength: 1048576, MaxCompletionTokens: 131072, InputModalities: []string{"text", "image"}, ReasoningLevels: []string{"minimal", "low", "medium", "high", "xhigh"}, DisplayName: "Muse Spark 1.3 Free"},
	"muse-spark-1.2-contributor-free": {ContextLength: 1048576, MaxCompletionTokens: 131072, InputModalities: []string{"text", "image"}, ReasoningLevels: []string{"minimal", "low", "medium", "high", "xhigh"}, DisplayName: "Muse Spark 1.2 Free"},
	"longcat-2.5-preview-free":        {ContextLength: 1000000, MaxCompletionTokens: 131072, InputModalities: []string{"text", "image"}, DisplayName: "LongCat 2.5 Preview Free"},
	"space-bunny-free":                {ContextLength: 1048576, MaxCompletionTokens: 524288, InputModalities: []string{"text", "image"}, ReasoningLevels: []string{"low", "medium", "high", "xhigh", "max"}, DisplayName: "Space Bunny Free"},
	"big-pickle":                      {ContextLength: 200000, MaxCompletionTokens: 32000, DisplayName: "Big Pickle"},
	"deepseek-v4-flash-free":          {ContextLength: 200000, MaxCompletionTokens: 128000, ReasoningLevels: []string{"low", "high", "max"}, DisplayName: "DeepSeek V4 Flash Free"},
}

// zenModelSpec looks up the built-in capability table for a model.
func zenModelSpec(model string) (modelCapabilities, bool) {
	caps, ok := defaultModelCapabilities[strings.ToLower(strings.TrimSpace(model))]
	return caps, ok
}

// resolveModelCapabilities merges built-in defaults with explicit route
// configuration; explicit values win. Known models always announce thinking
// support (all free-tier models reason); effort-type models list their levels.
func resolveModelCapabilities(m modelRoute) modelCapabilities {
	caps, _ := zenModelSpec(m.Model)
	if m.ContextLength > 0 {
		caps.ContextLength = m.ContextLength
	}
	if m.MaxCompletionTokens > 0 {
		caps.MaxCompletionTokens = m.MaxCompletionTokens
	}
	if len(m.InputModalities) > 0 {
		caps.InputModalities = m.InputModalities
	}
	if len(m.ReasoningLevels) > 0 {
		caps.ReasoningLevels = m.ReasoningLevels
	}
	if _, known := zenModelSpec(m.Model); known || len(m.ReasoningLevels) > 0 {
		caps.Thinking = &thinkingSupport{
			ZeroAllowed:    true,
			DynamicAllowed: true,
			Levels:         caps.ReasoningLevels,
		}
	}
	return caps
}

// discoveredZenModels returns the cached free-tier model IDs fetched from
// the zen /models endpoint, refreshing synchronously (at most once per TTL)
// when a key is known. Discovery failures silently fall back to the cache.
var discoveredModels struct {
	mu        sync.RWMutex
	ids       []string
	fetchedAt time.Time
}

// seenAuthKeys accumulates zen API keys observed by auth.parse; discovery
// needs one to query /models.
var seenAuthKeys struct {
	mu   sync.Mutex
	keys []string
}

func rememberAuthKey(key string) {
	if strings.TrimSpace(key) == "" {
		return
	}
	seenAuthKeys.mu.Lock()
	defer seenAuthKeys.mu.Unlock()
	for _, k := range seenAuthKeys.keys {
		if k == key {
			return
		}
	}
	seenAuthKeys.keys = append(seenAuthKeys.keys, key)
}

func anyKnownKey(cfg pluginConfig) string {
	if len(cfg.APIKeys) > 0 {
		return cfg.APIKeys[0]
	}
	seenAuthKeys.mu.Lock()
	defer seenAuthKeys.mu.Unlock()
	if len(seenAuthKeys.keys) > 0 {
		return seenAuthKeys.keys[0]
	}
	return ""
}

// authBaseURL picks the base URL for discovery: the auth file's custom
// service address if present, else plugin config, else the zen default.
func authBaseURL(stored map[string]any, cfg pluginConfig) string {
	for _, field := range []string{"base_url", "url"} {
		if v, ok := stored[field].(string); ok {
			if u := strings.TrimRight(strings.TrimSpace(v), "/"); u != "" {
				return u
			}
		}
	}
	if cfg.BaseURL != "" {
		return cfg.BaseURL
	}
	return zenBaseURL
}

// fetchZenModels lists free-tier model IDs from the zen /models endpoint.
// ponytail: plain http.Client instead of host.http.do — model.register has no
// host callback context; add host-routed discovery if proxy support matters.
func fetchZenModels(baseURL, apiKey string) ([]string, error) {
	client := &http.Client{Timeout: 10 * time.Second}
	req, err := http.NewRequest(http.MethodGet, baseURL+"/models", nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("zen /models status %d: %s", resp.StatusCode, truncate(body, 256))
	}
	var list struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &list); err != nil {
		return nil, err
	}
	ids := make([]string, 0, len(list.Data))
	for _, m := range list.Data {
		if id := strings.TrimSpace(m.ID); id != "" && strings.HasSuffix(strings.ToLower(id), "-free") {
			ids = append(ids, id)
		}
	}
	return ids, nil
}

// refreshDiscoveredModels updates the discovery cache (best effort).
func refreshDiscoveredModels(baseURL, apiKey string) {
	ids, err := fetchZenModels(baseURL, apiKey)
	if err != nil {
		return
	}
	discoveredModels.mu.Lock()
	discoveredModels.ids, discoveredModels.fetchedAt = ids, time.Now()
	discoveredModels.mu.Unlock()
}

// discoveredZenModels returns the cached free-tier IDs, refreshing the cache
// synchronously at most once per TTL when a key is known.
func discoveredZenModels(cfg pluginConfig) []string {
	discoveredModels.mu.RLock()
	fresh := time.Since(discoveredModels.fetchedAt) < 10*time.Minute
	ids := discoveredModels.ids
	discoveredModels.mu.RUnlock()
	if fresh {
		return ids
	}
	key := anyKnownKey(cfg)
	if key == "" {
		return ids
	}
	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = zenBaseURL
	}
	refreshDiscoveredModels(baseURL, key)
	discoveredModels.mu.RLock()
	defer discoveredModels.mu.RUnlock()
	return discoveredModels.ids
}

// modelExcluded reports whether a discovered model ID is filtered out by
// the exclude-models plugin configuration (case-insensitive).
func modelExcluded(cfg pluginConfig, model string) bool {
	for _, ex := range cfg.ExcludeModels {
		if strings.EqualFold(strings.TrimSpace(ex), model) {
			return true
		}
	}
	return false
}

// modelRegistration announces supported Zen models to CPA.
// If the user configured custom models in plugins.configs.zen, it announces those;
// otherwise it announces the default set of Zen free-tier models plus any
// free-tier models discovered live from the zen /models endpoint.
func modelRegistration() ([]byte, error) {
	cfg := loadedConfig()
	declared := cfg.Models
	if len(declared) == 0 {
		declared = defaultZenModels
		for _, id := range discoveredZenModels(cfg) {
			if modelExcluded(cfg, id) {
				continue
			}
			dup := false
			for _, m := range declared {
				if strings.EqualFold(m.Model, id) {
					dup = true
					break
				}
			}
			if !dup {
				// Endpoint is a hint: the official endpoint table when we know
				// the model, a name-based guess otherwise; execute()/runStream()
				// retry the dialects anyway. Models on dialects this plugin
				// cannot speak are not announced at all — selecting them could
				// never work.
				if ep, known := officialZenEndpoint(id); known {
					if ep == endpointUnsupported {
						continue
					}
					declared = append(declared, modelRoute{Model: id, Alias: id, Endpoint: ep})
					continue
				}
				declared = append(declared, modelRoute{Model: id, Alias: id, Endpoint: inferEndpoint(id)})
			}
		}
	}
	models := make([]modelInfo, 0, len(declared))
	for _, m := range declared {
		if m.Model == "" {
			continue
		}
		// configure() and defaultZenModels guarantee Alias is non-empty.
		alias := m.Alias
		caps := resolveModelCapabilities(m)
		displayName := caps.DisplayName
		if displayName == "" {
			displayName = m.Model
		}
		models = append(models, modelInfo{
			ID:                       alias,
			Object:                   "model",
			Created:                  1735689600,
			OwnedBy:                  cfg.Provider,
			Type:                     "openai",
			DisplayName:              displayName,
			ContextLength:            caps.ContextLength,
			MaxCompletionTokens:      caps.MaxCompletionTokens,
			SupportedInputModalities: caps.InputModalities,
			Thinking:                 caps.Thinking,
		})
	}
	return okEnvelopeJSON(modelRegistrationResponse{
		Provider: cfg.Provider,
		Models:   models,
	})
}

// ---------------------------------------------------------------------------
// execution
// ---------------------------------------------------------------------------

// execute implements executor.execute / executor.execute_stream.
//
// The host translates the client request into one of the declared input
// formats (chat-completions or responses) before calling us, and translates
// our output back to the client protocol. We:
//
//  1. resolve the model route (endpoint split: muse-spark → /responses,
//     chat-only free models → /chat/completions)
//  2. apply the zen free-tier gate (canonical ses_/msg_ ids, identity
//     headers, bash/read tools, stream:true)
//  3. POST to zen via the host HTTP client (proxy + request-log aware)
//  4. stream SSE chunks back through the plugin stream bridge, or fold the
//     whole SSE answer into one JSON payload for non-streaming clients.
func execute(payload []byte, stream bool) ([]byte, error) {
	var req executorRequest
	// Some clients (and hosts that forward verbatim) prepend a UTF-8 BOM;
	// strip it so the envelope parses.
	payload = trimJSONEnvelope(payload)
	if len(payload) > 0 {
		if err := json.Unmarshal(payload, &req); err != nil {
			return nil, err
		}
	}
	cfg := loadedConfig()
	if !cfg.Enabled {
		return nil, fmt.Errorf("zen provider plugin is disabled")
	}
	route, ok := routeForModel(cfg, req.Model)
	if !ok {
		return nil, fmt.Errorf("zen provider has no model %q", req.Model)
	}
	if !routeConfigured(cfg, route.Model) {
		if ep, known := officialZenEndpoint(route.Model); known && ep == endpointUnsupported {
			return nil, fmt.Errorf("zen model %q is served on a dialect this plugin cannot speak (Anthropic Messages / systemone / AI-SDK path); set an explicit models endpoint in plugins.configs.zen to override", route.Model)
		}
	}

	apiKey := apiKeyForRequest(req, cfg)
	if apiKey == "" {
		return nil, fmt.Errorf("zen provider has no api-key for request (please configure in AI Provider panel or plugins.configs.zen)")
	}

	baseURL := baseURLForRequest(req, cfg)
	headers := gateHeaders(req)

	if !stream {
		return executeNonStream(req, cfg, route, baseURL, headers, apiKey)
	}

	streamID := strings.TrimSpace(req.StreamID)
	if streamID == "" {
		return nil, fmt.Errorf("stream_id is required for executor.execute_stream")
	}
	go runStream(req, cfg, route, baseURL, headers, apiKey, streamID)
	return okEnvelopeJSON(executorStreamResponse{
		Headers: map[string][]string{"Content-Type": {"text/event-stream"}},
	})
}

// executeNonStream streams from zen and folds the SSE answer into one JSON
// payload. Without an explicitly configured endpoint, a path-level failure
// (400/404/405) on the first candidate dialect is retried once against the
// opposite endpoint and the winner is remembered.
func executeNonStream(req executorRequest, cfg pluginConfig, route modelRoute, baseURL string, headers map[string][]string, apiKey string) ([]byte, error) {
	endpoints := candidateEndpoints(cfg, route)
	var lastErr error
	authRetry := false
	// transportFailed marks errors a fresh connection may fix (host HTTP
	// failures such as connection resets / unexpected EOF); status errors
	// and payload-shape errors never are.
	transportFailed := false
	run := func(key string, hdrs map[string][]string) ([]byte, error) {
		transportFailed = false
		for _, ep := range endpoints {
			r := route
			r.Endpoint = ep
			// Gate rule 4: zen only answers streaming requests; the SSE answer
			// is folded below.
			upstreamBody, err := prepareUpstreamBody(req, r)
			if err != nil {
				return nil, err
			}
			body, respHeaders, status, err := doUpstream(req.HostCallbackID, http.MethodPost, baseURL+r.EndpointPath(), hdrs, key, upstreamBody)
			if err != nil {
				transportFailed = true
				return nil, err
			}
			if status < 200 || status >= 300 {
				lastErr = fmt.Errorf("zen upstream status %d: %s", status, truncate(body, 512))
				if isEndpointMismatch(status) && ep == endpoints[0] && len(endpoints) > 1 {
					continue // model may only be served on the other endpoint
				}
				// Auth-level failures (401 credits/auth, 403 region/policy):
				// retried once by the caller with a rotated key and a fresh
				// session.
				authRetry = status == 401 || status == 403
				return nil, lastErr
			}
			if ep != endpoints[0] {
				rememberEndpoint(route.Model, ep)
			}
			folded, err := foldSSEToJSON(body, ep)
			if err != nil {
				return nil, err
			}
			if ep == "responses" {
				// The plugin declares chat-completions output, so the folded
				// responses object must become a chat.completion before the host
				// translates it for the requesting client.
				folded, err = responsesCompletionToChat(folded, req.Model)
				if err != nil {
					return nil, err
				}
			}
			return okEnvelopeJSON(executorResponse{Payload: folded, Headers: respHeaders})
		}
		return nil, lastErr
	}
	out, err := run(apiKey, headers)
	if err != nil && authRetry {
		apiKey = nextAPIKey(cfg)
		refreshGateSession(headers)
		out, err = run(apiKey, headers)
	}
	// Transport-level failures (connection reset, unexpected EOF) are
	// transient: retry the whole fold with backoff so a broken connection
	// never surfaces as a failed turn.
	for i := 0; err != nil && transportFailed && i < cfg.StreamRetries; i++ {
		time.Sleep(streamRetryDelay(i + 1))
		out, err = run(apiKey, headers)
	}
	// Empty turn: the upstream completed but delivered no content and no
	// tool call. Nothing reached the client, so replaying once is safe.
	if err == nil && cfg.StreamRetries > 0 && isEmptyCompletion(out) {
		time.Sleep(streamRetryDelay(1))
		replay, replayErr := run(apiKey, headers)
		if replayErr == nil {
			out = replay
		}
	}
	return out, err
}

// isEmptyCompletion reports whether a folded chat.completion carries neither
// text nor tool calls: an empty turn (reasoning-only or abandoned).
func isEmptyCompletion(folded []byte) bool {
	var root struct {
		Choices []struct {
			Message *struct {
				Content   string `json:"content"`
				ToolCalls []any  `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(folded, &root); err != nil {
		return false
	}
	for _, c := range root.Choices {
		if c.Message == nil {
			continue
		}
		if strings.TrimSpace(c.Message.Content) != "" || len(c.Message.ToolCalls) > 0 {
			return false
		}
	}
	return len(root.Choices) > 0
}

// baseURLForRequest resolves the upstream base URL. It checks the host-selected
// auth record (StorageJSON) first for any custom service address, falling back to
// plugin config or the default https://opencode.ai/zen/v1.
func baseURLForRequest(req executorRequest, cfg pluginConfig) string {
	if len(req.StorageJSON) > 0 {
		var stored struct {
			BaseURL string `json:"base_url"`
			URL     string `json:"url"`
		}
		if err := json.Unmarshal(req.StorageJSON, &stored); err == nil {
			u := strings.TrimRight(strings.TrimSpace(stored.BaseURL), "/")
			if u == "" {
				u = strings.TrimRight(strings.TrimSpace(stored.URL), "/")
			}
			if u != "" {
				return u
			}
		}
	}
	if cfg.BaseURL != "" {
		return cfg.BaseURL
	}
	return zenBaseURL
}

// apiKeyForRequest resolves the zen API key for an executor request. It
// prefers the key carried by the host-selected auth record (StorageJSON) and
// falls back to the plugin-config rotation when the host provides none.
func apiKeyForRequest(req executorRequest, cfg pluginConfig) string {
	if len(req.StorageJSON) > 0 {
		var stored struct {
			APIKey string `json:"api_key"`
			Key    string `json:"key"`
		}
		if err := json.Unmarshal(req.StorageJSON, &stored); err == nil {
			if k := strings.TrimSpace(stored.APIKey); k != "" {
				return k
			}
			if k := strings.TrimSpace(stored.Key); k != "" {
				return k
			}
		}
	}
	return nextAPIKey(cfg)
}

// ---------------------------------------------------------------------------
// stream resume (automatic reconnect of a broken upstream stream)
// ---------------------------------------------------------------------------

// streamResumeMaxPartialBytes caps the partial assistant text replayed into
// a continuation prompt; the tail is what matters for resuming mid-answer.
const streamResumeMaxPartialBytes = 8000

// streamResumeInstruction is the user turn appended after the partial
// assistant answer when resuming a stream that broke mid-answer.
const streamResumeInstruction = "Your previous response was interrupted mid-stream by a network error. " +
	"Continue the unfinished task exactly from where you stopped, resuming mid-sentence if needed. " +
	"Do not repeat, rephrase, or summarize anything you already output, and do not add any preamble, " +
	"apology, or meta-commentary about the interruption."

// upstreamIdleTimeout gives an upstream that connects but then sends nothing
// (or stalls mid-generation far longer than any muse pause) before the
// plugin abandons the stream and reconnects.
const upstreamIdleTimeout = 60 * time.Second

// streamRetryDelay is the backoff before reconnect attempt n (1-based).
func streamRetryDelay(n int) time.Duration {
	d := time.Second << (n - 1)
	if d > 8*time.Second {
		d = 8 * time.Second
	}
	return d
}

// streamRetryableStatus reports whether an upstream HTTP status may succeed
// on a fresh connection (server-side or transient gateway trouble). Auth,
// quota, and request-shape errors never are.
func streamRetryableStatus(status int) bool {
	return status >= 500 || status == 408
}

// streamResumeMonitor records what the downstream client has already seen of
// the in-flight answer so a broken upstream stream can be resumed instead of
// silently truncated: whether the terminal event was forwarded, whether any
// tool-call frame was forwarded (a half-sent tool call cannot be resumed),
// and the assistant text emitted so far.
type streamResumeMonitor struct {
	sawFinish   bool
	sawToolCall bool
	partial     strings.Builder
}

// observeChatFrame inspects one outbound chat-completions chunk frame (both
// the chat passthrough and the responses->chat converter emit this shape).
func (m *streamResumeMonitor) observeChatFrame(frame []byte) {
	payload := strings.TrimSpace(stripSSEDataFraming(frame))
	if payload == "" || isDoneTerminator(payload) {
		return
	}
	var chunk struct {
		Choices []struct {
			Delta struct {
				Content   string            `json:"content"`
				ToolCalls []json.RawMessage `json:"tool_calls"`
			} `json:"delta"`
			FinishReason *string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
		return
	}
	for _, choice := range chunk.Choices {
		if choice.FinishReason != nil && *choice.FinishReason != "" {
			m.sawFinish = true
		}
		if len(choice.Delta.ToolCalls) > 0 {
			m.sawToolCall = true
		}
		if choice.Delta.Content != "" {
			m.partial.WriteString(choice.Delta.Content)
			// Keep only the tail once the answer outgrows the cap so a
			// very long stream cannot grow the resume prompt unbounded.
			if m.partial.Len() > 2*streamResumeMaxPartialBytes {
				kept := m.partial.String()
				m.partial.Reset()
				m.partial.WriteString(streamResumeElidedMarker + resumeTailString(kept, streamResumeMaxPartialBytes))
			}
		}
	}
}

// streamResumeElidedMarker prefixes a tail-only partial answer.
const streamResumeElidedMarker = "[... earlier output omitted; continue from the text below ...]\n"

// resumeTailString keeps the last n bytes of s on a UTF-8 rune boundary.
func resumeTailString(s string, n int) string {
	if len(s) <= n {
		return s
	}
	tail := s[len(s)-n:]
	for len(tail) > 0 && !utf8.RuneStart(tail[0]) {
		tail = tail[1:]
	}
	return tail
}

// trimJSONEnvelope strips a UTF-8 BOM and surrounding whitespace from an
// incoming request body. Clients that emit a BOM (and hosts that forward it
// verbatim) otherwise present a body that is not valid JSON.
func trimJSONEnvelope(b []byte) []byte {
	if len(b) >= 3 && b[0] == 0xEF && b[1] == 0xBB && b[2] == 0xBF {
		b = b[3:]
	}
	return []byte(strings.TrimSpace(string(b)))
}

// partialText returns the assistant text emitted downstream so far.
func (m *streamResumeMonitor) partialText() string {
	return strings.TrimSpace(m.partial.String())
}

// anyOutput reports whether anything reached the client at all. A stream
// that completes with neither text nor a tool call produced an empty turn
// (reasoning-only or silently abandoned); runStream reconnects once because
// nothing was delivered, so the replay cannot duplicate content.
func (m *streamResumeMonitor) anyOutput() bool {
	return m.sawToolCall || strings.TrimSpace(m.partial.String()) != ""
}

// danglingLeadIn reports whether the emitted text ends with a lead-in marker
// (a colon or comma, or an unclosed code fence): the model announced a next
// step but the stream stopped before producing it.
func (m *streamResumeMonitor) danglingLeadIn() bool {
	s := strings.TrimSpace(m.partial.String())
	if s == "" {
		return false
	}
	if strings.Count(s, "```")%2 == 1 {
		return true
	}
	runes := []rune(s)
	switch runes[len(runes)-1] {
	case '：', ':', '，', ',', '、', '（', '(', '【':
		return true
	}
	return false
}

// appendContinuationPayload returns the request payload extended with the
// partial answer (as an assistant turn) and the resume instruction (as a
// user turn), so the reconnected upstream continues the broken stream
// instead of restarting the answer. The payload's own dialect is preserved;
// prepareUpstreamBody converts it to the route endpoint as usual. An empty
// partial yields the payload unchanged (transparent replay).
func appendContinuationPayload(payload []byte, partial string) ([]byte, error) {
	trimmed := strings.TrimSpace(string(payload))
	if trimmed == "" || trimmed[0] != '{' {
		return nil, fmt.Errorf("zen stream resume: unsupported payload shape")
	}
	var root map[string]any
	if err := json.Unmarshal(payload, &root); err != nil {
		return nil, fmt.Errorf("zen stream resume: payload is not valid JSON: %w", err)
	}
	partial = strings.TrimSpace(partial)
	if partial == "" {
		// Nothing was forwarded downstream yet: replay the request as-is.
		return payload, nil
	}
	if len(partial) > streamResumeMaxPartialBytes {
		partial = streamResumeElidedMarker + resumeTailString(partial, streamResumeMaxPartialBytes)
	}
	switch items := root["messages"].(type) {
	case []any:
		if partial != "" {
			items = append(items, map[string]any{"role": "assistant", "content": partial})
		}
		items = append(items, map[string]any{"role": "user", "content": streamResumeInstruction})
		root["messages"] = items
	default:
		switch input := root["input"].(type) {
		case []any:
			if partial != "" {
				input = append(input, map[string]any{
					"type":    "message",
					"role":    "assistant",
					"content": []any{map[string]any{"type": "output_text", "text": partial}},
				})
			}
			input = append(input, map[string]any{
				"type":    "message",
				"role":    "user",
				"content": []any{map[string]any{"type": "input_text", "text": streamResumeInstruction}},
			})
			root["input"] = input
		case string:
			joined := input
			if partial != "" {
				joined += "\n\n[assistant, interrupted]\n" + partial
			}
			joined += "\n\n[user]\n" + streamResumeInstruction
			root["input"] = joined
		default:
			return nil, fmt.Errorf("zen stream resume: payload has no messages/input to extend")
		}
	}
	out, err := json.Marshal(root)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// runStream performs the upstream call and forwards SSE frames through the
// host stream bridge until done, then closes the plugin stream.
func runStream(req executorRequest, cfg pluginConfig, route modelRoute, baseURL string, headers map[string][]string, apiKey string, streamID string) {
	closeStream := func(errMsg string) {
		_, _ = callHost("host.stream.close", map[string]any{"stream_id": streamID, "error": strings.TrimSpace(errMsg)})
	}
	defer func() {
		if recovered := recover(); recovered != nil {
			closeStream(fmt.Sprintf("zen stream panic: %v", recovered))
		}
	}()

	endpoints := candidateEndpoints(cfg, route)
	maxRetries := cfg.StreamRetries

	drainHostStream := func(id string) {
		for i := 0; i < 8; i++ {
			chunk, errRead := readHostStream(id)
			if errRead != nil || chunk.Done || chunk.Error != "" {
				break
			}
		}
	}

	// The upstream payload is rebuilt on every attempt: a resumed stream
	// carries the partial answer plus a continuation instruction.
	payload := append([]byte(nil), req.Payload...)
	if len(payload) == 0 {
		payload = append([]byte(nil), req.OriginalRequest...)
	}

	// Endpoint auto-detection: without an explicitly configured endpoint, a
	// path-level failure (400/404/405) from the first candidate is retried
	// once against the opposite dialect and the winner is remembered. The
	// bool result separates retryable transport errors (fresh connection may
	// succeed) from payload-shape errors (never succeed).
	selectUpstream := func(key string, hdrs map[string][]string) (*hostStreamHandle, string, string, bool) {
		rreq := req
		rreq.Payload = payload
		var handle *hostStreamHandle
		used := ""
		for _, ep := range endpoints {
			r := route
			r.Endpoint = ep
			// Gate rule 4: zen only answers streaming requests.
			body, err := prepareUpstreamBody(rreq, r)
			if err != nil {
				return nil, "", err.Error(), false
			}
			h, err := doUpstreamStream(rreq.HostCallbackID, http.MethodPost, baseURL+r.EndpointPath(), hdrs, key, body)
			if err != nil {
				return nil, "", err.Error(), true
			}
			handle = h
			used = ep
			if isEndpointMismatch(h.StatusCode) && ep == endpoints[0] && len(endpoints) > 1 {
				// Drain the error body so the abandoned host stream is reaped.
				drainHostStream(h.StreamID)
				continue
			}
			break
		}
		return handle, used, "", true
	}

	key := apiKey
	// connectOne opens an upstream stream, rotating the key and minting a
	// fresh session once on auth-level failure (401 credits/auth, 403
	// region/policy). It returns the final handle or a classified error.
	connectOne := func() (*hostStreamHandle, string, string, bool) {
		h, used, errMsg, retryable := selectUpstream(key, headers)
		if errMsg != "" {
			return nil, "", errMsg, retryable
		}
		if h == nil {
			return nil, "", "zen upstream: no endpoint attempted", false
		}
		if h.StatusCode == 401 || h.StatusCode == 403 {
			drainHostStream(h.StreamID)
			key = nextAPIKey(cfg)
			refreshGateSession(headers)
			h, used, errMsg, retryable = selectUpstream(key, headers)
			if errMsg != "" {
				return nil, "", errMsg, retryable
			}
			if h == nil {
				return nil, "", "zen upstream: no endpoint attempted", false
			}
		}
		if used != "" && used != endpoints[0] {
			rememberEndpoint(route.Model, used)
		}
		return h, used, "", true
	}

	// Chat-completions clients (SourceFormat "openai" / "chat-completions") get
	// their SSE framing applied by the host itself: when outputFormat equals the
	// requested format the host passes plugin chunks through verbatim and the
	// handler writes the "data: " prefix, so pre-framed lines would double it.
	// Responses/claude/gemini clients go through TranslateStream, whose
	// translators expect "data:" framed input and strip the prefix themselves.
	src := strings.ToLower(strings.TrimSpace(req.SourceFormat))
	rawData := src == "openai" || strings.Contains(src, "chat")

	// The monitor watches every outbound chat frame: both the chat
	// passthrough and the responses->chat converter emit chat chunks, so
	// terminal detection (finish_reason) and partial-text capture are
	// dialect-uniform. Heartbeats bypass the monitor on purpose.
	monitor := &streamResumeMonitor{}
	const heartbeatInterval = 10 * time.Second
	lastEmit := time.Now()

	emitChat := func(frame []byte) error {
		monitor.observeChatFrame(frame)
		lastEmit = time.Now()
		return emitStreamFrame(streamID, frame)
	}
	emit := emitChat
	if route.Endpoint == "responses" {
		// The plugin declares chat-completions output, so the host expects chat
		// chunks; upstream /responses events must be converted first.
		conv := newResponsesChatStreamConverter(req.Model)
		emit = func(frame []byte) error {
			upstream := stripSSEDataFraming(frame)
			outs, err := conv.convert([]byte(upstream))
			if err != nil {
				return err
			}
			for _, out := range outs {
				framed := out
				if !rawData {
					framed = []byte("data: " + string(out) + "\n\n")
				}
				if err := emitChat(framed); err != nil {
					return err
				}
			}
			return nil
		}
	}

	type pumpOutcome struct {
		terminal  bool
		retryable bool
		empty     bool // ended cleanly but delivered nothing to the client
		errMsg    string
	}

	// pump forwards one upstream stream until it ends. A stream that stops
	// without the terminal event (finish_reason forwarded or [DONE] seen) is
	// a truncation, not a completion: it is reported as retryable so the
	// caller can reconnect and resume mid-answer.
	pump := func(h *hostStreamHandle) pumpOutcome {
		reassembler := &sseReassembler{
			rawData: rawData,
			emit:    emit,
		}
		terminal := func() bool { return monitor.sawFinish || reassembler.sawDone }
		// completed reports a clean end. A clean end that delivered nothing
		// (reasoning-only turn, or an upstream that stopped before emitting)
		// is an empty turn: the caller replays it once, which cannot
		// duplicate content because the client received none.
		completed := func() pumpOutcome {
			if !monitor.anyOutput() {
				return pumpOutcome{empty: true, retryable: true, errMsg: "zen upstream completed without producing any output"}
			}
			if !monitor.sawToolCall && monitor.danglingLeadIn() {
				// The answer stops right after a lead-in ("...：" / "```"):
				// the model announced the next step but the stream ended
				// before producing it. Resume instead of accepting a
				// half-finished turn. The wording includes "stream
				// interrupted" so client-side guards that auto-continue on
				// interrupted streams also engage.
				return pumpOutcome{retryable: true, errMsg: "stream interrupted: zen upstream stopped mid-plan after a dangling lead-in"}
			}
			return pumpOutcome{terminal: true}
		}
		truncated := func() pumpOutcome {
			// Emit the tail event when its data lines were complete; a
			// fragment cut mid-line would corrupt the downstream stream.
			if reassembler.lineBuf.Len() == 0 {
				_ = reassembler.flushPending()
			}
			if terminal() {
				return completed()
			}
			return pumpOutcome{
				retryable: true,
				errMsg:    "zen upstream stream ended before the finish marker",
			}
		}
		type upstreamChunk struct {
			chunk httpStreamChunk
			err   error
		}
		chunks := make(chan upstreamChunk, 8)
		readerDone := make(chan struct{})
		defer close(readerDone)
		go func() {
			defer close(chunks)
			for {
				chunk, err := readHostStream(h.StreamID)
				if err != nil {
					select {
					case chunks <- upstreamChunk{err: err}:
					case <-readerDone:
					}
					return
				}
				select {
				case chunks <- upstreamChunk{chunk: chunk}:
				case <-readerDone:
					return
				}
				if chunk.Done || chunk.Error != "" {
					return
				}
			}
		}()
		ticker := time.NewTicker(heartbeatInterval)
		defer ticker.Stop()
		lastRead := time.Now()
		for {
			select {
			case sc, ok := <-chunks:
				if !ok {
					return truncated()
				}
				lastRead = time.Now()
				if sc.err != nil {
					if terminal() {
						// The answer was complete; the trailing read error
						// only means the body closed uncleanly.
						return completed()
					}
					return pumpOutcome{retryable: true, errMsg: sc.err.Error()}
				}
				if sc.chunk.Error != "" {
					if terminal() {
						return completed()
					}
					return pumpOutcome{retryable: true, errMsg: sc.chunk.Error}
				}
				// Process the payload before honoring Done: the host bridge may
				// deliver the final body bytes together with the done marker in one
				// chunk, and dropping them would truncate the tail of the answer.
				if len(sc.chunk.Payload) > 0 {
					if err := reassembler.write(sc.chunk.Payload); err != nil {
						// Downstream emit failure or an upstream-declared
						// response failure: reconnecting cannot help.
						return pumpOutcome{errMsg: err.Error()}
					}
				}
				if sc.chunk.Done {
					return truncated()
				}
			case <-ticker.C:
				if time.Since(lastRead) > upstreamIdleTimeout {
					return pumpOutcome{
						retryable: true,
						errMsg:    "zen upstream stalled: no bytes for " + upstreamIdleTimeout.String(),
					}
				}
				if time.Since(lastEmit) < heartbeatInterval {
					continue
				}
				if err := emitHeartbeat(streamID, rawData, req.Model); err != nil {
					return pumpOutcome{errMsg: err.Error()}
				}
				lastEmit = time.Now()
			}
		}
	}

	// attempt loop: each iteration opens a fresh upstream stream. A broken
	// one is retried with backoff while the downstream bridge stays open;
	// heartbeats keep it warm during the waits.
	emptyReplay := 0
	for attempt := 0; ; attempt++ {
		giveUp := func(reason string) {
			if monitor.partialText() != "" && attempt > 0 {
				reason += fmt.Sprintf(" (zen auto-reconnect gave up after %d attempt(s); the answer above may be incomplete)", attempt)
			}
			closeStream(reason)
		}

		h, _, errMsg, retryable := connectOne()
		if errMsg != "" {
			if retryable && attempt < maxRetries {
				time.Sleep(streamRetryDelay(attempt + 1))
				continue
			}
			giveUp(errMsg)
			return
		}
		if h.StatusCode < 200 || h.StatusCode >= 300 {
			// Drain a bit of the error body for the message, then decide.
			errBody := ""
			if chunk, errRead := readHostStream(h.StreamID); errRead == nil && len(chunk.Payload) > 0 {
				errBody = truncate(chunk.Payload, 512)
			} else {
				drainHostStream(h.StreamID)
			}
			msg := fmt.Sprintf("zen upstream status %d: %s", h.StatusCode, errBody)
			if streamRetryableStatus(h.StatusCode) && attempt < maxRetries {
				time.Sleep(streamRetryDelay(attempt + 1))
				continue
			}
			giveUp(msg)
			return
		}

		outcome := pump(h)
		if outcome.terminal {
			closeStream("")
			return
		}
		drainHostStream(h.StreamID)
		if outcome.empty {
			// Empty turn: nothing reached the client, so a replay can never
			// duplicate content. Try once (stream-retries permitting).
			if emptyReplay < 1 && attempt < maxRetries {
				emptyReplay++
				time.Sleep(streamRetryDelay(attempt + 1))
				continue
			}
			giveUp(outcome.errMsg)
			return
		}
		if !outcome.retryable || monitor.sawToolCall || attempt >= maxRetries {
			giveUp(outcome.errMsg)
			return
		}

		// Resume: replay transparently when nothing was forwarded yet, else
		// continue the unfinished answer with a fresh upstream request.
		if partial := monitor.partialText(); partial != "" {
			next, err := appendContinuationPayload(payload, partial)
			if err != nil {
				giveUp(fmt.Sprintf("%s (zen resume impossible: %s)", outcome.errMsg, err.Error()))
				return
			}
			payload = next
		}
		// A continuation is a new upstream request: mint a fresh canonical
		// msg_ id so the gate never sees a duplicate request identity.
		headers[targetRequestHeader] = []string{
			canonicalID("msg", fmt.Sprintf("%s\x00resume-%d", requestIdentity(req), attempt+1)),
		}
		time.Sleep(streamRetryDelay(attempt + 1))
	}
}

// emitHeartbeat forwards one no-op chat.completion.chunk downstream to keep
// the connection warm while the upstream is quiet. The empty delta is the
// standard keep-alive shape used by OpenAI-compatible providers.
func emitHeartbeat(streamID string, rawData bool, model string) error {
	payload, err := json.Marshal(map[string]any{
		"id":      canonicalID("chatcmpl", model),
		"object":  "chat.completion.chunk",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []any{map[string]any{
			"index":         0,
			"delta":         map[string]any{},
			"finish_reason": nil,
		}},
	})
	if err != nil {
		return nil
	}
	if rawData {
		return emitStreamFrame(streamID, payload)
	}
	return emitStreamFrame(streamID, []byte("data: "+string(payload)+"\n\n"))
}

// ---------------------------------------------------------------------------
// gate emulation
// ---------------------------------------------------------------------------

// gateHeaders builds the upstream header set for the zen free-tier gate.
func gateHeaders(req executorRequest) map[string][]string {
	cfg := loadedConfig()
	headers := map[string][]string{
		"User-Agent":        {defaultUserAgent},
		"Accept":            {"*/*"},
		"Content-Type":      {"application/json"},
		targetProjectHeader: {cfg.Project},
		targetRequestHeader: {canonicalID("msg", requestIdentity(req))},
	}
	if v, ok := headerValue(req.Headers, targetClientHeader); ok && v != "" {
		headers[targetClientHeader] = []string{v}
	} else {
		headers[targetClientHeader] = []string{cfg.Client}
	}
	if session, ok := resolveSession(&req); ok {
		headers[targetSessionHeader] = []string{session}
		// The official client mirrors the session on both affinity headers.
		headers["X-Session-Affinity"] = []string{session}
		headers["X-Session-Id"] = []string{session}
	}
	return headers
}

// requestIdentity derives a stable per-execution identity for the msg_ id.
func requestIdentity(req executorRequest) string {
	if id, ok := req.Metadata["request_id"].(string); ok && strings.TrimSpace(id) != "" {
		return id
	}
	return fmt.Sprintf("%s\x00%s", req.Model, string(req.Payload))
}

// resolveSession returns the canonical ses_ id for this conversation.
// Priority: existing canonical client value → first present client session
// header → metadata canonical_session_id → content-derived fallback.
//
// The returned id always satisfies the free-tier gate invariant
// (ses_ + 12 lowercase hex chars whose low 12 bits are 0xFFE). Client values
// that only look canonical are re-minted instead of being forwarded, and
// minting is cached per conversation source so multi-turn requests keep the
// same id (session affinity + upstream prompt cache).
func resolveSession(req *executorRequest) (string, bool) {
	if existing, exists := headerValue(req.Headers, targetSessionHeader); exists {
		if existing == "" {
			return "", false
		}
		if isGateSessionID(existing) {
			return existing, true
		}
		return sessionIDForSource("client:" + existing), true
	}
	for _, src := range sessionSources {
		if v, exists := headerValue(req.Headers, src); exists {
			if v == "" {
				return "", false
			}
			return sessionIDForSource("client:" + v), true
		}
	}
	if src := sessionFallback(req.Metadata); src != "" {
		return sessionIDForSource("meta:" + src), true
	}
	if src := contentFallback(req); src != "" {
		return sessionIDForSource("content:" + src), true
	}
	return "", false
}

func sessionFallback(meta map[string]any) string {
	if meta == nil {
		return ""
	}
	s, ok := meta["canonical_session_id"].(string)
	if !ok {
		return ""
	}
	normalized, ok := normalizeSessionID(s)
	if !ok {
		return ""
	}
	return normalized
}

// contentFallback derives a stable conversation key from the model plus the
// first user message (stateless, like the host's canonical_session_id).
func contentFallback(req *executorRequest) string {
	text := firstUserText(req.Payload)
	if text == "" {
		return ""
	}
	return "content:" + req.Model + "\x00" + text
}

// ---------------------------------------------------------------------------
// upstream body preparation
// ---------------------------------------------------------------------------

// prepareUpstreamBody applies the body-level gate rules to the (already
// host-translated) payload:
//   - "stream": true (zen rejects stream:false with 403 FreeTierError)
//   - tools contains bash + read in the dialect of the payload
func prepareUpstreamBody(req executorRequest, route modelRoute) ([]byte, error) {
	payload := trimJSONEnvelope(req.Payload)
	if len(payload) == 0 {
		payload = trimJSONEnvelope(req.OriginalRequest)
	}
	trimmed := strings.TrimSpace(string(payload))
	if trimmed == "" || trimmed[0] != '{' {
		return nil, fmt.Errorf("zen executor: unsupported payload shape (payload=%d bytes, original=%d bytes, head=%q)",
			len(req.Payload), len(req.OriginalRequest), truncate(req.Payload, 64))
	}
	var root map[string]any
	if err := json.Unmarshal(payload, &root); err != nil {
		return nil, fmt.Errorf("zen executor: payload is not valid JSON: %w", err)
	}

	// The host translated the request into chat-completions or responses
	// format; keep the model pointing at the configured upstream name.
	if route.Model != "" {
		root["model"] = route.Model
	}
	if err := convertBodyDialect(root, route.Endpoint); err != nil {
		return nil, err
	}
	// Repair host-translated payloads before they reach zen: the host's
	// protocol translation can emit shapes zen's strict validator rejects
	// with 400 (text parts without a usable "text" field, roleless messages
	// from reasoning items, null content).
	sanitizeChatMessages(root)
	root["stream"] = true

	dialect, known := bodyDialect(root)
	if known {
		ensureGateTools(root, dialect)
	}
	if route.Endpoint == "responses" {
		// The zen gateway keeps no server-side response state: replayed
		// reasoning items (rs_ ids) and previous_response_id answer 400
		// "Referenced reasoning item was not found or has expired".
		sanitizeResponsesUpstreamInput(root)
	}
	normalizeReasoningEffort(root, route.Endpoint, route.Model)
	clampMaxOutput(root, route.Model)

	out, err := json.Marshal(root)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// sanitizeChatMessages repairs a chat-completions payload in place so zen's
// strict validator never sees malformed content. It drops messages without a
// usable role (artifacts of translated reasoning items), drops content parts
// whose declared text is missing or empty, and normalizes null content to ""
// (zen rejects null content). Valid parts arrays are preserved as arrays.
func sanitizeChatMessages(root map[string]any) {
	items, ok := root["messages"].([]any)
	if !ok {
		return
	}
	out := make([]any, 0, len(items))
	for _, item := range items {
		msg, ok := item.(map[string]any)
		if !ok {
			continue
		}
		role, _ := msg["role"].(string)
		if role = strings.TrimSpace(role); role == "" {
			continue
		}
		msg["role"] = role
		switch content := msg["content"].(type) {
		case []any:
			kept := make([]any, 0, len(content))
			for _, p := range content {
				part, ok := p.(map[string]any)
				if !ok {
					continue
				}
				partType, _ := part["type"].(string)
				switch partType {
				case "text", "input_text", "output_text":
					text, ok := part["text"].(string)
					if !ok || strings.TrimSpace(text) == "" {
						continue
					}
					part["type"] = "text"
					delete(part, "annotations")
					kept = append(kept, part)
				default:
					kept = append(kept, part)
				}
			}
			if len(kept) == 0 {
				msg["content"] = ""
			} else {
				msg["content"] = kept
			}
		case nil:
			msg["content"] = ""
		}
		out = append(out, msg)
	}
	if len(out) > 0 {
		root["messages"] = out
	}
}

// convertBodyDialect converts a host-translated request between the
// chat-completions and responses payload shapes. CPA may hand us either
// format regardless of the endpoint a model requires, so the upstream body
// must always match route.Endpoint.
func convertBodyDialect(root map[string]any, endpoint string) error {
	dialect, known := bodyDialect(root)
	if !known || dialect == endpoint {
		return nil
	}
	if endpoint == "chat" {
		return responsesToChat(root)
	}
	return chatToResponses(root)
}

func responsesToChat(root map[string]any) error {
	input, ok := root["input"]
	if !ok {
		return nil
	}
	delete(root, "input")

	var messages []any
	if instructions, ok := root["instructions"].(string); ok && strings.TrimSpace(instructions) != "" {
		messages = append(messages, map[string]any{
			"role":    "system",
			"content": instructions,
		})
	}
	delete(root, "instructions")

	switch source := input.(type) {
	case string:
		messages = append(messages, map[string]any{"role": "user", "content": source})
	case []any:
		for _, item := range source {
			entry, ok := item.(map[string]any)
			if !ok {
				continue
			}
			itemType, _ := entry["type"].(string)
			// Reasoning items have no chat-completions representation. The
			// host's translation of them produces malformed messages, so drop
			// them here when the payload arrives in responses form.
			if strings.EqualFold(itemType, "reasoning") {
				continue
			}
			// Tool traffic in responses form (function_call /
			// function_call_output, plus the custom_tool variants) carries no
			// "role" field. Letting it fall through to the generic role copy
			// below produced roleless messages that sanitizeChatMessages then
			// dropped, silently erasing the whole tool history: the upstream
			// model never saw its own tool calls or any tool results and
			// re-issued the same calls every turn (agent loops). Convert them
			// to their chat-completions equivalents instead.
			if strings.EqualFold(itemType, "function_call") || strings.EqualFold(itemType, "custom_tool_call") {
				callID, _ := entry["call_id"].(string)
				if callID == "" {
					callID, _ = entry["id"].(string)
				}
				name, _ := entry["name"].(string)
				arguments, _ := entry["arguments"].(string)
				if arguments == "" {
					if raw := entry["input"]; raw != nil {
						if s, ok := raw.(string); ok {
							arguments = s
						} else if b, err := json.Marshal(raw); err == nil {
							arguments = string(b)
						}
					}
				}
				call := map[string]any{
					"id":   callID,
					"type": "function",
					"function": map[string]any{
						"name":      name,
						"arguments": arguments,
					},
				}
				// Parallel calls arrive as consecutive function_call items;
				// merge them into one assistant message (the canonical
				// chat-completions shape) instead of emitting several
				// back-to-back assistant messages.
				if n := len(messages); n > 0 {
					if prev, ok := messages[n-1].(map[string]any); ok {
						if role, _ := prev["role"].(string); role == "assistant" {
							if calls, ok := prev["tool_calls"].([]any); ok {
								prev["tool_calls"] = append(calls, call)
								continue
							}
						}
					}
				}
				messages = append(messages, map[string]any{
					"role":       "assistant",
					"content":    "",
					"tool_calls": []any{call},
				})
				continue
			}
			if strings.EqualFold(itemType, "function_call_output") || strings.EqualFold(itemType, "custom_tool_call_output") {
				callID, _ := entry["call_id"].(string)
				if callID == "" {
					callID, _ = entry["id"].(string)
				}
				messages = append(messages, map[string]any{
					"role":         "tool",
					"tool_call_id": callID,
					"content":      toolOutputContent(entry["output"]),
				})
				continue
			}
			message := map[string]any{"role": entry["role"]}
			// chat-completions has no "developer" role; strict upstreams
			// reject it outright.
			if r, ok := message["role"].(string); ok && strings.EqualFold(r, "developer") {
				message["role"] = "system"
			}
			if content, ok := entry["content"]; ok {
				message["content"] = responsesPartsToChat(content)
			}
			messages = append(messages, message)
		}
	default:
		return fmt.Errorf("zen executor: unsupported responses input shape")
	}
	if len(messages) == 0 {
		return fmt.Errorf("zen executor: responses input is empty")
	}
	root["messages"] = messages

	if tools, ok := root["tools"].([]any); ok {
		root["tools"] = responsesToolsToChat(tools)
	}
	if text, ok := root["text"].(map[string]any); ok {
		if format, ok := text["format"]; ok {
			root["response_format"] = format
		}
		delete(root, "text")
	}
	applyResponsesToChatFields(root)
	return nil
}

// responsesOnlyRequestFields are Responses API request fields with no
// chat-completions equivalent. Strict chat upstreams (zen included, it
// already rejects strict:null inside tool schemas) answer 400
// unknown_parameter when they leak through.
//
// "reasoning" is deliberately kept: chat dialects disagree on the
// replacement (reasoning_effort vs nothing) and current payloads already
// work end to end.
var responsesOnlyRequestFields = []string{
	"store", "background", "include", "prompt_cache_key",
	"safety_identifier", "max_tool_calls", "truncation",
}

// chatOnlyRequestFields are chat-completions request fields with no
// Responses API equivalent; strict /responses upstreams reject them.
var chatOnlyRequestFields = []string{
	"stream_options", "logit_bias", "logprobs", "top_logprobs",
	"n", "prediction", "max_tokens", "max_completion_tokens",
}

// applyResponsesToChatFields normalizes top-level request parameters when
// converting a Responses payload for a chat-completions upstream.
func applyResponsesToChatFields(root map[string]any) {
	if v, ok := root["max_output_tokens"]; ok {
		if _, has := root["max_tokens"]; !has {
			root["max_tokens"] = v
		}
		delete(root, "max_output_tokens")
	}
	for _, k := range responsesOnlyRequestFields {
		delete(root, k)
	}
	reshapeToolChoice(root, "chat")
}

// applyChatToResponsesFields normalizes top-level request parameters when
// converting a chat-completions payload for a /responses upstream.
func applyChatToResponsesFields(root map[string]any) {
	// Legacy functions → tools (the flat functions shape is already the
	// responses tool shape, minus the type tag).
	if _, ok := root["tools"]; !ok {
		if fns, ok := root["functions"].([]any); ok && len(fns) > 0 {
			tools := make([]any, 0, len(fns))
			for _, fn := range fns {
				if m, ok := fn.(map[string]any); ok {
					inner := make(map[string]any, len(m))
					for k, v := range m {
						if k == "type" {
							continue
						}
						inner[k] = v
					}
					tools = append(tools, map[string]any{"type": "function", "function": inner})
					continue
				}
				tools = append(tools, fn)
			}
			root["tools"] = tools
		}
	}
	// Legacy function_call maps onto tool_choice where it still can.
	if _, has := root["tool_choice"]; !has {
		if fc, ok := root["function_call"].(string); ok && (fc == "none" || fc == "required") {
			root["tool_choice"] = fc
		}
	}
	delete(root, "functions")
	delete(root, "function_call")
	// max tokens: the Responses API only knows max_output_tokens. Prefer
	// max_completion_tokens (reasoning models reject max_tokens on chat).
	if v, ok := root["max_completion_tokens"]; ok {
		root["max_output_tokens"] = v
	} else if v, ok := root["max_tokens"]; ok {
		root["max_output_tokens"] = v
	}
	for _, k := range chatOnlyRequestFields {
		delete(root, k)
	}
	// chat json_schema response_format nests under "json_schema"; the
	// responses format object carries the same fields flat.
	if rf, ok := root["response_format"].(map[string]any); ok {
		if t, _ := rf["type"].(string); t == "json_schema" {
			if inner, ok := rf["json_schema"].(map[string]any); ok {
				flat := make(map[string]any, len(inner)+1)
				for k, v := range inner {
					flat[k] = v
				}
				flat["type"] = "json_schema"
				root["response_format"] = flat
			}
		}
	}
	reshapeToolChoice(root, "responses")
}

// reshapeToolChoice converts the function tool_choice object between the
// chat-completions ({"type":"function","function":{"name":...}}) and
// responses ({"type":"function","name":...}) shapes. Other values
// (auto/none/required, hosted tools) pass through unchanged.
func reshapeToolChoice(root map[string]any, to string) {
	tc, ok := root["tool_choice"].(map[string]any)
	if !ok {
		return
	}
	if t, _ := tc["type"].(string); t != "function" {
		return
	}
	switch to {
	case "chat":
		if _, nested := tc["function"].(map[string]any); nested {
			return
		}
		name, _ := tc["name"].(string)
		if name == "" {
			return
		}
		root["tool_choice"] = map[string]any{
			"type":     "function",
			"function": map[string]any{"name": name},
		}
	case "responses":
		fn, nested := tc["function"].(map[string]any)
		if !nested {
			return
		}
		name, _ := fn["name"].(string)
		if name == "" {
			return
		}
		out := map[string]any{"type": "function", "name": name}
		if strict, ok := fn["strict"]; ok && strict != nil {
			// nil-valued strict is dropped like everywhere else: strict
			// upstreams reject strict:null.
			out["strict"] = strict
		}
		root["tool_choice"] = out
	}
}

// toolOutputContent flattens a responses function_call_output "output"
// value into the string content a chat-completions tool message expects.
// The spec shape is a plain string, but some clients send an object with a
// content field; both are handled.
func toolOutputContent(output any) string {
	switch o := output.(type) {
	case string:
		return o
	case nil:
		return ""
	case map[string]any:
		if content, ok := o["content"]; ok && content != nil {
			if s, ok := content.(string); ok {
				return s
			}
			if b, err := json.Marshal(content); err == nil {
				return string(b)
			}
			return ""
		}
		if b, err := json.Marshal(o); err == nil {
			return string(b)
		}
		return ""
	default:
		if b, err := json.Marshal(o); err == nil {
			return string(b)
		}
		return ""
	}
}

// renamePartType rewrites content parts whose "type" matches one of
// fromTypes to toType, copying all other fields verbatim. Content is
// returned unchanged when no part matched.
func renamePartType(content any, toType string, fromTypes ...string) any {
	parts, ok := content.([]any)
	if !ok {
		return content
	}
	out := make([]any, 0, len(parts))
	changed := false
	for _, item := range parts {
		part, ok := item.(map[string]any)
		if !ok {
			out = append(out, item)
			continue
		}
		partType, _ := part["type"].(string)
		matched := false
		for _, want := range fromTypes {
			if partType == want {
				matched = true
				break
			}
		}
		if !matched {
			out = append(out, part)
			continue
		}
		converted := make(map[string]any, len(part))
		for k, v := range part {
			if k == "type" {
				converted[k] = toType
			} else {
				converted[k] = v
			}
		}
		out = append(out, converted)
		changed = true
	}
	if !changed {
		return content
	}
	return out
}

// assistantText extracts an assistant message's plain text from either a
// string content or a parts array (text / output_text parts).
func assistantText(content any) string {
	switch c := content.(type) {
	case string:
		if strings.TrimSpace(c) == "" {
			return ""
		}
		return c
	case []any:
		var b strings.Builder
		for _, item := range c {
			part, ok := item.(map[string]any)
			if !ok {
				continue
			}
			if pt, _ := part["type"].(string); pt != "text" && pt != "output_text" {
				continue
			}
			if s, ok := part["text"].(string); ok {
				b.WriteString(s)
			}
		}
		return b.String()
	default:
		return ""
	}
}

// responsesPartsToChat normalizes Responses API content parts for a
// chat-completions upstream: text part types are renamed and input images
// become image_url parts (with "detail" moved inside the image_url object).
// Malformed image parts are passed through untouched.
func responsesPartsToChat(content any) any {
	renamed := renamePartType(content, "text", "input_text", "output_text")
	parts, ok := renamed.([]any)
	if !ok {
		return renamed
	}
	out := make([]any, 0, len(parts))
	changed := false
	for _, item := range parts {
		part, ok := item.(map[string]any)
		if !ok {
			out = append(out, item)
			continue
		}
		if pt, _ := part["type"].(string); !strings.EqualFold(pt, "input_image") {
			out = append(out, item)
			continue
		}
		var urlObj map[string]any
		switch u := part["image_url"].(type) {
		case string:
			urlObj = map[string]any{"url": u}
		case map[string]any:
			urlObj = make(map[string]any, len(u)+1)
			for k, v := range u {
				urlObj[k] = v
			}
		default:
			out = append(out, item)
			continue
		}
		if detail := part["detail"]; detail != nil {
			urlObj["detail"] = detail
		}
		converted := make(map[string]any, len(part))
		for k, v := range part {
			switch k {
			case "type", "image_url", "detail":
				continue
			default:
				converted[k] = v
			}
		}
		converted["type"] = "image_url"
		converted["image_url"] = urlObj
		out = append(out, converted)
		changed = true
	}
	if !changed {
		return renamed
	}
	return out
}

func responsesToolsToChat(tools []any) []any {
	out := make([]any, 0, len(tools))
	for _, item := range tools {
		tool, ok := item.(map[string]any)
		if !ok {
			out = append(out, item)
			continue
		}
		if fn, ok := tool["function"].(map[string]any); ok {
			for k, v := range fn {
				if v == nil {
					delete(fn, k)
				}
			}
			out = append(out, tool)
			continue
		}
		name, _ := tool["name"].(string)
		if name == "" {
			out = append(out, tool)
			continue
		}
		// Flat responses tool → wrap into the chat function shape. (The
		// tool["function"] branch above already handled chat-shaped input,
		// so this branch only ever sees flat entries.)
		fn := make(map[string]any, len(tool))
		for k, v := range tool {
			if k == "type" {
				continue
			}
			fn[k] = v
		}
		// zen rejects "strict": null (pi sends it on every tool); drop null-valued keys.
		for k, v := range fn {
			if v == nil {
				delete(fn, k)
			}
		}
		out = append(out, map[string]any{"type": "function", "function": fn})
	}
	return out
}

func chatToResponses(root map[string]any) error {
	source, ok := root["messages"].([]any)
	if !ok {
		return nil
	}
	delete(root, "messages")

	input := make([]any, 0, len(source))
	for _, item := range source {
		entry, ok := item.(map[string]any)
		if !ok {
			continue
		}
		role, _ := entry["role"].(string)

		// Tool result messages become function_call_output items; a bare
		// {"role":"tool"} object is not a valid responses input item and the
		// tool result would be lost.
		if role == "tool" {
			callID, _ := entry["tool_call_id"].(string)
			content := ""
			switch c := entry["content"].(type) {
			case string:
				content = c
			case nil:
			default:
				if b, err := json.Marshal(c); err == nil {
					content = string(b)
				}
			}
			input = append(input, map[string]any{
				"type":    "function_call_output",
				"call_id": callID,
				"output":  content,
			})
			continue
		}

		// Assistant tool invocations become function_call items; dropping
		// tool_calls here would erase the model's own calls from history.
		if role == "assistant" {
			if calls, ok := entry["tool_calls"].([]any); ok && len(calls) > 0 {
				// The assistant's text must survive alongside the calls;
				// string and parts-array contents are both common.
				if text := assistantText(entry["content"]); text != "" {
					input = append(input, map[string]any{
						"type":    "message",
						"role":    "assistant",
						"content": []any{map[string]any{"type": "output_text", "text": text}},
					})
				}
				for _, c := range calls {
					call, ok := c.(map[string]any)
					if !ok {
						continue
					}
					callID, _ := call["id"].(string)
					fn, _ := call["function"].(map[string]any)
					name, _ := fn["name"].(string)
					arguments, _ := fn["arguments"].(string)
					input = append(input, map[string]any{
						"type":      "function_call",
						"call_id":   callID,
						"name":      name,
						"arguments": arguments,
					})
				}
				continue
			}
		}

		message := map[string]any{"type": "message", "role": entry["role"]}
		if content, ok := entry["content"]; ok {
			if role == "assistant" {
				message["content"] = renamePartType(content, "output_text", "text")
			} else {
				message["content"] = renamePartType(content, "input_text", "text")
			}
		}
		input = append(input, message)
	}
	if len(input) == 0 {
		return fmt.Errorf("zen executor: chat messages are empty")
	}
	root["input"] = input

	// Parameter hygiene must run before the tools dialect conversion so the
	// legacy functions fallback participates in it.
	applyChatToResponsesFields(root)

	if tools, ok := root["tools"].([]any); ok {
		root["tools"] = chatToolsToResponses(tools)
	}
	if format, ok := root["response_format"]; ok {
		root["text"] = map[string]any{"format": format}
		delete(root, "response_format")
	}
	return nil
}

func chatToolsToResponses(tools []any) []any {
	out := make([]any, 0, len(tools))
	for _, item := range tools {
		tool, ok := item.(map[string]any)
		if !ok {
			out = append(out, item)
			continue
		}
		fn, ok := tool["function"].(map[string]any)
		if !ok {
			out = append(out, tool)
			continue
		}
		converted := make(map[string]any, len(fn)+1)
		for k, v := range fn {
			// Same rule as the other direction: strict upstreams reject
			// null-valued keys (pi sends strict:null on every tool).
			if v == nil {
				continue
			}
			converted[k] = v
		}
		converted["type"] = "function"
		out = append(out, converted)
	}
	return out
}

func bodyDialect(root map[string]any) (string, bool) {
	if _, ok := root["messages"]; ok {
		return "chat", true
	}
	if _, ok := root["input"]; ok {
		return "responses", true
	}
	return "", false
}

// officialGateTool describes one of the OpenCode client's six built-in
// tools (bash, edit, glob, grep, read, write) with the official description
// and parameter schema, used to fill gaps in client tool sets.
type officialGateTool struct {
	name        string
	description string
	parameters  map[string]any
}

// officialGateTools mirrors the OpenCode client's built-in tool set. The
// zen free tier expects this set to be present; requests carrying no tools
// at all (compaction/summarization, plain chat clients) get the full set
// injected with tool_choice "none" so the gate passes while the model is
// still barred from calling tools the client cannot execute.
var officialGateTools = []officialGateTool{
	{
		name:        "bash",
		description: "Execute bash commands in the workspace environment",
		parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"command": map[string]any{"type": "string", "description": "Shell command string to execute"},
				"workdir": map[string]any{"type": "string", "description": "Working directory. Defaults to the active Location; relative paths resolve within it."},
			},
			"required": []any{"command"},
		},
	},
	{
		name:        "edit",
		description: "Edit a file by replacing text",
		parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path":       map[string]any{"type": "string", "description": "File path to edit. Relative paths resolve within the active Location."},
				"oldString":  map[string]any{"type": "string", "description": "The string in the file to be replaced"},
				"newString":  map[string]any{"type": "string", "description": "The string to replace oldString with"},
				"replaceAll": map[string]any{"type": "boolean", "description": "Replace all occurrences of oldString (default false)"},
			},
			"required": []any{"path", "oldString", "newString"},
		},
	},
	{
		name:        "glob",
		description: "Find files matching a glob pattern",
		parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"pattern": map[string]any{"type": "string", "description": "File pattern to include in the search (e.g. \"*.js\", \"*.{ts,tsx}\")"},
				"path":    map[string]any{"type": "string", "description": "Relative directory to search in. Defaults to the active Location."},
			},
			"required": []any{"pattern"},
		},
	},
	{
		name:        "grep",
		description: "Search file contents using regular expressions",
		parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"pattern": map[string]any{"type": "string", "description": "Regex pattern to search for in file contents"},
				"path":    map[string]any{"type": "string", "description": "Relative directory to search in. Defaults to the active Location."},
			},
			"required": []any{"pattern"},
		},
	},
	{
		name:        "read",
		description: "Read file contents",
		parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path":   map[string]any{"type": "string", "description": "File path to read. Relative paths resolve within the active Location."},
				"offset": map[string]any{"type": "number", "description": "The 1-based directory entry or text line offset"},
				"limit":  map[string]any{"type": "number", "description": "The maximum number of lines to read (defaults to 2000)"},
			},
			"required": []any{"path"},
		},
	},
	{
		name:        "write",
		description: "Write or overwrite file contents",
		parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path":    map[string]any{"type": "string", "description": "File path to write. Relative paths resolve within the active Location."},
				"content": map[string]any{"type": "string", "description": "Content to write to the file"},
			},
			"required": []any{"path", "content"},
		},
	},
}

// ensureGateTools aligns the tool set with the official client: any of the
// six core tools missing from the request is appended (dialect-appropriate
// shape) and the list is sorted by function name, matching the official
// client's alphabetical ordering. Requests without any tools get the full
// set plus tool_choice "none".
func ensureGateTools(root map[string]any, dialect string) {
	tools, _ := root["tools"].([]any)
	hadTools := len(tools) > 0
	names := map[string]bool{}
	for _, item := range tools {
		entry, ok := item.(map[string]any)
		if !ok {
			continue
		}
		names[toolEntryName(entry, dialect)] = true
	}
	for i := range officialGateTools {
		if !names[officialGateTools[i].name] {
			tools = append(tools, gateToolEntry(&officialGateTools[i], dialect))
		}
	}
	// Official clients send tools sorted by function name.
	sort.SliceStable(tools, func(i, j int) bool {
		a, _ := tools[i].(map[string]any)
		b, _ := tools[j].(map[string]any)
		return toolEntryName(a, dialect) < toolEntryName(b, dialect)
	})
	root["tools"] = tools
	if !hadTools {
		if _, exists := root["tool_choice"]; !exists {
			root["tool_choice"] = "none"
		}
	}
}

func toolEntryName(entry map[string]any, dialect string) string {
	if dialect == "chat" {
		if fn, ok := entry["function"].(map[string]any); ok {
			if name, ok := fn["name"].(string); ok {
				return strings.TrimSpace(name)
			}
		}
		return ""
	}
	if name, ok := entry["name"].(string); ok {
		return strings.TrimSpace(name)
	}
	return ""
}

func gateToolEntry(tool *officialGateTool, dialect string) map[string]any {
	if dialect == "chat" {
		return map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        tool.name,
				"description": tool.description,
				"parameters":  tool.parameters,
			},
		}
	}
	return map[string]any{
		"type":        "function",
		"name":        tool.name,
		"description": tool.description,
		"parameters":  tool.parameters,
	}
}

// sanitizeResponsesUpstreamInput strips server-state references the zen
// gateway does not retain: replayed reasoning items (their rs_ ids expire
// upstream and answer 400 "Referenced reasoning item was not found or has
// expired") and previous_response_id. Without this, agent clients that
// replay reasoning items (pi via the codex responses dialect) fail every
// follow-up turn and the repeated 400s get the credential marked
// unavailable upstream (auth_unavailable cascade).
func sanitizeResponsesUpstreamInput(root map[string]any) {
	delete(root, "previous_response_id")
	items, ok := root["input"].([]any)
	if !ok {
		return
	}
	out := make([]any, 0, len(items))
	for _, item := range items {
		entry, ok := item.(map[string]any)
		if !ok {
			out = append(out, item)
			continue
		}
		if t, _ := entry["type"].(string); strings.EqualFold(t, "reasoning") {
			continue
		}
		out = append(out, item)
	}
	root["input"] = out
}

// normalizeReasoningEffort clamps the requested reasoning effort to what
// the model supports. The zen gateway resolves the effort into a provider
// variant (reasoningEffort / reasoning_effort / reasoning.effort are all
// read), and unsupported values answer 400 "Invalid request parameters".
// Effort-type models keep values from their catalog level set; toggle-only
// models (mimo, nemotron, ...) drop the parameter entirely; unknown models
// fall back to the safe low/medium/high set. The anthropic-style top-level
// "thinking" object is always removed.
func normalizeReasoningEffort(root map[string]any, endpoint, model string) {
	delete(root, "thinking")

	effort := ""
	if s, ok := root["reasoning_effort"].(string); ok {
		effort = strings.ToLower(strings.TrimSpace(s))
	}
	if s, ok := root["reasoningEffort"].(string); ok && effort == "" {
		effort = strings.ToLower(strings.TrimSpace(s))
	}
	reasoning, _ := root["reasoning"].(map[string]any)
	if effort == "" {
		if s, ok := reasoning["effort"].(string); ok {
			effort = strings.ToLower(strings.TrimSpace(s))
		}
	}
	delete(root, "reasoning_effort")
	delete(root, "reasoningEffort")

	spec, known := zenModelSpec(model)
	if effort != "" {
		switch {
		case known && len(spec.ReasoningLevels) == 0:
			// Toggle-only thinking: the effort parameter is rejected.
			effort = ""
		case known:
			effort = clampEffortToLevels(effort, spec.ReasoningLevels)
		default:
			// Unknown model: only the universally safe set.
			effort = clampEffortToLevels(effort, []string{"low", "medium", "high"})
		}
	}

	if endpoint == "responses" {
		if reasoning == nil && effort != "" {
			reasoning = map[string]any{}
		}
		if reasoning != nil {
			if effort != "" {
				reasoning["effort"] = effort
			} else {
				delete(reasoning, "effort")
			}
			root["reasoning"] = reasoning
		}
		return
	}
	// Chat endpoint: the effort rides the top-level field; a leftover
	// reasoning object has no chat-completions meaning.
	delete(root, "reasoning")
	if effort != "" {
		root["reasoning_effort"] = effort
	}
}

// clampEffortToLevels maps an effort value onto the supported level set.
// An empty result means "drop the parameter".
func clampEffortToLevels(effort string, levels []string) string {
	has := func(v string) bool {
		for _, l := range levels {
			if l == v {
				return true
			}
		}
		return false
	}
	if has(effort) {
		return effort
	}
	if effort == "off" || effort == "none" || effort == "" {
		return ""
	}
	// Alias mapping: nearest supported level wins.
	aliases := map[string][]string{
		"max":     {"xhigh", "high"},
		"xhigh":   {"high"},
		"ultra":   {"high"},
		"minimal": {"low"},
		"mini":    {"low"},
	}
	for _, candidate := range aliases[effort] {
		if has(candidate) {
			return candidate
		}
	}
	for _, fallback := range []string{"high", "medium", "low"} {
		if has(fallback) {
			return fallback
		}
	}
	return ""
}

// clampMaxOutput caps the requested output tokens at the model's catalog
// limit (65536 for unknown models) so oversized requests cannot fail with
// 400 invalid parameters.
func clampMaxOutput(root map[string]any, model string) {
	limit := int64(65536)
	if spec, known := zenModelSpec(model); known && spec.MaxCompletionTokens > 0 {
		limit = spec.MaxCompletionTokens
	}
	for _, field := range []string{"max_tokens", "max_completion_tokens", "max_output_tokens"} {
		if v, ok := root[field].(float64); ok && v > float64(limit) {
			root[field] = float64(limit)
		}
	}
}

// officialIDAlphabet is the Base62 alphabet used by the OpenCode client's
// Identifier ids.
const officialIDAlphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

// gateSessionLowHex is the invariant the free-tier endpoint checks on the
// session id: the 12 hex chars carry 0xFFE in their low 12 bits.
const gateSessionLowHex = "ffe"

// zenSessionEpochMillis is the constant the official client counts down from:
// the 12 hex chars are ((zenSessionEpochMillis - unixMilli) << 12) | 0xFFE, so
// the low 12 bits stay pinned to 0xFFE and the high 36 bits descend as time
// passes. Once the clock passes the epoch the field wraps, which the client
// treats as a wrapped 36-bit counter.
const zenSessionEpochMillis = 27 << 36

// maxCachedSessionIDs bounds the per-source session id cache.
const maxCachedSessionIDs = 4096

// isGateSessionID reports whether an id satisfies the free-tier gate: the
// canonical shape plus 0xFFE in the low 12 bits of the hex field.
func isGateSessionID(id string) bool {
	if !canonicalSessionRe.MatchString(id) {
		return false
	}
	return id[13:16] == gateSessionLowHex
}

var (
	sessionIDCacheMu sync.Mutex
	sessionIDCache   = map[string]string{}
)

// sessionIDForSource returns a gate-valid ses_ id for one conversation source,
// minting it on first use and reusing it afterwards so every turn of a
// conversation keeps the same session (affinity + upstream prompt cache).
func sessionIDForSource(source string) string {
	sessionIDCacheMu.Lock()
	defer sessionIDCacheMu.Unlock()
	if id, ok := sessionIDCache[source]; ok {
		return id
	}
	if len(sessionIDCache) >= maxCachedSessionIDs {
		sessionIDCache = map[string]string{}
	}
	id := newOfficialSessionID()
	sessionIDCache[source] = id
	return id
}

// newOfficialSessionID mints a session id in the official descending format
// (ses_ + 12 hex chars of the descending millisecond timestamp + 14 base62
// chars) so the id reads as freshly created by the official client, including
// the 0xFFE low bits the free-tier gate requires. Used when a retry needs a
// session the upstream has never seen.
func newOfficialSessionID() string {
	value := uint64((zenSessionEpochMillis-time.Now().UnixMilli())<<12|0xFFE) & 0xFFFFFFFFFFFF
	const hexDigits = "0123456789abcdef"
	var hexPart [12]byte
	for i := 0; i < 6; i++ {
		b := byte(value >> (40 - 8*i))
		hexPart[i*2] = hexDigits[b>>4]
		hexPart[i*2+1] = hexDigits[b&0xf]
	}
	randBuf := make([]byte, 14)
	_, _ = rand.Read(randBuf)
	var suffix [14]byte
	for i, b := range randBuf {
		suffix[i] = officialIDAlphabet[int(b)%len(officialIDAlphabet)]
	}
	return "ses_" + string(hexPart[:]) + string(suffix[:])
}

// refreshGateSession replaces the session headers with a freshly minted
// official id; used when retrying after auth-level failures.
func refreshGateSession(headers map[string][]string) string {
	session := newOfficialSessionID()
	headers[targetSessionHeader] = []string{session}
	headers["X-Session-Affinity"] = []string{session}
	headers["X-Session-Id"] = []string{session}
	return session
}

// firstUserText extracts the first user message text for the content-derived
// session fallback. Understands both chat and responses payloads.
func firstUserText(body []byte) string {
	trimmed := strings.TrimSpace(string(body))
	if trimmed == "" || trimmed[0] != '{' {
		return ""
	}
	var root map[string]any
	if err := json.Unmarshal(body, &root); err != nil {
		return ""
	}
	if items, ok := root["messages"].([]any); ok {
		return firstUserTextFromItems(items)
	}
	switch input := root["input"].(type) {
	case string:
		return input
	case []any:
		return firstUserTextFromItems(input)
	}
	return ""
}

// firstUserTextFromItems returns the text of the first user message in a
// chat-style message list (string content or text parts).
func firstUserTextFromItems(items []any) string {
	for _, item := range items {
		msg, ok := item.(map[string]any)
		if !ok || msg["role"] != "user" {
			continue
		}
		if s, ok := msg["content"].(string); ok && s != "" {
			return s
		}
		if parts, ok := msg["content"].([]any); ok {
			for _, part := range parts {
				pm, ok := part.(map[string]any)
				if !ok {
					continue
				}
				if s, ok := pm["text"].(string); ok && s != "" {
					return s
				}
			}
		}
	}
	return ""
}

// ---------------------------------------------------------------------------
// canonical id minting
// ---------------------------------------------------------------------------

// canonicalID deterministically maps an arbitrary source string onto zen's
// canonical id shapes (msg_/chatcmpl_/call_ + 12 hex + 14 alphanumerics,
// 30 chars). Session ids must not be minted here: the free-tier gate reads
// the low 12 bits of the hex field, which a hash does not satisfy. Use
// sessionIDForSource for ses_ ids.
func canonicalID(prefix, source string) string {
	sum := sha256.Sum256([]byte(source))
	hexPart := hex.EncodeToString(sum[:6])
	const alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
	suffix := make([]byte, 14)
	for i := 0; i < 14; i++ {
		suffix[i] = alphabet[int(sum[6+i])%len(alphabet)]
	}
	return prefix + "_" + hexPart + string(suffix)
}

// ---------------------------------------------------------------------------
// header helpers
// ---------------------------------------------------------------------------

// headerValue returns the trimmed value for a header name and whether the
// header was present. Conflicting repeated or case-variant values fail
// closed (present, empty).
func headerValue(headers map[string][]string, name string) (string, bool) {
	if headers == nil {
		return "", false
	}
	var found bool
	var value string
	for key, values := range headers {
		if !strings.EqualFold(key, name) {
			continue
		}
		for _, v := range values {
			v = strings.TrimSpace(v)
			if v == "" {
				continue
			}
			if found && v != value {
				return "", true
			}
			if !found {
				found = true
				value = v
			}
		}
	}
	if !found {
		return "", false
	}
	normalized, ok := normalizeSessionID(value)
	if !ok {
		return "", true
	}
	return normalized, true
}

// normalizeSessionID trims, enforces the size cap, and rejects Unicode
// control characters.
func normalizeSessionID(value string) (string, bool) {
	value = strings.TrimSpace(value)
	if value == "" || len(value) > maxSessionIDBytes {
		return "", false
	}
	for _, r := range value {
		if unicode.IsControl(r) {
			return "", false
		}
	}
	return value, true
}

// ---------------------------------------------------------------------------
// host callbacks (HTTP + stream bridge)
// ---------------------------------------------------------------------------

type hostHTTPResponse struct {
	StatusCode int                 `json:"StatusCode"`
	Headers    map[string][]string `json:"Headers"`
	Body       []byte              `json:"Body"`
}

type hostHTTPStreamResponse struct {
	StatusCode int                 `json:"status_code"`
	Headers    map[string][]string `json:"headers,omitempty"`
	StreamID   string              `json:"stream_id,omitempty"`
}

// callHost invokes a host callback method and decodes the result envelope.
func callHost(method string, payload any) (json.RawMessage, error) {
	rawPayload, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal host callback payload %s: %w", method, err)
	}
	cMethod := C.CString(method)
	defer C.free(unsafe.Pointer(cMethod))

	var response C.cliproxy_buffer
	var requestPtr *C.uint8_t
	if len(rawPayload) > 0 {
		cPayload := C.CBytes(rawPayload)
		if cPayload == nil {
			return nil, fmt.Errorf("allocate host callback payload %s", method)
		}
		defer C.free(cPayload)
		requestPtr = (*C.uint8_t)(cPayload)
	}
	callCode := C.call_host_api(cMethod, requestPtr, C.size_t(len(rawPayload)), &response)
	var rawResponse []byte
	if response.ptr != nil && response.len > 0 {
		rawResponse = C.GoBytes(response.ptr, C.int(response.len))
	}
	if response.ptr != nil {
		C.free_host_buffer(response.ptr, response.len)
	}
	if len(rawResponse) == 0 {
		return nil, fmt.Errorf("host callback %s returned no response, code=%d", method, int(callCode))
	}
	var env envelope
	if err := json.Unmarshal(rawResponse, &env); err != nil {
		return nil, fmt.Errorf("decode host callback envelope %s: %w", method, err)
	}
	if !env.OK {
		if env.Error != nil {
			return nil, fmt.Errorf("%s: %s", env.Error.Code, env.Error.Message)
		}
		return nil, fmt.Errorf("host callback %s failed", method)
	}
	return append(json.RawMessage(nil), env.Result...), nil
}

// keyRotation rotates zen API keys round-robin.
var keyRotation struct {
	mu    sync.Mutex
	index int
}

func nextAPIKey(cfg pluginConfig) string {
	if len(cfg.APIKeys) == 0 {
		return ""
	}
	keyRotation.mu.Lock()
	defer keyRotation.mu.Unlock()
	key := cfg.APIKeys[keyRotation.index%len(cfg.APIKeys)]
	keyRotation.index++
	return key
}

// doUpstream performs a blocking host HTTP call.
func doUpstream(hostCallbackID, method, url string, headers map[string][]string, apiKey string, body []byte) ([]byte, map[string][]string, int, error) {
	req := map[string]any{
		"Method":  method,
		"URL":     url,
		"Headers": withAuth(headers, apiKey),
		"Body":    body,
	}
	if hostCallbackID != "" {
		req["host_callback_id"] = hostCallbackID
	}
	raw, err := callHost("host.http.do", req)
	if err != nil {
		return nil, nil, 0, err
	}
	var resp hostHTTPResponse
	if err := json.Unmarshal(raw, &resp); err != nil {
		return nil, nil, 0, err
	}
	return resp.Body, resp.Headers, resp.StatusCode, nil
}

// hostStreamHandle is an opened host HTTP stream.
type hostStreamHandle struct {
	StatusCode int
	Headers    map[string][]string
	StreamID   string
}

// doUpstreamStream opens a host HTTP stream and returns immediately.
func doUpstreamStream(hostCallbackID, method, url string, headers map[string][]string, apiKey string, body []byte) (*hostStreamHandle, error) {
	req := map[string]any{
		"Method":  method,
		"URL":     url,
		"Headers": withAuth(headers, apiKey),
		"Body":    body,
	}
	if hostCallbackID != "" {
		req["host_callback_id"] = hostCallbackID
	}
	raw, err := callHost("host.http.do_stream", req)
	if err != nil {
		return nil, err
	}
	var resp hostHTTPStreamResponse
	if err := json.Unmarshal(raw, &resp); err != nil {
		return nil, err
	}
	if resp.StreamID == "" {
		return nil, fmt.Errorf("host http stream bridge is unavailable")
	}
	return &hostStreamHandle{
		StatusCode: resp.StatusCode,
		Headers:    resp.Headers,
		StreamID:   resp.StreamID,
	}, nil
}

// readHostStream reads the next chunk from an open host HTTP stream.
func readHostStream(streamID string) (httpStreamChunk, error) {
	raw, err := callHost("host.http.stream_read", map[string]any{"stream_id": streamID})
	if err != nil {
		return httpStreamChunk{}, err
	}
	var chunk httpStreamChunk
	if err := json.Unmarshal(raw, &chunk); err != nil {
		return httpStreamChunk{}, err
	}
	return chunk, nil
}

// emitStreamFrame forwards one prepared payload frame through the plugin
// stream bridge. "[DONE]" terminators are dropped: the host appends its own
// done tail after the plugin stream closes (adapters_executors.go
// emitTranslatedExecutorStreamTail / executorStreamDonePayload), so
// forwarding the upstream terminator would duplicate it.
func emitStreamFrame(streamID string, frame []byte) error {
	if len(frame) == 0 {
		return nil
	}
	if isDoneTerminator(string(frame)) {
		return nil
	}
	_, err := callHost("host.stream.emit", map[string]any{
		"stream_id": streamID,
		"payload":   frame,
	})
	return err
}

// sseReassembler turns chunked upstream SSE bytes into complete plugin
// stream emissions. It buffers partial lines across chunk boundaries and
// reassembles the data fields of one event per the SSE specification
// (multiple "data:" lines of an event are joined with "\n") before
// emitting, so a payload split across several data lines reaches the
// client as one frame instead of invalid fragments.
type sseReassembler struct {
	rawData bool
	lineBuf bytes.Buffer
	pending []string // data payloads of the event currently being assembled
	emit    func(frame []byte) error

	// sawDone records that an upstream [DONE] terminator was observed and
	// dropped (the host appends its own done tail). runStream treats it as
	// proof the stream ended cleanly even when a finish_reason chunk was
	// never forwarded.
	sawDone bool
}

// write feeds one upstream chunk into the reassembler.
func (r *sseReassembler) write(p []byte) error {
	r.lineBuf.Write(p)
	for {
		data := r.lineBuf.Bytes()
		idx := bytes.IndexByte(data, '\n')
		if idx < 0 {
			return nil
		}
		line := make([]byte, idx)
		copy(line, data[:idx])
		r.lineBuf.Next(idx + 1)
		if err := r.handleLine(line); err != nil {
			return err
		}
	}
}

// flush emits any buffered partial line and the pending event. It must be
// called once the upstream stream is finished so the tail of the answer is
// never dropped.
func (r *sseReassembler) flush() error {
	if r.lineBuf.Len() > 0 {
		line := make([]byte, r.lineBuf.Len())
		copy(line, r.lineBuf.Bytes())
		r.lineBuf.Reset()
		if err := r.handleLine(line); err != nil {
			return err
		}
	}
	return r.flushPending()
}

// handleLine processes one complete SSE line.
func (r *sseReassembler) handleLine(line []byte) error {
	s := strings.TrimSpace(strings.TrimRight(string(line), "\r"))
	if s == "" {
		// A blank line terminates the current event.
		return r.flushPending()
	}
	if after, ok := strings.CutPrefix(s, "data:"); ok {
		payload := strings.TrimLeft(after, " \t")
		if payload == "" || strings.HasPrefix(payload, ":") {
			return nil
		}
		r.pending = append(r.pending, payload)
		return nil
	}
	// Non-data lines flush the pending event first to preserve ordering,
	// then keep the single-line passthrough behavior (comments dropped,
	// event/id/retry and bare JSON lines normalized).
	if err := r.flushPending(); err != nil {
		return err
	}
	frame := normalizeSSEFrame(r.rawData, []byte(s))
	if len(frame) == 0 || isDoneTerminator(string(frame)) {
		if isDoneTerminator(string(frame)) {
			r.sawDone = true
		}
		return nil
	}
	return r.emit(frame)
}

// flushPending emits the joined data payload of the current event.
// "[DONE]" terminators are dropped: the host appends its own done tail
// after the plugin stream closes, so forwarding the upstream terminator
// would duplicate it.
func (r *sseReassembler) flushPending() error {
	if len(r.pending) == 0 {
		return nil
	}
	joined := strings.Join(r.pending, "\n")
	r.pending = nil
	if isDoneTerminator(joined) {
		r.sawDone = true
		return nil
	}
	if r.rawData {
		return r.emit([]byte(joined))
	}
	return r.emit([]byte("data: " + joined + "\n\n"))
}

// isDoneTerminator reports whether a payload is an SSE [DONE] terminator.
func isDoneTerminator(s string) bool {
	s = strings.TrimSpace(s)
	return s == "[DONE]" || s == "data: [DONE]"
}

// normalizeSSEFrame converts one non-data upstream SSE line into the
// payload to forward through the host stream bridge. Empty lines and
// keep-alive comments are dropped; event/id/retry lines pass through framed
// (dropped in raw mode); a bare JSON object line is treated as an unframed
// data payload. data: lines and [DONE] terminators are handled by the
// caller (sseReassembler.handleLine).
func normalizeSSEFrame(rawData bool, raw []byte) []byte {
	s := strings.TrimSpace(string(raw))
	if s == "" || strings.HasPrefix(s, ":") {
		return nil
	}
	if strings.HasPrefix(s, "event:") || strings.HasPrefix(s, "id:") || strings.HasPrefix(s, "retry:") {
		if rawData {
			return nil
		}
		return []byte(s + "\n\n")
	}
	if strings.HasPrefix(s, "{") && strings.HasSuffix(s, "}") {
		if rawData {
			return []byte(s)
		}
		return []byte("data: " + s + "\n\n")
	}
	return nil
}

func withAuth(headers map[string][]string, apiKey string) map[string][]string {
	out := make(map[string][]string, len(headers)+1)
	for k, v := range headers {
		out[k] = v
	}
	if apiKey != "" {
		out["Authorization"] = []string{"Bearer " + apiKey}
	}
	return out
}

// ---------------------------------------------------------------------------
// SSE folding (non-streaming clients)
// ---------------------------------------------------------------------------

// foldSSEToJSON reassembles an SSE answer into a single JSON payload in the
// dialect of the endpoint (chat.completion / response object).
func foldSSEToJSON(sse []byte, endpoint string) ([]byte, error) {
	switch endpoint {
	case "responses":
		return foldResponsesSSE(sse)
	default:
		return foldChatSSE(sse)
	}
}

// foldChatSSE merges chat.completion.chunk frames into one chat.completion.
func foldChatSSE(sse []byte) ([]byte, error) {
	var (
		id, model, role, finish string
		content                 strings.Builder
		toolCalls               = map[int]*bytes.Buffer{}
		toolNames               = map[int]string{}
		toolIDs                 = map[int]string{}
		toolOrder               []int
		usage                   json.RawMessage
	)
	for _, frame := range sseFrames(sse) {
		if frame == "[DONE]" {
			break
		}
		var chunk struct {
			ID      string `json:"id"`
			Model   string `json:"model"`
			Choices []struct {
				Index int `json:"index"`
				Delta struct {
					Role      string          `json:"role"`
					Content   string          `json:"content"`
					ToolCalls []toolCallDelta `json:"tool_calls"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
			Usage json.RawMessage `json:"usage"`
		}
		if err := json.Unmarshal([]byte(frame), &chunk); err != nil {
			continue // skip non-JSON frames (keep-alives, metadata)
		}
		if chunk.ID != "" {
			id = chunk.ID
		}
		if chunk.Model != "" {
			model = chunk.Model
		}
		if len(chunk.Usage) > 0 && string(chunk.Usage) != "null" {
			usage = chunk.Usage
		}
		for _, choice := range chunk.Choices {
			if choice.Delta.Role != "" {
				role = choice.Delta.Role
			}
			content.WriteString(choice.Delta.Content)
			for _, tc := range choice.Delta.ToolCalls {
				buf := toolCalls[tc.Index]
				if buf == nil {
					buf = &bytes.Buffer{}
					toolCalls[tc.Index] = buf
					toolOrder = append(toolOrder, tc.Index)
				}
				buf.WriteString(tc.Function.Arguments)
				if tc.Function.Name != "" {
					toolNames[tc.Index] = tc.Function.Name
				}
				// Preserve the upstream tool call id: fabricated ids break
				// call_id pairing in responses dialects and in agent clients.
				if tc.ID != "" {
					toolIDs[tc.Index] = tc.ID
				}
			}
			if choice.FinishReason != nil && *choice.FinishReason != "" {
				finish = *choice.FinishReason
			}
		}
	}
	// Some upstreams never emit a role delta; the folded message must still
	// carry a role or downstream translators drop it as malformed (which
	// reads as an empty reply to the client).
	if role == "" {
		role = "assistant"
	}
	message := map[string]any{"role": role}
	if content.Len() > 0 {
		message["content"] = content.String()
	} else if len(toolCalls) == 0 {
		message["content"] = nil
	}
	if len(toolCalls) > 0 {
		calls := make([]any, 0, len(toolCalls))
		for _, idx := range toolOrder {
			id := toolIDs[idx]
			if id == "" {
				id = fmt.Sprintf("call_%d", idx)
			}
			calls = append(calls, map[string]any{
				"id":   id,
				"type": "function",
				"function": map[string]any{
					"name":      toolNames[idx],
					"arguments": toolCalls[idx].String(),
				},
			})
		}
		message["tool_calls"] = calls
		if finish == "" {
			finish = "tool_calls"
		}
	}
	if finish == "" {
		finish = "stop"
	}
	out := map[string]any{
		"id":     id,
		"object": "chat.completion",
		"model":  model,
		"choices": []any{map[string]any{
			"index":         0,
			"message":       message,
			"finish_reason": finish,
		}},
	}
	if usage != nil {
		out["usage"] = json.RawMessage(usage)
	}
	return json.Marshal(out)
}

type toolCallDelta struct {
	Index    int    `json:"index"`
	ID       string `json:"id"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// foldResponsesSSE merges response.* events into one response object.
func foldResponsesSSE(sse []byte) ([]byte, error) {
	var (
		id, model, status string
		output            []json.RawMessage
		usage             json.RawMessage
	)
	for _, frame := range sseFrames(sse) {
		if frame == "[DONE]" {
			break
		}
		var ev struct {
			Type   string          `json:"type"`
			ID     string          `json:"id"`
			Model  string          `json:"model"`
			Status string          `json:"status"`
			Item   json.RawMessage `json:"item"`
			Usage  json.RawMessage `json:"usage"`
		}
		if err := json.Unmarshal([]byte(frame), &ev); err != nil {
			continue
		}
		switch ev.Type {
		case "response.created":
			if ev.ID != "" {
				id = ev.ID
			}
		case "response.output_item.added":
			if len(ev.Item) > 0 {
				output = append(output, ev.Item)
			}
		case "response.completed", "response.incomplete", "response.failed":
			// Terminal events carry the full response object; prefer it.
			// incomplete/failed must do the same as completed, or a truncated
			// answer is folded with status "completed" and no usage, silently
			// reporting a partial reply as a complete one.
			var terminal struct {
				Response json.RawMessage `json:"response"`
			}
			if err := json.Unmarshal([]byte(frame), &terminal); err == nil && len(terminal.Response) > 0 {
				return terminal.Response, nil
			}
			if len(ev.Usage) > 0 && string(ev.Usage) != "null" {
				usage = ev.Usage
			}
		}
		if ev.Model != "" {
			model = ev.Model
		}
		if ev.Status != "" {
			status = ev.Status
		}
	}
	if status == "" {
		status = "completed"
	}
	out := map[string]any{
		"id":     id,
		"object": "response",
		"status": status,
		"model":  model,
		"output": output,
	}
	if usage != nil {
		out["usage"] = json.RawMessage(usage)
	}
	return json.Marshal(out)
}

// sseFrames splits an SSE byte stream into individual data frame payloads.
func sseFrames(sse []byte) []string {
	var frames []string
	for _, line := range strings.Split(string(sse), "\n") {
		line = strings.TrimRight(line, "\r")
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" {
			continue
		}
		frames = append(frames, data)
	}
	return frames
}

// ---------------------------------------------------------------------------
// responses → chat conversion
//
// The plugin declares chat-completions as its executor output format, so the
// host expects chat-format payloads from the plugin and translates them for
// responses/claude/gemini clients itself. Models routed to the /responses
// endpoint (muse-spark) return Responses API events instead, so both the
// stream and the folded non-stream payload must be converted to chat format
// before they reach the host; otherwise the host translators drop the frames
// (stream) or produce garbage (non-stream).
// ---------------------------------------------------------------------------

// responsesChatStreamConverter converts Responses API stream events into
// chat.completion.chunk payloads, statefully.
type responsesChatStreamConverter struct {
	model     string
	id        string
	created   int64
	sawDelta  map[string]bool
	toolIdx   map[string]int // responses item id -> chat tool_calls index
	nextTool  int
	toolArgs  map[string]bool // item id -> arguments already streamed
	toolCallN int
	textN     int
}

func newResponsesChatStreamConverter(model string) *responsesChatStreamConverter {
	return &responsesChatStreamConverter{
		model:    model,
		sawDelta: map[string]bool{},
		toolIdx:  map[string]int{},
		toolArgs: map[string]bool{},
	}
}

// responsesOutputItem is one entry of a responses completion payload's
// output array.
type responsesOutputItem struct {
	ID        string          `json:"id"`
	Type      string          `json:"type"`
	CallID    string          `json:"call_id"`
	Name      string          `json:"name"`
	Arguments string          `json:"arguments"`
	Input     json.RawMessage `json:"input"`
	Content   []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
}

// recoverFinalItems emits tool calls and text that arrived only inside the
// final completion payload instead of as per-item streaming events. Some
// gateways deliver a short streamed prefix and then the authoritative items
// in response.completed; without this recovery a finished turn would reach
// the client with its tool call (or its whole answer) silently missing while
// still looking complete.
func (c *responsesChatStreamConverter) recoverFinalItems(output []responsesOutputItem) [][]byte {
	var outs [][]byte
	if c.toolCallN == 0 {
		for _, item := range output {
			if item.Type != "function_call" && item.Type != "custom_tool_call" {
				continue
			}
			id := item.CallID
			if id == "" {
				id = item.ID
			}
			args := item.Arguments
			if args == "" {
				args = rawString(item.Input)
			}
			c.toolCallN++
			outs = append(outs, c.chunk(map[string]any{
				"tool_calls": []any{map[string]any{
					"index": c.toolIndexFor(item.ID),
					"id":    id,
					"type":  "function",
					"function": map[string]any{
						"name":      item.Name,
						"arguments": args,
					},
				}},
			}, nil, nil))
		}
	}
	if c.textN == 0 {
		var text strings.Builder
		for _, item := range output {
			if item.Type != "message" {
				continue
			}
			for _, part := range item.Content {
				if part.Type == "output_text" || part.Type == "text" {
					text.WriteString(part.Text)
				}
			}
		}
		if text.Len() > 0 {
			c.textN++
			outs = append(outs, c.chunk(map[string]any{"content": text.String()}, nil, nil))
		}
	}
	return outs
}

// convert maps one SSE payload (responses event JSON, framing stripped) to
// zero or more chat chunk JSON payloads. A non-JSON payload is dropped.
func (c *responsesChatStreamConverter) convert(payload []byte) ([][]byte, error) {
	var ev struct {
		Type      string          `json:"type"`
		ItemID    string          `json:"item_id"`
		Delta     string          `json:"delta"`
		Arguments string          `json:"arguments"`
		Input     json.RawMessage `json:"input"`
		Response  struct {
			ID        string `json:"id"`
			CreatedAt int64  `json:"created_at"`
			Status    string `json:"status"`

			IncompleteDetails *struct {
				Reason string `json:"reason"`
			} `json:"incomplete_details"`
			Error *struct {
				Message string `json:"message"`
			} `json:"error"`
			Usage *struct {
				InputTokens  int64 `json:"input_tokens"`
				OutputTokens int64 `json:"output_tokens"`
				TotalTokens  int64 `json:"total_tokens"`
			} `json:"usage"`
			Output []responsesOutputItem `json:"output"`
		} `json:"response"`
		Item struct {
			ID        string          `json:"id"`
			Type      string          `json:"type"`
			CallID    string          `json:"call_id"`
			Name      string          `json:"name"`
			Arguments string          `json:"arguments"`
			Input     json.RawMessage `json:"input"`
			Content   []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
		} `json:"item"`
	}
	if err := json.Unmarshal(payload, &ev); err != nil {
		return nil, nil
	}
	switch ev.Type {
	case "response.created":
		if ev.Response.ID != "" {
			c.id = ev.Response.ID
		}
		if ev.Response.CreatedAt != 0 {
			c.created = ev.Response.CreatedAt
		}
		return [][]byte{c.chunk(map[string]any{"role": "assistant"}, nil, nil)}, nil
	case "response.output_item.added":
		// A function_call item announces the tool invocation: emit the
		// chat-completions tool_calls head (id + name, empty arguments) so
		// downstream translators can map it to a function_call item.
		if ev.Item.Type == "function_call" || ev.Item.Type == "custom_tool_call" {
			idx := c.toolIndexFor(ev.Item.ID)
			c.toolCallN++
			id := ev.Item.CallID
			if id == "" {
				id = ev.Item.ID
			}
			return [][]byte{c.chunk(map[string]any{
				"tool_calls": []any{map[string]any{
					"index": idx,
					"id":    id,
					"type":  "function",
					"function": map[string]any{
						"name":      ev.Item.Name,
						"arguments": "",
					},
				}},
			}, nil, nil)}, nil
		}
		return nil, nil
	case "response.function_call_arguments.delta", "response.custom_tool_call_input.delta":
		if ev.ItemID != "" {
			c.sawDelta[ev.ItemID] = true
			c.toolArgs[ev.ItemID] = true
		}
		if ev.Delta == "" {
			return nil, nil
		}
		return [][]byte{c.chunk(map[string]any{
			"tool_calls": []any{map[string]any{
				"index": c.toolIndexFor(ev.ItemID),
				"function": map[string]any{
					"arguments": ev.Delta,
				},
			}},
		}, nil, nil)}, nil
	case "response.function_call_arguments.done", "response.custom_tool_call_input.done":
		// Upstreams that do not stream argument fragments deliver the full
		// argument string once here; emit it unless deltas already covered it.
		if ev.ItemID != "" && !c.toolArgs[ev.ItemID] {
			c.toolArgs[ev.ItemID] = true
			c.sawDelta[ev.ItemID] = true
			if args := ev.Arguments; args != "" {
				return [][]byte{c.chunk(map[string]any{
					"tool_calls": []any{map[string]any{
						"index": c.toolIndexFor(ev.ItemID),
						"function": map[string]any{
							"arguments": args,
						},
					}},
				}, nil, nil)}, nil
			} else if args := rawString(ev.Input); args != "" {
				return [][]byte{c.chunk(map[string]any{
					"tool_calls": []any{map[string]any{
						"index": c.toolIndexFor(ev.ItemID),
						"function": map[string]any{
							"arguments": args,
						},
					}},
				}, nil, nil)}, nil
			}
		}
		return nil, nil
	case "response.output_text.delta":
		if ev.ItemID != "" {
			c.sawDelta[ev.ItemID] = true
		}
		if ev.Delta == "" {
			return nil, nil
		}
		c.textN++
		return [][]byte{c.chunk(map[string]any{"content": ev.Delta}, nil, nil)}, nil
	case "response.reasoning_summary_text.delta", "response.reasoning_text.delta":
		if ev.ItemID != "" {
			c.sawDelta[ev.ItemID] = true
		}
		if ev.Delta == "" {
			return nil, nil
		}
		return [][]byte{c.chunk(map[string]any{"reasoning_content": ev.Delta}, nil, nil)}, nil
	case "response.output_item.done":
		// Fallback: an item completed without streamed deltas (some upstreams
		// only send the final item). Emit its text once.
		if ev.Item.Type != "function_call" && ev.Item.Type != "custom_tool_call" {
			if ev.Item.Type != "message" || ev.Item.ID == "" || c.sawDelta[ev.Item.ID] {
				return nil, nil
			}
			var text strings.Builder
			for _, part := range ev.Item.Content {
				if part.Type == "output_text" || part.Type == "text" {
					text.WriteString(part.Text)
				}
			}
			if text.Len() == 0 {
				return nil, nil
			}
			c.textN++
			return [][]byte{c.chunk(map[string]any{"content": text.String()}, nil, nil)}, nil
		}
		if ev.Item.ID == "" || c.sawDelta[ev.Item.ID] {
			return nil, nil
		}
		c.sawDelta[ev.Item.ID] = true
		c.toolArgs[ev.Item.ID] = true
		c.toolCallN++
		id := ev.Item.CallID
		if id == "" {
			id = ev.Item.ID
		}
		args := ev.Item.Arguments
		if args == "" {
			args = rawString(ev.Item.Input)
		}
		return [][]byte{c.chunk(map[string]any{
			"tool_calls": []any{map[string]any{
				"index": c.toolIndexFor(ev.Item.ID),
				"id":    id,
				"type":  "function",
				"function": map[string]any{
					"name":      ev.Item.Name,
					"arguments": args,
				},
			}},
		}, nil, nil)}, nil
	case "response.completed":
		var usage map[string]any
		if ev.Response.Usage != nil {
			usage = map[string]any{
				"prompt_tokens":     ev.Response.Usage.InputTokens,
				"completion_tokens": ev.Response.Usage.OutputTokens,
				"total_tokens":      ev.Response.Usage.TotalTokens,
			}
		}
		outs := c.recoverFinalItems(ev.Response.Output)
		finish := "stop"
		if c.toolCallN > 0 {
			finish = "tool_calls"
		}
		outs = append(outs, c.chunk(map[string]any{}, strPtr(finish), usage))
		return outs, nil
	case "response.incomplete":
		// Truncated upstream responses are not failures: everything streamed
		// so far is valid, so finish the chat stream normally instead of
		// erroring the turn away.
		var usage map[string]any
		if ev.Response.Usage != nil {
			usage = map[string]any{
				"prompt_tokens":     ev.Response.Usage.InputTokens,
				"completion_tokens": ev.Response.Usage.OutputTokens,
				"total_tokens":      ev.Response.Usage.TotalTokens,
			}
		}
		outs := c.recoverFinalItems(ev.Response.Output)
		finish := "length"
		if r := ev.Response.IncompleteDetails; r != nil && r.Reason != "" && r.Reason != "max_output_tokens" {
			finish = "content_filter"
		}
		outs = append(outs, c.chunk(map[string]any{}, strPtr(finish), usage))
		return outs, nil
	case "response.failed":
		msg := "upstream response failed"
		if ev.Response.Error != nil && ev.Response.Error.Message != "" {
			msg = ev.Response.Error.Message
		} else if ev.Response.Status != "" {
			msg = "upstream response " + ev.Response.Status
		}
		return nil, fmt.Errorf("zen responses stream failed: %s", msg)
	}
	return nil, nil
}

// toolIndexFor maps a responses item id to a stable chat tool_calls index,
// assigning the next slot on first sight.
func (c *responsesChatStreamConverter) toolIndexFor(itemID string) int {
	if idx, ok := c.toolIdx[itemID]; ok {
		return idx
	}
	idx := c.nextTool
	c.nextTool++
	c.toolIdx[itemID] = idx
	return idx
}

// chunk renders one chat.completion.chunk payload.
func (c *responsesChatStreamConverter) chunk(delta map[string]any, finish *string, usage map[string]any) []byte {
	id := c.id
	if id == "" {
		id = canonicalID("chatcmpl", c.model)
	}
	created := c.created
	if created == 0 {
		created = time.Now().Unix()
	}
	m := map[string]any{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": created,
		"model":   c.model,
		"choices": []any{map[string]any{
			"index":         0,
			"delta":         delta,
			"finish_reason": finish,
		}},
	}
	if usage != nil {
		m["usage"] = usage
	}
	b, err := json.Marshal(m)
	if err != nil {
		return []byte("{}")
	}
	return b
}

// responsesCompletionToChat converts a folded Responses API response object
// into a chat.completion object for the non-streaming path.
func responsesCompletionToChat(body []byte, model string) ([]byte, error) {
	var resp struct {
		ID                string `json:"id"`
		Model             string `json:"model"`
		Status            string `json:"status"`
		CreatedAt         int64  `json:"created_at"`
		IncompleteDetails *struct {
			Reason string `json:"reason"`
		} `json:"incomplete_details"`
		Output []struct {
			Type      string          `json:"type"`
			Role      string          `json:"role"`
			CallID    string          `json:"call_id"`
			Name      string          `json:"name"`
			Arguments string          `json:"arguments"`
			Input     json.RawMessage `json:"input"`
			Content   []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
			Summary []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"summary"`
		} `json:"output"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
		Usage *struct {
			InputTokens  int64 `json:"input_tokens"`
			OutputTokens int64 `json:"output_tokens"`
			TotalTokens  int64 `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("zen responses payload is not valid JSON: %w", err)
	}
	if resp.Status == "failed" || resp.Error != nil {
		msg := "upstream response failed"
		if resp.Error != nil && resp.Error.Message != "" {
			msg = resp.Error.Message
		}
		return nil, fmt.Errorf("zen responses stream failed: %s", msg)
	}
	var content, reasoning strings.Builder
	var toolCalls []any
	for _, item := range resp.Output {
		switch item.Type {
		case "message":
			for _, part := range item.Content {
				if part.Type == "output_text" || part.Type == "text" {
					content.WriteString(part.Text)
				}
			}
		case "reasoning":
			for _, sum := range item.Summary {
				reasoning.WriteString(sum.Text)
			}
		case "function_call", "custom_tool_call":
			// Tool invocations must survive the responses -> chat fold, or
			// agent clients never see the model's tool calls.
			id := item.CallID
			if id == "" {
				id = canonicalID("call", item.Name+item.Arguments)
			}
			args := item.Arguments
			if args == "" {
				args = rawString(item.Input)
			}
			toolCalls = append(toolCalls, map[string]any{
				"id":   id,
				"type": "function",
				"function": map[string]any{
					"name":      item.Name,
					"arguments": args,
				},
			})
		}
	}
	if resp.Model != "" {
		model = resp.Model
	}
	id := resp.ID
	if id == "" {
		id = canonicalID("chatcmpl", model)
	}
	created := resp.CreatedAt
	if created == 0 {
		created = time.Now().Unix()
	}
	message := map[string]any{"role": "assistant", "content": content.String()}
	if reasoning.Len() > 0 {
		message["reasoning_content"] = reasoning.String()
	}
	finish := "stop"
	if len(toolCalls) > 0 {
		message["tool_calls"] = toolCalls
		finish = "tool_calls"
	} else if resp.Status == "incomplete" {
		// Truncated (not failed) upstream: report the chat finish reason
		// that matches the truncation cause instead of claiming "stop".
		finish = "length"
		if r := resp.IncompleteDetails; r != nil && r.Reason != "" && r.Reason != "max_output_tokens" {
			finish = "content_filter"
		}
	}
	m := map[string]any{
		"id":      id,
		"object":  "chat.completion",
		"created": created,
		"model":   model,
		"choices": []any{map[string]any{
			"index":         0,
			"message":       message,
			"finish_reason": finish,
		}},
	}
	if resp.Usage != nil {
		m["usage"] = map[string]any{
			"prompt_tokens":     resp.Usage.InputTokens,
			"completion_tokens": resp.Usage.OutputTokens,
			"total_tokens":      resp.Usage.TotalTokens,
		}
	}
	return json.Marshal(m)
}

// stripSSEDataFraming removes the "data:" prefix and surrounding whitespace
// from one emitted SSE frame, yielding the raw payload.
func stripSSEDataFraming(frame []byte) string {
	s := strings.TrimSpace(string(frame))
	if after, ok := strings.CutPrefix(s, "data:"); ok {
		s = strings.TrimSpace(after)
	}
	return s
}

// strPtr is a tiny helper for optional string fields.
func strPtr(s string) *string { return &s }

// rawString decodes an event field that is a JSON string in the spec but
// may arrive as an arbitrary JSON value (custom tool inputs). Objects are
// returned as their compact JSON text.
func rawString(raw json.RawMessage) string {
	if len(raw) == 0 || string(raw) == "null" {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	return string(raw)
}

func truncate(b []byte, n int) string {
	s := strings.TrimSpace(string(b))
	if len(s) > n {
		return s[:n] + "..."
	}
	return s
}

// ---------------------------------------------------------------------------
// envelope helpers
// ---------------------------------------------------------------------------

func okEnvelopeJSON(result any) ([]byte, error) {
	raw, err := json.Marshal(result)
	if err != nil {
		return nil, err
	}
	return json.Marshal(envelope{OK: true, Result: raw})
}

func errorEnvelope(code, message string) []byte {
	raw, _ := json.Marshal(envelope{OK: false, Error: &envelopeError{Code: code, Message: message}})
	return raw
}

func writeResponse(response *C.cliproxy_buffer, raw []byte) {
	if response == nil || len(raw) == 0 {
		return
	}
	ptr := C.CBytes(raw)
	if ptr == nil {
		return
	}
	response.ptr = ptr
	response.len = C.size_t(len(raw))
}
