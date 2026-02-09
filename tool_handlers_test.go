package main

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

// newTestHandlers creates a ToolHandlers with a mock Ollama client.
func newTestHandlers(mock OllamaClient) *ToolHandlers {
	store := NewTaskStore()
	pool := newTestPool(store, 2, mock)
	return &ToolHandlers{store: store, pool: pool}
}

// ---------------------------------------------------------------------------
// submit_tasks
// ---------------------------------------------------------------------------

func TestHandleSubmitTasks(t *testing.T) {
	mock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			fn(api.ChatResponse{Message: api.Message{Content: "done"}})
			return nil
		},
	}
	h := newTestHandlers(mock)

	args := SubmitTasksArgs{
		Tasks: []TaskSpec{
			{SystemPrompt: "sys", Prompt: "p1", Tag: "batch1"},
			{SystemPrompt: "sys", Prompt: "p2", Tag: "batch1"},
		},
	}

	_, out, err := h.handleSubmitTasks(context.Background(), nil, args)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(out.TaskIDs) != 2 {
		t.Fatalf("expected 2 task IDs, got %d", len(out.TaskIDs))
	}
	// IDs should be non-empty UUIDs
	for _, id := range out.TaskIDs {
		if len(id) < 10 {
			t.Fatalf("task ID looks invalid: %q", id)
		}
	}

	// Wait for tasks to complete
	for _, id := range out.TaskIDs {
		waitForStatus(t, h.store, id, 2*time.Second, "completed")
	}
}

// ---------------------------------------------------------------------------
// submit_tasks batch limit
// ---------------------------------------------------------------------------

func TestHandleSubmitTasksBatchLimit(t *testing.T) {
	h := newTestHandlers(&mockOllamaClient{})

	tasks := make([]TaskSpec, 501)
	for i := range tasks {
		tasks[i] = TaskSpec{SystemPrompt: "sys", Prompt: "p"}
	}

	_, _, err := h.handleSubmitTasks(context.Background(), nil, SubmitTasksArgs{Tasks: tasks})
	if err == nil {
		t.Fatal("expected error for batch > 500")
	}
}

// ---------------------------------------------------------------------------
// check_tasks
// ---------------------------------------------------------------------------

func TestHandleCheckTasks(t *testing.T) {
	h := newTestHandlers(&mockOllamaClient{})

	// Manually populate the store
	h.store.Add([]*Task{
		makeTask("a", "grp", "pending"),
		makeTask("b", "grp", "pending"),
	})
	h.store.SetRunning("b")
	h.store.SetCompleted("b", "result")

	_, out, err := h.handleCheckTasks(context.Background(), nil, CheckTasksArgs{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out.Summary.Total != 2 {
		t.Fatalf("expected total 2, got %d", out.Summary.Total)
	}
	if out.Summary.Pending != 1 || out.Summary.Completed != 1 {
		t.Fatalf("unexpected summary: %+v", out.Summary)
	}
	if len(out.Tasks) != 2 {
		t.Fatalf("expected 2 task statuses, got %d", len(out.Tasks))
	}
}

// ---------------------------------------------------------------------------
// check_tasks filtering
// ---------------------------------------------------------------------------

func TestHandleCheckTasksFilterByTag(t *testing.T) {
	h := newTestHandlers(&mockOllamaClient{})
	h.store.Add([]*Task{
		makeTask("a", "alpha", "pending"),
		makeTask("b", "beta", "pending"),
		makeTask("c", "alpha", "pending"),
	})

	_, out, err := h.handleCheckTasks(context.Background(), nil, CheckTasksArgs{Tag: "alpha"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out.Summary.Total != 2 {
		t.Fatalf("expected 2 for tag alpha, got %d", out.Summary.Total)
	}
}

func TestHandleCheckTasksFilterByIDs(t *testing.T) {
	h := newTestHandlers(&mockOllamaClient{})
	h.store.Add([]*Task{
		makeTask("a", "", "pending"),
		makeTask("b", "", "pending"),
		makeTask("c", "", "pending"),
	})

	_, out, err := h.handleCheckTasks(context.Background(), nil, CheckTasksArgs{TaskIDs: []string{"a", "c"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out.Summary.Total != 2 {
		t.Fatalf("expected 2, got %d", out.Summary.Total)
	}
}

// ---------------------------------------------------------------------------
// get_result
// ---------------------------------------------------------------------------

func TestHandleGetResult(t *testing.T) {
	h := newTestHandlers(&mockOllamaClient{})
	h.store.Add([]*Task{makeTask("a", "", "pending")})
	h.store.SetRunning("a")
	h.store.SetCompleted("a", "the answer")

	_, out, err := h.handleGetResult(context.Background(), nil, GetResultArgs{TaskIDs: []string{"a"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(out.Results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(out.Results))
	}
	if out.Results[0].Content != "the answer" {
		t.Fatalf("expected 'the answer', got %q", out.Results[0].Content)
	}
	if out.Results[0].Status != "completed" {
		t.Fatalf("expected completed, got %s", out.Results[0].Status)
	}
}

// ---------------------------------------------------------------------------
// get_result not found
// ---------------------------------------------------------------------------

func TestHandleGetResultNotFound(t *testing.T) {
	h := newTestHandlers(&mockOllamaClient{})

	_, out, err := h.handleGetResult(context.Background(), nil, GetResultArgs{TaskIDs: []string{"nope", "nah"}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(out.Results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(out.Results))
	}
	for _, r := range out.Results {
		if r.Status != "not_found" {
			t.Fatalf("expected not_found, got %s", r.Status)
		}
	}
}

// ---------------------------------------------------------------------------
// cancel_tasks
// ---------------------------------------------------------------------------

func TestHandleCancelTasks(t *testing.T) {
	h := newTestHandlers(&mockOllamaClient{})
	h.store.Add([]*Task{
		makeTask("a", "x", "pending"),
		makeTask("b", "x", "pending"),
		makeTask("c", "y", "pending"),
	})

	_, out, err := h.handleCancelTasks(context.Background(), nil, CancelTasksArgs{Tag: "x"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out.Cancelled != 2 {
		t.Fatalf("expected 2 cancelled, got %d", out.Cancelled)
	}
	if h.store.Get("c").Status != "pending" {
		t.Fatal("task c should still be pending")
	}
}

// ---------------------------------------------------------------------------
// list_models
// ---------------------------------------------------------------------------

func TestHandleListModels(t *testing.T) {
	mock := &mockOllamaClient{
		listFn: func(ctx context.Context) (*api.ListResponse, error) {
			return &api.ListResponse{
				Models: []api.ListModelResponse{
					{
						Name: "qwen2.5:14b",
						Size: 8_000_000_000,
						Details: api.ModelDetails{
							ParameterSize:     "14B",
							QuantizationLevel: "Q4_K_M",
							Family:            "qwen2",
						},
					},
					{
						Name: "llama3:8b",
						Size: 4_000_000_000,
						Details: api.ModelDetails{
							ParameterSize:     "8B",
							QuantizationLevel: "Q4_0",
							Family:            "llama",
						},
					},
				},
			}, nil
		},
	}
	h := newTestHandlers(mock)

	_, out, err := h.handleListModels(context.Background(), nil, ListModelsArgs{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(out.Models) != 2 {
		t.Fatalf("expected 2 models, got %d", len(out.Models))
	}
	if out.Models[0].Name != "qwen2.5:14b" {
		t.Fatalf("expected qwen2.5:14b, got %s", out.Models[0].Name)
	}
	if out.Models[0].ParameterSize != "14B" {
		t.Fatalf("expected 14B, got %s", out.Models[0].ParameterSize)
	}
	if out.Models[1].Family != "llama" {
		t.Fatalf("expected llama, got %s", out.Models[1].Family)
	}
}

// ---------------------------------------------------------------------------
// submit_tasks propagates input_file to Ollama
// ---------------------------------------------------------------------------

func TestHandleSubmitTasksInputFile(t *testing.T) {
	var capturedUserMsg string
	mock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			capturedUserMsg = req.Messages[1].Content
			fn(api.ChatResponse{Message: api.Message{Content: "done"}})
			return nil
		},
	}
	h := newTestHandlers(mock)

	// Create a temp file with content
	tmpFile, err := os.CreateTemp("", "test-input-*.txt")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())
	if _, err := tmpFile.WriteString("package main"); err != nil {
		t.Fatal(err)
	}
	tmpFile.Close()

	args := SubmitTasksArgs{
		Tasks: []TaskSpec{
			{
				SystemPrompt: "sys",
				Prompt:       "Review this",
				InputFile:    tmpFile.Name(),
			},
		},
	}
	_, out, err := h.handleSubmitTasks(context.Background(), nil, args)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	waitForStatus(t, h.store, out.TaskIDs[0], 2*time.Second, "completed")

	if capturedUserMsg != "Review this\n\npackage main" {
		t.Fatalf("expected concatenated prompt+file content, got %q", capturedUserMsg)
	}
}

func TestHandleListModelsError(t *testing.T) {
	mock := &mockOllamaClient{
		listFn: func(ctx context.Context) (*api.ListResponse, error) {
			return nil, context.DeadlineExceeded
		},
	}
	h := newTestHandlers(mock)

	_, _, err := h.handleListModels(context.Background(), nil, ListModelsArgs{})
	if err == nil {
		t.Fatal("expected error from list_models")
	}
}

// ---------------------------------------------------------------------------
// submit_tasks: empty batch
// ---------------------------------------------------------------------------

func TestHandleSubmitTasksEmptyBatch(t *testing.T) {
	h := newTestHandlers(&mockOllamaClient{})

	_, out, err := h.handleSubmitTasks(context.Background(), nil, SubmitTasksArgs{Tasks: []TaskSpec{}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(out.TaskIDs) != 0 {
		t.Fatalf("expected 0 task IDs, got %d", len(out.TaskIDs))
	}
}

// ---------------------------------------------------------------------------
// submit_tasks: exact batch limit (maxBatchSize)
// ---------------------------------------------------------------------------

func TestHandleSubmitTasksExactLimit(t *testing.T) {
	mock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			fn(api.ChatResponse{Message: api.Message{Content: "ok"}})
			return nil
		},
	}
	h := newTestHandlers(mock)

	tasks := make([]TaskSpec, maxBatchSize)
	for i := range tasks {
		tasks[i] = TaskSpec{SystemPrompt: "sys", Prompt: "p"}
	}

	_, out, err := h.handleSubmitTasks(context.Background(), nil, SubmitTasksArgs{Tasks: tasks})
	if err != nil {
		t.Fatalf("expected no error for %d tasks, got: %v", maxBatchSize, err)
	}
	if len(out.TaskIDs) != maxBatchSize {
		t.Fatalf("expected %d IDs, got %d", maxBatchSize, len(out.TaskIDs))
	}
}

// ---------------------------------------------------------------------------
// submit_tasks: default model applied
// ---------------------------------------------------------------------------

func TestHandleSubmitTasksDefaultModel(t *testing.T) {
	var capturedModel string
	mock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			capturedModel = req.Model
			fn(api.ChatResponse{Message: api.Message{Content: "ok"}})
			return nil
		},
	}
	h := newTestHandlers(mock)

	args := SubmitTasksArgs{
		Tasks: []TaskSpec{
			{SystemPrompt: "sys", Prompt: "p"}, // no model specified
		},
	}
	_, out, err := h.handleSubmitTasks(context.Background(), nil, args)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	waitForStatus(t, h.store, out.TaskIDs[0], 2*time.Second, "completed")

	if capturedModel != getDefaultModel() {
		t.Fatalf("expected default model %q, got %q", getDefaultModel(), capturedModel)
	}
}

// ---------------------------------------------------------------------------
// submit_tasks: default response_hint
// ---------------------------------------------------------------------------

func TestHandleSubmitTasksDefaultResponseHint(t *testing.T) {
	mock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			fn(api.ChatResponse{Message: api.Message{Content: "ok"}})
			return nil
		},
	}
	h := newTestHandlers(mock)

	args := SubmitTasksArgs{
		Tasks: []TaskSpec{
			{SystemPrompt: "sys", Prompt: "p"}, // no response_hint
		},
	}
	_, out, err := h.handleSubmitTasks(context.Background(), nil, args)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	waitForStatus(t, h.store, out.TaskIDs[0], 2*time.Second, "completed")

	// ResponseHint is not cleared on completion, so we can check it
	task := h.store.Get(out.TaskIDs[0])
	if task.ResponseHint != "content" {
		t.Fatalf("expected default response_hint 'content', got %q", task.ResponseHint)
	}
}

// ---------------------------------------------------------------------------
// getDefaultModel env var
// ---------------------------------------------------------------------------

func TestGetDefaultModelEnvVar(t *testing.T) {
	t.Setenv("DEFAULT_MODEL", "custom-model:7b")
	if got := getDefaultModel(); got != "custom-model:7b" {
		t.Fatalf("expected custom-model:7b, got %s", got)
	}
}

func TestGetDefaultModelDefault(t *testing.T) {
	t.Setenv("DEFAULT_MODEL", "")
	if got := getDefaultModel(); got != defaultModel {
		t.Fatalf("expected %s, got %s", defaultModel, got)
	}
}

// ---------------------------------------------------------------------------
// cancel_tasks: cancel all (empty filters)
// ---------------------------------------------------------------------------

func TestHandleCancelTasksAll(t *testing.T) {
	h := newTestHandlers(&mockOllamaClient{})
	h.store.Add([]*Task{
		makeTask("a", "x", "pending"),
		makeTask("b", "y", "pending"),
		makeTask("c", "", "pending"),
	})

	_, out, err := h.handleCancelTasks(context.Background(), nil, CancelTasksArgs{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out.Cancelled != 3 {
		t.Fatalf("expected 3 cancelled, got %d", out.Cancelled)
	}
}

// ---------------------------------------------------------------------------
// get_result: empty task IDs
// ---------------------------------------------------------------------------

func TestHandleGetResultEmptyIDs(t *testing.T) {
	h := newTestHandlers(&mockOllamaClient{})

	_, out, err := h.handleGetResult(context.Background(), nil, GetResultArgs{TaskIDs: []string{}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(out.Results) != 0 {
		t.Fatalf("expected 0 results, got %d", len(out.Results))
	}
}

// ---------------------------------------------------------------------------
// check_tasks: empty store
// ---------------------------------------------------------------------------

func TestHandleCheckTasksEmptyStore(t *testing.T) {
	h := newTestHandlers(&mockOllamaClient{})

	_, out, err := h.handleCheckTasks(context.Background(), nil, CheckTasksArgs{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out.Summary.Total != 0 {
		t.Fatalf("expected 0 total, got %d", out.Summary.Total)
	}
	if len(out.Tasks) != 0 {
		t.Fatalf("expected 0 tasks, got %d", len(out.Tasks))
	}
}

// ---------------------------------------------------------------------------
// submit_tasks: absolute path validation
// ---------------------------------------------------------------------------

func TestHandleSubmitTasksAbsolutePathValidation(t *testing.T) {
	h := newTestHandlers(&mockOllamaClient{})

	// Relative input_file should be rejected
	_, _, err := h.handleSubmitTasks(context.Background(), nil, SubmitTasksArgs{
		Tasks: []TaskSpec{
			{SystemPrompt: "sys", Prompt: "p", InputFile: "relative/path.go"},
		},
	})
	if err == nil {
		t.Fatal("expected error for relative input_file")
	}

	// Relative output_file should be rejected
	_, _, err = h.handleSubmitTasks(context.Background(), nil, SubmitTasksArgs{
		Tasks: []TaskSpec{
			{SystemPrompt: "sys", Prompt: "p", OutputFile: "relative/output.go"},
		},
	})
	if err == nil {
		t.Fatal("expected error for relative output_file")
	}

	// Absolute paths should be accepted (no error at submit time)
	mock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			fn(api.ChatResponse{Message: api.Message{Content: "ok"}})
			return nil
		},
	}
	h2 := newTestHandlers(mock)
	_, _, err = h2.handleSubmitTasks(context.Background(), nil, SubmitTasksArgs{
		Tasks: []TaskSpec{
			{SystemPrompt: "sys", Prompt: "p", InputFile: "/abs/path.go", OutputFile: "/abs/output.go"},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error for absolute paths: %v", err)
	}
}

// ---------------------------------------------------------------------------
// submit_tasks: StripMarkdownFences default (nil → true)
// ---------------------------------------------------------------------------

func TestHandleSubmitTasksStripMarkdownFencesDefault(t *testing.T) {
	mock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			fn(api.ChatResponse{Message: api.Message{Content: "ok"}})
			return nil
		},
	}
	h := newTestHandlers(mock)

	args := SubmitTasksArgs{
		Tasks: []TaskSpec{
			{SystemPrompt: "sys", Prompt: "p"}, // StripMarkdownFences not set (nil)
		},
	}
	_, out, err := h.handleSubmitTasks(context.Background(), nil, args)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	waitForStatus(t, h.store, out.TaskIDs[0], 2*time.Second, "completed")

	task := h.store.Get(out.TaskIDs[0])
	if !task.StripMarkdownFences {
		t.Fatal("StripMarkdownFences should default to true when nil")
	}
}

// ---------------------------------------------------------------------------
// submit_tasks: StripMarkdownFences explicit true
// ---------------------------------------------------------------------------

func TestHandleSubmitTasksStripMarkdownFencesExplicitTrue(t *testing.T) {
	mock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			fn(api.ChatResponse{Message: api.Message{Content: "ok"}})
			return nil
		},
	}
	h := newTestHandlers(mock)

	trueVal := true
	args := SubmitTasksArgs{
		Tasks: []TaskSpec{
			{SystemPrompt: "sys", Prompt: "p", StripMarkdownFences: &trueVal},
		},
	}
	_, out, err := h.handleSubmitTasks(context.Background(), nil, args)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	waitForStatus(t, h.store, out.TaskIDs[0], 2*time.Second, "completed")

	task := h.store.Get(out.TaskIDs[0])
	if !task.StripMarkdownFences {
		t.Fatal("StripMarkdownFences should be true when explicitly set to true")
	}
}

// ---------------------------------------------------------------------------
// check_tasks: elapsed_seconds populated after completion
// ---------------------------------------------------------------------------

func TestHandleCheckTasksElapsedSeconds(t *testing.T) {
	mock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			fn(api.ChatResponse{Message: api.Message{Content: "done"}})
			return nil
		},
	}
	h := newTestHandlers(mock)

	args := SubmitTasksArgs{
		Tasks: []TaskSpec{
			{SystemPrompt: "sys", Prompt: "p1", Tag: "elapsed"},
		},
	}
	_, submitOut, err := h.handleSubmitTasks(context.Background(), nil, args)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	waitForStatus(t, h.store, submitOut.TaskIDs[0], 2*time.Second, "completed")

	_, checkOut, err := h.handleCheckTasks(context.Background(), nil, CheckTasksArgs{Tag: "elapsed"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(checkOut.Tasks) != 1 {
		t.Fatalf("expected 1 task, got %d", len(checkOut.Tasks))
	}
	// elapsed_seconds should be >= 0 (task completes nearly instantly with mock)
	if checkOut.Tasks[0].ElapsedSeconds < 0 {
		t.Fatalf("elapsed_seconds should be >= 0, got %d", checkOut.Tasks[0].ElapsedSeconds)
	}
}

// ---------------------------------------------------------------------------
// submit_tasks: StripMarkdownFences explicit false
// ---------------------------------------------------------------------------

func TestHandleSubmitTasksStripMarkdownFencesExplicitFalse(t *testing.T) {
	mock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			fn(api.ChatResponse{Message: api.Message{Content: "ok"}})
			return nil
		},
	}
	h := newTestHandlers(mock)

	falseVal := false
	args := SubmitTasksArgs{
		Tasks: []TaskSpec{
			{SystemPrompt: "sys", Prompt: "p", StripMarkdownFences: &falseVal},
		},
	}
	_, out, err := h.handleSubmitTasks(context.Background(), nil, args)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	waitForStatus(t, h.store, out.TaskIDs[0], 2*time.Second, "completed")

	task := h.store.Get(out.TaskIDs[0])
	if task.StripMarkdownFences {
		t.Fatal("StripMarkdownFences should be false when explicitly set to false")
	}
}

// ---------------------------------------------------------------------------
// submit_tasks: concurrency parameter
// ---------------------------------------------------------------------------

func TestHandleSubmitTasksConcurrency(t *testing.T) {
	mock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			fn(api.ChatResponse{Message: api.Message{Content: "ok"}})
			return nil
		},
	}
	h := newTestHandlers(mock)

	concurrency := 5
	args := SubmitTasksArgs{
		Concurrency: &concurrency,
		Tasks: []TaskSpec{
			{SystemPrompt: "sys", Prompt: "p"},
		},
	}
	_, out, err := h.handleSubmitTasks(context.Background(), nil, args)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	waitForStatus(t, h.store, out.TaskIDs[0], 2*time.Second, "completed")

	if h.pool.Concurrency() != 5 {
		t.Fatalf("expected concurrency 5, got %d", h.pool.Concurrency())
	}
}

func TestHandleSubmitTasksConcurrencyZeroRejected(t *testing.T) {
	h := newTestHandlers(&mockOllamaClient{})

	concurrency := 0
	args := SubmitTasksArgs{
		Concurrency: &concurrency,
		Tasks: []TaskSpec{
			{SystemPrompt: "sys", Prompt: "p"},
		},
	}
	_, _, err := h.handleSubmitTasks(context.Background(), nil, args)
	if err == nil {
		t.Fatal("expected error for concurrency 0")
	}
}

func TestHandleSubmitTasksConcurrencyNegativeRejected(t *testing.T) {
	h := newTestHandlers(&mockOllamaClient{})

	concurrency := -1
	args := SubmitTasksArgs{
		Concurrency: &concurrency,
		Tasks: []TaskSpec{
			{SystemPrompt: "sys", Prompt: "p"},
		},
	}
	_, _, err := h.handleSubmitTasks(context.Background(), nil, args)
	if err == nil {
		t.Fatal("expected error for negative concurrency")
	}
}

func TestHandleSubmitTasksConcurrencyNilIgnored(t *testing.T) {
	mock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			fn(api.ChatResponse{Message: api.Message{Content: "ok"}})
			return nil
		},
	}
	h := newTestHandlers(mock)

	// Pool starts with concurrency 2 (from newTestHandlers)
	originalConcurrency := h.pool.Concurrency()

	args := SubmitTasksArgs{
		// Concurrency is nil (not set)
		Tasks: []TaskSpec{
			{SystemPrompt: "sys", Prompt: "p"},
		},
	}
	_, out, err := h.handleSubmitTasks(context.Background(), nil, args)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	waitForStatus(t, h.store, out.TaskIDs[0], 2*time.Second, "completed")

	if h.pool.Concurrency() != originalConcurrency {
		t.Fatalf("expected concurrency to remain %d, got %d", originalConcurrency, h.pool.Concurrency())
	}
}
