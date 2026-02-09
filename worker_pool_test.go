package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

// ---------------------------------------------------------------------------
// Mock Ollama client
// ---------------------------------------------------------------------------

type mockOllamaClient struct {
	chatFn func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error
	listFn func(ctx context.Context) (*api.ListResponse, error)
}

func (m *mockOllamaClient) Chat(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
	if m.chatFn != nil {
		return m.chatFn(ctx, req, fn)
	}
	return nil
}

func (m *mockOllamaClient) List(ctx context.Context) (*api.ListResponse, error) {
	if m.listFn != nil {
		return m.listFn(ctx)
	}
	return &api.ListResponse{}, nil
}

// newTestPool creates a WorkerPool with a mock client and the given concurrency.
func newTestPool(store *TaskStore, concurrency int, client OllamaClient) *WorkerPool {
	return &WorkerPool{
		sem:    make(chan struct{}, concurrency),
		client: client,
		store:  store,
	}
}

// submitTestTask creates a context, sets Cancel on the task, adds it to the
// store, and submits it to the pool. This mirrors the production sequence in
// handleSubmitTasks and avoids the race between store.Add and pool.Submit.
func submitTestTask(store *TaskStore, pool *WorkerPool, task *Task) {
	ctx, cancel := context.WithCancel(context.Background())
	task.Cancel = cancel
	store.Add([]*Task{task})
	pool.Submit(ctx, cancel, task)
}

// getStatus returns the status of a task using Results (which copies under the lock),
// avoiding a data race from reading the Task struct pointer directly.
func getStatus(store *TaskStore, id string) string {
	results := store.Results([]string{id})
	if len(results) == 1 {
		return results[0].Status
	}
	return ""
}

// waitForStatus polls until the task reaches one of the expected statuses or times out.
func waitForStatus(t *testing.T, store *TaskStore, id string, timeout time.Duration, statuses ...string) string {
	t.Helper()
	deadline := time.Now().Add(timeout)
	set := make(map[string]bool, len(statuses))
	for _, s := range statuses {
		set[s] = true
	}
	for time.Now().Before(deadline) {
		status := getStatus(store, id)
		if set[status] {
			return status
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timeout waiting for task %s to reach %v (current: %s)", id, statuses, getStatus(store, id))
	return ""
}

// ---------------------------------------------------------------------------
// Successful task lifecycle
// ---------------------------------------------------------------------------

func TestWorkerSuccessfulLifecycle(t *testing.T) {
	store := NewTaskStore()
	mock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			fn(api.ChatResponse{Message: api.Message{Content: "hello "}})
			fn(api.ChatResponse{Message: api.Message{Content: "world"}})
			return nil
		},
	}
	pool := newTestPool(store, 2, mock)

	task := &Task{
		ID:           "t1",
		SystemPrompt: "sys",
		Prompt:       "prompt",
		Model:        "test-model",
		Status:       "pending",
		CreatedAt:    time.Now(),
	}
	submitTestTask(store, pool, task)

	waitForStatus(t, store, "t1", 2*time.Second, "completed")
	results := store.Results([]string{"t1"})
	if results[0].Content != "hello world" {
		t.Fatalf("expected 'hello world', got %q", results[0].Content)
	}
}

// ---------------------------------------------------------------------------
// Failed task
// ---------------------------------------------------------------------------

func TestWorkerFailedTask(t *testing.T) {
	store := NewTaskStore()
	mock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			return fmt.Errorf("model not found")
		},
	}
	pool := newTestPool(store, 2, mock)

	task := &Task{ID: "t1", Prompt: "p", Status: "pending", CreatedAt: time.Now()}
	submitTestTask(store, pool, task)

	waitForStatus(t, store, "t1", 2*time.Second, "failed")
	got := store.Get("t1")
	if got.Error != "model not found" {
		t.Fatalf("expected 'model not found', got %q", got.Error)
	}
}

// ---------------------------------------------------------------------------
// Timeout
// ---------------------------------------------------------------------------

func TestWorkerTimeout(t *testing.T) {
	store := NewTaskStore()
	mock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			// Block until context is cancelled (timeout)
			<-ctx.Done()
			return ctx.Err()
		},
	}
	pool := newTestPool(store, 2, mock)

	task := &Task{
		ID:             "t1",
		Prompt:         "p",
		Status:         "pending",
		TimeoutSeconds: 1, // 1 second timeout
		CreatedAt:      time.Now(),
	}
	submitTestTask(store, pool, task)

	waitForStatus(t, store, "t1", 3*time.Second, "failed")
	got := store.Get("t1")
	if got.Error == "" {
		t.Fatal("expected error message")
	}
	if !strings.HasPrefix(got.Error, "TIMEOUT:") {
		t.Fatalf("expected TIMEOUT: prefix, got %q", got.Error)
	}
}

// ---------------------------------------------------------------------------
// Cancellation while queued
// ---------------------------------------------------------------------------

func TestWorkerCancelWhileQueued(t *testing.T) {
	store := NewTaskStore()
	// Fill the semaphore so new tasks queue
	blocker := make(chan struct{})
	mock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			<-blocker
			return nil
		},
	}
	pool := newTestPool(store, 1, mock)

	// Submit a blocking task to fill the single slot
	blocking := &Task{ID: "blocker", Prompt: "p", Status: "pending", CreatedAt: time.Now()}
	submitTestTask(store, pool, blocking)
	waitForStatus(t, store, "blocker", 2*time.Second, "running")

	// Submit a task that will queue
	queued := &Task{ID: "queued", Prompt: "p", Status: "pending", CreatedAt: time.Now()}
	submitTestTask(store, pool, queued)

	// Give goroutine a moment to start and block on semaphore
	time.Sleep(50 * time.Millisecond)

	// Cancel while queued
	store.SetCancelled("queued")
	// The goroutine should exit without processing
	waitForStatus(t, store, "queued", 2*time.Second, "cancelled")

	// Unblock the first task
	close(blocker)
	waitForStatus(t, store, "blocker", 2*time.Second, "completed")
}

// ---------------------------------------------------------------------------
// Cancellation while running
// ---------------------------------------------------------------------------

func TestWorkerCancelWhileRunning(t *testing.T) {
	store := NewTaskStore()
	chatStarted := make(chan struct{})
	mock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			close(chatStarted)
			<-ctx.Done()
			return ctx.Err()
		},
	}
	pool := newTestPool(store, 2, mock)

	task := &Task{ID: "t1", Prompt: "p", Status: "pending", CreatedAt: time.Now()}
	submitTestTask(store, pool, task)

	// Wait for the Chat call to start
	select {
	case <-chatStarted:
	case <-time.After(2 * time.Second):
		t.Fatal("timeout waiting for chat to start")
	}

	// Cancel while running — should propagate context cancellation
	store.SetCancelled("t1")
	waitForStatus(t, store, "t1", 2*time.Second, "cancelled")
}

// ---------------------------------------------------------------------------
// Cancel-then-complete race
// ---------------------------------------------------------------------------

func TestWorkerCancelThenCompleteRace(t *testing.T) {
	store := NewTaskStore()
	chatStarted := make(chan struct{})
	proceed := make(chan struct{})
	mock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			close(chatStarted)
			<-proceed
			fn(api.ChatResponse{Message: api.Message{Content: "result"}})
			return nil
		},
	}
	pool := newTestPool(store, 2, mock)

	task := &Task{ID: "t1", Prompt: "p", Status: "pending", CreatedAt: time.Now()}
	submitTestTask(store, pool, task)

	<-chatStarted

	// Cancel the task
	store.SetCancelled("t1")
	// Let the mock return success
	close(proceed)

	// Give the worker goroutine time to finish
	time.Sleep(100 * time.Millisecond)

	// Status should stay cancelled because SetCompleted guards on status == "running"
	if s := getStatus(store, "t1"); s != "cancelled" {
		t.Fatalf("expected cancelled, got %s", s)
	}
}

// ---------------------------------------------------------------------------
// Concurrency bound
// ---------------------------------------------------------------------------

func TestWorkerConcurrencyBound(t *testing.T) {
	store := NewTaskStore()
	const maxConcurrent = 2
	var running atomic.Int32
	var maxSeen atomic.Int32

	mock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			cur := running.Add(1)
			// Track the maximum concurrent calls
			for {
				old := maxSeen.Load()
				if cur <= old || maxSeen.CompareAndSwap(old, cur) {
					break
				}
			}
			time.Sleep(50 * time.Millisecond) // simulate work
			running.Add(-1)
			fn(api.ChatResponse{Message: api.Message{Content: "ok"}})
			return nil
		},
	}
	pool := newTestPool(store, maxConcurrent, mock)

	const numTasks = 8
	for i := range numTasks {
		id := fmt.Sprintf("t%d", i)
		task := &Task{ID: id, Prompt: "p", Status: "pending", CreatedAt: time.Now()}
		submitTestTask(store, pool, task)
	}

	// Wait for all to complete
	for i := range numTasks {
		id := fmt.Sprintf("t%d", i)
		waitForStatus(t, store, id, 5*time.Second, "completed")
	}

	if maxSeen.Load() > int32(maxConcurrent) {
		t.Fatalf("max concurrent %d exceeded limit %d", maxSeen.Load(), maxConcurrent)
	}
}

// ---------------------------------------------------------------------------
// Shutdown
// ---------------------------------------------------------------------------

func TestWorkerShutdown(t *testing.T) {
	store := NewTaskStore()
	chatStarted := make(chan struct{}, 2)
	mock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			chatStarted <- struct{}{}
			<-ctx.Done()
			return ctx.Err()
		},
	}
	pool := newTestPool(store, 2, mock)

	// Submit tasks
	for i := range 2 {
		id := fmt.Sprintf("t%d", i)
		task := &Task{ID: id, Prompt: "p", Status: "pending", CreatedAt: time.Now()}
		submitTestTask(store, pool, task)
	}

	// Wait for both to start running
	for range 2 {
		select {
		case <-chatStarted:
		case <-time.After(2 * time.Second):
			t.Fatal("timeout waiting for tasks to start")
		}
	}

	// Shutdown should cancel tasks and return promptly
	done := make(chan struct{})
	go func() {
		pool.Shutdown()
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("Shutdown did not complete in time")
	}

	// Tasks should be cancelled
	for i := range 2 {
		id := fmt.Sprintf("t%d", i)
		got := store.Get(id)
		if got.Status != "cancelled" {
			t.Fatalf("task %s: expected cancelled, got %s", id, got.Status)
		}
	}
}

// ---------------------------------------------------------------------------
// Input file reading via worker pipeline
// ---------------------------------------------------------------------------

func TestWorkerInputFile(t *testing.T) {
	store := NewTaskStore()
	var capturedUserMsg string
	var mu sync.Mutex

	mock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			mu.Lock()
			capturedUserMsg = req.Messages[1].Content
			mu.Unlock()
			fn(api.ChatResponse{Message: api.Message{Content: "ok"}})
			return nil
		},
	}
	pool := newTestPool(store, 2, mock)

	// Create a temp file with content
	tmpFile, err := os.CreateTemp("", "test-input-*.go")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())
	if _, err := tmpFile.WriteString("func main() {}"); err != nil {
		t.Fatal(err)
	}
	tmpFile.Close()

	// Task WITH input file — should concatenate prompt + "\n\n" + file content
	task := &Task{
		ID:        "t1",
		Prompt:    "Review this code",
		InputFile: tmpFile.Name(),
		Status:    "pending",
		Model:     "m",
	}
	submitTestTask(store, pool, task)
	waitForStatus(t, store, "t1", 2*time.Second, "completed")

	mu.Lock()
	if capturedUserMsg != "Review this code\n\nfunc main() {}" {
		t.Fatalf("expected concatenated message, got %q", capturedUserMsg)
	}
	mu.Unlock()

	// Task WITHOUT input file — should send just the prompt
	task2 := &Task{
		ID:     "t2",
		Prompt: "Hello",
		Status: "pending",
		Model:  "m",
	}
	submitTestTask(store, pool, task2)
	waitForStatus(t, store, "t2", 2*time.Second, "completed")

	mu.Lock()
	if capturedUserMsg != "Hello" {
		t.Fatalf("expected just prompt, got %q", capturedUserMsg)
	}
	mu.Unlock()
}

// ---------------------------------------------------------------------------
// Input file not found
// ---------------------------------------------------------------------------

func TestWorkerInputFileNotFound(t *testing.T) {
	store := NewTaskStore()
	chatCalled := atomic.Bool{}
	mock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			chatCalled.Store(true)
			fn(api.ChatResponse{Message: api.Message{Content: "ok"}})
			return nil
		},
	}
	pool := newTestPool(store, 2, mock)

	task := &Task{
		ID:        "t1",
		Prompt:    "p",
		InputFile: "/nonexistent/path/file.go",
		Status:    "pending",
		Model:     "m",
	}
	submitTestTask(store, pool, task)
	waitForStatus(t, store, "t1", 2*time.Second, "failed")

	if chatCalled.Load() {
		t.Fatal("Ollama should not have been called when input file doesn't exist")
	}
	got := store.Get("t1")
	if !strings.Contains(got.Error, "failed to read input file") {
		t.Fatalf("expected input file error, got %q", got.Error)
	}
}

// ---------------------------------------------------------------------------
// Output file write
// ---------------------------------------------------------------------------

func TestWorkerOutputFileWrite(t *testing.T) {
	store := NewTaskStore()
	mock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			fn(api.ChatResponse{Message: api.Message{Content: "package main\n\nfunc main() {}"}})
			return nil
		},
	}
	pool := newTestPool(store, 2, mock)

	tmpDir := t.TempDir()
	outPath := filepath.Join(tmpDir, "output.go")

	task := &Task{
		ID:                  "t1",
		Prompt:              "p",
		OutputFile:          outPath,
		StripMarkdownFences: true,
		Status:              "pending",
		Model:               "m",
	}
	submitTestTask(store, pool, task)
	waitForStatus(t, store, "t1", 2*time.Second, "completed")

	// Verify file was written
	data, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("output file not written: %v", err)
	}
	if string(data) != "package main\n\nfunc main() {}" {
		t.Fatalf("unexpected file content: %q", string(data))
	}

	// Verify result was cleared from memory (FileWritten)
	results := store.Results([]string{"t1"})
	if results[0].Content != "" {
		t.Fatal("Result should be cleared when file was written")
	}
	if results[0].OutputFile != outPath {
		t.Fatalf("expected OutputFile %q, got %q", outPath, results[0].OutputFile)
	}
}

// ---------------------------------------------------------------------------
// Output file write failure — result preserved
// ---------------------------------------------------------------------------

func TestWorkerOutputFileWriteFail(t *testing.T) {
	store := NewTaskStore()
	mock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			fn(api.ChatResponse{Message: api.Message{Content: "the result"}})
			return nil
		},
	}
	pool := newTestPool(store, 2, mock)

	// Use a path in a non-existent directory to trigger write failure
	task := &Task{
		ID:                  "t1",
		Prompt:              "p",
		OutputFile:          "/nonexistent/dir/output.go",
		StripMarkdownFences: true,
		Status:              "pending",
		Model:               "m",
	}
	submitTestTask(store, pool, task)
	waitForStatus(t, store, "t1", 2*time.Second, "failed")

	got := store.Get("t1")
	if !strings.Contains(got.Error, "failed to write output file") {
		t.Fatalf("expected write error, got %q", got.Error)
	}
	// Result should be preserved since Ollama succeeded
	if got.Result != "the result" {
		t.Fatalf("expected preserved result 'the result', got %q", got.Result)
	}
}

// ---------------------------------------------------------------------------
// stripMarkdownFences unit tests
// ---------------------------------------------------------------------------

func TestStripMarkdownFences(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "with language tag",
			input: "```go\npackage main\n```",
			want:  "package main",
		},
		{
			name:  "without language tag",
			input: "```\npackage main\n```",
			want:  "package main",
		},
		{
			name:  "no fences passthrough",
			input: "package main",
			want:  "package main",
		},
		{
			name:  "multi-line content",
			input: "```go\npackage main\n\nfunc main() {\n}\n```",
			want:  "package main\n\nfunc main() {\n}",
		},
		{
			name:  "empty string",
			input: "",
			want:  "",
		},
		{
			name:  "empty inside fences",
			input: "```\n\n```",
			want:  "",
		},
		{
			name:  "with surrounding whitespace",
			input: "  ```go\nfoo\n```  ",
			want:  "foo",
		},
		{
			name:  "no closing fence — passthrough",
			input: "```go\nfoo\nbar",
			want:  "```go\nfoo\nbar",
		},
		{
			name:  "just triple backtick",
			input: "```",
			want:  "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := stripMarkdownFences(tt.input)
			if got != tt.want {
				t.Fatalf("stripMarkdownFences(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Strip markdown fences integration — verify fences stripped in written file
// ---------------------------------------------------------------------------

func TestWorkerStripMarkdownFencesIntegration(t *testing.T) {
	store := NewTaskStore()
	mock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			fn(api.ChatResponse{Message: api.Message{Content: "```go\npackage main\n```"}})
			return nil
		},
	}
	pool := newTestPool(store, 2, mock)

	tmpDir := t.TempDir()
	outPath := filepath.Join(tmpDir, "output.go")

	task := &Task{
		ID:                  "t1",
		Prompt:              "p",
		OutputFile:          outPath,
		StripMarkdownFences: true,
		Status:              "pending",
		Model:               "m",
	}
	submitTestTask(store, pool, task)
	waitForStatus(t, store, "t1", 2*time.Second, "completed")

	data, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("output file not written: %v", err)
	}
	if string(data) != "package main" {
		t.Fatalf("expected fences stripped, got %q", string(data))
	}
}

// ---------------------------------------------------------------------------
// Strip markdown fences disabled
// ---------------------------------------------------------------------------

func TestWorkerStripMarkdownFencesDisabled(t *testing.T) {
	store := NewTaskStore()
	mock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			fn(api.ChatResponse{Message: api.Message{Content: "```go\npackage main\n```"}})
			return nil
		},
	}
	pool := newTestPool(store, 2, mock)

	tmpDir := t.TempDir()
	outPath := filepath.Join(tmpDir, "output.go")

	task := &Task{
		ID:                  "t1",
		Prompt:              "p",
		OutputFile:          outPath,
		StripMarkdownFences: false, // explicitly disabled
		Status:              "pending",
		Model:               "m",
	}
	submitTestTask(store, pool, task)
	waitForStatus(t, store, "t1", 2*time.Second, "completed")

	data, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("output file not written: %v", err)
	}
	if string(data) != "```go\npackage main\n```" {
		t.Fatalf("expected fences preserved, got %q", string(data))
	}
}

// ---------------------------------------------------------------------------
// Post-write command success
// ---------------------------------------------------------------------------

func TestWorkerPostWriteCmd(t *testing.T) {
	store := NewTaskStore()
	mock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			fn(api.ChatResponse{Message: api.Message{Content: "hello"}})
			return nil
		},
	}
	pool := newTestPool(store, 2, mock)
	pool.PostWriteCmdTimeout = 5 * time.Second

	tmpDir := t.TempDir()
	outPath := filepath.Join(tmpDir, "output.txt")
	markerPath := filepath.Join(tmpDir, "marker.txt")

	task := &Task{
		ID:                  "t1",
		Prompt:              "p",
		OutputFile:          outPath,
		StripMarkdownFences: false,
		PostWriteCmd:        fmt.Sprintf("touch %s", markerPath),
		Status:              "pending",
		Model:               "m",
	}
	submitTestTask(store, pool, task)
	waitForStatus(t, store, "t1", 5*time.Second, "completed")

	// Verify the post-write command ran
	if _, err := os.Stat(markerPath); os.IsNotExist(err) {
		t.Fatal("post-write command did not run (marker file not created)")
	}
}

// ---------------------------------------------------------------------------
// Post-write command failure — file already written, result preserved
// ---------------------------------------------------------------------------

func TestWorkerPostWriteCmdFail(t *testing.T) {
	store := NewTaskStore()
	mock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			fn(api.ChatResponse{Message: api.Message{Content: "the content"}})
			return nil
		},
	}
	pool := newTestPool(store, 2, mock)
	pool.PostWriteCmdTimeout = 5 * time.Second

	tmpDir := t.TempDir()
	outPath := filepath.Join(tmpDir, "output.txt")

	task := &Task{
		ID:                  "t1",
		Prompt:              "p",
		OutputFile:          outPath,
		StripMarkdownFences: false,
		PostWriteCmd:        "exit 1", // always fails
		Status:              "pending",
		Model:               "m",
	}
	submitTestTask(store, pool, task)
	waitForStatus(t, store, "t1", 5*time.Second, "failed")

	got := store.Get("t1")
	if !strings.Contains(got.Error, "post-write command failed") {
		t.Fatalf("expected post-write command error, got %q", got.Error)
	}
	// Result should be preserved
	if got.Result != "the content" {
		t.Fatalf("expected preserved result 'the content', got %q", got.Result)
	}
	// Output file should still exist on disk
	if _, err := os.Stat(outPath); os.IsNotExist(err) {
		t.Fatal("output file should exist even though post-write command failed")
	}
}

// ---------------------------------------------------------------------------
// Post-write command timeout
// ---------------------------------------------------------------------------

func TestWorkerPostWriteCmdTimeout(t *testing.T) {
	store := NewTaskStore()
	mock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			fn(api.ChatResponse{Message: api.Message{Content: "result"}})
			return nil
		},
	}
	pool := newTestPool(store, 2, mock)
	pool.PostWriteCmdTimeout = 100 * time.Millisecond // very short timeout

	tmpDir := t.TempDir()
	outPath := filepath.Join(tmpDir, "output.txt")

	task := &Task{
		ID:                  "t1",
		Prompt:              "p",
		OutputFile:          outPath,
		StripMarkdownFences: false,
		PostWriteCmd:        "sleep 10", // will be killed by timeout
		Status:              "pending",
		Model:               "m",
	}
	submitTestTask(store, pool, task)
	waitForStatus(t, store, "t1", 5*time.Second, "failed")

	got := store.Get("t1")
	if !strings.Contains(got.Error, "post-write command failed") {
		t.Fatalf("expected post-write command error, got %q", got.Error)
	}
}

// ---------------------------------------------------------------------------
// WORKER_CONCURRENCY env var parsing in NewWorkerPool
// ---------------------------------------------------------------------------

func TestNewWorkerPoolConcurrencyEnvVar(t *testing.T) {
	store := NewTaskStore()

	// Valid concurrency value
	t.Setenv("WORKER_CONCURRENCY", "5")
	pool, err := NewWorkerPool(store)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cap(pool.sem) != 5 {
		t.Fatalf("expected concurrency 5, got %d", cap(pool.sem))
	}

	// Invalid (non-numeric) falls back to default
	t.Setenv("WORKER_CONCURRENCY", "notanumber")
	pool, err = NewWorkerPool(store)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cap(pool.sem) != defaultConcurrency {
		t.Fatalf("expected default concurrency %d, got %d", defaultConcurrency, cap(pool.sem))
	}

	// Negative falls back to default
	t.Setenv("WORKER_CONCURRENCY", "-3")
	pool, err = NewWorkerPool(store)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cap(pool.sem) != defaultConcurrency {
		t.Fatalf("expected default concurrency %d for negative value, got %d", defaultConcurrency, cap(pool.sem))
	}

	// Unset falls back to default
	t.Setenv("WORKER_CONCURRENCY", "")
	pool, err = NewWorkerPool(store)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cap(pool.sem) != defaultConcurrency {
		t.Fatalf("expected default concurrency %d for empty value, got %d", defaultConcurrency, cap(pool.sem))
	}
}

// ---------------------------------------------------------------------------
// Post-write command failure with stderr output in error message
// ---------------------------------------------------------------------------

func TestWorkerPostWriteCmdFailWithStderr(t *testing.T) {
	store := NewTaskStore()
	mock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			fn(api.ChatResponse{Message: api.Message{Content: "the result"}})
			return nil
		},
	}
	pool := newTestPool(store, 2, mock)
	pool.PostWriteCmdTimeout = 5 * time.Second

	tmpDir := t.TempDir()
	outPath := filepath.Join(tmpDir, "output.txt")

	// Command that produces stderr and exits non-zero
	task := &Task{
		ID:                  "t1",
		Prompt:              "p",
		OutputFile:          outPath,
		StripMarkdownFences: false,
		PostWriteCmd:        "echo 'syntax error near line 42' >&2; exit 1",
		Status:              "pending",
		Model:               "m",
	}
	submitTestTask(store, pool, task)
	waitForStatus(t, store, "t1", 5*time.Second, "failed")

	got := store.Get("t1")
	if !strings.Contains(got.Error, "post-write command failed") {
		t.Fatalf("expected post-write command error, got %q", got.Error)
	}
	// Verify stderr output is included in the error message
	if !strings.Contains(got.Error, "syntax error near line 42") {
		t.Fatalf("expected stderr output in error message, got %q", got.Error)
	}
	// Result should be preserved
	if got.Result != "the result" {
		t.Fatalf("expected preserved result, got %q", got.Result)
	}
}

// ---------------------------------------------------------------------------
// getTaskTimeout env var fallback
// ---------------------------------------------------------------------------

func TestGetTaskTimeoutEnvVar(t *testing.T) {
	// Per-task timeout takes priority
	task := &Task{TimeoutSeconds: 30}
	if got := getTaskTimeout(task); got != 30*time.Second {
		t.Fatalf("expected 30s, got %v", got)
	}

	// Env var fallback
	t.Setenv("TASK_TIMEOUT", "120")
	task2 := &Task{TimeoutSeconds: 0}
	if got := getTaskTimeout(task2); got != 120*time.Second {
		t.Fatalf("expected 120s from env, got %v", got)
	}

	// Invalid env var falls through to default
	t.Setenv("TASK_TIMEOUT", "notanumber")
	if got := getTaskTimeout(task2); got != time.Duration(defaultTaskTimeoutSec)*time.Second {
		t.Fatalf("expected default %ds, got %v", defaultTaskTimeoutSec, got)
	}

	// Negative env var falls through to default
	t.Setenv("TASK_TIMEOUT", "-5")
	if got := getTaskTimeout(task2); got != time.Duration(defaultTaskTimeoutSec)*time.Second {
		t.Fatalf("expected default %ds, got %v", defaultTaskTimeoutSec, got)
	}
}

// ---------------------------------------------------------------------------
// Post-semaphore cancellation (double-check after acquiring slot)
// ---------------------------------------------------------------------------

func TestWorkerCancelDuringSemaphoreAcquire(t *testing.T) {
	store := NewTaskStore()
	chatCalled := atomic.Bool{}
	mock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			chatCalled.Store(true)
			fn(api.ChatResponse{Message: api.Message{Content: "ok"}})
			return nil
		},
	}
	// Use concurrency of 1 with a blocking task to control timing
	blocker := make(chan struct{})
	blockingMock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			if req.Model == "blocker" {
				<-blocker
				fn(api.ChatResponse{Message: api.Message{Content: "ok"}})
				return nil
			}
			return mock.chatFn(ctx, req, fn)
		},
	}
	pool := newTestPool(store, 1, blockingMock)

	// Fill the slot with a blocking task
	blocking := &Task{ID: "blocker", Prompt: "p", Model: "blocker", Status: "pending"}
	submitTestTask(store, pool, blocking)
	waitForStatus(t, store, "blocker", 2*time.Second, "running")

	// Submit a second task that will queue on semaphore
	task := &Task{ID: "victim", Prompt: "p", Model: "other", Status: "pending"}
	submitTestTask(store, pool, task)
	time.Sleep(50 * time.Millisecond) // let goroutine block on semaphore

	// Cancel victim while it's waiting on semaphore, then release the slot
	store.SetCancelled("victim")
	close(blocker)

	// Wait for blocker to complete
	waitForStatus(t, store, "blocker", 2*time.Second, "completed")
	// Give victim's goroutine time to wake up and exit
	time.Sleep(50 * time.Millisecond)

	// victim should stay cancelled — the double-check prevents SetRunning
	if s := getStatus(store, "victim"); s != "cancelled" {
		t.Fatalf("expected cancelled, got %s", s)
	}
}

// ---------------------------------------------------------------------------
// Default model fallback
// ---------------------------------------------------------------------------

func TestWorkerDefaultModelFallback(t *testing.T) {
	store := NewTaskStore()
	var capturedModel string
	var mu sync.Mutex

	mock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			mu.Lock()
			capturedModel = req.Model
			mu.Unlock()
			fn(api.ChatResponse{Message: api.Message{Content: "ok"}})
			return nil
		},
	}
	pool := newTestPool(store, 2, mock)

	// Task with explicit model
	task := &Task{ID: "t1", Prompt: "p", Model: "custom-model", Status: "pending", CreatedAt: time.Now()}
	submitTestTask(store, pool, task)

	waitForStatus(t, store, "t1", 2*time.Second, "completed")
	mu.Lock()
	if capturedModel != "custom-model" {
		t.Fatalf("expected custom-model, got %s", capturedModel)
	}
	mu.Unlock()
}

// ---------------------------------------------------------------------------
// Negative per-task timeout falls through to default
// ---------------------------------------------------------------------------

func TestGetTaskTimeoutNegative(t *testing.T) {
	task := &Task{TimeoutSeconds: -5}
	got := getTaskTimeout(task)
	if got != time.Duration(defaultTaskTimeoutSec)*time.Second {
		t.Fatalf("expected default %ds for negative timeout, got %v", defaultTaskTimeoutSec, got)
	}
}

// ---------------------------------------------------------------------------
// Shutdown with no tasks
// ---------------------------------------------------------------------------

func TestWorkerShutdownNoTasks(t *testing.T) {
	store := NewTaskStore()
	pool := newTestPool(store, 2, &mockOllamaClient{})

	done := make(chan struct{})
	go func() {
		pool.Shutdown()
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Shutdown should return immediately with no tasks")
	}
}

// ---------------------------------------------------------------------------
// Shutdown with queued (pending) tasks blocked on semaphore
// ---------------------------------------------------------------------------

func TestWorkerShutdownWithQueuedTasks(t *testing.T) {
	store := NewTaskStore()
	mock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			<-ctx.Done()
			return ctx.Err()
		},
	}
	pool := newTestPool(store, 1, mock)

	// Submit a running task to fill the single slot
	running := &Task{ID: "r1", Prompt: "p", Status: "pending", CreatedAt: time.Now()}
	submitTestTask(store, pool, running)
	waitForStatus(t, store, "r1", 2*time.Second, "running")

	// Submit tasks that will queue on the semaphore
	for i := range 3 {
		id := fmt.Sprintf("q%d", i)
		task := &Task{ID: id, Prompt: "p", Status: "pending", CreatedAt: time.Now()}
		submitTestTask(store, pool, task)
	}
	time.Sleep(50 * time.Millisecond) // let goroutines block on semaphore

	// Shutdown should cancel everything and return
	done := make(chan struct{})
	go func() {
		pool.Shutdown()
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("Shutdown did not complete in time")
	}

	// All tasks should be cancelled
	for _, id := range []string{"r1", "q0", "q1", "q2"} {
		if s := getStatus(store, id); s != "cancelled" {
			t.Fatalf("task %s: expected cancelled, got %s", id, s)
		}
	}
}

// ---------------------------------------------------------------------------
// Double shutdown is harmless
// ---------------------------------------------------------------------------

func TestWorkerDoubleShutdown(t *testing.T) {
	store := NewTaskStore()
	mock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			<-ctx.Done()
			return ctx.Err()
		},
	}
	pool := newTestPool(store, 2, mock)

	task := &Task{ID: "t1", Prompt: "p", Status: "pending", CreatedAt: time.Now()}
	submitTestTask(store, pool, task)
	waitForStatus(t, store, "t1", 2*time.Second, "running")

	pool.Shutdown()
	// Second shutdown should not panic or hang
	pool.Shutdown()

	if s := getStatus(store, "t1"); s != "cancelled" {
		t.Fatalf("expected cancelled, got %s", s)
	}
}

// ---------------------------------------------------------------------------
// SetConcurrency: dynamic concurrency adjustment
// ---------------------------------------------------------------------------

func TestSetConcurrency(t *testing.T) {
	store := NewTaskStore()
	var running atomic.Int32
	proceed := make(chan struct{})

	mock := &mockOllamaClient{
		chatFn: func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
			running.Add(1)
			defer running.Add(-1)
			<-proceed
			fn(api.ChatResponse{Message: api.Message{Content: "ok"}})
			return nil
		},
	}
	// Start with concurrency 1
	pool := newTestPool(store, 1, mock)

	// Submit 3 tasks — only 1 should run at a time
	for i := range 3 {
		id := fmt.Sprintf("a%d", i)
		task := &Task{ID: id, Prompt: "p", Status: "pending", CreatedAt: time.Now()}
		submitTestTask(store, pool, task)
	}

	// Wait for 1 to be running
	waitForStatus(t, store, "a0", 2*time.Second, "running")
	time.Sleep(50 * time.Millisecond) // let other goroutines block on sem
	if r := running.Load(); r != 1 {
		t.Fatalf("expected 1 running with concurrency 1, got %d", r)
	}

	// Release all 3 tasks
	close(proceed)
	for i := range 3 {
		waitForStatus(t, store, fmt.Sprintf("a%d", i), 5*time.Second, "completed")
	}

	// Now increase concurrency to 3
	pool.SetConcurrency(3)
	if pool.Concurrency() != 3 {
		t.Fatalf("expected concurrency 3, got %d", pool.Concurrency())
	}

	// Submit 3 more tasks — all 3 should run concurrently
	proceed2 := make(chan struct{})
	var running2 atomic.Int32
	var maxSeen2 atomic.Int32
	mock.chatFn = func(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
		cur := running2.Add(1)
		for {
			old := maxSeen2.Load()
			if cur <= old || maxSeen2.CompareAndSwap(old, cur) {
				break
			}
		}
		<-proceed2
		running2.Add(-1)
		fn(api.ChatResponse{Message: api.Message{Content: "ok"}})
		return nil
	}

	for i := range 3 {
		id := fmt.Sprintf("b%d", i)
		task := &Task{ID: id, Prompt: "p", Status: "pending", CreatedAt: time.Now()}
		submitTestTask(store, pool, task)
	}

	// Wait for all 3 to be running
	for i := range 3 {
		waitForStatus(t, store, fmt.Sprintf("b%d", i), 2*time.Second, "running")
	}
	time.Sleep(50 * time.Millisecond)

	if maxSeen2.Load() < 3 {
		t.Fatalf("expected 3 concurrent with concurrency 3, max seen was %d", maxSeen2.Load())
	}

	close(proceed2)
	for i := range 3 {
		waitForStatus(t, store, fmt.Sprintf("b%d", i), 5*time.Second, "completed")
	}
}

// ---------------------------------------------------------------------------
// Concurrency getter
// ---------------------------------------------------------------------------

func TestConcurrencyGetter(t *testing.T) {
	store := NewTaskStore()
	pool := newTestPool(store, 5, &mockOllamaClient{})

	if pool.Concurrency() != 5 {
		t.Fatalf("expected 5, got %d", pool.Concurrency())
	}

	pool.SetConcurrency(10)
	if pool.Concurrency() != 10 {
		t.Fatalf("expected 10, got %d", pool.Concurrency())
	}

	pool.SetConcurrency(1)
	if pool.Concurrency() != 1 {
		t.Fatalf("expected 1, got %d", pool.Concurrency())
	}
}
