// worker.go implements the worker pool that processes tasks against Ollama.
//
// Each submitted task gets its own goroutine, but a semaphore (buffered channel)
// limits how many tasks can call Ollama concurrently. This keeps things simple:
//   - No dispatcher loop or work channel needed
//   - Each goroutine manages its own lifecycle including cancellation
//   - The semaphore is the only coordination primitive
//
// Concurrency is bounded because Ollama model inference is GPU-bound. On an
// M3 Pro with 36GB RAM running a 14B model, 2 concurrent workers is a safe
// default. Adjust via the WORKER_CONCURRENCY environment variable.
package main

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
)

const (
	defaultConcurrency         = 3                   // max parallel Ollama requests (GPU-bound)
	defaultModel               = "qwen2.5-coder:14b" // fallback when task doesn't specify a model
	defaultTaskTimeoutSec      = 600                  // 10 minutes per task
	defaultPostWriteCmdTimeout = 30 * time.Second     // timeout for post-write commands (e.g. gofmt)
)

// OllamaClient is the subset of the Ollama API client used by WorkerPool.
// Defined as an interface so tests can inject a mock.
type OllamaClient interface {
	Chat(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error
	List(ctx context.Context) (*api.ListResponse, error)
}

// WorkerPool manages concurrent Ollama inference requests.
type WorkerPool struct {
	sem                chan struct{}   // semaphore: buffered to max concurrent workers
	semMu              sync.RWMutex   // guards sem for hot-swap via SetConcurrency
	client             OllamaClient   // Ollama API client, created once and reused
	store              *TaskStore     // shared task store for status updates
	wg                 sync.WaitGroup // tracks in-flight goroutines for graceful shutdown
	PostWriteCmdTimeout time.Duration // timeout for post-write commands; 0 means use default
}

// NewWorkerPool creates a worker pool connected to the local Ollama instance.
// The Ollama client connects to http://127.0.0.1:11434 by default, or to
// whatever is specified in the OLLAMA_HOST environment variable.
func NewWorkerPool(store *TaskStore) (*WorkerPool, error) {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return nil, fmt.Errorf("failed to connect to Ollama: %v", err)
	}

	concurrency := defaultConcurrency
	if v := os.Getenv("WORKER_CONCURRENCY"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			concurrency = n
		}
	}

	return &WorkerPool{
		sem:    make(chan struct{}, concurrency),
		client: client,
		store:  store,
	}, nil
}

// SetConcurrency replaces the semaphore with a new one of capacity n.
// In-flight goroutines that already acquired a slot on the old semaphore
// will release back to it when done — they drain naturally. New tasks
// will acquire from the new semaphore.
func (p *WorkerPool) SetConcurrency(n int) {
	p.semMu.Lock()
	p.sem = make(chan struct{}, n)
	p.semMu.Unlock()
}

// Concurrency returns the current worker concurrency (semaphore capacity).
func (p *WorkerPool) Concurrency() int {
	p.semMu.RLock()
	c := cap(p.sem)
	p.semMu.RUnlock()
	return c
}

// Submit starts a background goroutine to process the task. The goroutine
// will block on the semaphore until a worker slot is available, then call
// Ollama and update the task store with the result.
//
// The caller must create the task context and set task.Cancel BEFORE adding
// the task to the store. This ensures the cancel function is visible to
// concurrent cancel_tasks calls from the moment the task enters the store.
// The context should be derived from context.Background(), not the request
// context, because tasks must outlive the submit_tasks MCP request.
func (p *WorkerPool) Submit(ctx context.Context, cancel context.CancelFunc, task *Task) {
	p.wg.Add(1)
	go func() {
		defer p.wg.Done()
		defer cancel()
		p.run(ctx, task)
	}()
}

// Shutdown cancels all pending/running tasks and waits up to 5 seconds for
// worker goroutines to finish. Called when the MCP server stops.
func (p *WorkerPool) Shutdown() {
	p.store.Cancel(nil, "")
	done := make(chan struct{})
	go func() {
		p.wg.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
	}
}

// getTaskTimeout returns the timeout duration for a task. It checks (in order):
// the per-task TimeoutSeconds, the TASK_TIMEOUT env var, then the compiled default.
func getTaskTimeout(task *Task) time.Duration {
	if task.TimeoutSeconds > 0 {
		return time.Duration(task.TimeoutSeconds) * time.Second
	}
	if v := os.Getenv("TASK_TIMEOUT"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return time.Duration(n) * time.Second
		}
	}
	return time.Duration(defaultTaskTimeoutSec) * time.Second
}

// postWriteCmdTimeout returns the configured post-write command timeout.
// The PostWriteCmdTimeout field can be set directly on WorkerPool to override
// the default (used by tests to avoid 30-second waits).
func (p *WorkerPool) postWriteCmdTimeout() time.Duration {
	if p.PostWriteCmdTimeout > 0 {
		return p.PostWriteCmdTimeout
	}
	return defaultPostWriteCmdTimeout
}

// run is the goroutine body for a single task. It acquires a semaphore slot,
// reads input files, calls Ollama with a timeout, optionally strips fences,
// writes output files, runs post-write commands, and updates the store.
func (p *WorkerPool) run(ctx context.Context, task *Task) {
	// Capture the current semaphore under RLock. If SetConcurrency swaps
	// in a new channel between now and when we release, we drain the old
	// one correctly — no slots are leaked.
	p.semMu.RLock()
	sem := p.sem
	p.semMu.RUnlock()

	// Acquire a worker slot. Blocks here if all slots are in use — this is
	// effectively the queue. The task stays in "pending" status while waiting.
	select {
	case sem <- struct{}{}:
		defer func() { <-sem }()
	case <-ctx.Done():
		// Task was cancelled while waiting in the queue
		return
	}

	// Double-check cancellation after acquiring the slot
	if ctx.Err() != nil {
		return
	}

	if !p.store.SetRunning(task.ID) {
		return // task was cancelled while waiting in the queue
	}

	// Apply per-task timeout so a hung Ollama call doesn't block a
	// semaphore slot forever.
	timeout := getTaskTimeout(task)
	timeoutCtx, timeoutCancel := context.WithTimeout(ctx, timeout)
	defer timeoutCancel()

	// Step 1: Read input file if specified
	var fileContent string
	if task.InputFile != "" {
		var err error
		fileContent, err = readInputFile(task.InputFile)
		if err != nil {
			p.store.SetFailed(task.ID, fmt.Sprintf("failed to read input file: %v", err))
			return
		}
	}

	// Step 2: Call Ollama
	result, err := p.callOllama(timeoutCtx, task, fileContent)
	if err != nil {
		if ctx.Err() != nil {
			// Parent context was cancelled (user called cancel_tasks).
			// The store.Cancel method already set the status.
			return
		}
		if timeoutCtx.Err() == context.DeadlineExceeded {
			p.store.SetFailed(task.ID, fmt.Sprintf(
				"TIMEOUT: task exceeded %d second limit. Resubmit with a larger timeout_seconds value if the task needs more time.",
				int(timeout.Seconds()),
			))
			return
		}
		p.store.SetFailed(task.ID, err.Error())
		return
	}

	// Step 3: Strip markdown fences if configured
	output := result
	if task.StripMarkdownFences {
		output = stripMarkdownFences(result)
	}

	// Step 4: Write output file if specified
	if task.OutputFile != "" {
		if err := writeOutputFile(task.OutputFile, output); err != nil {
			p.store.SetFailedWithResult(task.ID, result, fmt.Sprintf("failed to write output file: %v", err))
			return
		}
		p.store.SetFileWritten(task.ID)
	}

	// Step 5: Run post-write command if specified
	if task.PostWriteCmd != "" {
		if cmdOutput, err := runPostWriteCmd(task.PostWriteCmd, p.postWriteCmdTimeout()); err != nil {
			errMsg := fmt.Sprintf("post-write command failed: %v", err)
			if cmdOutput != "" {
				errMsg += ": " + cmdOutput
			}
			p.store.SetFailedWithResult(task.ID, result, errMsg)
			return
		}
	}

	// Step 6: Mark completed (uses the raw Ollama result, not stripped —
	// stripped version is already on disk if OutputFile was set)
	p.store.SetCompleted(task.ID, result)
}

// callOllama sends the task to Ollama using the Chat API and streams the
// response. The Chat API is used instead of Generate because it cleanly
// separates system and user messages, which maps naturally to how the
// caller (Opus) structures its prompts.
func (p *WorkerPool) callOllama(ctx context.Context, task *Task, fileContent string) (string, error) {
	// Build the user message: prompt first, then file content (if any)
	// separated by a blank line.
	userMessage := task.Prompt
	if fileContent != "" {
		userMessage = task.Prompt + "\n\n" + fileContent
	}

	messages := []api.Message{
		{Role: "system", Content: task.SystemPrompt},
		{Role: "user", Content: userMessage},
	}

	// Stream the response, accumulating chunks into a string builder.
	var result strings.Builder
	err := p.client.Chat(ctx, &api.ChatRequest{
		Model:    task.Model,
		Messages: messages,
	}, func(resp api.ChatResponse) error {
		result.WriteString(resp.Message.Content)
		return nil
	})

	if err != nil {
		return "", err
	}
	return result.String(), nil
}

// readInputFile reads a file from disk and returns its contents as a string.
// Called by run() when a task specifies InputFile. The contents are appended
// to the prompt before sending to Ollama, so the orchestrating agent never
// needs to read the file into its own context window.
func readInputFile(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// stripMarkdownFences removes markdown code fences from the output.
// Handles: ```lang\n...\n```, ```\n...\n```, no fences (passthrough),
// empty content inside fences.
func stripMarkdownFences(s string) string {
	trimmed := strings.TrimSpace(s)
	if !strings.HasPrefix(trimmed, "```") {
		return s
	}

	// Find the end of the opening fence line
	firstNewline := strings.Index(trimmed, "\n")
	if firstNewline == -1 {
		// Just "```" with nothing else — return empty
		return ""
	}

	// Check for closing fence
	if !strings.HasSuffix(trimmed, "```") {
		return s
	}

	// Extract content between fences
	inner := trimmed[firstNewline+1:]
	// Remove the closing ```
	inner = inner[:len(inner)-3]

	// Trim trailing newline before closing fence (if any)
	inner = strings.TrimRight(inner, "\n")

	return inner
}

// writeOutputFile writes the Ollama response to disk at the specified path.
// Called by run() when a task specifies OutputFile. After a successful write,
// the result is cleared from memory (via SetFileWritten + SetCompleted) since
// the content is on disk. Creates or overwrites the file with mode 0644.
func writeOutputFile(path, content string) error {
	return os.WriteFile(path, []byte(content), 0644)
}

// runPostWriteCmd runs a shell command after a successful file write (e.g.
// "gofmt -w /path/to/file.go"). Executes via "sh -c" with the given timeout.
// Returns the command's combined stdout/stderr on failure for error reporting.
// If the command fails, the output file has already been written — the task
// is marked as failed with the Ollama result preserved via SetFailedWithResult.
func runPostWriteCmd(cmdStr string, timeout time.Duration) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, "sh", "-c", cmdStr)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return strings.TrimSpace(string(output)), err
	}
	return "", nil
}
