// main.go is the entrypoint for the Ollama worker MCP server.
//
// This server is designed to be spawned by Claude Code as an MCP server
// process. It communicates over stdin/stdout using the MCP protocol (JSON-RPC).
// Claude Code starts this process automatically when a new session begins —
// there is no separate daemon to manage.
//
// The server exposes five tools:
//   - list_models:   discover available Ollama models and their capabilities
//   - submit_tasks:  submit a batch of work items for Ollama to process
//   - check_tasks:   poll task status (lightweight, no result content)
//   - get_result:    retrieve full results for specific completed tasks
//   - cancel_tasks:  cancel pending or running tasks
//
// Configuration via environment variables:
//   - OLLAMA_HOST:         Ollama API address (default: http://127.0.0.1:11434)
//   - WORKER_CONCURRENCY:  max parallel Ollama requests (default: 3)
//   - DEFAULT_MODEL:       fallback model when tasks don't specify one (default: qwen2.5-coder:14b)
//   - TASK_TIMEOUT:        default per-task timeout in seconds (default: 600)
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// getDefaultModel returns the default Ollama model, checking the DEFAULT_MODEL
// env var first, then falling back to the compiled-in default.
func getDefaultModel() string {
	if m := os.Getenv("DEFAULT_MODEL"); m != "" {
		return m
	}
	return defaultModel
}

func main() {
	// Initialize shared state: the task store and worker pool.
	store := NewTaskStore()

	pool, err := NewWorkerPool(store)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize worker pool: %v\n", err)
		os.Exit(1)
	}

	handlers := &ToolHandlers{store: store, pool: pool}

	// Create the MCP server using the official SDK. The Instructions field
	// is sent to Claude during initialization and teaches it how to use
	// the worker tools effectively.
	s := mcp.NewServer(&mcp.Implementation{
		Name:    "OpusGoLlama",
		Version: "3.0.0",
	}, &mcp.ServerOptions{
		Instructions: serverInstructions,
	})

	// Register all five tools. The SDK auto-generates JSON Schema for each
	// tool's input/output from the struct tags on the arg/output types.
	mcp.AddTool(s, &mcp.Tool{
		Name: "list_models",
		Description: "List available Ollama models with their capabilities. Call this at the start of each session to discover what's available. " +
			"Returns model name, parameter size (e.g. 14B), quantization level (e.g. Q4_K_M), and family (e.g. qwen2). " +
			"Use this to decide which model to target for tasks and to calibrate expectations for worker capability.",
	}, handlers.handleListModels)

	mcp.AddTool(s, &mcp.Tool{
		Name: "submit_tasks",
		Description: "Submit one or more tasks for local Ollama workers to process. Returns task IDs immediately — work runs in the background. " +
			"Each task needs a system_prompt and prompt. Use input_file to read file contents directly (keeps them out of your context) " +
			"and output_file to write results to disk. strip_markdown_fences (default: true) removes code fences before writing. " +
			"post_write_cmd runs a shell command after writing (e.g. a formatter). " +
			"You can specify model, tag (for grouping/filtering), response_hint (status_only|content|json), and timeout_seconds (default 600). " +
			"Set concurrency to adjust the number of parallel Ollama requests (e.g. lower for larger models, higher for lightweight tasks). " +
			"Always test with 2-3 tasks first before submitting a full batch.",
	}, handlers.handleSubmitTasks)

	mcp.AddTool(s, &mcp.Tool{
		Name: "check_tasks",
		Description: "Lightweight status poll. Returns aggregate counts (pending/running/completed/failed/cancelled) and per-task status without full result content. " +
			"Use this for monitoring progress — it's cheap on your context window. " +
			"Filter by task_ids or tag. Failed tasks include a brief error message — look for 'TIMEOUT:' prefix to identify tasks that need a longer timeout_seconds. " +
			"Tasks with output_file show the path in the status.",
	}, handlers.handleCheckTasks)

	mcp.AddTool(s, &mcp.Tool{
		Name: "get_result",
		Description: "Retrieve the full Ollama response content for specific completed or failed tasks. " +
			"Use selectively — spot-check a few results or investigate failures rather than retrieving everything. " +
			"Takes a list of task_ids. Returns full content, status, and any error message for each. " +
			"Note: tasks with output_file have their result written to disk — content will be empty but output_file path is returned.",
	}, handlers.handleGetResult)

	mcp.AddTool(s, &mcp.Tool{
		Name: "cancel_tasks",
		Description: "Cancel pending or running tasks. Running tasks have their in-flight Ollama request aborted. " +
			"Filter by task_ids or tag. If both are empty, cancels all pending/running tasks.",
	}, handlers.handleCancelTasks)

	// Run the server over stdio. Claude Code communicates with this process
	// via stdin/stdout using JSON-RPC (the MCP transport protocol).
	// This blocks until the client disconnects (i.e. the Claude Code session ends).
	serverErr := s.Run(context.Background(), &mcp.StdioTransport{})

	// Graceful shutdown: cancel in-flight tasks and wait for goroutines to drain
	// so we don't leave orphaned Ollama requests consuming GPU time.
	pool.Shutdown()

	if serverErr != nil {
		fmt.Fprintf(os.Stderr, "Server error: %v\n", serverErr)
		os.Exit(1)
	}
}
