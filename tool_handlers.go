// tools.go contains the MCP tool handler functions.
//
// Each function corresponds to one of the five MCP tools exposed by the server.
// The handlers are methods on ToolHandlers so they share access to the task
// store and worker pool.
//
// Handler signatures follow the official MCP SDK convention:
//
//	func(ctx, *mcp.CallToolRequest, TypedArgs) (*mcp.CallToolResult, TypedOutput, error)
//
// The SDK automatically:
//   - Generates JSON Schema for the tool from the arg struct tags
//   - Unmarshals incoming JSON arguments into the typed arg struct
//   - Marshals the typed output back to JSON in the response
package main

import (
	"context"
	"fmt"
	"path/filepath"
	"time"

	"github.com/google/uuid"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// ToolHandlers holds references to shared state needed by all tool handlers.
type ToolHandlers struct {
	store *TaskStore
	pool  *WorkerPool
}

// maxBatchSize caps the number of tasks in a single submit_tasks call.
// Prevents accidental submission of enormous batches that could overwhelm
// the worker pool or exhaust memory.
const maxBatchSize = 500

// handleSubmitTasks accepts a batch of tasks, enqueues them in the store,
// and kicks off a worker goroutine for each one. Returns the assigned task
// IDs immediately — the actual work happens asynchronously.
func (h *ToolHandlers) handleSubmitTasks(ctx context.Context, req *mcp.CallToolRequest, args SubmitTasksArgs) (*mcp.CallToolResult, SubmitTasksOutput, error) {
	if len(args.Tasks) > maxBatchSize {
		return nil, SubmitTasksOutput{}, fmt.Errorf("batch too large: %d tasks exceeds maximum of %d", len(args.Tasks), maxBatchSize)
	}

	// Apply concurrency change if requested (before task creation so the
	// new semaphore is active when workers start)
	if args.Concurrency != nil {
		if *args.Concurrency <= 0 {
			return nil, SubmitTasksOutput{}, fmt.Errorf("concurrency must be > 0, got %d", *args.Concurrency)
		}
		h.pool.SetConcurrency(*args.Concurrency)
	}

	// Validate all paths are absolute before creating any tasks (fail fast)
	for i, spec := range args.Tasks {
		if spec.InputFile != "" && !filepath.IsAbs(spec.InputFile) {
			return nil, SubmitTasksOutput{}, fmt.Errorf("task %d: input_file must be an absolute path, got %q", i, spec.InputFile)
		}
		if spec.OutputFile != "" && !filepath.IsAbs(spec.OutputFile) {
			return nil, SubmitTasksOutput{}, fmt.Errorf("task %d: output_file must be an absolute path, got %q", i, spec.OutputFile)
		}
	}

	tasks := make([]*Task, 0, len(args.Tasks))
	taskCtxs := make([]context.Context, 0, len(args.Tasks))
	taskCancels := make([]context.CancelFunc, 0, len(args.Tasks))
	ids := make([]string, 0, len(args.Tasks))

	for _, spec := range args.Tasks {
		id := uuid.New().String()
		model := spec.Model
		if model == "" {
			model = getDefaultModel()
		}
		hint := spec.ResponseHint
		if hint == "" {
			hint = "content"
		}

		// Resolve StripMarkdownFences: nil (not set) → true, explicit value → use it
		stripFences := true
		if spec.StripMarkdownFences != nil {
			stripFences = *spec.StripMarkdownFences
		}

		taskCtx, cancel := context.WithCancel(context.Background())

		task := &Task{
			ID:                  id,
			Tag:                 spec.Tag,
			SystemPrompt:        spec.SystemPrompt,
			Prompt:              spec.Prompt,
			InputFile:           spec.InputFile,
			OutputFile:          spec.OutputFile,
			StripMarkdownFences: stripFences,
			PostWriteCmd:        spec.PostWriteCmd,
			Model:               model,
			ResponseHint:        hint,
			TimeoutSeconds:      spec.TimeoutSeconds,
			Status:              "pending",
			CreatedAt:           time.Now(),
			Cancel:              cancel,
		}
		tasks = append(tasks, task)
		taskCtxs = append(taskCtxs, taskCtx)
		taskCancels = append(taskCancels, cancel)
		ids = append(ids, id)
	}

	h.store.Add(tasks)

	// Start a worker goroutine for each task. They'll block on the semaphore
	// if all worker slots are occupied.
	for i, task := range tasks {
		h.pool.Submit(taskCtxs[i], taskCancels[i], task)
	}

	return nil, SubmitTasksOutput{TaskIDs: ids}, nil
}

// handleCheckTasks returns a compact status overview: aggregate counts plus
// per-task status (without full result content). This is the primary polling
// tool — designed to be cheap on the caller's context window.
func (h *ToolHandlers) handleCheckTasks(_ context.Context, _ *mcp.CallToolRequest, args CheckTasksArgs) (*mcp.CallToolResult, CheckTasksOutput, error) {
	summary, statuses := h.store.Summary(args.TaskIDs, args.Tag)
	return nil, CheckTasksOutput{
		Summary: summary,
		Tasks:   statuses,
	}, nil
}

// handleGetResult retrieves the full Ollama response content for specific
// tasks. The caller should use this selectively — e.g. spot-checking a few
// results or investigating failures — rather than retrieving everything.
func (h *ToolHandlers) handleGetResult(_ context.Context, _ *mcp.CallToolRequest, args GetResultArgs) (*mcp.CallToolResult, GetResultOutput, error) {
	results := h.store.Results(args.TaskIDs)
	return nil, GetResultOutput{Results: results}, nil
}

// handleCancelTasks cancels pending or running tasks. Running tasks have
// their context cancelled, which aborts the in-flight Ollama request.
func (h *ToolHandlers) handleCancelTasks(_ context.Context, _ *mcp.CallToolRequest, args CancelTasksArgs) (*mcp.CallToolResult, CancelTasksOutput, error) {
	count := h.store.Cancel(args.TaskIDs, args.Tag)
	return nil, CancelTasksOutput{Cancelled: count}, nil
}

// handleListModels queries the local Ollama instance for available models.
// Claude should call this at the start of each session to understand what
// models are available and calibrate expectations for worker capability.
func (h *ToolHandlers) handleListModels(ctx context.Context, _ *mcp.CallToolRequest, _ ListModelsArgs) (*mcp.CallToolResult, ListModelsOutput, error) {
	resp, err := h.pool.client.List(ctx)
	if err != nil {
		return nil, ListModelsOutput{}, fmt.Errorf("failed to list Ollama models: %v", err)
	}

	models := make([]ModelInfo, 0, len(resp.Models))
	for _, m := range resp.Models {
		models = append(models, ModelInfo{
			Name:              m.Name,
			Size:              m.Size,
			ParameterSize:     m.Details.ParameterSize,
			QuantizationLevel: m.Details.QuantizationLevel,
			Family:            m.Details.Family,
		})
	}

	return nil, ListModelsOutput{Models: models}, nil
}
