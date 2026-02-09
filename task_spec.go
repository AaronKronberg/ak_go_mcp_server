// task_spec.go defines the submit_tasks tool types: the specification for
// creating tasks and the synchronous response containing task IDs.
package main

// SubmitTasksArgs is the input for the submit_tasks tool.
type SubmitTasksArgs struct {
	// Concurrency optionally adjusts the worker pool size for this and all
	// subsequent batches. Use a lower value when running a larger model (to
	// avoid OOM) or a higher value for lightweight tasks. If nil, the current
	// concurrency is unchanged.
	Concurrency *int `json:"concurrency,omitempty" jsonschema:"Set worker pool concurrency (number of parallel Ollama requests). Persists until changed again. Omit to keep current value."`

	Tasks []TaskSpec `json:"tasks" jsonschema:"List of tasks to submit"`
}

// TaskSpec describes a single unit of work to send to an Ollama worker.
// The caller (e.g. Claude Opus) defines the prompts — this keeps the server
// completely generic. The same tool handles refactoring, summarization,
// code generation, or any other LLM task.
type TaskSpec struct {
	// Tag is an optional caller-defined label for grouping (e.g. "refactor_batch_1").
	// Used to filter tasks in check_tasks and cancel_tasks.
	Tag string `json:"tag,omitempty" jsonschema:"Caller-defined grouping label"`

	// SystemPrompt sets the persona/instructions for the Ollama model.
	// Example: "You are a Go refactoring worker. Return ONLY modified code."
	SystemPrompt string `json:"system_prompt" jsonschema:"System prompt for the worker"`

	// Prompt is the main instruction or question sent to the model.
	Prompt string `json:"prompt" jsonschema:"The user prompt / instructions"`

	// InputFile is an optional absolute path to a file whose contents are read
	// by the server and appended to the prompt. File contents never enter the
	// orchestrating agent's context window — the server reads from disk and
	// passes directly to Ollama.
	InputFile string `json:"input_file,omitempty" jsonschema:"Absolute path to file whose contents are read and appended to the prompt"`

	// OutputFile is an optional absolute path where the worker's response will
	// be written. When set, the result is written to disk and cleared from
	// memory (use get_result to see the output_file path, not the content).
	OutputFile string `json:"output_file,omitempty" jsonschema:"Absolute path where the worker's response will be written"`

	// StripMarkdownFences controls whether markdown code fences are stripped
	// from the output before writing to output_file. Default is true (nil → true).
	// Set explicitly to false to preserve fences.
	StripMarkdownFences *bool `json:"strip_markdown_fences,omitempty" jsonschema:"Strip markdown code fences from output before writing (default: true)"`

	// PostWriteCmd is an optional shell command to run after writing output_file
	// (e.g. "gofmt -w" or "prettier --write"). Runs with a 30-second timeout.
	PostWriteCmd string `json:"post_write_cmd,omitempty" jsonschema:"Shell command to run after writing output_file (30s timeout)"`

	// Model specifies which Ollama model to use. Defaults to DEFAULT_MODEL env
	// var or "qwen2.5-coder:14b" if unset.
	Model string `json:"model,omitempty" jsonschema:"Ollama model to use (default: qwen2.5-coder:14b)"`

	// ResponseHint tells the caller what kind of result to expect. The server
	// always stores the full Ollama response — this hint is metadata that helps
	// the caller decide whether to retrieve full results or just check status.
	//   "status_only" — caller only needs pass/fail (skip get_result)
	//   "content"     — caller will retrieve the full response text
	//   "json"        — caller expects structured JSON output
	ResponseHint string `json:"response_hint,omitempty" jsonschema:"What the caller wants back: status_only|content|json (default: content)"`

	// TimeoutSeconds sets a per-task timeout in seconds. If the Ollama model
	// doesn't respond within this duration, the task fails with a timeout error.
	// Default is 600 (10 minutes), configurable via TASK_TIMEOUT env var.
	// Increase for large inputs or complex generation tasks.
	TimeoutSeconds int `json:"timeout_seconds,omitempty" jsonschema:"Per-task timeout in seconds. Default 600 (10 min). Increase for large/complex tasks. Tasks that hit this limit fail with a clear timeout error so you can retry with a longer value."`
}

// SubmitTasksOutput is returned synchronously from submit_tasks.
type SubmitTasksOutput struct {
	TaskIDs []string `json:"task_ids"`
}
