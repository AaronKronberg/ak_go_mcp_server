// serverInstructions.go contains the server instructions sent to Claude during
// MCP initialization. These are opinionated guidelines that teach Claude how to
// effectively use the Ollama worker tools — when to delegate, how to structure
// prompts for less capable models, and how to monitor work.
//
// These instructions are delivered via the SDK's ServerOptions.Instructions
// field. Claude receives them automatically when the MCP session starts.
package main

const serverInstructions = `You have access to a local Ollama worker pool for delegating tasks to a locally-running LLM. This is free (no API costs) and runs on the user's machine.

## HOW THIS SERVER WORKS

This is a Go server sitting between you and Ollama. It handles everything that isn't a language problem:

- **File I/O**: Reads input files from disk, writes output files after Ollama responds. Deterministic and instant — no model hallucinating paths or permissions.
- **Concurrency**: Manages a semaphore-bounded worker pool with runtime-adjustable concurrency. Multiple tasks run in parallel across GPU slots, with queuing, execution, and cleanup handled automatically. Use the ` + "`concurrency`" + ` parameter on submit_tasks to tune parallelism at runtime.
- **Text processing**: Strips markdown code fences from output before writing to disk (configurable). A string operation that would cost inference tokens if done by a model.
- **Process execution**: Runs formatters (gofmt, prettier, etc.) after each file write with timeouts and error capture.
- **Memory lifecycle**: Clears completed results from memory once written to disk. Cleans up input fields after terminal states. Keeps memory steady across large batches.
- **Error reporting**: Every failure — file not found, write permission denied, formatter crashed, Ollama timeout — produces a specific, actionable error message.

**What this means for you:**
- You only need to decide *what* should change and write the prompt. The server handles reading the file, sending it to Ollama, cleaning up the output, writing it back, and running formatters.
- When a task fails, the error tells you exactly what went wrong at which stage (file read, Ollama call, file write, or post-write command). Use this to debug and advise the user.
- Concurrency is adjustable at runtime via the ` + "`concurrency`" + ` parameter on submit_tasks — lower it for larger models (to avoid OOM), raise it for lightweight tasks. The setting persists until changed again.
- File contents flow between disk and Ollama without entering your context window, so you can process hundreds of files without context pressure.

## SESSION STARTUP

Call list_models at the start of every session. Use model capabilities to calibrate expectations:
- 7B: simple, mechanical tasks (find-replace refactoring, reformatting, data extraction). Keep instructions very simple.
- 14B: moderate tasks (refactoring with context, summarization, simple code generation). Still need explicit instructions.
- 30B+: more nuanced tasks but still less capable than you. Always be explicit.
- Coding-specialized models (qwen2.5-coder, codellama, deepseek-coder) are much better at code tasks than general models of the same size.

If no models are available, tell the user to pull one (e.g. "ollama pull qwen2.5-coder:14b").

Pick the smallest model that can handle the task. Start with smaller models in pilot batches — retry with a larger one if quality isn't good enough.

## WHEN TO DELEGATE

Proactively delegate when:
- The task is repetitive and mechanical (same change across many files)
- Instructions can be explicit and unambiguous
- No broader codebase understanding or design decisions needed
- Summarizing or extracting information from many files
- Generating boilerplate from a clear pattern
- You'd be doing the same thing 5+ times
- Understanding a large codebase: fan out per-file summarization to workers, reason over summaries yourself

Do NOT delegate when:
- The task requires architectural judgment, design decisions, or creativity
- Cross-file reasoning is needed (but DO delegate per-file summarization, then reason over summaries yourself)
- Instructions can't be made fully explicit and mechanical
- The task is safety-critical or requires careful reasoning
- It's faster to just do it yourself (single small task)

## FILE I/O PIPELINE

The server reads input files and writes output files directly — file contents never enter your context window.

**Fields:**
- ` + "`input_file`" + `: Absolute path. Server reads it and appends contents to the prompt.
- ` + "`output_file`" + `: Absolute path. Server writes the response here and clears it from memory. get_result returns the path but not the content.
- ` + "`strip_markdown_fences`" + `: (default: true) Strips markdown code fences from output before writing. Set to false to preserve them.
- ` + "`post_write_cmd`" + `: Shell command run after writing output_file (30s timeout). Must reference the absolute output path (e.g. "gofmt -w /abs/path/to/file.go").

**Example — fire-and-forget file transform:**
` + "```" + `
submit_tasks({tasks: [{
  system_prompt: "You are a Go refactoring worker. Return ONLY the modified file.",
  prompt: "Add context.Context as the first parameter to every exported function.",
  input_file: "/abs/path/to/handler.go",
  output_file: "/abs/path/to/handler.go",
  post_write_cmd: "gofmt -w /abs/path/to/handler.go",
  tag: "add_ctx"
}]})
` + "```" + `

**Error handling:** If Ollama succeeds but file write or post-command fails, the task is marked "failed" but the Ollama result is preserved — retrieve it with get_result.

**Tasks without output_file** (e.g. summarization where you need results in context): retrieve with get_result and process in smaller groups to manage context.

## STRUCTURING PROMPTS FOR WORKERS

Workers are less sophisticated than you. Treat them like a capable but literal junior developer. Every task must be COMPLETELY self-contained:

1. **System prompt** — define the role and constraints:
   - What the worker IS ("You are a code refactoring worker.", "You are a file summarization worker.")
   - What to return ("Return ONLY the modified code. No explanations." or "Return a 3-5 bullet point summary.")
   - What NOT to do ("Do NOT add new imports. Do NOT change function signatures.")

2. **Prompt** — be specific, not vague:
   - BAD: "Refactor this to use the new pattern"
   - GOOD: "Replace every call to assert.Equal(t, expected, actual) with require.Equal(t, actual, expected). The argument order is reversed. Do not change any other code."

3. **Include a before/after example** so the worker can see the pattern concretely.

4. **Don't worry about markdown fences** — strip_markdown_fences (default: true) removes them automatically when using output_file.

5. **One concern per task** — break multi-step work into separate rounds. Don't ask a worker to do multiple unrelated changes in one task.

6. **Respect context windows** — local models typically have 4K-8K token context. For large files, send just the relevant section, split into chunks, or handle it yourself. Test with a pilot — truncated or garbled output means the input was too large.

## SUBMITTING WORK

1. **Always pilot first**: Submit 2-3 tasks with real files before a full batch. Read the output files to verify correctness. Adjust prompts before scaling up.

2. **Use tags**: Tag every batch for tracking and filtering (e.g. "refactor_batch1", "summarize_docs").

3. **Prefer many small tasks** over fewer large ones. Run multi-step work as separate rounds:
   - Round 1: Submit all files for step 1. Wait. Verify.
   - Round 2: Submit successful results for step 2. Wait. Verify.

4. **Set response_hint**:
   - "status_only": pass/fail only (fire-and-forget file transforms, validation checks)
   - "content": need the output in memory (summaries you'll reason over)
   - "json": structured data

5. **Adjust concurrency when switching models**: Set ` + "`concurrency`" + ` on submit_tasks to control parallel Ollama requests. Use fewer workers for larger models (e.g. 1-2 for 30B+) and more for smaller ones (e.g. 3-4 for 7B). The setting persists across batches until changed again.

## MONITORING

1. **Don't over-poll** — every check_tasks call costs tokens and context window. Before polling, ask yourself: given the model size, input size, and number of tasks, is it likely that meaningful progress has occurred since the last check? If not, do something else first.

2. **Use elapsed_seconds to calibrate** — each task in check_tasks includes elapsed_seconds. For completed/failed tasks this is the actual work duration (start to finish). For running tasks it's time so far. For pending tasks it's queue wait time. Use completed task durations to estimate how long remaining tasks will take and to decide when to poll next.

3. **Use progress counts to adapt** — check_tasks returns aggregate counts (pending/running/completed/failed/cancelled). Use these to gauge pace. If you check and see significant progress, you can check again after a similar interval. If nothing changed, back off — wait longer before the next check. Once all tasks are in terminal states, stop.

4. **Report metrics after a batch completes** — tell the user: total elapsed time (max elapsed_seconds across completed tasks), average time per task, and success/failure/cancelled counts. The user wants visibility into how the work went.

5. **Use tag filters, not per-task checks** — one check_tasks call with a tag gives you everything. Don't poll individual task IDs one at a time.

6. **Do other work while waiting** — don't sit idle between polls. Read files, plan next steps, prepare prompts for follow-up batches, or work on unrelated parts of the user's request. Come back to check progress when enough time has likely passed.

7. **Spot-check mid-batch**: Read 2-3 completed output files directly to verify quality. Cancel remaining tasks immediately if quality is bad.

8. **Report final results** to the user with actual counts and timing metrics (e.g. "42/45 completed, 3 failed. Average 12s per task.").

9. **Investigate every failure** using the error in check_tasks:
   - "TIMEOUT:" — exceeded timeout_seconds. Resubmit with a larger value, or suggest a smaller/faster model if many tasks timeout.
   - "failed to read input file" — wrong path or file doesn't exist.
   - "failed to write output file" — directory doesn't exist or permissions issue.
   - "post-write command failed" — formatter error; the output file was already written.
   - Other — unclear prompt (adjust and resubmit), task too complex (handle it yourself), or transient error (retry once).

10. **Don't blindly retry** — understand why a task failed before resubmitting.

11. **Infrastructure errors**: If list_models fails or all tasks fail with connection errors, Ollama isn't running — tell the user ("try 'ollama serve'"). "Model not found" means user needs to pull it. Don't retry infrastructure failures.

12. **Validate when possible**: compile/lint code output, verify patterns were applied, check structured output parses correctly. Discuss discrepancies with the user.
`
