# gen_demo_art_experiment_1 — report_results

> Phase: `gen_paper_repo` · `gen_demo_art`
> Run: `run_S_3Y_l0tX7Xe` — Sharpness-Aware Minimization and Test NLL Beyond Temperature Scaling on Fashion-MNIST
>
> Full, verbatim transcript of this agent task — every system/user prompt, assistant response, thinking block, tool call and tool result — in the order they occurred. Nothing truncated.

## Task: `gen_demo_art_experiment_1` (terminal_claude_agent, claude-haiku-5-5)

### [1] CONFIG · 2026-11-25 09:27:20 UTC

```
model: Claude Haiku 5.5 | effort: high | permission: bypassPermissions
```

### [2] SYSTEM-USER prompt · 2026-11-25 09:27:32 UTC

````
<conversion_philosophy>
**MINIMAL CHANGES — PRESERVE THE ORIGINAL CODE**

The goal is to make the artifact's code READABLE, UNDERSTANDABLE, and RUNNABLE in a short time
to someone reviewing the research, with the option to easily scale parameters back to original
values for a full run (which can take much longer). Think of this as annotating and reformatting,
not refactoring.

**DO:**
- Split the original script into logical notebook cells (imports, setup, processing, results)
- Open with a markdown cell that explains the setup: the question this artifact answers, the
  data and model it runs on, and what the reader will see when it runs
- Add markdown cells BETWEEN code cells explaining what each section does and why
- Add inline comments where the logic is non-obvious
- Add a visualization/summary cell at the end showing key outputs, plotted from the real numbers
  the code computes
- Close with a markdown cell that interprets the results: what the printed numbers and plots
  show, and what they mean for the claim the artifact tests
- Fix hardcoded file paths to use the GitHub data loading pattern

**DO NOT:**
- Rewrite functions or change algorithms
- Rename variables or restructure logic
- Add error handling, type hints, or "improvements" that weren't in the original
- Simplify or "clean up" the original code
- Remove any original comments or logic
- Change the computational approach

The reader should recognize the original script when looking at the notebook — it's the
same code, just split into cells with explanatory markdown between sections.
</conversion_philosophy>

<system_reminder>
Do not ask follow up questions and do not ask the user anything. Execute all steps independently.
You must follow the todo list provided in each prompt exactly as written.
No placeholders, stubs, or incomplete code — all code must be complete and functional.
</system_reminder>

<process_isolation>
CRITICAL: Multiple pipeline runs may execute simultaneously on this machine. `ps aux | grep method.py` matches ALL runs, not just yours.
- NEVER kill processes by name (`killall`, `pkill -f`, `ps aux | grep ... | xargs kill`). This kills OTHER runs' processes.
- NEVER monitor processes by name (`ps aux | grep method.py`). You will see other runs' processes and get confused.
- ALWAYS use PID-based process management:
  Run: `uv run method.py & PID=$!` or `timeout <seconds> uv run method.py & PID=$!`
  Check: `kill -0 $PID 2>/dev/null && echo "Running" || echo "Ended"`
  Stop: `kill $PID`
  Wait: `wait $PID; echo "Exit code: $?"`
  Monitor: `tail -f logs/run.log & TAIL_PID=$!` then `kill $TAIL_PID` when done
</process_isolation>

<flex_environment>
You run in your own macOS account in one long-lived macOS virtual machine on Apple silicon, which other agents share at the same time. The Apple GPU is reached through `mps` in torch and through MLX. CUDA does not exist here, there is no Docker and no Linux binary runs. Run your GPU processes one at a time. Keep temporary files under $TMPDIR. Your own processes reach each other on any local port (port 0 picks a free one; the $AII_PORT_COUNT ports from $AII_PORT_BASE are given to no other agent); other agents' ports are closed to you.
Your pod type is `cpu_basic`: 2 vCPUs intended (the guest's 16 are shared with the other agents, not reserved for you); 16 GB of memory in total, and going over the total stops your job alone; a 40 GB disk quota. You have no GPU: plan CPU-only work.
Start every script that uses torch with the device line: `DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")`. Read the `aii-use-hardware` skill before planning compute.
Instead of each CUDA habit: vLLM -> mlx-lm or llama-server; FP8 -> bf16 or MLX 8-bit; bitsandbytes -> MLX 4-bit or 8-bit; Triton kernels -> torch ops or an MLX Metal kernel; float64 on the GPU -> the CPU; torch.cuda.* -> torch.mps; .cuda() -> .to(DEVICE); "cuda" if torch.cuda.is_available() else "cpu" -> DEVICE (that test silently picks the CPU); flash-attn -> sdpa; several GPUs -> one device; torch.compile -> eager mode.
Known slow spots: LSTMs, small MLPs, ops that fall back to the CPU (warned in the log), the first MPS call's warm-up, and model downloads over a home link; every step fetches its inputs again.
What executor steps get: `cpu_basic`: Flex pod type: 2 vCPUs (shared), 16 GB of memory in total, no GPU, a 40 GB disk; `cpu_plus`: Flex pod type: 4 vCPUs (shared), 32 GB of memory in total, no GPU, a 40 GB disk; `gpu_basic`: Flex pod type: 7 vCPUs (shared), 29 GB of memory in total, of which up to 20 GB may be GPU memory (Apple unified memory: GPU memory is not extra), the Apple GPU, a 40 GB disk; `gpu_plus`: Flex pod type: 8 vCPUs (shared), 60 GB of memory in total, of which up to 40 GB may be GPU memory (Apple unified memory: GPU memory is not extra), the Apple GPU, a 60 GB disk. Text agents get `text`; an executor step gets the pod type its plan's `runpod_compute_profile` names, capped by the run's compute tier.
One step's inputs (the run's files plus its sessions) stay under 20 GB: keep large models and datasets out of the run's files.
`python` is a read-only venv with a pod's packages but no torch and no MLX: for those and any other package make your own `uv venv`, then `uv pip install torch` (it takes the guest's MPS build: pass no index URL) or `uv pip install mlx`; never upgrade torch.
</flex_environment>

<workspace>
Your workspace: `/Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/4_gen_paper_repo/_4_gen_demo_art/notebook_workspaces/iter_1/art_SL9gPfE09HlN`

CRITICAL: Every file you create, write, or save MUST be inside this workspace directory (subdirectories OK). You MUST NOT write files anywhere outside this path — external paths are READ-ONLY. Use absolute paths for all file operations.

EVERY file write MUST start with `/Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/4_gen_paper_repo/_4_gen_demo_art/notebook_workspaces/iter_1/art_SL9gPfE09HlN/`:
GOOD: `/Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/4_gen_paper_repo/_4_gen_demo_art/notebook_workspaces/iter_1/art_SL9gPfE09HlN/file.py`, `/Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/4_gen_paper_repo/_4_gen_demo_art/notebook_workspaces/iter_1/art_SL9gPfE09HlN/results/out.json`
BAD: `/tmp/file.py`, `~/output.json`, `./file.py`, any path outside the workspace

Put downloads, unpacked archives, caches and intermediate files in `/Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/4_gen_paper_repo/_4_gen_demo_art/notebook_workspaces/iter_1/art_SL9gPfE09HlN/scratch/`. A folder named `scratch` is never kept, anywhere: whatever a later step or a reader needs goes elsewhere in the workspace.
</workspace>
<disposable_outputs>
A SHARED CACHE ALREADY EXISTS FOR THIS RUN: `/Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache`
`HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE`,
`TORCH_HOME`, `PIP_CACHE_DIR` and `UV_CACHE_DIR` are ALREADY set to point
there. Every step and every iteration of this run shares it, so a model or
dataset an earlier experiment downloaded is already on disk for you.

DO NOT override those variables. In particular do NOT write the common
pattern `os.environ["HF_HOME"] = <workspace>/hf_cache` — `HF_HOME` and
`TRANSFORMERS_CACHE` are read differently by `huggingface_hub` (one has
`/hub` appended, the other does not), so pointing both at one directory
stores every weight TWICE. That mistake cost one run 25 GB of identical
blobs. If you must set them, use the values above verbatim.

YOUR WORKING DIRECTORY IS A DELIVERABLE. When this module ends it must read
like a GitHub repository someone else can fork, resume and run — and the bulk
it holds must be either worth keeping or restorable. This run shares a storage
volume with the database; a run that fills it stops every other run on the box.

So before you finish, produce TWO files:

1. `.aii/manifest.yaml` — one entry per heavy path, each with EXACTLY ONE decision.
   The `.aii/` directory ALREADY EXISTS in your cwd: write the file into
   it. Do not create, replace or `touch` `.aii` itself — a plain file by
   that name makes the manifest unwritable for the rest of the module.

```yaml
entries:
  - path: results/
    keep: six GPU-hours of sweep output, not reproducible inside this run
  - path: hf_cache/
    delete: redownloadable
    source: "huggingface-cli download meta-llama/Llama-3-8B"
  - path: checkpoints/
    delete: regenerable
    source: "uv run train.py --epochs 3 --seed 0"
```

   - `keep:` takes a ONE-LINE reason. Use it for the expensive and the
     irreproducible: trained weights, long-running results, datasets you
     collected yourself.
   - `delete:` takes `redownloadable` (and a `source:` naming the repo id, URL
     or command) or `regenerable` (and a `source:` that is the command which
     rebuilds it). These are deleted AFTER the round ends, never mid-step.
   - Every path is RELATIVE TO YOUR CWD and must resolve INSIDE it. Absolute
     paths, `..`, and anything resolving outside are rejected.
   - Globs and whole directories are fine. A whole `hf_cache/` is ONE entry —
     do not list files individually.

2. `README.md` — written as if your cwd were a GitHub repository: what you
   did, the layout with a line per important file/directory, how to run it,
   and a **"Restoring removed files"** section giving the install/download
   command for EVERY `delete` entry. An `install.sh` or `restore.sh` beside it
   is welcome.

A CHECKER RUNS WHEN YOU SUBMIT. If anything heavy has no decision it fails
your submission and hands you the uncovered list, grouped by directory with
sizes, and you fix the manifest and submit again.

WHAT NEEDS NO DECISION — do not write entries for these:
- text and code files, at ANY size (source, JSON, CSV, YAML, logs, markdown);
- anything under the auto-keep floor (10 MB), whatever it holds.
Only large binaries and cache directories (`hf_cache/`, `.venv/`,
`node_modules/`, `checkpoints/`, `wandb/`, `__pycache__/`, …) need one.

NEVER mark your results, figures, papers, code, logs or anything a later step
reads as `delete`. If a later step needs it, it is a `keep`.

WHAT A `keep` BUYS YOU. Anything you do not mark `delete` stays exactly where
you wrote it, on this run's storage volume, at the path it already has — it is
not moved, renamed or copied. A later round reads it there, by that absolute
workspace path, so a checkpoint you keep is a checkpoint the next round can
load instead of retraining. It is also the ONLY copy: the publish step pushes
your cwd to GitHub but skips every file of 100 MB or
more, so trained weights and large binary artifacts never leave the volume.
Name each kept artifact in your results and your `README.md` by its path
RELATIVE to your cwd, and say it stays on the run's volume rather than in the
published repository. Never write an absolute server path into a file that is
published: a reader's machine has none of them.
</disposable_outputs>

<tool_use>
Maximize parallel tool calls. Parallelize independent operations, only sequentialize dependencies.
- Multiple searches/fetches on different topics → parallel in one turn
- Search then fetch results → sequential (need URLs first)
</tool_use>

<task>
Convert this artifact's Python script into a demo notebook with MINIMAL changes to the original code.
Split into cells, add markdown explanations between sections, add a visualization cell at the end.
Output: mini_demo_data.json + code_demo.ipynb (notebook that loads data from GitHub URL)
</task>

<artifact_info>
id: art_SL9gPfE09HlN
type: experiment
title: Does SAM Improve Calibration on Fashion-MNIST?
summary: >-
  Artifact: a controlled comparison of plain SGD (momentum 0.9) against sharpness-aware minimization (SAM) on a two-block
  CNN trained on Fashion-MNIST. Both optimizers share architecture, data split (50k/10k/10k from the 60k training images,
  official 10k test), batch size, schedule, weight decay and per-seed initialization and batch order. Learning rates (and
  SAM's rho) were chosen on validation NLL only, using seed 100. Final results use 5 paired seeds (0-4) at 40 epochs, plus
  compressed 5- and 15-epoch schedules to test whether the calibration gap grows with training length. A temperature-scaled
  SGD control is fitted on validation logits. Reported metrics: accuracy, ECE (15 equal-width bins), ACE, NLL, Brier, AURC,
  with paired t-intervals (df=4), Holm correction and reliability diagrams. Key results at epoch 40 (SAM minus SGD): accuracy
  +0.0015 with CI [-0.0007, +0.0036] (no detectable difference); ECE -0.0378 [-0.0414, -0.0341]; NLL -0.0980 [-0.1092, -0.0867];
  Brier -0.0136; AURC -0.0015. Against temperature-scaled SGD, SAM's NLL is still lower (-0.0209 [-0.0266, -0.0151], Holm
  p=0.0027), but its ECE advantage is not significant after Holm (p=0.079). Caveats: SAM is under-fit at epoch 40 (training
  loss 0.107 against 0.008 for SGD), so part of its calibration advantage may come from a less-fit model; both selected learning
  rates (0.01) sit at the lower edge of the grid; one tuning configuration collapsed to chance (SAM lr 0.03, rho 0.2) and
  was not selected; the 60-epoch extension was not run. Scope is one dataset and one small architecture, with five seeds,
  so power is low. Independent re-derivation: the headline paired differences and CIs were recomputed directly from the saved
  test logits with separate code (explicit log-softmax and bin loops, scipy ttest_rel), and matched to the reported values;
  placebo checks (identical models, permuted labels) did not reject. Not re-derived: the ECE bin loop and ACE/Brier/AURC for
  every run were not independently recomputed, and the training itself was not repeated.
workspace_path: >-
  /Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/3_invention_loop/iter_1/gen_art/gen_art_experiment_1
out_expected_files:
- method.py
- full_method_out.json
- mini_method_out.json
- preview_method_out.json
- reproducibility.md
out_demo_files:
- path: method.py
  description: Research methodology implementation
</artifact_info>

<github_repo>
Repo URL: https://github.com/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small
Raw data URL: https://raw.githubusercontent.com/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small/main/round-1/experiment-1/demo/mini_demo_data.json

URLs won't work yet — files pushed to GitHub AFTER notebook creation.
Use local fallback pattern so notebook works locally (now) and in Colab (after deployment).
</github_repo>

<data_file_sizes>
Data files come in three sizes:
- preview_*_out.json — READ THIS to inspect the data structure
- mini_*_out.json (~3 examples) — use for prototyping/testing
- full_*_out.json (complete) — use for the final production run. NEVER open it directly (too large to read into context). Instead, extract values programmatically with shell commands (e.g. grep) or a Python script (use aii-long-running-tasks skill for scripts).
</data_file_sizes>

<install_dependencies_pattern>
Follow the aii-colab skill exactly. It has the install cell pattern, pre-installed package list, numpy 2.0 compat shims, and all Colab-specific rules.
</install_dependencies_pattern>

<data_loading_pattern>
`mini_demo_data.json` = curated subset for the demo.
Use this pattern for Colab compatibility (GitHub URL with local fallback):
```python
GITHUB_DATA_URL = "https://raw.githubusercontent.com/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small/main/round-1/experiment-1/demo/mini_demo_data.json"
import http.client
import json
import urllib.request
from pathlib import Path

def load_data():
    try:
        with urllib.request.urlopen(GITHUB_DATA_URL, timeout=15) as response:
            return json.loads(response.read().decode())
    # Network errors, timeouts, a placeholder URL, bad JSON, a cut-off response.
    except (OSError, ValueError, http.client.HTTPException) as e:
        print("GitHub data unavailable, using local mini_demo_data.json:", e)
    local = Path("mini_demo_data.json")
    if local.exists(): return json.loads(local.read_text())
    raise FileNotFoundError("Could not load mini_demo_data.json")
```
</data_loading_pattern>

<figure_style>
Every plot in every notebook of this run shares one style. Paste this block, verbatim, at the end
of the imports cell:
```python
import matplotlib.pyplot as plt

# One colour per role, the same in every notebook of this run and in the paper's figures.
COLOURS = {
    "method": "#0173B2",  # the paper's own method, model or condition
    "baseline": "#949494",  # the main baseline, control or reference
    "other_1": "#DE8F05",  # the first further comparator
    "other_2": "#029E73",  # the second
    "other_3": "#CC78BC",  # the third
    "other_4": "#CA9161",  # the fourth
    "other_5": "#56B4E9",  # the fifth
}
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#CCCCCC",
    "grid.alpha": 0.6,
    "axes.axisbelow": True,
    "axes.prop_cycle": plt.cycler(color=list(COLOURS.values())),
    "image.cmap": "cividis",
    "legend.frameon": False,
    "figure.constrained_layout.use": True,
    "figure.dpi": 120,
})
```
Colour each series by its ROLE from `COLOURS` (the proposed method `COLOURS["method"]`, the main
baseline `COLOURS["baseline"]`, further comparators `other_1`, `other_2`, ... in order). Use the
default colormap (cividis) for magnitudes and "RdBu_r" for signed values. Set no colours, fonts,
sizes or style sheets of your own.
</figure_style>

<notebook_structure>
--- Setup ---
Cell 1 (markdown): Title, description, what this artifact does.
Cell 2 (code): Install dependencies — follow the aii-colab skill's install cell pattern exactly. Fill in all packages imported by the artifact's code.
Cell 3 (code): Imports — copy original import block as-is, plus any additional imports needed for the notebook (e.g. matplotlib for visualization), then the <figure_style> block.
Cell 4 (code): Data loading helper — use the <data_loading_pattern> above.
Cell 5 (code): `data = load_data()`

--- Config ---
Config cell (code): Define ALL tunable parameters (iterations, epochs, n_samples, hidden_size, etc.) as variables at the top of this cell. Start with the ABSOLUTE MINIMUM values — the smallest that produce any output at all (e.g. 1 iteration, 2 samples, smallest array size). These get gradually increased during testing — see TODOs.

--- Processing ---
Remaining cells: One code cell per logical section of the original script. Add a markdown cell BEFORE each code cell. Copy code as closely as possible, with these changes:
  1. Replace file paths to use the loaded `data` variable.
  2. Use the config variables from the config cell (NOT hardcoded values).
  3. Minimal fixes are allowed if something doesn't work in notebook context (e.g. adjusting paths, removing CLI args, fixing imports), but keep changes to the absolute minimum.

--- Results ---
Visualization cell (code): Print key results in a readable table, plot numeric data with matplotlib if appropriate, in the <figure_style> style.
</notebook_structure>

<priority>
WORKING > OPTIMIZED. A small-scale demo that runs correctly is the goal. Once the notebook passes with minimum config values, scale up only if time permits — do NOT spend multiple retries chasing larger parameters. If a working version exists, finish and move on.
</priority>

<max_notebook_total_runtime>600s (10 min)</max_notebook_total_runtime>

<test_environment>
To test-run the notebook in a clean environment (simulating Colab), use the disposable venv at `/Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache/nb_test_envs/art_SL9gPfE09HlN-13a6bcbdff23`. It is on this machine's local disk, outside your workspace, which is on a slow network volume: never create a venv inside your workspace.
```bash
UV_VENV_CLEAR=1 UV_CACHE_DIR=/Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache/uv uv venv -q --seed --python 3.12 /Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache/nb_test_envs/art_SL9gPfE09HlN-13a6bcbdff23
UV_CACHE_DIR=/Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache/uv uv pip install -q --python /Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache/nb_test_envs/art_SL9gPfE09HlN-13a6bcbdff23/bin/python jupyter ipykernel
/Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache/nb_test_envs/art_SL9gPfE09HlN-13a6bcbdff23/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 code_demo.ipynb --output code_demo.ipynb
```
The timeout is set to <max_notebook_total_runtime>. The entire notebook must finish within this time.
`UV_VENV_CLEAR=1` recreates the venv empty at the start of every test, so each test starts clean. Do NOT delete it yourself: the pipeline removes it after you finish. If you run a test in the background, wait for its result before starting the next test.

What happens: the venv starts with only pip, jupyter and ipykernel. When the notebook's install cell runs, `google.colab` is NOT in sys.modules, so ALL packages get installed — non-Colab packages unconditionally, and Colab packages (numpy, pandas, etc.) at Colab's exact versions via the guard block. The result mirrors Colab's environment as closely as possible. If a cell fails, fix the notebook and re-run.
</test_environment>

FIRST, add ALL of these to your todo list using your task/todo-tracking tool:

CRITICAL: Todo content must be copied exactly as is written here, with NO CHANGES. These todos are intentionally detailed so that another LLM could read each one without any external context and understand exactly what it has to do.


<todos>
TODO 1. Read and STRICTLY follow these skills: aii-colab, aii-long-running-tasks.
TODO 2. Read demo file and relevant preview_* files (preview only). Understand script structure: imports, setup, processing, output. Identify ALL tunable parameters (iterations, epochs, n_samples, hidden_size, batch_size, etc.) — these go in the config cell.
TODO 3. Create `mini_demo_data.json`: curated subset from at most ONE dataset (no more than 100 diverse examples). CRITICAL: do NOT read/grep full output file — may crash. Use `head -c 5000` or stream first entries with Python to pick examples.
TODO 4. Create `code_demo.ipynb` via NotebookEdit following <notebook_structure>. Set ALL config parameters to ABSOLUTE MINIMUM values — the smallest that produce any output (e.g. 1 iteration, 2 samples, smallest array sizes). Test-run using <test_environment>. Fix all errors until it passes.
TODO 5. GRADUALLY SCALE (but don't overdo it): increase config params step by step (e.g. ~2x each round). After each increase: test-run, record runtime, fix errors. STOP SCALING as soon as results look meaningful — a working small-scale demo beats a failed large-scale one. If full original params fit within <max_notebook_total_runtime> (10% margin), use them. Otherwise keep whatever works and comment out the true original values. Do NOT spend more than 2-3 scaling rounds.
TODO 6. Verify: (1) code_demo.ipynb contains GITHUB_DATA_URL = "https://raw.githubusercontent.com/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small/main/round-1/experiment-1/demo/mini_demo_data.json" exactly, (2) mini_demo_data.json exists, (3) uses GitHub URL pattern not just open('mini_demo_data.json').
</todos>

---

Output the result as JSON to: `./.terminal_claude_agent_struct_out.json`

JSON Schema:
```json
{
  "$defs": {
    "DemoExpectedFiles": {
      "description": "Expected output files from code demo notebook generation.",
      "properties": {
        "notebook": {
          "default": "",
          "description": "Path to the generated demo notebook. Example: 'code_demo.ipynb'",
          "title": "Notebook",
          "type": "string"
        },
        "mini_data_file": {
          "default": "",
          "description": "Path to the mini demo data JSON (curated subset). Example: 'mini_demo_data.json'",
          "title": "Mini Data File",
          "type": "string"
        }
      },
      "title": "DemoExpectedFiles",
      "type": "object"
    }
  },
  "description": "Dataset/experiment/evaluation \u2192 Jupyter notebook demo.\n\nTitle and summary come from the parent artifact (gen_art step),\nnot from the demo agent. The agent only outputs expected files.",
  "properties": {
    "out_expected_files": {
      "$ref": "#/$defs/DemoExpectedFiles",
      "description": "All output files you created. Must include the demo notebook."
    }
  },
  "required": [],
  "title": "CodeDemo",
  "type": "object"
}
```

IMPORTANT: this task is NOT complete until `./.terminal_claude_agent_struct_out.json` exists and contains JSON matching the schema above.

Does sharpness-aware minimization improve test accuracy and calibration over plain SGD for a small convolutional network on Fashion-MNIST? Train the models and compare.
````

### [3] TOOL CALL — Skill · 2026-11-25 09:27:34 UTC

```
aii-colab
```

### [4] TOOL RESULT — Skill · 2026-11-25 09:27:34 UTC

```
{"success": true, "commandName": "aii-colab"}
```

### [5] TOOL CALL — Skill · 2026-11-25 09:27:34 UTC

```
aii-long-running-tasks
```

### [6] TOOL RESULT — Skill · 2026-11-25 09:27:34 UTC

```
{"success": true, "commandName": "aii-long-running-tasks"}
```

### [7] TOOL CALL — Bash · 2026-11-25 09:27:34 UTC

```
List workspace and source artifact directory:
ls -la && ls -la .aii 2>/dev/null; ls -la /Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/3_invention_loop/iter_1/gen_art/gen_art_experiment_1 2>/dev/null
```

### [8] TOOL RESULT — Bash · 2026-11-25 09:27:36 UTC

```
{"stdout": "total 20\ndrwx------ 4 s20002 s20002   128 Nov 25 09:27 .\ndrwx------ 3 s20002 s20002    96 Nov 25 09:27 ..\ndrwxrwx--- 2 s20002 s20002    64 Nov 25 09:27 .aii\n-rw-r----- 1 root   s20002 19238 Nov 25 09:27 .repl_agent.ptylog\ntotal 0\ndrwxrwx--- 2 s20002 s20002  64 Nov 25 09:27 .\ndrwx------ 4 s20002 s20002 128 Nov 25 09:27 ..\ntotal 18488\ndrwx------ 21 s20002 s20002     672 Nov 25 09:27 .\ndrwx------  5 s20002 s20002     160 Nov 25 09:27 ..\ndrwx------  4 s20002 s20002     128 Nov 25 09:27 .aii\n-rw-------  1 s20002 s20002 1372368 Jan  1  1970 .repl_agent.ptylog\n-rw-------  1 s20002 s20002    2637 Jan  1  1970 .terminal_claude_agent_struct_out.json\n-rw-------  1 s20002 s20002   20910 Jan  1  1970 analyze.py\n-rw-------  1 s20002 s20002    5742 Jan  1  1970 audit_rederive.py\ndrwx------  6 s20002 s20002     192 Nov 25 09:27 figures\n-rw-------  1 s20002 s20002 8681928 Jan  1  1970 full_method_out.json\n-rw-------  1 s20002 s20002 8681928 Jan  1  1970 method_out.json\n-rw-------  1 s20002 s20002   24315 Jan  1  1970 method.py\n-rw-------  1 s20002 s20002    8212 Jan  1  1970 metrics.py\n-rw-------  1 s20002 s20002   25442 Jan  1  1970 mini_method_out.json\n-rw-------  1 s20002 s20002   25254 Jan  1  1970 preview_method_out.json\n-rw-------  1 s20002 s20002    1138 Jan  1  1970 pyproject.toml\n-rw-------  1 s20002 s20002    6115 Jan  1  1970 README.md\n-rw-------  1 s20002 s20002    7308 Jan  1  1970 reproducibility.md\ndrwx------ 18 s20002 s20002     576 Nov 25 09:27 results\n-rwx------  1 s20002 s20002     347 Jan  1  1970 run_all.sh\ndrwx------  3 s20002 s20002      96 Nov 25 09:27 tests\n-rw-------  1 s20002 s20002   29584 Jan  1  1970 uv.lock", "stderr": "", "interrupted": false, "isImage": false, "noOutputExpected": false}
```

### [9] ASSISTANT · 2026-11-25 09:27:38 UTC

```
Both skills are loaded. Next I'll read the source script and the preview data, which is the input to the notebook.
```

### [10] TOOL CALL — Read · 2026-11-25 09:27:38 UTC

```
/Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/3_invention_loop/iter_1/gen_art/gen_art_experiment_1/method.py
```

### [11] TOOL RESULT — Read · 2026-11-25 09:27:38 UTC

```
{"type": "text", "file": {"filePath": "/Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/3_invention_loop/iter_1/gen_art/gen_art_experiment_1/method.py", "content": "#!/usr/bin/env python3\n\"\"\"SAM vs SGD on a small CNN (Fashion-MNIST): training, tuning and final runs.\n\nModes (run in this order; each mode skips runs whose metrics.json already exists):\n  smoke       2-epoch SGD and SAM runs written to scratch/ (pipeline check)\n  timing      time one epoch per condition, apply the... [truncated, 25765 chars total]
```

### [12] TOOL CALL — Bash · 2026-11-25 09:27:38 UTC

```
Preview data structure and project config:
cd /Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/3_invention_loop/iter_1/gen_art/gen_art_experiment_1 && head -c 3000 preview_method_out.json; echo; echo ---; python3 -c "import json;d=json.load(open('mini_method_out.json'));print(type(d), list(d.keys()) if isinstance(d,dict) else len(d))"; cat run_all.sh; cat pyproject.toml
```

### [13] TOOL RESULT — Bash · 2026-11-25 09:27:38 UTC

```
{"stdout": "{\n  \"metadata\": {\n    \"method_name\": \"SAM vs SGD calibration on a small CNN (Fashion-MNIST)\",\n    \"description\": \"Plain SGD (momentum 0.9) and sharpness-aware minimisation (SAM) on the same base optimiser, matched architecture, data split, batch size, schedule and seeds. Hyperparameters chosen on validation only....\",\n    \"chosen_hparams\": {\n      \"selection_rule\": \"minimum validation NLL of the final-epoch model, seed 100\",\n      \"sgd\": {\n        \"lr\": 0.01,\n        \"val_nll_final\": 0.26497\n      },\n      \"sam\": {\n        \"lr\": 0.01,\n        \"rho\": 0.1,\n        \"val_nll_final\": 0.18404\n      },\n      \"grid\": {\n        \"lr\": [\n          0.01,\n          0.03,\n          0.1\n        ],\n        \"rho\": [\n          0.05,\n          0.1,\n          0.2\n        ],\n        \"compressed\": true,\n        \"decisions\": [\n          \"no cut: full grid fits the timing gate\"\n        ],\n        \"projected_total_h\": 0.5861708631166341,\n        \"t_sgd_epoch_s\": 1.4025094159997025,\n        \"t_sam_epoch_s\": 2.304789625000012,\n        \"projected_sam_40ep_run_s\": 92.19158500000049\n      },\n      \"tuning_table\": \"results/tuning_table.csv\"\n    },\n    \"split\": \"60k official training images: fixed 50k train / 10k validation (seed 0); 10k official test set used only in final evaluation\",\n    \"architecture\": \"Conv(1,32,3,p1)-BN-ReLU-Pool, Conv(32,64,3,p1)-BN-ReLU-Pool, FC 3136-128-10\",\n    \"training\": {\n      \"batch_size\": 128,\n      \"epochs\": 40,\n      \"compressed_epochs\": [\n        5,\n        15\n      ],\n      \"scheduler\": \"CosineAnnealingLR(T_max=E) stepped per epoch\",\n      \"momentum\": 0.9,\n      \"weight_decay\": 0.0005,\n      \"augmentation\": \"none\",\n      \"normalisation\": \"x/255 then train-split mean/std\"\n    },\n    \"timing_gate\": {\n      \"lr\": [\n        0.01,\n        0.03,\n        0.1\n      ],\n      \"rho\": [\n        0.05,\n        0.1,\n        0.2\n      ],\n      \"compressed\": true,\n      \"decisions\": [\n        \"no cut: full grid fits the timing gate\"\n      ],\n      \"projected_total_h\": 0.5861708631166341,\n      \"t_sgd_epoch_s\": 1.4025094159997025,\n      \"t_sam_epoch_s\": 2.304789625000012,\n      \"projected_sam_40ep_run_s\": 92.19158500000049\n    },\n    \"verdicts\": {\n      \"a_accuracy\": {\n        \"criterion\": \"SAM test accuracy differs from SGD at epoch 40 (paired, seeds 0-4)\",\n        \"verdict\": \"no detectable difference (CI includes zero)\",\n        \"mean_diff\": 0.001460000000000039,\n        \"ci95\": [\n          -0.000707761856975732,\n          0.00362776185697581\n        ],\n        \"holm_p\": 0.13483865104196946\n      },\n      \"b_calibration_vs_sgd\": {\n        \"criterion\": \"SAM ECE is lower than SGD ECE (paired, seeds 0-4)\",\n        \"verdict\": \"SAM lower: favours SAM (CI excludes zero)\",\n        \"mean_diff\": -0.03778507472240959,\n        \"ci95\": [\n          -0.041445905732531846,\n          -0.03412424371228734\n        ],\n        \"holm_p\": 7.060005749967155e-05\n      },\n      \"c_survives_temperature_scaling\": {\n        \"criterion\": \"SAM NLL is lower than temperature-scaled SGD NL\n---\n<class 'dict'> ['metadata', 'datasets']\n#!/usr/bin/env bash\n# Full pipeline: timing gate -> tuning (seed 100) -> final (seeds 0-4) -> compressed (5/15 epochs).\nset -euo pipefail\ncd \"$(dirname \"$0\")\"\nfor mode in timing tune final compressed; do\n  echo \"=== mode: $mode ===\"\n  .venv/bin/python method.py \"$mode\"\ndone\necho \"=== analyze ===\"\n.venv/bin/python analyze.py\necho \"PIPELINE_DONE\"\n[project]\nname = \"sam-sgd-calibration\"\nversion = \"1.0.0\"\ndescription = \"SAM vs SGD calibration on a small CNN (Fashion-MNIST)\"\nrequires-python = \"==3.12.*\"\ndependencies = [\n    \"attrs==26.1.0\",\n    \"contourpy==1.4.0\",\n    \"cycler==0.12.1\",\n    \"filelock==4.3.1\",\n    \"fonttools==4.68.0\",\n    \"fsspec==2026.9.0\",\n    \"jinja2==3.1.6\",\n    \"jsonschema==4.26.0\",\n    \"jsonschema-specifications==2025.9.1\",\n    \"kiwisolver==1.5.1\",\n    \"loguru==0.8.0\",\n    \"markupsafe==3.0.4\",\n    \"matplotlib==3.11.2\",\n    \"mpmath==1.3.0\",\n    \"networkx==3.7\",\n    \"numpy==2.5.5\",\n    \"packaging==26.4\",\n    \"pillow==13.0.0\",\n    \"psutil==7.2.2\",\n    \"pyparsing==3.3.3\",\n    \"python-dateutil==4.0.0\",\n    \"referencing==0.37.0\",\n    \"rpds-py==2026.9.1\",\n    \"scipy==1.18.1\",\n    \"sympy==1.14.0\",\n    \"torch==2.16.0.dev20261116+aii1\",\n    \"torchvision==0.30.0.dev20261116\",\n    \"typing-extensions==4.16.0\",\n    \"tzdata==2026.6\",\n]\n\n[tool.uv]\npackage = false\n# The pinned torch build comes from the run's package mirror and needs a Linux-only\n# triton; resolve for the macOS environment the results were produced on.\nenvironments = [\"sys_platform == 'darwin'\"]", "stderr": "\nShell cwd was reset to /Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/4_gen_paper_repo/_4_gen_demo_art/notebook_workspaces/iter_1/art_SL9gPfE09HlN", "interrupted": false, "isImage": false, "noOutputExpected": false}
```

### [14] ASSISTANT · 2026-11-25 09:27:56 UTC

```
The source is a 570-line training pipeline (tuning grid, 5 seeds, 40 epochs, torchvision download). The notebook has to shrink that to a fast demo, so I need the helper module it imports and a Fashion-MNIST source to build the 100-example subset from.
```

### [15] TOOL CALL — Read · 2026-11-25 09:27:56 UTC

```
/Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/3_invention_loop/iter_1/gen_art/gen_art_experiment_1/metrics.py
```

### [16] TOOL RESULT — Read · 2026-11-25 09:27:56 UTC

```
{"type": "text", "file": {"filePath": "/Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/3_invention_loop/iter_1/gen_art/gen_art_experiment_1/metrics.py", "content": "\"\"\"Calibration and accuracy metrics computed from saved logits (numpy only).\n\nConventions\n- Confidence is the maximum softmax probability (top-label confidence).\n- ECE: 15 equal-width bins on confidence, bins are (lo, hi]; first bin includes 0.\n- ACE: adaptive calibration error, 15 equal-mass bins (sorted by confidence),\n  unweighted mean over bins of |accuracy - mean confidence|.\n- NLL: mean negative log-likelihood of the true class.\n- Brier: multiclass Brier score, sum over classes, mean over examples.\n- AURC: area under the risk-coverage curve; examples ranked by confidence\n  (descending), risk at coverage k = mean error among the k most confident.\n\"\"\"\n\nfrom __future__ import annotations\n\nimport numpy as np\nfrom scipy.optimize import minimize_scalar\nfrom scipy.special import logsumexp\n\nN_BINS = 15\nTEMP_LO, TEMP_HI = 0.05, 20.0\n\n\ndef softmax64(logits: np.ndarray) -> np.ndarray:\n    z = logits.astype(np.float64)\n    z = z - z.max(axis=1, keepdims=True)\n    e = np.exp(z)\n    return e / e.sum(axis=1, keepdims=True)\n\n\ndef nll_per_example(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> np.ndarray:\n    z = logits.astype(np.float64) / temperature\n    lse = logsumexp(z, axis=1)\n    return lse - z[np.arange(len(z)), labels]\n\n\ndef confidence_and_correct(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0):\n    probs = softmax64(logits / temperature)\n    conf = probs.max(axis=1)\n    correct = (probs.argmax(axis=1) == labels).astype(np.float64)\n    return conf, correct, probs\n\n\ndef brier_per_example(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:\n    onehot = np.eye(probs.shape[1])[labels]\n    return np.sum((probs - onehot) ** 2, axis=1)\n\n\ndef ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:\n    \"\"\"Expected calibration error with equal-width bins over (0, 1].\"\"\"\n    n = len(conf)\n    edges = np.linspace(0.0, 1.0, n_bins + 1)\n    idx = np.clip(np.searchsorted(edges, conf, side=\"left\") - 1, 0, n_bins - 1)\n    cnt = np.bincount(idx, minlength=n_bins).astype(np.float64)\n    sum_conf = np.bincount(idx, weights=conf, minlength=n_bins)\n    sum_acc = np.bincount(idx, weights=correct, minlength=n_bins)\n    nz = cnt > 0\n    return float(np.abs(sum_acc[nz] - sum_conf[nz]).sum() / n)\n\n\ndef reliability_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> dict:\n    \"\"\"Per-bin counts, mean confidence and accuracy (equal-width bins).\"\"\"\n    edges = np.linspace(0.0, 1.0, n_bins + 1)\n    idx = np.clip(np.searchsorted(edges, conf, side=\"left\") - 1, 0, n_bins - 1)\n    cnt = np.bincount(idx, minlength=n_bins)\n    sum_conf = np.bincount(idx, weights=conf, minlength=n_bins)\n    sum_acc = np.bincount(idx, weights=correct, minlength=n_bins)\n    with np.errstate(invalid=\"ignore\", divide=\"ignore\"):\n        mean_conf = np.where(cnt > 0, sum_conf / np.maximum(cnt, 1), np.nan)\n        acc = np.where(cnt > 0, sum_acc / np.maximum(cnt, 1), np.nan)\n    return {\n        \"edges\": edges.tolist(),\n        \"count\": cnt.tolist(),\n        \"mean_confidence\": mean_conf.tolist(),\n        \"accuracy\": acc.tolist(),\n    }\n\n\ndef ace(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:\n    \"\"\"Adaptive calibration error: equal-mass bins, unweighted mean over bins.\"\"\"\n    order = np.argsort(conf, kind=\"stable\")\n    groups = np.array_split(order, n_bins)\n    vals = [abs(correct[g].mean() - conf[g].mean()) for g in groups if len(g) > 0]\n    return float(np.mean(vals))\n\n\ndef aurc(conf: np.ndarray, correct: np.ndarray) -> float:\n    \"\"\"Area under the risk-coverage curve (ranking by confidence, descending).\"\"\"\n    order = np.argsort(-conf, kind=\"stable\")\n    err = (1.0 - correct)[order]\n    cum = np.cumsum(err)\n    risk = cum / np.arange(1, len(err) + 1)\n    return float(risk.mean())\n\n\ndef all_metrics(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> dict:\n    conf, correct, probs = confidence_and_correct(logits, labels, temperature)\n    return {\n        \"accuracy\": float(correct.mean()),\n        \"ece\": ece(conf, correct),\n        \"ace\": ace(conf, correct),\n        \"nll\": float(nll_per_example(logits, labels, temperature).mean()),\n        \"brier\": float(brier_per_example(probs, labels).mean()),\n        \"aurc\": aurc(conf, correct),\n    }\n\n\ndef bootstrap_ci(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0,\n                 n_boot: int = 1000, seed: int = 0) -> dict:\n    \"\"\"95% percentile bootstrap CI over test examples, one resample per replicate.\"\"\"\n    conf, correct, probs = confidence_and_correct(logits, labels, temperature)\n    nll_pe = nll_per_example(logits, labels, temperature)\n    brier_pe = brier_per_example(probs, labels)\n    rng = np.random.default_rng(seed)\n    n = len(labels)\n    samples = {k: np.empty(n_boot) for k in (\"accuracy\", \"ece\", \"ace\", \"nll\", \"brier\", \"aurc\")}\n    for b in range(n_boot):\n        i = rng.integers(0, n, size=n)\n        c, r = conf[i], correct[i]\n        samples[\"accuracy\"][b] = r.mean()\n        samples[\"ece\"][b] = ece(c, r)\n        samples[\"ace\"][b] = ace(c, r)\n        samples[\"nll\"][b] = nll_pe[i].mean()\n        samples[\"brier\"][b] = brier_pe[i].mean()\n        samples[\"aurc\"][b] = aurc(c, r)\n    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]\n            for k, v in samples.items()}\n\n\ndef fit_temperature(logits: np.ndarray, labels: np.ndarray) -> dict:\n    \"\"\"Fit one temperature on validation logits by minimising validation NLL.\n\n    Bounded scalar minimisation over [0.05, 20]. A coarse log-grid is also\n    evaluated so that a non-global bounded-search optimum is visible.\n    \"\"\"\n    def obj(t: float) -> float:\n        return float(nll_per_example(logits, labels, t).mean())\n\n    res = minimize_scalar(obj, bounds=(TEMP_LO, TEMP_HI), method=\"bounded\",\n                          options={\"xatol\": 1e-6})\n    grid = np.exp(np.linspace(np.log(TEMP_LO), np.log(TEMP_HI), 400))\n    grid_vals = np.array([obj(t) for t in grid])\n    best_grid = float(grid[np.argmin(grid_vals)])\n    t_hat = float(res.x)\n    at_bound = bool(t_hat <= TEMP_LO * 1.001 or t_hat >= TEMP_HI * 0.999)\n    return {\n        \"temperature\": t_hat,\n        \"val_nll_at_T\": obj(t_hat),\n        \"val_nll_at_bounds\": {str(TEMP_LO): obj(TEMP_LO), str(TEMP_HI): obj(TEMP_HI)},\n        \"grid_best_T\": best_grid,\n        \"grid_best_val_nll\": float(grid_vals.min()),\n        \"at_bound\": at_bound,\n    }\n\n\n# ---------------------------------------------------------------- statistics\n\ndef paired_t_interval(d: np.ndarray, alpha: float = 0.05) -> dict:\n    \"\"\"Mean of paired differences with a t-interval (df = n - 1) and two-sided p.\"\"\"\n    from scipy import stats\n\n    d = np.asarray(d, dtype=np.float64)\n    n = len(d)\n    mean = float(d.mean())\n    sd = float(d.std(ddof=1)) if n > 1 else 0.0\n    se = sd / np.sqrt(n)\n    tcrit = float(stats.t.ppf(1 - alpha / 2, n - 1))\n    if se == 0.0:\n        p = 1.0 if mean == 0.0 else 0.0\n        t_stat = float(\"inf\") if mean != 0.0 else 0.0\n    else:\n        t_stat = mean / se\n        p = float(2 * stats.t.sf(abs(t_stat), n - 1))\n    return {\n        \"n\": n,\n        \"df\": n - 1,\n        \"mean\": mean,\n        \"sd\": sd,\n        \"ci95\": [mean - tcrit * se, mean + tcrit * se],\n        \"t\": float(t_stat),\n        \"p\": p,\n    }\n\n\ndef holm(pvals: dict[str, float]) -> dict[str, float]:\n    \"\"\"Holm step-down adjusted p-values over a family.\"\"\"\n    items = sorted(pvals.items(), key=lambda kv: kv[1])\n    m = len(items)\n    adj: dict[str, float] = {}\n    running = 0.0\n    for rank, (name, p) in enumerate(items):\n        val = min(1.0, (m - rank) * p)\n        running = max(running, val)\n        adj[name] = running\n    return adj\n\n\n# ---------------------------------------------------------------- self-test helpers\n\ndef ece_known_example() -> float:\n    \"\"\"Hand-built case with known ECE = 0.1.\n\n    Four examples at confidence 0.9 with 3 correct: bin accuracy 0.75, gap 0.15,\n    weight 4/6, contribution 0.1. Two examples at confidence 0.5 with 1 correct:\n    bin accuracy 0.5 equals confidence, gap 0. Total ECE = 0.1.\n    \"\"\"\n    conf = np.array([0.9, 0.9, 0.9, 0.9, 0.5, 0.5])\n    correct = np.array([1, 1, 1, 0, 1, 0], dtype=np.float64)\n    return ece(conf, correct)\n", "numLines": 210, "startLine": 1, "totalLines": 210}}
```

### [17] TOOL CALL — Bash · 2026-11-25 09:27:56 UTC

```
Read metrics module and search for Fashion-MNIST data locally:
ls /Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache 2>/dev/null; find /Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache -iname "*fashion*" 2>/dev/null | head; find /Volumes/aii-s20002/work -maxdepth 6 -iname "*fashion*" 2>/dev/null | head -5; timeout 20 curl -sI https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-images-idx3-ubyte.gz | head -1; timeout 20 curl -sI https://huggingface.co | head -1; echo done
```

### [18] TOOL RESULT — Bash · 2026-11-25 09:27:58 UTC

```
{"stdout": "HTTP/2 200 \r\nHTTP/2 200 \r\ndone", "stderr": "", "interrupted": false, "isImage": false, "noOutputExpected": false}
```

### [19] ASSISTANT · 2026-11-25 09:28:03 UTC

```
The helper module is clear, and the network is reachable. I'll fetch the raw Fashion-MNIST IDX files, then write a script that builds a class-balanced 100-example subset (60 train, 20 val, 20 test) into `mini_demo_data.json`.
```

### [20] TOOL CALL — Bash · 2026-11-25 09:28:03 UTC

```
Download Fashion-MNIST IDX files into scratch:
mkdir -p scratch/data && cd scratch/data && for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do curl -sSL -o $f.gz https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/$f.gz || echo FAIL $f; done; ls -la
```

### [21] TOOL RESULT — Bash · 2026-11-25 09:28:07 UTC

```
{"stdout": "total 30196\ndrwx------ 6 s20002 s20002      192 Nov 25 09:28 .\ndrwx------ 3 s20002 s20002       96 Nov 25 09:28 ..\n-rw------- 1 s20002 s20002  4422102 Nov 25 09:28 t10k-images-idx3-ubyte.gz\n-rw------- 1 s20002 s20002     5148 Nov 25 09:28 t10k-labels-idx1-ubyte.gz\n-rw------- 1 s20002 s20002 26421880 Nov 25 09:28 train-images-idx3-ubyte.gz\n-rw------- 1 s20002 s20002    29515 Nov 25 09:28 train-labels-idx1-ubyte.gz", "stderr": "", "interrupted": false, "isImage": false, "noOutputExpected": false}
```

### [22] TOOL CALL — Write · 2026-11-25 09:28:13 UTC

```
File: /Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/4_gen_paper_repo/_4_gen_demo_art/notebook_workspaces/iter_1/art_SL9gPfE09HlN/make_mini_demo_data.py

#!/usr/bin/env python3
"""Build mini_demo_data.json: a class-balanced 100-image subset of Fashion-MNIST.

Reads the official IDX files (already downloaded into scratch/data) and writes
60 train, 20 validation and 20 test images, 6/2/2 per class, chosen with seed 0.
Pixels are stored as integers 0-255 (flattened 28x28), labels as 0-9.
"""

import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "scratch" / "data"
OUT = ROOT / "mini_demo_data.json"
PER_CLASS = {"train": 6, "val": 2, "test": 2}


def read_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        buf = f.read()
    return np.frombuffer(buf, dtype=np.uint8, offset=16).reshape(-1, 784)


def read_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        buf = f.read()
    return np.frombuffer(buf, dtype=np.uint8, offset=8).astype(np.int64)


def main() -> None:
    rng = np.random.default_rng(0)
    splits = {
        "train": (read_images(RAW / "train-images-idx3-ubyte.gz"),
                  read_labels(RAW / "train-labels-idx1-ubyte.gz")),
        "test": (read_images(RAW / "t10k-images-idx3-ubyte.gz"),
                 read_labels(RAW / "t10k-labels-idx1-ubyte.gz")),
    }
    # Validation images come from the official training split, disjoint from the
    # train subset, mirroring the original 50k/10k split of the 60k training images.
    tr_x, tr_y = splits["train"]
    out = {}
    used = set()
    chosen = {"train": [], "val": [], "test": []}
    for c in range(10):
        pool = rng.permutation(np.where(tr_y == c)[0])
        n_tr, n_va = PER_CLASS["train"], PER_CLASS["val"]
        chosen["train"] += list(pool[:n_tr])
        chosen["val"] += list(pool[n_tr:n_tr + n_va])
        used.update(pool[:n_tr + n_va].tolist())
        te_x, te_y = splits["test"]
        tpool = rng.permutation(np.where(te_y == c)[0])
        chosen["test"] += list(tpool[:PER_CLASS["test"]])

    for name in ("train", "val"):
        idx = np.array(chosen[name])
        out[name] = {"x": tr_x[idx].tolist(), "y": tr_y[idx].tolist()}
    te_x, te_y = splits["test"]
    idx = np.array(chosen["test"])
    out["test"] = {"x": te_x[idx].tolist(), "y": te_y[idx].tolist()}

    out["metadata"] = {
        "dataset": "Fashion-MNIST (zalandoresearch/fashion-mnist, official IDX files)",
        "selection": "10 classes, 6 train + 2 val from official training images, 2 test from official test images; seed 0",
        "pixels": "uint8 integers 0-255, flattened 28x28 row-major",
        "n_train": len(out["train"]["y"]),
        "n_val": len(out["val"]["y"]),
        "n_test": len(out["test"]["y"]),
    }
    OUT.write_text(json.dumps(out))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)", out["metadata"])


if __name__ == "__main__":
    main()
```

### [23] TOOL RESULT — Write · 2026-11-25 09:28:13 UTC

```
{"type": "create", "filePath": "/Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/4_gen_paper_repo/_4_gen_demo_art/notebook_workspaces/iter_1/art_SL9gPfE09HlN/make_mini_demo_data.py", "content": "#!/usr/bin/env python3\n\"\"\"Build mini_demo_data.json: a class-balanced 100-image subset of Fashion-MNIST.\n\nReads the official IDX files (already downloaded into scratch/data) and writes\n60 train, 20 validation and 20 test images, 6/2/2 per class, chosen with seed 0.\nPixels are stored as integers 0-255 (flattened 28x28), labels as 0-9.\n\"\"\"\n\nimport gzip\nimport json\nfrom pathlib import Path\n\nimport numpy as np\n\nROOT = Path(__file__).resolve().parent\nRAW = ROOT / \"scratch\" / \"data\"\nOUT = ROOT / \"mini_demo_data.json\"\nPER_CLASS = {\"train\": 6, \"val\": 2, \"test\": 2}\n\n\ndef read_images(path: Path) -> np.ndarray:\n    with gzip.open(path, \"rb\") as f:\n        buf = f.read()\n    return np.frombuffer(buf, dtype=np.uint8, offset=16).reshape(-1, 784)\n\n\ndef read_labels(path: Path) -> np.ndarray:\n    with gzip.open(path, \"rb\") as f:\n        buf = f.read()\n    return np.frombuffer(buf, dtype=np.uint8, offset=8).astype(np.int64)\n\n\ndef main() -> None:\n    rng = np.random.default_rng(0)\n    splits = {\n        \"train\": (read_images(RAW / \"train-images-idx3-ubyte.gz\"),\n                  read_labels(RAW / \"train-labels-idx1-ubyte.gz\")),\n        \"test\": (read_images(RAW / \"t10k-images-idx3-ubyte.gz\"),\n                 read_labels(RAW / \"t10k-labels-idx1-ubyte.gz\")),\n    }\n    # Validation images come from the official training split, disjoint from the\n    # train subset, mirroring the original 50k/10k split of the 60k training images.\n    tr_x, tr_y = splits[\"train\"]\n    out = {}\n    used = set()\n    chosen = {\"train\": [], \"val\": [], \"test\": []}\n    for c in range(10):\n        pool = rng.permutation(np.where(tr_y == c)[0])\n        n_tr, n_va = PER_CLASS[\"train\"], PER_CLASS[\"val\"]\n        chosen[\"train\"] += list(pool[:n_tr])\n        chosen[\"val\"] += list(pool[n_tr:n_tr + n_va])\n        used.update(pool[:n_tr + n_va].tolist())\n        te_x, te_y = splits[\"test\"]\n        tpool = rng.permutation(np.where(te_y == c)[0])\n        chosen[\"test\"] += list(tpool[:PER_CLASS[\"test\"]])\n\n    for name in (\"train\", \"val\"):\n        idx = np.array(chosen[name])\n        out[name] = {\"x\": tr_x[idx].tolist(), \"y\": tr_y[idx].tolist()}\n    te_x, te_y = splits[\"test\"]\n    idx = np.array(chosen[\"test\"])\n    out[\"test\"] = {\"x\": te_x[idx].tolist(), \"y\": te_y[idx].tolist()}\n\n    out[\"metadata\"] = {\n        \"dataset\": \"Fashion-MNIST (zalandoresearch/fashion-mnist, official IDX files)\",\n        \"selection\": \"10 classes, 6 train + 2 val from official training images, 2 test from official test images; seed 0\",\n        \"pixels\": \"uint8 integers 0-255, flattened 28x28 row-major\",\n        \"n_train\": len(out[\"train\"][\"y\"]),\n        \"n_val\": len(out[\"val\"][\"y\"]),\n        \"n_test\": len(out[\"test\"][\"y\"]),\n    }\n    OUT.write_text(json.dumps(out))\n    print(f\"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)\", out[\"metadata\"])\n\n\nif __name__ == \"__main__\":\n    main()\n", "structuredPatch": [], "originalFile": null, "userModified": false}
```

### [24] TOOL CALL — Bash · 2026-11-25 09:28:17 UTC

```
Build the 100-example mini demo JSON:
cd /Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/4_gen_paper_repo/_4_gen_demo_art/notebook_workspaces/iter_1/art_SL9gPfE09HlN && (python3 -c "import numpy" 2>/dev/null && python3 make_mini_demo_data.py || (uv run --no-project --with numpy python make_mini_demo_data.py)) && ls -la mini_demo_data.json
```

### [25] TOOL RESULT — Bash · 2026-11-25 09:28:19 UTC

```
{"stdout": "wrote /Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/4_gen_paper_repo/_4_gen_demo_art/notebook_workspaces/iter_1/art_SL9gPfE09HlN/mini_demo_data.json (296.7 KB) {'dataset': 'Fashion-MNIST (zalandoresearch/fashion-mnist, official IDX files)', 'selection': '10 classes, 6 train + 2 val from official training images, 2 test from official test images; seed 0', 'pixels': 'uint8 integers 0-255, flattened 28x28 row-major', 'n_train': 60, 'n_val': 20, 'n_test': 20}\n-rw------- 1 s20002 s20002 303772 Nov 25 09:28 mini_demo_data.json", "stderr": "", "interrupted": false, "isImage": false, "noOutputExpected": false}
```

### [26] ASSISTANT · 2026-11-25 09:29:47 UTC

```
The subset is built: 100 images, 296 KB. Next I'll write the notebook. I'm generating it with a small builder that pulls the training and metric code verbatim from `method.py` and `metrics.py`, so the notebook matches the original code instead of being retyped by hand.
```

### [27] TOOL CALL — Write · 2026-11-25 09:29:47 UTC

```
File: /Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/4_gen_paper_repo/_4_gen_demo_art/notebook_workspaces/iter_1/art_SL9gPfE09HlN/scratch/make_notebook.py

#!/usr/bin/env python3
"""Assemble code_demo.ipynb from the original method.py and metrics.py.

Model and training code is copied verbatim from method.py (line blocks found by
their section headers); the only edits are the ones listed in the asserts below.
"""

import json
from pathlib import Path

SRC = Path("/Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/3_invention_loop/iter_1/gen_art/gen_art_experiment_1")
WS = Path("/Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/4_gen_paper_repo/_4_gen_demo_art/notebook_workspaces/iter_1/art_SL9gPfE09HlN")
URL = ("https://raw.githubusercontent.com/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small/"
       "main/round-1/experiment-1/demo/mini_demo_data.json")

method = (SRC / "method.py").read_text()
metrics_src = (SRC / "metrics.py").read_text()


def between(text: str, start: str, end: str) -> str:
    i = text.index(start)
    j = text.index(end, i)
    return text[i:j]


HDR_MODEL = "# ---------------------------------------------------------------- model and steps"
HDR_TRAIN = "# ---------------------------------------------------------------- one training run"
HDR_HELP = "# ---------------------------------------------------------------- helpers"

model_block = between(method, HDR_MODEL, HDR_TRAIN)
train_block = between(method, HDR_TRAIN, HDR_HELP)

# Edit 1: drop the per-run file log sink (notebook logs to stderr only).
sink_line = ('    sink = logger.add(LOGS / f"{name}.log", level="DEBUG",\n'
             '                      filter=lambda r: r["extra"].get("run") == name)\n')
finally_block = "    finally:\n        logger.remove(sink)\n"
assert train_block.count(sink_line) == 1 and train_block.count(finally_block) == 1
train_block = train_block.replace(sink_line, "").replace(finally_block, "")
# Edit 2: the notebook has no separate training-label file; labels come from `data`.
assert "M." not in model_block and "M." not in train_block

# metrics.py is imported as `M` in method.py; here its functions are defined in-cell.
metrics_code = metrics_src[metrics_src.index("from __future__"):]
assert "import metrics" not in method
method_refs_M = ["M.nll_per_example"]
train_block = train_block.replace("M.nll_per_example", "nll_per_example")
assert "M." not in train_block


def cell(kind: str, text: str) -> dict:
    lines = text.splitlines(keepends=True)
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": lines}


imports_cell = '''from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

import matplotlib.pyplot as plt

# One colour per role, the same in every notebook of this run and in the paper's figures.
COLOURS = {
    "method": "#0173B2",  # the paper's own method, model or condition
    "baseline": "#949494",  # the main baseline, control or reference
    "other_1": "#DE8F05",  # the first further comparator
    "other_2": "#029E73",  # the second
    "other_3": "#CC78BC",  # the third
    "other_4": "#CA9161",  # the fourth
    "other_5": "#56B4E9",  # the fifth
}
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#CCCCCC",
    "grid.alpha": 0.6,
    "axes.axisbelow": True,
    "axes.prop_cycle": plt.cycler(color=list(COLOURS.values())),
    "image.cmap": "cividis",
    "legend.frameon": False,
    "figure.constrained_layout.use": True,
    "figure.dpi": 120,
})
'''

install_cell = '''import subprocess, sys
def _pip(*a): subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *a])

# Not pre-installed on Colab: always install
_pip('loguru==0.8.0')

# Core packages (pre-installed on Colab, install locally to match Colab env)
if 'google.colab' not in sys.modules:
    _pip('numpy==2.0.2', 'scipy==1.16.3', 'matplotlib==3.10.0', 'torch==2.9.0')
'''

data_cell = f'''GITHUB_DATA_URL = "{URL}"
import http.client
import json
import urllib.request
from pathlib import Path

def load_data():
    try:
        with urllib.request.urlopen(GITHUB_DATA_URL, timeout=15) as response:
            return json.loads(response.read().decode())
    # Network errors, timeouts, a placeholder URL, bad JSON, a cut-off response.
    except (OSError, ValueError, http.client.HTTPException) as e:
        print("GitHub data unavailable, using local mini_demo_data.json:", e)
    local = Path("mini_demo_data.json")
    if local.exists(): return json.loads(local.read_text())
    raise FileNotFoundError("Could not load mini_demo_data.json")

data = load_data()
print({{k: len(v["y"]) for k, v in data.items() if k != "metadata"}}, "examples per split")
'''

config_cell = '''DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH = 128
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
EPOCHS = 2            # original: 40 (paper setting); smallest demo value: 2
SEEDS = (0,)          # original: (0, 1, 2, 3, 4)
LR_SGD = 0.01         # selected on validation NLL (seed 100) in the original tuning sweep
LR_SAM = 0.01         # selected on validation NLL (seed 100) in the original tuning sweep
RHO = 0.1             # SAM neighbourhood radius, selected on validation NLL
RESULTS = Path("results")
RUNS = RESULTS / "runs"
LOGS = RESULTS / "logs"
CKPTS = RESULTS / "checkpoints"
LOGS.mkdir(parents=True, exist_ok=True)
'''

prepare_cell = '''def prepare_data(raw: dict) -> dict:
    """Turn the JSON lists into tensors, normalised with train-split statistics.

    Replaces the original load_data(), which downloaded Fashion-MNIST: the subset
    is already split (train/val/test) and stored as integer pixels 0-255.
    """
    def to_tensors(split: str):
        x = torch.tensor(raw[split]["x"], dtype=torch.float32).div(255.0).view(-1, 1, 28, 28)
        y = torch.tensor(raw[split]["y"], dtype=torch.long)
        return x, y

    x_tr, y_tr = to_tensors("train")
    x_va, y_va = to_tensors("val")
    x_te, y_te = to_tensors("test")
    mean = float(x_tr.mean())
    std = float(x_tr.std())
    print(f"Train-split normalisation: mean={mean:.4f} std={std:.4f}")

    def norm(x: torch.Tensor) -> torch.Tensor:
        return ((x - mean) / std).contiguous()

    return {
        "x_train": norm(x_tr).to(DEVICE), "y_train": y_tr.to(DEVICE),
        "x_val": norm(x_va).to(DEVICE), "y_val": y_va.to(DEVICE),
        "x_test": norm(x_te).to(DEVICE), "y_test": y_te.to(DEVICE),
    }


data = prepare_data(data)
print({k: tuple(v.shape) for k, v in data.items()})
'''

run_name_cell = '''def run_name(condition: str, lr: float, rho: float | None, seed: int, epochs: int) -> str:
    rho_part = "" if rho is None else f"_rho{rho}"
    return f"{condition}_lr{lr}{rho_part}_E{epochs}_s{seed}"
'''

final_cell = '''def run_final(data: dict) -> None:
    """Train SGD and SAM for each seed, then fit temperature scaling on validation logits.

    Mirrors mode_final in method.py. The learning rates and rho come from the config
    cell instead of chosen_hparams.json, and only the test logits of each run are
    saved as files.
    """
    val_y = data["y_val"].cpu().numpy()
    for seed in SEEDS:
        for cond, lr, r in (("sgd", LR_SGD, None), ("sam", LR_SAM, RHO)):
            name = run_name(cond, lr, r, seed, EPOCHS)
            out = RUNS / f"final_{name}"
            s = run_or_load(data=data, condition=cond, lr=lr, rho=r, seed=seed,
                            epochs=EPOCHS, save_epochs=(EPOCHS,), out_dir=out,
                            include_test=True, checkpoint=False, name=f"final_{name}")
            z = np.load(out / f"logits_ep{EPOCHS}.npz")
            temp = fit_temperature(z["val_logits"], val_y)
            T = temp["temperature"]
            np.savez(out / "ts_test_logits.npz",
                     test_logits=(z["test_logits"].astype(np.float64) / T).astype(np.float32))
            ts = {"seed": seed, "condition": cond, **temp}
            (out / "temperature.json").write_text(json.dumps(ts, indent=2))
            logger.info(f"final {name}: T={T:.4f} at_bound={temp['at_bound']} "
                        f"final_train_loss={s['final_train_loss_eval']:.5f} "
                        f"test_acc@{EPOCHS}={s['saved'][str(EPOCHS)]['test_acc']:.4f}")


run_final(data)
'''

results_cell = '''test_y = data["y_test"].cpu().numpy()
METRICS = ["accuracy", "ece", "ace", "nll", "brier", "aurc"]
per_seed = []
for seed in SEEDS:
    sgd_dir = RUNS / f"final_{run_name('sgd', LR_SGD, None, seed, EPOCHS)}"
    sam_dir = RUNS / f"final_{run_name('sam', LR_SAM, RHO, seed, EPOCHS)}"
    z_sgd = np.load(sgd_dir / f"logits_ep{EPOCHS}.npz")["test_logits"]
    z_sam = np.load(sam_dir / f"logits_ep{EPOCHS}.npz")["test_logits"]
    T = json.loads((sgd_dir / "temperature.json").read_text())["temperature"]
    per_seed.append({
        "seed": seed,
        "logits": {"SGD": z_sgd, "SAM": z_sam},
        "SGD": all_metrics(z_sgd, test_y),
        "SAM": all_metrics(z_sam, test_y),
        "SGD+TS": all_metrics(z_sgd, test_y, temperature=T),
    })

# Paired SAM-minus-SGD differences over seeds, t-interval and Holm correction over metrics.
diffs = {m: paired_t_interval(np.array([r["SAM"][m] - r["SGD"][m] for r in per_seed]))
         for m in METRICS}
holm_p = holm({m: diffs[m]["p"] for m in METRICS})

print(f"{'metric':<10}{'SGD':>9}{'SAM':>9}{'SGD+TS':>9}{'SAM-SGD':>10}{'95% CI':>24}{'Holm p':>9}")
for m in METRICS:
    mean = {k: np.mean([r[k][m] for r in per_seed]) for k in ("SGD", "SAM", "SGD+TS")}
    ci = diffs[m]["ci95"]
    print(f"{m:<10}{mean['SGD']:>9.4f}{mean['SAM']:>9.4f}{mean['SGD+TS']:>9.4f}"
          f"{diffs[m]['mean']:>+10.4f}  [{ci[0]:+.4f}, {ci[1]:+.4f}]{holm_p[m]:>9.4f}")
'''

viz_cell = '''labels_m = ["SGD", "SAM", "SGD+TS"]
colours = [COLOURS["baseline"], COLOURS["method"], COLOURS["other_1"]]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, m in zip(axes[:2], ["ece", "nll"]):
    vals = [np.mean([r[k][m] for r in per_seed]) for k in labels_m]
    ax.bar(labels_m, vals, color=colours)
    ax.set_title(f"Test {m.upper()} (mean over seeds)")

# Reliability diagram on the first seed: accuracy against mean confidence per bin.
ax = axes[2]
first = per_seed[0]["logits"]
for name, key in (("SGD", "SGD"), ("SAM", "SAM")):
    conf, correct, _ = confidence_and_correct(first[key], test_y)
    rb = reliability_bins(conf, correct)
    keep = [i for i, c in enumerate(rb["count"]) if c > 0]
    ax.plot([rb["mean_confidence"][i] for i in keep], [rb["accuracy"][i] for i in keep],
            "o-", color=COLOURS["method"] if key == "SAM" else COLOURS["baseline"], label=name)
ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
ax.set_xlabel("Mean confidence")
ax.set_ylabel("Accuracy")
ax.set_title("Reliability (seed 0)")
ax.legend()
plt.show()
'''

processing_md = {
    "model": "## Model and training step\nThe CNN and the two optimiser steps, copied from `method.py`: `sgd_step` is plain SGD; `sam_step` perturbs the weights by `rho` along the normalised gradient, takes a second gradient there with BatchNorm statistics frozen, and then applies the base SGD step to the perturbed-point gradient.",
    "train": "## Training runs\nThe original `train_run` with two changes: the per-run file log sink is removed, and the checkpoint path is unused in this notebook. Each run saves logits on the validation and test sets at the final epoch.",
    "final": "## Final runs with temperature scaling\nFor each seed, train SGD and SAM. A single temperature is fitted on the validation logits of the SGD model by minimising validation NLL (the SGD+TS control). The fitted temperature is applied to the SGD test logits.",
    "results": "## Test-set results\nAccuracy, ECE (15 equal-width bins), ACE, NLL, Brier and AURC for SGD, SAM and temperature-scaled SGD. The SAM-minus-SGD differences use paired t-intervals over seeds with df = n-1 (df = 0 with one seed, so the interval is undefined: use more seeds for a CI).",
    "viz": "## Plots\nMean ECE and NLL per method, and a reliability diagram for the first seed.",
}

cells = [
    cell("markdown", "# Does sharpness-aware minimisation (SAM) improve calibration over SGD on Fashion-MNIST?\n\n"
         "This notebook demonstrates the artifact `method.py`. It trains a small two-block CNN on Fashion-MNIST with "
         "plain SGD (momentum 0.9) and with SAM on the same base optimiser, using the same architecture, data split, "
         "batch order and per-seed initialisation. It then compares accuracy, ECE, NLL, Brier and AURC on the test set, "
         "and checks whether SAM's calibration advantage survives temperature scaling of the SGD model.\n\n"
         "**Demo scope.** The notebook runs on a 100-image, class-balanced subset of Fashion-MNIST "
         "(`mini_demo_data.json`: 60 train, 20 validation, 20 test) so that it finishes in minutes. The numbers it prints "
         "are therefore not the paper's results: the paper trains on 50,000 images and evaluates on the official 10,000-image "
         "test set. The config cell sets the scale.\n\n"
         "**Selected hyperparameters.** The learning rates (SGD 0.01, SAM 0.01) and SAM's rho (0.1) are the values the original "
         "pipeline selected on validation NLL with seed 100. The demo uses them directly and skips the tuning grid."),
    cell("markdown", "## Install dependencies\nInstall the packages the notebook imports. On Colab the core scientific "
         "packages are already installed and must not be reinstalled, so they are only installed locally."),
    cell("code", install_cell),
    cell("markdown", "## Imports\nThe original import block (without the torchvision and `metrics` imports, which are replaced "
         "by the JSON loader and the in-notebook metrics below), followed by the shared figure style."),
    cell("code", imports_cell),
    cell("markdown", "## Metric functions\nThe calibration and accuracy metrics from `metrics.py`, the module the original script "
         "imports as `M`. They are defined here so the notebook needs no extra file; calls use the bare names instead of `M.`."),
    cell("code", metrics_code),
    cell("markdown", "## Load the data\n`data` is the curated 100-image subset. `load_data()` reads it from GitHub and falls "
         "back to the local `mini_demo_data.json`."),
    cell("code", data_cell),
    cell("markdown", "## Config\nAll tunable values. The first run uses the smallest values (2 epochs, one seed); they are "
         "increased in later rounds, see the scaling note at the end."),
    cell("code", config_cell),
    cell("markdown", "## Prepare the data\nConvert the lists to tensors and normalise with the training-split mean and standard "
         "deviation, as the original `load_data()` did."),
    cell("code", prepare_cell),
    cell("markdown", processing_md["model"]),
    cell("code", model_block),
    cell("markdown", processing_md["train"]),
    cell("code", run_name_cell + "\n" + train_block),
    cell("markdown", processing_md["final"]),
    cell("code", final_cell),
    cell("markdown", processing_md["results"]),
    cell("code", results_cell),
    cell("markdown", processing_md["viz"]),
    cell("code", viz_cell),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}
(WS / "code_demo.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote", WS / "code_demo.ipynb", len(cells), "cells")
```

### [28] TOOL RESULT — Write · 2026-11-25 09:29:47 UTC

```
{"type": "create", "filePath": "/Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/4_gen_paper_repo/_4_gen_demo_art/notebook_workspaces/iter_1/art_SL9gPfE09HlN/scratch/make_notebook.py", "content": "#!/usr/bin/env python3\n\"\"\"Assemble code_demo.ipynb from the original method.py and metrics.py.\n\nModel and training code is copied verbatim from method.py (line blocks found by\ntheir section headers); the only edits are the ones listed in the asserts below.\n\"\"\"\n\nimport json\nfrom pathlib import Path\n\nSRC = Path(\"/Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/3_invention_loop/iter_1/gen_art/gen_art_experiment_1\")\nWS = Path(\"/Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/4_gen_paper_repo/_4_gen_demo_art/notebook_workspaces/iter_1/art_SL9gPfE09HlN\")\nURL = (\"https://raw.githubusercontent.com/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small/\"\n       \"main/round-1/experiment-1/demo/mini_demo_data.json\")\n\nmethod = (SRC / \"method.py\").read_text()\nmetrics_src = (SRC / \"metrics.py\").read_text()\n\n\ndef between(text: str, start: str, end: str) -> str:\n    i = text.index(start)\n    j = text.index(end, i)\n    return text[i:j]\n\n\nHDR_MODEL = \"# ---------------------------------------------------------------- model and steps\"\nHDR_TRAIN = \"# ---------------------------------------------------------------- one training run\"\nHDR_HELP = \"# ---------------------------------------------------------------- helpers\"\n\nmodel_block = between(method, HDR_MODEL, HDR_TRAIN)\ntrain_block = between(method, HDR_TRAIN, HDR_HELP)\n\n# Edit 1: drop the per-run file log sink (notebook logs to stderr only).\nsink_line = ('    sink = logger.add(LOGS / f\"{name}.log\", level=\"DEBUG\",\\n'\n             '                      filter=lambda r: r[\"extra\"].get(\"run\") == name)\\n')\nfinally_block = \"    finally:\\n        logger.remove(sink)\\n\"\nassert train_block.count(sink_line) == 1 and train_block.count(finally_block) == 1\ntrain_block = train_block.replace(sink_line, \"\").replace(finally_block, \"\")\n# Edit 2: the notebook has no separate training-label file; labels come from `data`.\nassert \"M.\" not in model_block and \"M.\" not in train_block\n\n# metrics.py is imported as `M` in method.py; here its functions are defined in-cell.\nmetrics_code = metrics_src[metrics_src.index(\"from __future__\"):]\nassert \"import metrics\" not in method\nmethod_refs_M = [\"M.nll_per_example\"]\ntrain_block = train_block.replace(\"M.nll_per_example\", \"nll_per_example\")\nassert \"M.\" not in train_block\n\n\ndef cell(kind: str, text: str) -> dict:\n    lines = text.splitlines(keepends=True)\n    if kind == \"markdown\":\n        return {\"cell_type\": \"markdown\", \"metadata\": {}, \"source\": lines}\n    return {\"cell_type\": \"code\", \"execution_count\": None, \"metadata\": {},\n            \"outputs\": [], \"source\": lines}\n\n\nimports_cell = '''from __future__ import annotations\n\nimport gc\nimport json\nimport sys\nimport time\nfrom pathlib import Path\n\nimport numpy as np\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom loguru import logger\n\nimport matplotlib.pyplot as plt\n\n# One colour per role, the same in every notebook of this run and in the paper's figures.\nCOLOURS = {\n    \"method\": \"#0173B2\",  # the paper's own method, model or condition\n    \"baseline\": \"#949494\",  # the main baseline, control or reference\n    \"other_1\": \"#DE8F05\",  # the first further comparator\n    \"other_2\": \"#029E73\",  # the second\n    \"other_3\": \"#CC78BC\",  # the third\n    \"other_4\": \"#CA9161\",  # the fourth\n    \"other_5\": \"#56B4E9\",  # the fifth\n}\nplt.rcParams.update({\n    \"font.family\": \"serif\",\n    \"font.serif\": [\"CMU Serif\", \"Computer Modern Roman\", \"DejaVu Serif\"],\n    \"mathtext.fontset\": \"cm\",\n    \"font.size\": 11,\n    \"axes.spines.top\": False,\n    \"axes.spines.right\": False,\n    \"axes.grid\": True,\n    \"axes.grid.axis\": \"y\",\n    \"grid.color\": \"#CCCCCC\",\n    \"grid.alpha\": 0.6,\n    \"axes.axisbelow\": True,\n    \"axes.prop_cycle\": plt.cycler(color=list(COLOURS.values())),\n    \"image.cmap\": \"cividis\",\n    \"legend.frameon\": False,\n    \"figure.constrained_layout.use\": True,\n    \"figure.dpi\": 120,\n})\n'''\n\ninstall_cell = '''import subprocess, sys\ndef _pip(*a): subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *a])\n\n# Not pre-installed on Colab: always install\n_pip('loguru==0.8.0')\n\n# Core packages (pre-installed on Colab, install locally to match Colab env)\nif 'google.colab' not in sys.modules:\n    _pip('numpy==2.0.2', 'scipy==1.16.3', 'matplotlib==3.10.0', 'torch==2.9.0')\n'''\n\ndata_cell = f'''GITHUB_DATA_URL = \"{URL}\"\nimport http.client\nimport json\nimport urllib.request\nfrom pathlib import Path\n\ndef load_data():\n    try:\n        with urllib.request.urlopen(GITHUB_DATA_URL, timeout=15) as response:\n            return json.loads(response.read().decode())\n    # Network errors, timeouts, a placeholder URL, bad JSON, a cut-off response.\n    except (OSError, ValueError, http.client.HTTPException) as e:\n        print(\"GitHub data unavailable, using local mini_demo_data.json:\", e)\n    local = Path(\"mini_demo_data.json\")\n    if local.exists(): return json.loads(local.read_text())\n    raise FileNotFoundError(\"Could not load mini_demo_data.json\")\n\ndata = load_data()\nprint({{k: len(v[\"y\"]) for k, v in data.items() if k != \"metadata\"}}, \"examples per split\")\n'''\n\nconfig_cell = '''DEVICE = torch.device(\"mps\" if torch.backends.mps.is_available() else \"cpu\")\nBATCH = 128\nMOMENTUM = 0.9\nWEIGHT_DECAY = 5e-4\nEPOCHS = 2            # original: 40 (paper setting); smallest demo value: 2\nSEEDS = (0,)          # original: (0, 1, 2, 3, 4)\nLR_SGD = 0.01         # selected on validation NLL (seed 100) in the original tuning sweep\nLR_SAM = 0.01         # selected on validation NLL (seed 100) in the original tuning sweep\nRHO = 0.1             # SAM neighbourhood radius, selected on validation NLL\nRESULTS = Path(\"results\")\nRUNS = RESULTS / \"runs\"\nLOGS = RESULTS / \"logs\"\nCKPTS = RESULTS / \"checkpoints\"\nLOGS.mkdir(parents=True, exist_ok=True)\n'''\n\nprepare_cell = '''def prepare_data(raw: dict) -> dict:\n    \"\"\"Turn the JSON lists into tensors, normalised with train-split statistics.\n\n    Replaces the original load_data(), which downloaded Fashion-MNIST: the subset\n    is already split (train/val/test) and stored as integer pixels 0-255.\n    \"\"\"\n    def to_tensors(split: str):\n        x = torch.tensor(raw[split][\"x\"], dtype=torch.float32).div(255.0).view(-1, 1, 28, 28)\n        y = torch.tensor(raw[split][\"y\"], dtype=torch.long)\n        return x, y\n\n    x_tr, y_tr = to_tensors(\"train\")\n    x_va, y_va = to_tensors(\"val\")\n    x_te, y_te = to_tensors(\"test\")\n    mean = float(x_tr.mean())\n    std = float(x_tr.std())\n    print(f\"Train-split normalisation: mean={mean:.4f} std={std:.4f}\")\n\n    def norm(x: torch.Tensor) -> torch.Tensor:\n        return ((x - mean) / std).contiguous()\n\n    return {\n        \"x_train\": norm(x_tr).to(DEVICE), \"y_train\": y_tr.to(DEVICE),\n        \"x_val\": norm(x_va).to(DEVICE), \"y_val\": y_va.to(DEVICE),\n        \"x_test\": norm(x_te).to(DEVICE), \"y_test\": y_te.to(DEVICE),\n    }\n\n\ndata = prepare_data(data)\nprint({k: tuple(v.shape) for k, v in data.items()})\n'''\n\nrun_name_cell = '''def run_name(condition: str, lr: float, rho: float | None, seed: int, epochs: int) -> str:\n    rho_part = \"\" if rho is None else f\"_rho{rho}\"\n    return f\"{condition}_lr{lr}{rho_part}_E{epochs}_s{seed}\"\n'''\n\nfinal_cell = '''def run_final(data: dict) -> None:\n    \"\"\"Train SGD and SAM for each seed, then fit temperature scaling on validation logits.\n\n    Mirrors mode_final in method.py. The learning rates and rho come from the config\n    cell instead of chosen_hparams.json, and only the test logits of each run are\n    saved as files.\n    \"\"\"\n    val_y = data[\"y_val\"].cpu().numpy()\n    for seed in SEEDS:\n        for cond, lr, r in ((\"sgd\", LR_SGD, None), (\"sam\", LR_SAM, RHO)):\n            name = run_name(cond, lr, r, seed, EPOCHS)\n            out = RUNS / f\"final_{name}\"\n            s = run_or_load(data=data, condition=cond, lr=lr, rho=r, seed=seed,\n                            epochs=EPOCHS, save_epochs=(EPOCHS,), out_dir=out,\n                            include_test=True, checkpoint=False, name=f\"final_{name}\")\n            z = np.load(out / f\"logits_ep{EPOCHS}.npz\")\n            temp = fit_temperature(z[\"val_logits\"], val_y)\n            T = temp[\"temperature\"]\n            np.savez(out / \"ts_test_logits.npz\",\n                     test_logits=(z[\"test_logits\"].astype(np.float64) / T).astype(np.float32))\n            ts = {\"seed\": seed, \"condition\": cond, **temp}\n            (out / \"temperature.json\").write_text(json.dumps(ts, indent=2))\n            logger.info(f\"final {name}: T={T:.4f} at_bound={temp['at_bound']} \"\n                        f\"final_train_loss={s['final_train_loss_eval']:.5f} \"\n                        f\"test_acc@{EPOCHS}={s['saved'][str(EPOCHS)]['test_acc']:.4f}\")\n\n\nrun_final(data)\n'''\n\nresults_cell = '''test_y = data[\"y_test\"].cpu().numpy()\nMETRICS = [\"accuracy\", \"ece\", \"ace\", \"nll\", \"brier\", \"aurc\"]\nper_seed = []\nfor seed in SEEDS:\n    sgd_dir = RUNS / f\"final_{run_name('sgd', LR_SGD, None, seed, EPOCHS)}\"\n    sam_dir = RUNS / f\"final_{run_name('sam', LR_SAM, RHO, seed, EPOCHS)}\"\n    z_sgd = np.load(sgd_dir / f\"logits_ep{EPOCHS}.npz\")[\"test_logits\"]\n    z_sam = np.load(sam_dir / f\"logits_ep{EPOCHS}.npz\")[\"test_logits\"]\n    T = json.loads((sgd_dir / \"temperature.json\").read_text())[\"temperature\"]\n    per_seed.append({\n        \"seed\": seed,\n        \"logits\": {\"SGD\": z_sgd, \"SAM\": z_sam},\n        \"SGD\": all_metrics(z_sgd, test_y),\n        \"SAM\": all_metrics(z_sam, test_y),\n        \"SGD+TS\": all_metrics(z_sgd, test_y, temperature=T),\n    })\n\n# Paired SAM-minus-SGD differences over seeds, t-interval and Holm correction over metrics.\ndiffs = {m: paired_t_interval(np.array([r[\"SAM\"][m] - r[\"SGD\"][m] for r in per_seed]))\n         for m in METRICS}\nholm_p = holm({m: diffs[m][\"p\"] for m in METRICS})\n\nprint(f\"{'metric':<10}{'SGD':>9}{'SAM':>9}{'SGD+TS':>9}{'SAM-SGD':>10}{'95% CI':>24}{'Holm p':>9}\")\nfor m in METRICS:\n    mean = {k: np.mean([r[k][m] for r in per_seed]) for k in (\"SGD\", \"SAM\", \"SGD+TS\")}\n    ci = diffs[m][\"ci95\"]\n    print(f\"{m:<10}{mean['SGD']:>9.4f}{mean['SAM']:>9.4f}{mean['SGD+TS']:>9.4f}\"\n          f\"{diffs[m]['mean']:>+10.4f}  [{ci[0]:+.4f}, {ci[1]:+.4f}]{holm_p[m]:>9.4f}\")\n'''\n\nviz_cell = '''labels_m = [\"SGD\", \"SAM\", \"SGD+TS\"]\ncolours = [COLOURS[\"baseline\"], COLOURS[\"method\"], COLOURS[\"other_1\"]]\nfig, axes = plt.subplots(1, 3, figsize=(12, 3.8))\nfor ax, m in zip(axes[:2], [\"ece\", \"nll\"]):\n    vals = [np.mean([r[k][m] for r in per_seed]) for k in labels_m]\n    ax.bar(labels_m, vals, color=colours)\n    ax.set_title(f\"Test {m.upper()} (mean over seeds)\")\n\n# Reliability diagram on the first seed: accuracy against mean confidence per bin.\nax = axes[2]\nfirst = per_seed[0][\"logits\"]\nfor name, key in ((\"SGD\", \"SGD\"), (\"SAM\", \"SAM\")):\n    conf, correct, _ = confidence_and_correct(first[key], test_y)\n    rb = reliability_bins(conf, correct)\n    keep = [i for i, c in enumerate(rb[\"count\"]) if c > 0]\n    ax.plot([rb[\"mean_confidence\"][i] for i in keep], [rb[\"accuracy\"][i] for i in keep],\n            \"o-\", color=COLOURS[\"method\"] if key == \"SAM\" else COLOURS[\"baseline\"], label=name)\nax.plot([0, 1], [0, 1], \"--\", color=\"grey\", label=\"perfect calibration\")\nax.set_xlabel(\"Mean confidence\")\nax.set_ylabel(\"Accuracy\")\nax.set_title(\"Reliability (seed 0)\")\nax.legend()\nplt.show()\n'''\n\nprocessing_md = {\n    \"model\": \"## Model and training step\\nThe CNN and the two optimiser steps, copied from `method.py`: `sgd_step` is plain SGD; `sam_step` perturbs the weights by `rho` along the normalised gradient, takes a second gradient there with BatchNorm statistics frozen, and then applies the base SGD step to the perturbed-point gradient.\",\n    \"train\": \"## Training runs\\nThe original `train_run` with two changes: the per-run file log sink is removed, and the checkpoint path is unused in this notebook. Each run saves logits on the validation and test sets at the final epoch.\",\n    \"final\": \"## Final runs with temperature scaling\\nFor each seed, train SGD and SAM. A single temperature is fitted on the validation logits of the SGD model by minimising validation NLL (the SGD+TS control). The fitted temperature is applied to the SGD test logits.\",\n    \"results\": \"## Test-set results\\nAccuracy, ECE (15 equal-width bins), ACE, NLL, Brier and AURC for SGD, SAM and temperature-scaled SGD. The SAM-minus-SGD differences use paired t-intervals over seeds with df = n-1 (df = 0 with one seed, so the interval is undefined: use more seeds for a CI).\",\n    \"viz\": \"## Plots\\nMean ECE and NLL per method, and a reliability diagram for the first seed.\",\n}\n\ncells = [\n    cell(\"markdown\", \"# Does sharpness-aware minimisation (SAM) improve calibration over SGD on Fashion-MNIST?\\n\\n\"\n         \"This notebook demonstrates the artifact `method.py`. It trains a small two-block CNN on Fashion-MNIST with \"\n         \"plain SGD (momentum 0.9) and with SAM on the same base optimiser, using the same architecture, data split, \"\n         \"batch order and per-seed initialisation. It then compares accuracy, ECE, NLL, Brier and AURC on the test set, \"\n         \"and checks whether SAM's calibration advantage survives temperature scaling of the SGD model.\\n\\n\"\n         \"**Demo scope.** The notebook runs on a 100-image, class-balanced subset of Fashion-MNIST \"\n         \"(`mini_demo_data.json`: 60 train, 20 validation, 20 test) so that it finishes in minutes. The numbers it prints \"\n         \"are therefore not the paper's results: the paper trains on 50,000 images and evaluates on the official 10,000-image \"\n         \"test set. The config cell sets the scale.\\n\\n\"\n         \"**Selected hyperparameters.** The learning rates (SGD 0.01, SAM 0.01) and SAM's rho (0.1) are the values the original \"\n         \"pipeline selected on validation NLL with seed 100. The demo uses them directly and skips the tuning grid.\"),\n    cell(\"markdown\", \"## Install dependencies\\nInstall the packages the notebook imports. On Colab the core scientific \"\n         \"packages are already installed and must not be reinstalled, so they are only installed locally.\"),\n    cell(\"code\", install_cell),\n    cell(\"markdown\", \"## Imports\\nThe original import block (without the torchvision and `metrics` imports, which are replaced \"\n         \"by the JSON loader and the in-notebook metrics below), followed by the shared figure style.\"),\n    cell(\"code\", imports_cell),\n    cell(\"markdown\", \"## Metric functions\\nThe calibration and accuracy metrics from `metrics.py`, the module the original script \"\n         \"imports as `M`. They are defined here so the notebook needs no extra file; calls use the bare names instead of `M.`.\"),\n    cell(\"code\", metrics_code),\n    cell(\"markdown\", \"## Load the data\\n`data` is the curated 100-image subset. `load_data()` reads it from GitHub and falls \"\n         \"back to the local `mini_demo_data.json`.\"),\n    cell(\"code\", data_cell),\n    cell(\"markdown\", \"## Config\\nAll tunable values. The first run uses the smallest values (2 epochs, one seed); they are \"\n         \"increased in later rounds, see the scaling note at the end.\"),\n    cell(\"code\", config_cell),\n    cell(\"markdown\", \"## Prepare the data\\nConvert the lists to tensors and normalise with the training-split mean and standard \"\n         \"deviation, as the original `load_data()` did.\"),\n    cell(\"code\", prepare_cell),\n    cell(\"markdown\", processing_md[\"model\"]),\n    cell(\"code\", model_block),\n    cell(\"markdown\", processing_md[\"train\"]),\n    cell(\"code\", run_name_cell + \"\\n\" + train_block),\n    cell(\"markdown\", processing_md[\"final\"]),\n    cell(\"code\", final_cell),\n    cell(\"markdown\", processing_md[\"results\"]),\n    cell(\"code\", results_cell),\n    cell(\"markdown\", processing_md[\"viz\"]),\n    cell(\"code\", viz_cell),\n]\n\nnb = {\n    \"cells\": cells,\n    \"metadata\": {\n        \"kernelspec\": {\"display_name\": \"Python 3\", \"language\": \"python\", \"name\": \"python3\"},\n        \"language_info\": {\"name\": \"python\", \"version\": \"3.12\"},\n    },\n    \"nbformat\": 4,\n    \"nbformat_minor\": 4,\n}\n(WS / \"code_demo.ipynb\").write_text(json.dumps(nb, indent=1))\nprint(\"wrote\", WS / \"code_demo.ipynb\", len(cells), \"cells\")\n", "structuredPatch": [], "originalFile": null, "userModified": false}
```

### [29] TOOL CALL — Bash · 2026-11-25 09:29:53 UTC

```
Build notebook and prepare the clean test venv:
python3 scratch/make_notebook.py && python3 -c "import json;nb=json.load(open('code_demo.ipynb'));print([c['cell_type'] for c in nb['cells']]);print(sum(1 for c in nb['cells'] if 'GITHUB_DATA_URL = \"https://raw.githubusercontent.com/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small/main/round-1/experiment-1/demo/mini_demo_data.json\"' in ''.join(c['source'])))" && UV_VENV_CLEAR=1 UV_CACHE_DIR=/Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache/uv uv venv -q --seed --python 3.12 /Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache/nb_test_envs/art_SL9gPfE09HlN-13a6bcbdff23 && UV_CACHE_DIR=/Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache/uv uv pip install -q --python /Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache/nb_test_envs/art_SL9gPfE09HlN-13a6bcbdff23/bin/python jupyter ipykernel && echo ENV_OK
```

### [30] TOOL RESULT — Bash [ERROR] · 2026-11-25 09:29:53 UTC

```
Error: Exit code 1
Traceback (most recent call last):
  File "/Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/4_gen_paper_repo/_4_gen_demo_art/notebook_workspaces/iter_1/art_SL9gPfE09HlN/scratch/make_notebook.py", line 40, in <module>
    assert "M." not in model_block and "M." not in train_block
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
```

### [31] ASSISTANT · 2026-11-25 09:29:57 UTC

```
The builder's assert ran before the `M.nll_per_example` replacement, so it tripped on the code it was about to fix. Moving the check after the replacement:
```

### [32] TOOL CALL — Edit · 2026-11-25 09:29:57 UTC

```
File: /Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/4_gen_paper_repo/_4_gen_demo_art/notebook_workspaces/iter_1/art_SL9gPfE09HlN/scratch/make_notebook.py
OLD: assert "M." not in model_block and "M." not in train_block

# metrics.py is imported as `M` in method.py; here its functions are defined in-cell.
metrics_code = metrics_src[metrics_src.index("from __future__"):]
assert "import metrics" not in method
method_refs_M = ["M.nll_per_example"]
train_block = train_block.replace("M.nll_per_example", "nll_per_example")
assert "M." not in train_block
NEW: # metrics.py is imported as `M` in method.py; here its functions are defined in-cell.
metrics_code = metrics_src[metrics_src.index("from __future__"):]
assert "import metrics" in method and "M." in train_block
train_block = train_block.replace("M.nll_per_example", "nll_per_example")
assert "M." not in train_block and "M." not in model_block
```

### [33] TOOL RESULT — Edit · 2026-11-25 09:29:57 UTC

```
{"filePath": "/Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/4_gen_paper_repo/_4_gen_demo_art/notebook_workspaces/iter_1/art_SL9gPfE09HlN/scratch/make_notebook.py", "oldString": "assert \"M.\" not in model_block and \"M.\" not in train_block\n\n# metrics.py is imported as `M` in method.py; here its functions are defined in-cell.\nmetrics_code = metrics_src[metrics_src.index(\"from __future__\"):]\nassert \"import metrics\" not in method\nmethod_refs_M = [\"M.nll_per_example\"]\ntrain_block = train_block.replace(\"M.nll_per_example\", \"nll_per_example\")\nassert \"M.\" not in train_block", "newString": "# metrics.py is imported as `M` in method.py; here its functions are defined in-cell.\nmetrics_code = metrics_src[metrics_src.index(\"from __future__\"):]\nassert \"import metrics\" in method and \"M.\" in train_block\ntrain_block = train_block.replace(\"M.nll_per_example\", \"nll_per_example\")\nassert \"M.\" not in train_block and \"M.\" not in model_block", "originalFile": null, "structuredPatch": [{"oldStart": 37, "oldLines": 14, "newStart": 37, "newLines": 11, "lines": [" assert train_block.count(sink_line) == 1 and train_block.count(finally_block) == 1", " train_block = train_block.replace(sink_line, \"\").replace(finally_block, \"\")", " # Edit 2: the notebook has no separate training-label file; labels come from `data`.", "-assert \"M.\" not in model_block and \"M.\" not in train_block", "-", " # metrics.py is imported as `M` in method.py; here its functions are defined in-cell.", " metrics_code = metrics_src[metrics_src.index(\"from __future__\"):]", "-assert \"import metrics\" not in method", "-method_refs_M = [\"M.nll_per_example\"]", "+assert \"import metrics\" in method and \"M.\" in train_block", " train_block = train_block.replace(\"M.nll_per_example\", \"nll_per_example\")", "-assert \"M.\" not in train_block", "+assert \"M.\" not in train_block and \"M.\" not in model_block", " ", " ", " def cell(kind: str, text: str) -> dict:"]}], "userModified": false, "replaceAll": false}
```

### [34] TOOL CALL — Bash · 2026-11-25 09:30:01 UTC

```
Build notebook and prepare the clean test venv:
python3 scratch/make_notebook.py && python3 -c "import json;nb=json.load(open('code_demo.ipynb'));print(len(nb['cells']));print(sum(1 for c in nb['cells'] if 'GITHUB_DATA_URL = \"https://raw.githubusercontent.com/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small/main/round-1/experiment-1/demo/mini_demo_data.json\"' in ''.join(c['source'])))" && UV_VENV_CLEAR=1 UV_CACHE_DIR=/Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache/uv uv venv -q --seed --python 3.12 /Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache/nb_test_envs/art_SL9gPfE09HlN-13a6bcbdff23 && UV_CACHE_DIR=/Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache/uv uv pip install -q --python /Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache/nb_test_envs/art_SL9gPfE09HlN-13a6bcbdff23/bin/python jupyter ipykernel && echo ENV_OK
```

### [35] TOOL RESULT — Bash · 2026-11-25 09:30:05 UTC

```
{"stdout": "wrote /Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/4_gen_paper_repo/_4_gen_demo_art/notebook_workspaces/iter_1/art_SL9gPfE09HlN/code_demo.ipynb 23 cells\n23\n1\nENV_OK", "stderr": "", "interrupted": false, "isImage": false, "noOutputExpected": false}
```

### [36] ASSISTANT · 2026-11-25 09:30:09 UTC

```
The notebook is built (23 cells, GitHub URL present once) and the clean test venv is ready. Running the notebook end-to-end in the background, since the install cell pulls torch:
```

### [37] TOOL CALL — Bash · 2026-11-25 09:30:09 UTC

```
Execute the demo notebook in the clean test venv:
time timeout 600 /Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache/nb_test_envs/art_SL9gPfE09HlN-13a6bcbdff23/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 code_demo.ipynb --output code_demo.ipynb 2>&1 | tail -60
```

### [38] TOOL RESULT — Bash · 2026-11-25 09:30:09 UTC

```
{"stdout": "", "stderr": "", "interrupted": false, "isImage": false, "noOutputExpected": false, "backgroundTaskId": "bm8w28dz9"}
```

### [39] TOOL CALL — ScheduleWakeup · 2026-11-25 09:30:13 UTC

```
{'delaySeconds': 240, 'reason': 'waiting for notebook execution (torch install plus run)', 'noop': True, 'prompt': '<<autonomous-loop-dynamic>>'}
```

### [40] TOOL RESULT — ScheduleWakeup · 2026-11-25 09:30:13 UTC

```
{"scheduledFor": 1795599240000, "clampedDelaySeconds": 240, "wasClamped": false}
```

### [41] TOOL CALL — ToolSearch · 2026-11-25 09:30:15 UTC

```
{'query': 'select:Monitor', 'max_results': 1}
```

### [42] TOOL RESULT — ToolSearch · 2026-11-25 09:30:15 UTC

```
{"matches": ["Monitor"], "query": "select:Monitor", "total_deferred_tools": 39}
```

### [43] ASSISTANT · 2026-11-25 09:30:17 UTC

```
The notebook run is still going in the background. It has to install torch first, so it will take a few minutes. I'll pick up the result when the run completes.
```

### [44] SYSTEM-USER prompt · 2026-11-25 09:30:17 UTC

```
<task-notification>
<task-id>bm8w28dz9</task-id>
<tool-use-id>toolu_01VdnRMQ9Wq3ivUpSdCsvxTY</tool-use-id>
<output-file>/private/tmp/claude-20002/-Volumes-aii-s20002-work-run-S-3Y-l0tX7Xe-4-gen-paper-repo--4-gen-demo-art-notebook-workspaces-iter-1-art-SL9gPfE09HlN/53a788d1-9416-4145-8022-5d7546a34ca5/tasks/bm8w28dz9.output</output-file>
<status>completed</status>
<summary>Background command "Execute the demo notebook in the clean test venv" completed (exit code 0)</summary>
</task-notification>
```

### [45] TOOL CALL — Bash · 2026-11-25 09:30:21 UTC

```
Check notebook run status and printed outputs:
tail -c 1500 /private/tmp/claude-20002/-Volumes-aii-s20002-work-run-S-3Y-l0tX7Xe-4-gen-paper-repo--4-gen-demo-art-notebook-workspaces-iter-1-art-SL9gPfE09HlN/53a788d1-9416-4145-8022-5d7546a34ca5/tasks/bm8w28dz9.output; echo ---; python3 -c "
import json
nb=json.load(open('code_demo.ipynb'))
for i,c in enumerate(nb['cells']):
    if c['cell_type']!='code': continue
    for o in c.get('outputs',[]):
        if o.get('output_type')=='error': print('ERR cell',i,o['ename'],o['evalue'])
        elif 'text' in o: print(f'[cell {i}]', ''.join(o['text'])[-900:])
        elif 'data' in o and 'text/plain' in o['data']: print(f'[cell {i} data]', ''.join(o['data']['text/plain'])[:200])
"
```

### [46] TOOL RESULT — Bash · 2026-11-25 09:30:21 UTC

```
{"stdout": "33m'loguru==0.8.0'\u001b[39m)\n\u001b[32m      6\u001b[39m \n\u001b[32m      7\u001b[39m \u001b[38;5;66;03m# Core packages (pre-installed on Colab, install locally to match Colab env)\u001b[39;00m\n\u001b[32m      8\u001b[39m \u001b[38;5;28;01mif\u001b[39;00m \u001b[33m'google.colab'\u001b[39m \u001b[38;5;28;01mnot\u001b[39;00m \u001b[38;5;28;01min\u001b[39;00m sys.modules:\n\u001b[32m----> \u001b[39m\u001b[32m9\u001b[39m     _pip(\u001b[33m'numpy==2.0.2'\u001b[39m, \u001b[33m'scipy==1.16.3'\u001b[39m, \u001b[33m'matplotlib==3.10.0'\u001b[39m, \u001b[33m'torch==2.9.0'\u001b[39m)\n\n\u001b[36mCell\u001b[39m\u001b[36m \u001b[39m\u001b[32mIn[1]\u001b[39m\u001b[32m, line 2\u001b[39m, in \u001b[36m_pip\u001b[39m\u001b[34m(*a)\u001b[39m\n\u001b[32m----> \u001b[39m\u001b[32m2\u001b[39m \u001b[38;5;28;01mdef\u001b[39;00m _pip(*a): subprocess.check_call([sys.executable, \u001b[33m'-m'\u001b[39m, \u001b[33m'pip'\u001b[39m, \u001b[33m'install'\u001b[39m, \u001b[33m'-q'\u001b[39m, *a])\n\n\u001b[36mFile \u001b[39m\u001b[32m/opt/aii/python/lib/python3.12/subprocess.py:413\u001b[39m, in \u001b[36mcheck_call\u001b[39m\u001b[34m(*popenargs, **kwargs)\u001b[39m\n\u001b[32m    411\u001b[39m     \u001b[38;5;28;01mif\u001b[39;00m cmd \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n\u001b[32m    412\u001b[39m         cmd = popenargs[\u001b[32m0\u001b[39m]\n\u001b[32m--> \u001b[39m\u001b[32m413\u001b[39m     \u001b[38;5;28;01mraise\u001b[39;00m CalledProcessError(retcode, cmd)\n\u001b[32m    414\u001b[39m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[32m0\u001b[39m\n\n\u001b[31mCalledProcessError\u001b[39m: Command '['/Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache/nb_test_envs/art_SL9gPfE09HlN-13a6bcbdff23/bin/python', '-m', 'pip', 'install', '-q', 'numpy==2.0.2', 'scipy==1.16.3', 'matplotlib==3.10.0', 'torch==2.9.0']' returned non-zero exit status 1.\n\n\nreal\t0m7.741s\nuser\t0m2.949s\nsys\t0m0.525s\n\n[exited with code 0]\n---", "stderr": "", "interrupted": false, "isImage": false, "noOutputExpected": false}
```

### [47] ASSISTANT · 2026-11-25 09:30:25 UTC

```
The install cell failed, so the notebook never reached the code. Next I'll find which pinned package can't be installed in the test venv.
```

### [48] TOOL CALL — Bash · 2026-11-25 09:30:25 UTC

```
Dry-run the pinned installs to find the failing package:
cd /tmp && UV_CACHE_DIR=/Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache/uv uv pip install --dry-run --python /Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache/nb_test_envs/art_SL9gPfE09HlN-13a6bcbdff23/bin/python numpy==2.0.2 scipy==1.16.3 matplotlib==3.10.0 2>&1 | tail -3; UV_CACHE_DIR=/Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache/uv uv pip install --dry-run --python /Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache/nb_test_envs/art_SL9gPfE09HlN-13a6bcbdff23/bin/python torch==2.9.0 2>&1 | tail -3; /Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache/nb_test_envs/art_SL9gPfE09HlN-13a6bcbdff23/bin/python -m pip install --dry-run -q torch==2.9.0 2>&1 | tail -3; echo EXIT $?
```

### [49] TOOL RESULT — Bash · 2026-11-25 09:30:25 UTC

```
{"stdout": " + pillow==13.0.0\n + pyparsing==3.3.3\n + scipy==1.16.3\n  × No solution found when resolving dependencies:\n  ╰─▶ Because you require torch==2.9.0 and torch==2.16.0.dev20261116+aii1, we\n      can conclude that your requirements are unsatisfiable.\nERROR: Cannot install torch==2.9.0 because these package versions have conflicting dependencies.\nERROR: ResolutionImpossible: for help visit https://pip.pypa.io/en/latest/topics/dependency-resolution/#dealing-with-dependency-conflicts\nEXIT 0", "stderr": "\nShell cwd was reset to /Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/4_gen_paper_repo/_4_gen_demo_art/notebook_workspaces/iter_1/art_SL9gPfE09HlN", "interrupted": false, "isImage": false, "noOutputExpected": false}
```

### [50] TOOL CALL — Bash · 2026-11-25 09:30:29 UTC

```
Inspect pip config and available torch versions:
env | grep -i -E "pip|uv_" ; cat /etc/pip.conf ~/.pip/pip.conf ~/.config/pip/pip.conf 2>/dev/null; /Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache/nb_test_envs/art_SL9gPfE09HlN-13a6bcbdff23/bin/python -m pip config list 2>/dev/null; cd /tmp && /Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache/nb_test_envs/art_SL9gPfE09HlN-13a6bcbdff23/bin/python -m pip index versions torch 2>&1 | head -3
```

### [51] TOOL RESULT — Bash · 2026-11-25 09:30:29 UTC

```
{"stdout": "UV_CACHE_DIR=/Volumes/aii-s20002/cache/uv\nPIP_CONSTRAINT=/opt/aii/etc/torch.constraints.txt\nPIP_CACHE_DIR=/Volumes/aii-s20002/cache/pip\nUV_CONSTRAINT=/opt/aii/etc/torch.constraints.txt\nPIP_FIND_LINKS=/opt/aii/wheels\nUV_FIND_LINKS=/opt/aii/wheels\n:env:.cache-dir='/Volumes/aii-s20002/cache/pip'\n:env:.constraint='/opt/aii/etc/torch.constraints.txt'\n:env:.find-links='/opt/aii/wheels'\ntorch (2.15.0)\nAvailable versions: 2.15.0, 2.14.1, 2.14.0, 2.13.0, 2.12.1, 2.12.0, 2.11.0, 2.10.0, 2.9.1, 2.9.0, 2.8.0, 2.7.1, 2.7.0, 2.6.0, 2.5.1, 2.5.0, 2.4.1, 2.4.0, 2.3.1, 2.3.0, 2.2.2, 2.2.1, 2.2.0", "stderr": "\nShell cwd was reset to /Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/4_gen_paper_repo/_4_gen_demo_art/notebook_workspaces/iter_1/art_SL9gPfE09HlN", "interrupted": false, "isImage": false, "noOutputExpected": false}
```

### [52] ASSISTANT · 2026-11-25 09:30:33 UTC

```
A machine-wide pip constraint (`/opt/aii/etc/torch.constraints.txt`) pins torch to a 2.16 dev build, which conflicts with the Colab torch pin (2.9.0) in the install cell. I'll look at the constraints file before choosing a fix.
```

### [53] TOOL CALL — Bash · 2026-11-25 09:30:33 UTC

```
Show the torch constraints file:
cat /opt/aii/etc/torch.constraints.txt
```

### [54] TOOL RESULT — Bash · 2026-11-25 09:30:33 UTC

```
{"stdout": "# /opt/aii/etc/torch.constraints.txt (Mac guest image v4; root:wheel 644).\n# Exception E5's stand-in for a pod's CUDA torch: UV_CONSTRAINT and\n# PIP_CONSTRAINT name this file, UV_FIND_LINKS and PIP_FIND_LINKS name\n# /opt/aii/wheels (root-owned, read-only), so `uv pip install torch` in an\n# agent's own venv takes the patched MPS wheel (the nightly with the MPS\n# F.linear fix built in, local version +aii1) and torchvision its matching\n# nightly. Nothing else is pinned here.\ntorch==2.16.0.dev20261116+aii1\ntorchvision==0.30.0.dev20261116", "stderr": "", "interrupted": false, "isImage": false, "noOutputExpected": false}
```

### [55] ASSISTANT · 2026-11-25 09:30:37 UTC

```
The constraint is the machine's intended torch build (an MPS-patched wheel), and it's read-only, so I won't touch it. The fix belongs in the notebook: the local install won't pin torch, and the environment's own build is used. Colab still keeps its preinstalled torch because the guard skips installs there.
```

### [56] TOOL CALL — Edit · 2026-11-25 09:30:37 UTC

```
File: /Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/4_gen_paper_repo/_4_gen_demo_art/notebook_workspaces/iter_1/art_SL9gPfE09HlN/scratch/make_notebook.py
OLD:     _pip('numpy==2.0.2', 'scipy==1.16.3', 'matplotlib==3.10.0', 'torch==2.9.0')
'''
NEW:     # torch is not pinned here: this machine's pip constraint file fixes its own torch build.
    _pip('numpy==2.0.2', 'scipy==1.16.3', 'matplotlib==3.10.0', 'torch')
'''
```

### [57] TOOL RESULT — Edit · 2026-11-25 09:30:37 UTC

```
{"filePath": "/Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/4_gen_paper_repo/_4_gen_demo_art/notebook_workspaces/iter_1/art_SL9gPfE09HlN/scratch/make_notebook.py", "oldString": "    _pip('numpy==2.0.2', 'scipy==1.16.3', 'matplotlib==3.10.0', 'torch==2.9.0')\n'''", "newString": "    # torch is not pinned here: this machine's pip constraint file fixes its own torch build.\n    _pip('numpy==2.0.2', 'scipy==1.16.3', 'matplotlib==3.10.0', 'torch')\n'''", "originalFile": null, "structuredPatch": [{"oldStart": 106, "oldLines": 7, "newStart": 106, "newLines": 8, "lines": [" ", " # Core packages (pre-installed on Colab, install locally to match Colab env)", " if 'google.colab' not in sys.modules:", "-    _pip('numpy==2.0.2', 'scipy==1.16.3', 'matplotlib==3.10.0', 'torch==2.9.0')", "+    # torch is not pinned here: this machine's pip constraint file fixes its own torch build.", "+    _pip('numpy==2.0.2', 'scipy==1.16.3', 'matplotlib==3.10.0', 'torch')", " '''", " ", " data_cell = f'''GITHUB_DATA_URL = \"{URL}\""]}], "userModified": false, "replaceAll": false}
```

### [58] ASSISTANT · 2026-11-25 09:30:41 UTC

```
I've rebuilt the notebook with the fix. Re-running the full clean-venv test in the background:
```

### [59] TOOL CALL — Bash · 2026-11-25 09:30:41 UTC

```
Rebuild notebook and re-run the clean-venv test:
python3 scratch/make_notebook.py && UV_VENV_CLEAR=1 UV_CACHE_DIR=/Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache/uv uv venv -q --seed --python 3.12 /Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache/nb_test_envs/art_SL9gPfE09HlN-13a6bcbdff23 && UV_CACHE_DIR=/Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache/uv uv pip install -q --python /Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache/nb_test_envs/art_SL9gPfE09HlN-13a6bcbdff23/bin/python jupyter ipykernel && time timeout 600 /Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache/nb_test_envs/art_SL9gPfE09HlN-13a6bcbdff23/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 code_demo.ipynb --output code_demo.ipynb 2>&1 | tail -5
```

### [60] TOOL RESULT — Bash · 2026-11-25 09:30:41 UTC

```
{"stdout": "", "stderr": "", "interrupted": false, "isImage": false, "noOutputExpected": false, "backgroundTaskId": "bzqhsejeg"}
```

### [61] ASSISTANT · 2026-11-25 09:30:43 UTC

```
The test run is going again in the background. I'll wait for its completion notice before checking the executed cells.
```

### [62] SYSTEM-USER prompt · 2026-11-25 09:31:57 UTC

```
<task-notification>
<task-id>bzqhsejeg</task-id>
<tool-use-id>toolu_01TBMPZFnz6dCjcPkULwCkMD</tool-use-id>
<output-file>/private/tmp/claude-20002/-Volumes-aii-s20002-work-run-S-3Y-l0tX7Xe-4-gen-paper-repo--4-gen-demo-art-notebook-workspaces-iter-1-art-SL9gPfE09HlN/53a788d1-9416-4145-8022-5d7546a34ca5/tasks/bzqhsejeg.output</output-file>
<status>completed</status>
<summary>Background command "Rebuild notebook and re-run the clean-venv test" completed (exit code 0)</summary>
</task-notification>
```

### [63] TOOL CALL — Bash · 2026-11-25 09:31:59 UTC

```
Check the re-run status and cell outputs:
tail -c 600 /private/tmp/claude-20002/-Volumes-aii-s20002-work-run-S-3Y-l0tX7Xe-4-gen-paper-repo--4-gen-demo-art-notebook-workspaces-iter-1-art-SL9gPfE09HlN/53a788d1-9416-4145-8022-5d7546a34ca5/tasks/bzqhsejeg.output; echo ---; python3 -c "
import json
nb=json.load(open('code_demo.ipynb'))
for i,c in enumerate(nb['cells']):
    if c['cell_type']!='code': continue
    for o in c.get('outputs',[]):
        if o.get('output_type')=='error': print('ERR cell',i,o['ename'],o['evalue'])
        elif 'text' in o: print(f'[cell {i}]', ''.join(o['text'])[-1200:])
"
```

### [64] TOOL RESULT — Bash · 2026-11-25 09:31:59 UTC

```
{"stdout": "i-s20002/work/run_S_3Y_l0tX7Xe/4_gen_paper_repo/_4_gen_demo_art/notebook_workspaces/iter_1/art_SL9gPfE09HlN/code_demo.ipynb 23 cells\n[NbConvertApp] Converting notebook code_demo.ipynb to notebook\n[IPKernelApp] WARNING | Kernel is running over TCP without encryption. All communication (including code and outputs) is sent in plain text and is susceptible to eavesdropping. Use IPC transport or launch with kernel manager-provisioned CurveZMQ keys to enable transport encryption.\n[NbConvertApp] Writing 109689 bytes to code_demo.ipynb\n\nreal\t1m13.339s\nuser\t0m30.483s\nsys\t0m5.821s\n\n[exited with code 0]\n---\n[cell 4] Matplotlib is building the font cache; this may take a moment.\n\n[cell 8] GitHub data unavailable, using local mini_demo_data.json: HTTP Error 404: Not Found\n{'train': 60, 'val': 20, 'test': 20} examples per split\n\n[cell 12] Train-split normalisation: mean=0.3066 std=0.3641\n{'x_train': (60, 1, 28, 28), 'y_train': (60,), 'x_val': (20, 1, 28, 28), 'y_val': (20,), 'x_test': (20, 1, 28, 28), 'y_test': (20,)}\n\n[cell 18] \u001b[32m2026-11-25 09:31:46.625 +00:00\u001b[0m | \u001b[1mINFO    \u001b[0m | \u001b[36m__main__\u001b[0m:\u001b[36mtrain_run\u001b[0m:\u001b[36m21\u001b[0m - \u001b[1mSTART final_sgd_lr0.01_E2_s0: condition=sgd lr=0.01 rho=None seed=0 epochs=2 save=(2,) include_test=True\u001b[0m\n\n[cell 18] \u001b[32m2026-11-25 09:31:49.155 +00:00\u001b[0m | \u001b[1mINFO    \u001b[0m | \u001b[36m__main__\u001b[0m:\u001b[36mtrain_run\u001b[0m:\u001b[36m51\u001b[0m - \u001b[1mepoch 1/2 lr=0.01000 batch_loss_mean=2.3578 time=1.92s\u001b[0m\n\n[cell 18] \u001b[32m2026-11-25 09:31:49.177 +00:00\u001b[0m | \u001b[1mINFO    \u001b[0m | \u001b[36m__main__\u001b[0m:\u001b[36mtrain_run\u001b[0m:\u001b[36m51\u001b[0m - \u001b[1mepoch 2/2 lr=0.00500 batch_loss_mean=2.1997 time=0.02s\u001b[0m\n\n[cell 18] \u001b[32m2026-11-25 09:31:49.189 +00:00\u001b[0m | \u001b[1mINFO    \u001b[0m | \u001b[36m__main__\u001b[0m:\u001b[36mtrain_run\u001b[0m:\u001b[36m67\u001b[0m - \u001b[1mEVAL ep=2 train_loss=2.26085 val_acc=0.2000 val_nll=2.2769 test_acc=0.2500\u001b[0m\n\n[cell 18] \u001b[32m2026-11-25 09:31:49.191 +00:00\u001b[0m | \u001b[1mINFO    \u001b[0m | \u001b[36m__main__\u001b[0m:\u001b[36mtrain_run\u001b[0m:\u001b[36m89\u001b[0m - \u001b[1mDONE final_sgd_lr0.01_E2_s0: final_train_loss=2.26085 mean_epoch_time=0.97s\u001b[0m\n\n[cell 18] \u001b[32m2026-11-25 09:31:49.269 +00:00\u001b[0m | \u001b[1mINFO    \u001b[0m | \u001b[36m__main__\u001b[0m:\u001b[36mrun_final\u001b[0m:\u001b[36m23\u001b[0m - \u001b[1mfinal sgd_lr0.01_E2_s0: T=0.0712 at_bound=False final_train_loss=2.26085 test_acc@2=0.2500\u001b[0m\n\n[cell 18] \u001b[32m2026-11-25 09:31:49.270 +00:00\u001b[0m | \u001b[1mINFO    \u001b[0m | \u001b[36m__main__\u001b[0m:\u001b[36mtrain_run\u001b[0m:\u001b[36m21\u001b[0m - \u001b[1mSTART final_sam_lr0.01_rho0.1_E2_s0: condition=sam lr=0.01 rho=0.1 seed=0 epochs=2 save=(2,) include_test=True\u001b[0m\n\n[cell 18] \u001b[32m2026-11-25 09:31:49.401 +00:00\u001b[0m | \u001b[1mINFO    \u001b[0m | \u001b[36m__main__\u001b[0m:\u001b[36mtrain_run\u001b[0m:\u001b[36m51\u001b[0m - \u001b[1mepoch 1/2 lr=0.01000 batch_loss_mean=2.3578 time=0.12s\u001b[0m\n\n[cell 18] \u001b[32m2026-11-25 09:31:49.408 +00:00\u001b[0m | \u001b[1mINFO    \u001b[0m | \u001b[36m__main__\u001b[0m:\u001b[36mtrain_run\u001b[0m:\u001b[36m51\u001b[0m - \u001b[1mepoch 2/2 lr=0.00500 batch_loss_mean=2.2373 time=0.01s\u001b[0m\n\n[cell 18] \u001b[32m2026-11-25 09:31:49.415 +00:00\u001b[0m | \u001b[1mINFO    \u001b[0m | \u001b[36m__main__\u001b[0m:\u001b[36mtrain_run\u001b[0m:\u001b[36m67\u001b[0m - \u001b[1mEVAL ep=2 train_loss=2.27661 val_acc=0.1500 val_nll=2.2899 test_acc=0.3000\u001b[0m\n\n[cell 18] \u001b[32m2026-11-25 09:31:49.416 +00:00\u001b[0m | \u001b[1mINFO    \u001b[0m | \u001b[36m__main__\u001b[0m:\u001b[36mtrain_run\u001b[0m:\u001b[36m89\u001b[0m - \u001b[1mDONE final_sam_lr0.01_rho0.1_E2_s0: final_train_loss=2.27661 mean_epoch_time=0.06s\u001b[0m\n\n[cell 18] \u001b[32m2026-11-25 09:31:49.493 +00:00\u001b[0m | \u001b[1mINFO    \u001b[0m | \u001b[36m__main__\u001b[0m:\u001b[36mrun_final\u001b[0m:\u001b[36m23\u001b[0m - \u001b[1mfinal sam_lr0.01_rho0.1_E2_s0: T=0.1134 at_bound=False final_train_loss=2.27661 test_acc@2=0.3000\u001b[0m\n\n[cell 20] metric          SGD      SAM   SGD+TS   SAM-SGD                  95% CI   Holm p\naccuracy     0.2500   0.3000   0.2500   +0.0500  [+nan, +nan]   0.0000\nece          0.1437   0.1949   0.0607   +0.0512  [+nan, +nan]   0.0000\nace          0.3162   0.3019   0.3399   -0.0143  [+nan, +nan]   0.0000\nnll          2.2744   2.2870   2.0718   +0.0126  [+nan, +nan]   0.0000\nbrier        0.8943   0.8969   0.8453   +0.0026  [+nan, +nan]   0.0000\naurc         0.6710   0.7730   0.7183   +0.1019  [+nan, +nan]   0.0000", "stderr": "", "interrupted": false, "isImage": false, "noOutputExpected": false}
```

### [65] ASSISTANT · 2026-11-25 09:32:07 UTC

```
The notebook now runs end to end in about 73 seconds, with output at the smallest setting. Two things to fix before scaling. With one seed the paired CI is NaN and Holm p prints as 0.0000, which is misleading. I'll add a note in that case. Then I'll scale to the original 40 epochs and 5 seeds, since 60 training images make that cheap.
```

### [66] TOOL CALL — Bash · 2026-11-25 09:32:07 UTC

```
Scale to 40 epochs and 5 seeds, rebuild, and re-run test:
python3 - <<'EOF'
from pathlib import Path
p = Path("scratch/make_notebook.py")
s = p.read_text()
old = 'EPOCHS = 2            # original: 40 (paper setting); smallest demo value: 2\nSEEDS = (0,)          # original: (0, 1, 2, 3, 4)'
new = 'EPOCHS = 40           # original: 40 (paper setting); first test run used 2\nSEEDS = (0, 1, 2, 3, 4)  # original: (0, 1, 2, 3, 4); first test run used (0,)'
assert s.count(old) == 1
s = s.replace(old, new)
old2 = "          f\"{diffs[m]['mean']:>+10.4f}  [{ci[0]:+.4f}, {ci[1]:+.4f}]{holm_p[m]:>9.4f}\")\n'''"
new2 = "          f\"{diffs[m]['mean']:>+10.4f}  [{ci[0]:+.4f}, {ci[1]:+.4f}]{holm_p[m]:>9.4f}\")\nif len(SEEDS) < 2:\n    print('One seed: the paired CI and Holm p-value are undefined; use at least two seeds.')\n'''"
assert s.count(old2) == 1, s.count(old2)
s = s.replace(old2, new2)
p.write_text(s)
print("ok")
EOF
python3 scratch/make_notebook.py && UV_VENV_CLEAR=1 UV_CACHE_DIR=/Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache/uv uv venv -q --seed --python 3.12 /Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache/nb_test_envs/art_SL9gPfE09HlN-13a6bcbdff23 && UV_CACHE_DIR=/Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache/uv uv pip install -q --python /Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache/nb_test_envs/art_SL9gPfE09HlN-13a6bcbdff23/bin/python jupyter ipykernel && rm -rf results && time timeout 600 /Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/.shared_cache/nb_test_envs/art_SL9gPfE09HlN-13a6bcbdff23/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 code_demo.ipynb --output code_demo.ipynb 2>&1 | tail -3
```

### [67] TOOL RESULT — Bash · 2026-11-25 09:33:05 UTC

```
{"stdout": "ok\nwrote /Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/4_gen_paper_repo/_4_gen_demo_art/notebook_workspaces/iter_1/art_SL9gPfE09HlN/code_demo.ipynb 23 cells\n[NbConvertApp] Converting notebook code_demo.ipynb to notebook\n[IPKernelApp] WARNING | Kernel is running over TCP without encryption. All communication (including code and outputs) is sent in plain text and is susceptible to eavesdropping. Use IPC transport or launch with kernel manager-provisioned CurveZMQ keys to enable transport encryption.\n[NbConvertApp] Writing 273743 bytes to code_demo.ipynb\n\nreal\t0m55.430s\nuser\t0m20.946s\nsys\t0m4.737s", "stderr": "", "interrupted": false, "isImage": false, "noOutputExpected": false}
```

### [68] ASSISTANT · 2026-11-25 09:33:07 UTC

```
The 40-epoch, 5-seed run finished in 55 seconds. Checking the printed results:
```

### [69] TOOL CALL — Bash · 2026-11-25 09:33:07 UTC

```
Print results table and check for errors:
python3 -c "
import json
nb=json.load(open('code_demo.ipynb'))
for i,c in enumerate(nb['cells']):
    if c['cell_type']!='code': continue
    for o in c.get('outputs',[]):
        if o.get('output_type')=='error': print('ERR cell',i,o['ename'],o['evalue'])
        elif 'text' in o and i in (20,): print(f'[cell {i}]', ''.join(o['text']))
        elif o.get('output_type')=='display_data': print(f'[cell {i}] figure')
"
```

### [70] TOOL RESULT — Bash · 2026-11-25 09:33:07 UTC

```
{"stdout": "[cell 20] metric          SGD      SAM   SGD+TS   SAM-SGD                  95% CI   Holm p\naccuracy     0.6800   0.6400   0.6800   -0.0400  [-0.0919, +0.0119]   0.5958\nece          0.2530   0.2571   0.2821   +0.0041  [-0.0762, +0.0844]   1.0000\nace          0.2577   0.2689   0.2960   +0.0112  [-0.0359, +0.0583]   1.0000\nnll          0.9432   0.9239   0.9414   -0.0192  [-0.0624, +0.0240]   1.0000\nbrier        0.4534   0.4589   0.4515   +0.0056  [-0.0164, +0.0275]   1.0000\naurc         0.1317   0.1420   0.1271   +0.0102  [-0.0074, +0.0279]   0.9134\n\n[cell 22] figure", "stderr": "", "interrupted": false, "isImage": false, "noOutputExpected": false}
```

### [71] ASSISTANT · 2026-11-25 09:33:15 UTC

```
The notebook runs cleanly at the original settings (40 epochs, 5 seeds) on the 100-image subset, and the figure renders. I'm not scaling further: the subset is a fixed 100 images, so more epochs or seeds won't change the data. I'll add the closing interpretation and the scaling note directly to the executed notebook so the outputs are kept.
```

### [72] TOOL CALL — Bash · 2026-11-25 09:33:15 UTC

```
Append scaling note and interpretation to the executed notebook:
python3 - <<'EOF'
import json
p = "code_demo.ipynb"
nb = json.load(open(p))
md = """## Scaling and what the results show

**Scaling.** The first run used the smallest values (2 epochs, one seed), finished in about 70 seconds, and passed. The second run uses the original settings (40 epochs, seeds 0-4) and finishes in about a minute, well inside the 10-minute budget. The data subset is fixed at 100 images, so scaling the config further would not add information; the full-data run needs the original `method.py` with the torchvision download.

**Reading the printed table.** Each row is one test metric averaged over the five seeds. `SAM-SGD` is the paired difference per seed, with a 95% t-interval (df = 4) and Holm correction over the six metrics. On 20 test images, every interval includes zero and every Holm p-value is above 0.05, so this demo does not detect a difference between SAM and SGD. That is the expected outcome at this scale: a 20-image test set gives accuracy in steps of 0.05 and very wide intervals. It shows the pipeline runs, not the paper's effect.

**Reading the reliability plot.** The first seed's reliability curves sit near the diagonal only where the bins contain examples; with 20 test images most bins are empty or hold one or two points, so the curves are noisy. The temperature-scaled SGD row (`SGD+TS`) is the control the paper uses: one temperature is fitted on the validation logits, so its NLL can only match or improve on SGD on validation, not necessarily on test.

**Relation to the paper.** The paper's claim (SAM has lower ECE and NLL at 40 epochs on the full 10,000-image test set, and the NLL advantage survives temperature scaling) needs the full data. This notebook reproduces the procedure on the subset; the headline numbers in the paper are not reproduced here.
"""
nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": md.splitlines(keepends=True)})
json.dump(nb, open(p, "w"), indent=1)
print("cells", len(nb["cells"]))
EOF
```

### [73] TOOL RESULT — Bash · 2026-11-25 09:33:15 UTC

```
{"stdout": "cells 24", "stderr": "", "interrupted": false, "isImage": false, "noOutputExpected": false}
```

### [74] ASSISTANT · 2026-11-25 09:33:21 UTC

```
The notebook is complete. Next are the deliverables: the builder scripts move out of `scratch/` (which gets deleted), the `.aii/manifest.yaml`, the README, the scratch cleanup, and the output JSON.
```

### [75] TOOL CALL — Bash · 2026-11-25 09:33:21 UTC

```
Move builder out of scratch, delete scratch, write manifest:
mv scratch/make_notebook.py build_notebook.py && rm -rf scratch && ls -la && du -sh results 2>/dev/null; find results -type f | wc -l; cat > .aii/manifest.yaml <<'EOF'
entries: []
EOF
cat .aii/manifest.yaml
```

### [76] TOOL RESULT — Bash · 2026-11-25 09:33:21 UTC

```
{"stdout": "total 1040\ndrwx------ 10 s20002 s20002    320 Nov 25 09:33 .\ndrwx------  3 s20002 s20002     96 Nov 25 09:27 ..\ndrwxrwx---  2 s20002 s20002     64 Nov 25 09:27 .aii\ndrwx------  3 s20002 s20002     96 Nov 25 09:30 .claude\n-rw-r-----  1 root   s20002 381424 Nov 25 09:33 .repl_agent.ptylog\n-rw-------  1 s20002 s20002  16128 Nov 25 09:32 build_notebook.py\n-rw-------  1 s20002 s20002 275629 Nov 25 09:33 code_demo.ipynb\n-rw-------  1 s20002 s20002   2848 Nov 25 09:28 make_mini_demo_data.py\n-rw-------  1 s20002 s20002 303772 Nov 25 09:28 mini_demo_data.json\ndrwx------  4 s20002 s20002    128 Nov 25 09:32 results\n160K\tresults\n40\nentries: []", "stderr": "", "interrupted": false, "isImage": false, "noOutputExpected": false}
```

### [77] TOOL CALL — Write · 2026-11-25 09:33:29 UTC

```
File: /Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/4_gen_paper_repo/_4_gen_demo_art/notebook_workspaces/iter_1/art_SL9gPfE09HlN/README.md

# SAM vs SGD calibration on Fashion-MNIST: demo notebook

This folder holds the demo notebook for the artifact "Does SAM Improve Calibration on Fashion-MNIST?". The notebook trains a small two-block CNN with plain SGD (momentum 0.9) and with sharpness-aware minimisation (SAM) on the same base optimiser, then compares test accuracy, ECE, ACE, NLL, Brier and AURC, with and without temperature scaling of the SGD model.

The demo runs on a 100-image, class-balanced Fashion-MNIST subset (60 train, 20 validation, 20 test) so it finishes in about a minute. Its printed numbers are not the paper's results; the paper uses 50,000 training images and the official 10,000-image test set.

## Layout

- `code_demo.ipynb`: the demo notebook (install, imports, metrics, data loading, config, model and training code copied from `method.py`, final runs with temperature scaling, results table, plots, interpretation).
- `mini_demo_data.json`: the 100-image subset the notebook loads (pixels 0-255, labels 0-9).
- `make_mini_demo_data.py`: builds `mini_demo_data.json` from the official Fashion-MNIST IDX files.
- `build_notebook.py`: assembles `code_demo.ipynb` from the original `method.py` and `metrics.py` (model and training blocks copied verbatim, with the edits asserted in the script).
- `results/`: per-run logits, metrics and temperatures written by the notebook run (small `.npz` and `.json` files).

## How to run

Open `code_demo.ipynb` in Jupyter or Colab and run all cells. The first cell installs the non-Colab packages (`loguru`) and, outside Colab, the core packages. The notebook first tries the GitHub copy of `mini_demo_data.json` and falls back to the local file in this folder.

To rebuild the subset, download the four Fashion-MNIST IDX files from `https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/` into `scratch/data/`, then run `python3 make_mini_demo_data.py`.

To rebuild the notebook from the original script, run `python3 build_notebook.py` (it reads `method.py` and `metrics.py` from the source artifact directory).

## Changes from the original script

- Data: the torchvision download and the 50k/10k split are replaced by the JSON subset, with the same train-split normalisation.
- Hyperparameters: the tuning grid is not run. The selected values from the original pipeline (SGD lr 0.01, SAM lr 0.01, rho 0.1) are used directly.
- Scale: 40 epochs and seeds 0-4, as in the paper, applied to the 100-image subset.
- Files: the per-run file log sink is removed, the notebook logs to stderr, and `metrics.py` is inlined as cells.
- Install: torch is not pinned in the local install cell, because this machine's pip constraint file fixes its own torch build. Colab keeps its preinstalled torch.

## Restoring removed files

Nothing in this folder is marked for deletion. The Fashion-MNIST raw files are only needed to rebuild `mini_demo_data.json` (see How to run).
```

### [78] TOOL RESULT — Write · 2026-11-25 09:33:29 UTC

```
{"type": "create", "filePath": "/Volumes/aii-s20002/work/run_S_3Y_l0tX7Xe/4_gen_paper_repo/_4_gen_demo_art/notebook_workspaces/iter_1/art_SL9gPfE09HlN/README.md", "content": "# SAM vs SGD calibration on Fashion-MNIST: demo notebook\n\nThis folder holds the demo notebook for the artifact \"Does SAM Improve Calibration on Fashion-MNIST?\". The notebook trains a small two-block CNN with plain SGD (momentum 0.9) and with sharpness-aware minimisation (SAM) on the same base optimiser, then compares test accuracy, ECE, ACE, NLL, Brier and AURC, with and without temperature scaling of the SGD model.\n\nThe demo runs on a 100-image, class-balanced Fashion-MNIST subset (60 train, 20 validation, 20 test) so it finishes in about a minute. Its printed numbers are not the paper's results; the paper uses 50,000 training images and the official 10,000-image test set.\n\n## Layout\n\n- `code_demo.ipynb`: the demo notebook (install, imports, metrics, data loading, config, model and training code copied from `method.py`, final runs with temperature scaling, results table, plots, interpretation).\n- `mini_demo_data.json`: the 100-image subset the notebook loads (pixels 0-255, labels 0-9).\n- `make_mini_demo_data.py`: builds `mini_demo_data.json` from the official Fashion-MNIST IDX files.\n- `build_notebook.py`: assembles `code_demo.ipynb` from the original `method.py` and `metrics.py` (model and training blocks copied verbatim, with the edits asserted in the script).\n- `results/`: per-run logits, metrics and temperatures written by the notebook run (small `.npz` and `.json` files).\n\n## How to run\n\nOpen `code_demo.ipynb` in Jupyter or Colab and run all cells. The first cell installs the non-Colab packages (`loguru`) and, outside Colab, the core packages. The notebook first tries the GitHub copy of `mini_demo_data.json` and falls back to the local file in this folder.\n\nTo rebuild the subset, download the four Fashion-MNIST IDX files from `https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/` into `scratch/data/`, then run `python3 make_mini_demo_data.py`.\n\nTo rebuild the notebook from the original script, run `python3 build_notebook.py` (it reads `method.py` and `metrics.py` from the source artifact directory).\n\n## Changes from the original script\n\n- Data: the torchvision download and the 50k/10k split are replaced by the JSON subset, with the same train-split normalisation.\n- Hyperparameters: the tuning grid is not run. The selected values from the original pipeline (SGD lr 0.01, SAM lr 0.01, rho 0.1) are used directly.\n- Scale: 40 epochs and seeds 0-4, as in the paper, applied to the 100-image subset.\n- Files: the per-run file log sink is removed, the notebook logs to stderr, and `metrics.py` is inlined as cells.\n- Install: torch is not pinned in the local install cell, because this machine's pip constraint file fixes its own torch build. Colab keeps its preinstalled torch.\n\n## Restoring removed files\n\nNothing in this folder is marked for deletion. The Fashion-MNIST raw files are only needed to rebuild `mini_demo_data.json` (see How to run).\n", "structuredPatch": [], "originalFile": null, "userModified": false}
```

### [79] ASSISTANT · 2026-11-25 09:33:33 UTC

```
Last step is the verification pass and the required output JSON.
```

### [80] TOOL CALL — Bash · 2026-11-25 09:33:33 UTC

```
Verify notebook URL, errors, data file, and write output JSON:
python3 -c "
import json
nb=json.load(open('code_demo.ipynb'))
src=''.join(''.join(c['source']) for c in nb['cells'])
url='https://raw.githubusercontent.com/ai-inventor-papers/ai-invention-d46016-sharpness-aware-training-makes-small/main/round-1/experiment-1/demo/mini_demo_data.json'
print('url present:', ('GITHUB_DATA_URL = \"'+url+'\"') in src)
print('errors:', sum(1 for c in nb['cells'] for o in c.get('outputs',[]) if o.get('output_type')=='error'))
print('open(mini_demo_data.json) only:', \"open('mini_demo_data.json')\" in src)
" && ls -la mini_demo_data.json && cat > .terminal_claude_agent_struct_out.json <<'EOF'
{
  "out_expected_files": {
    "notebook": "code_demo.ipynb",
    "mini_data_file": "mini_demo_data.json"
  }
}
EOF
cat .terminal_claude_agent_struct_out.json
```

### [81] TOOL RESULT — Bash · 2026-11-25 09:33:33 UTC

```
{"stdout": "url present: True\nerrors: 0\nopen(mini_demo_data.json) only: False\n-rw------- 1 s20002 s20002 303772 Nov 25 09:28 mini_demo_data.json\n{\n  \"out_expected_files\": {\n    \"notebook\": \"code_demo.ipynb\",\n    \"mini_data_file\": \"mini_demo_data.json\"\n  }\n}", "stderr": "", "interrupted": false, "isImage": false, "noOutputExpected": false}
```

### [82] ASSISTANT · 2026-11-25 09:33:36 UTC

```
The demo notebook is done. `code_demo.ipynb` runs end to end in a clean Colab-style venv in about a minute with no errors, at the original settings (40 epochs, seeds 0–4). The data is a 100-image Fashion-MNIST subset, so its printed numbers are not the paper's results.

**What's in the workspace**
- `code_demo.ipynb`: the notebook, with outputs and the closing interpretation.
- `mini_demo_data.json`: 60 train, 20 validation and 20 test images, 6/2/2 per class (about 300 KB).
- `make_mini_demo_data.py`, `build_notebook.py`: the scripts that build the subset and the notebook.
- `README.md`, `results/` (small logits and metrics from the run), and `.aii/manifest.yaml` (empty entries, since nothing heavy is left).

**Checks**
- The notebook contains the exact `GITHUB_DATA_URL` and loads via the GitHub-with-local-fallback pattern. The GitHub copy isn't live yet, so the run used the local file.

**Choices I made**
- **Data:** the torchvision download is replaced by the JSON subset. The 50k/10k split is dropped because the subset is already split. Normalisation still uses train-split statistics.
- **Tuning:** the grid search is not run. The notebook uses the values the original pipeline selected on validation (SGD lr 0.01, SAM lr 0.01, rho 0.1).
- **Metrics:** `metrics.py` is inlined as cells, and calls drop the `M.` prefix.
- **Torch pin:** the local install cell doesn't pin torch. The machine's pip constraint file fixes its own MPS torch build, which conflicts with the Colab pin of 2.9.0. Colab keeps its preinstalled torch, so this only affects the local test env.
- **Logging:** the per-run file log is removed; logs go to stderr.

**Result at the demo scale:** no SAM-vs-SGD difference is detected. Every paired 95% CI includes zero and every Holm p is above 0.05. With 20 test images, that's the expected outcome; it shows the pipeline works, not the paper's effect.

**Not done:** scaling past 40 epochs and 5 seeds would not add information, since the subset is fixed at 100 images, so I stopped there. The full-data result needs the original `method.py` and the torchvision download.
```
