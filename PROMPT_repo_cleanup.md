# Session Prompt — MiniRocketHLS Repository Cleanup & Documentation Overhaul

> Paste this whole file as the first message of a NEW Claude Code session in
> `/home/rdave009/minirocket-hls`. Run the orchestrator on **Claude Fable**.
> Repo: github.com/Rishi-Dave/MiniRocketHLS.

---

## 0. Operating model (READ FIRST — non-negotiable)

You are the **orchestrator**. You do **not** do task work yourself — **every actual
task (inventory, reading, git ops, edits, doc writing, verification) is dispatched to
a subagent** via the `Agent` tool. You plan, decompose, dispatch (parallel when
independent), verify each subagent's output, and integrate. Keep conclusions, not
dumps.

**Start in PLAN MODE.** Use `EnterPlanMode` immediately. Do all discovery
(read-only) first, then present ONE consolidated plan via `ExitPlanMode` for the
user to approve. **NOTHING destructive or history-/remote-changing happens before
the user approves the plan** — no `rm`, no `git rm`, no branch delete, no merge, no
push, no commit, no `.gitignore`-driven purge. Discovery only until approved.

Load skills before planning:
1. `superpowers:using-superpowers` (how to find/use skills).
2. `superpowers:subagent-driven-development` (execute the plan via subagents) +
   `superpowers:dispatching-parallel-agents` (fan out independent inventory tasks).
   If a superpowers skill is missing, STOP and tell the user to install the
   `superpowers` plugin.
3. Respect project `CLAUDE.md` rules (below). Consider `claude-md-management` /
   `revise-claude-md` if CLAUDE.md itself needs updating at the end.

**`gh` CLI is authorized with full permissions** for branch listing, PRs, merges,
pushes, and branch deletes — but every *destructive/remote* `gh`/`git` action is
still gated behind plan approval (§0) and the safety rules (§1).

---

## 1. PRIME DIRECTIVE — this is a RESEARCH repo. Nothing irreplaceable is lost. Ever.

This repo holds months of FPGA research. Losing the wrong file/branch/result can cost
hours-to-weeks of rebuilds or destroy HW-validated data that can't be regenerated.
Treat every deletion/merge/history op as potentially irreversible.

**Precious — never delete, always preserve/back-up first:**
- `*.xclbin` / `*.xo` — each is HOURS of synthesis. (CLAUDE.md: never delete.)
- HW-validated results / CSVs / benchmark data / saturation runs.
- **Unique code that exists only on a branch** (research branches carry divergent,
  important scripts — see §3). Assume every branch may hold something irreplaceable
  until proven otherwise.
- Trained model JSONs / weights that can't be trivially regenerated.
- The **current working state**: the active FPGA-to-FPGA network experiments
  (`saturation_harness/`, the ported deploy tooling, `fpga-network/`, the two
  `PROMPT_*.md` handoff files), and whatever branch is currently checked out.

**Hard safety rules (from CLAUDE.md + repo mechanics):**
- **No force-push. No history rewrite** (filter-repo/BFG/rebase-drop) unless the user
  *explicitly* approves it as a separate, coordinated step. History bloat from
  previously-committed large files is a KNOWN research-repo problem, but rewriting a
  shared history is high-risk — default to working-tree cleanup + `.gitignore` +
  `git rm --cached` (stops tracking, keeps history) and only *propose* history
  slimming as an opt-in.
- **Tag before you delete anything git-tracked.** A git tag is free immortality:
  `git tag archive/<branch-or-desc> <sha>` before deleting a branch, so its code is
  recoverable forever without keeping the branch.
- **Never commit `.xclbin`/`.xo`** or multi-GB data (there may be giant
  `*_test_data.json` files — some historically ~2 GB — committed or floating).
- **Review every diff** before commit; small, well-labeled commits; end commit
  messages with the Co-Authored-By line per CLAUDE.md.
- Back up anything questionable to a tag or an `archive/` area rather than deleting.

---

## 2. Do NOT trust priors — DISCOVER the real state (Phase 1, read-only)

The repo has drifted; assume any prior description (including anything in memory or
this prompt) may be stale. Build a **ground-truth inventory** via subagents before
proposing anything. Dispatch these in parallel (each returns a structured report,
not raw dumps):

- **Branch map:** `git branch -a` (local + remotes). For EACH branch: last commit,
  ahead/behind vs the main/default branch, and — critically — **what files/scripts
  exist only on it or differ meaningfully** (`git diff --stat <main>...<branch>`,
  `git log --oneline`, `git ls-tree`). Flag unique valuable code (build scripts,
  kernel sources, host code, result data) per branch. Identify the current branch and
  the true default/canonical branch.
- **Junk/artifact inventory:** find build-artifact dirs and files that flood the tree
  and almost certainly shouldn't be tracked — verify before proposing removal:
  `_x*`, `*_prj`, `.autopilot`, `.Xil`, `.run`, `package.hw*`, `build_dir*`,
  `*.jou`, `vivado*.log`, `*.backup.log`, `*_emu*.log`, `emulation_debug.log`,
  `*.pb`, `.ipcache`, `__pycache__`, plus obvious dup/backup sources
  (`* copy.cpp`, `*-old-for-ref*`, `*.bak`). Report sizes and whether each is
  git-tracked or untracked.
- **Large-file audit:** biggest files in the working tree AND biggest blobs in git
  history (e.g. committed xclbins / GB JSONs). Report tracked vs untracked, and which
  bloat history.
- **VC hygiene audit:** current `.gitignore` (or absence), `git status` (untracked
  that should be ignored/committed), tracked files that match junk categories
  (`git ls-files` ∩ junk), any uncommitted-but-important work in the working tree.
- **Docs & story audit:** every `README*`, `docs/`, `*.md`, RESULTS tables. What do
  they currently claim vs reality? (See §5 for what the story SHOULD be.)
- **Numbers source-of-truth:** read `memory/` (`benchmarks.md`, `paper-resubmission.md`,
  `network_bandwidth_2026_07_04.md`, `oct_cluster_state_2026_05_19.md`,
  `dpr_source_recovery_2026_06_04.md`, `MEMORY.md`) and `PaperTexFiles/` — these hold
  the CURRENT validated numbers + the FCCM story. Docs must match these, not older
  READMEs.

Synthesize into a single **repo state report** (branches & their unique value; junk
by category+size; large/committed-blob problems; VC issues; doc gaps). That report
seeds the plan.

---

## 3. Branch consolidation (the delicate part)

Research branches carry divergent important code — do NOT blindly merge (chaos) or
delete (loss). For the plan:
1. For each branch, decide: **unique-and-valuable** (must preserve its content),
   **superseded/duplicate** (content already elsewhere), or **stale/experimental**.
2. Choose the **canonical branch** (likely the current/most-recent working branch —
   confirm with the user; do not assume). Propose how each other branch's unique
   value gets preserved: cherry-pick / merge specific files into canonical, or
   **`git tag archive/<branch>` then delete** (tag keeps it recoverable).
3. Use `gh` for merges/PRs where a real merge is warranted; prefer small, reviewed
   PRs over a big-bang merge. Resolve conflicts deliberately (a subagent per branch,
   report conflicts, you adjudicate).
4. **Never lose the "different important code scripts" across branches** — the plan
   must explicitly account for every unique file before any branch is deleted, and
   every deleted branch must be tagged first.

---

## 4. Cleanup workstreams (execute only after plan approval, in phases with verification)

- **W-A Working-tree de-junk:** remove verified build-artifacts/dups/logs; add a
  proper `.gitignore` (HLS/Vitis artifacts, build dirs, `.xclbin`/`.xo`, large data,
  `__pycache__`, editor cruft) so it stays clean; `git rm --cached` anything tracked
  that should be ignored (keeps history, stops tracking). Verify the repo still
  builds / nothing sourced-but-untracked was removed.
- **W-B Structure:** propose a sane top-level layout (variants, fpga-network,
  saturation_harness, host, docs, paper) if the current tree is chaotic — but move,
  don't delete, and update references. Get approval for any moves (they touch paths in
  build scripts).
- **W-C Branch consolidation:** per §3.
- **W-D Large-file / history:** if committed xclbins/GB-JSONs bloat the repo, propose
  `git rm --cached` + ignore going forward (safe). Flag (do NOT auto-do) any history
  rewrite as an opt-in the user must explicitly bless.
- **W-E Documentation overhaul:** §5.

Each phase: dispatch subagents, verify, small labeled commits, push via `gh`/`git`
(no force). Preserve the active network-experiment work throughout.

---

## 5. Documentation overhaul (fix the story + the numbers)

The docs are stale and do NOT reflect current reality. Rewrite the repo's docs
(root `README`, per-area READMEs, a `RESULTS.md`, `docs/`) so a reviewer/collaborator
lands and immediately understands the project and its results. It must reflect **all
three threads** (currently missing/partial):

1. **The FCCM story & variant taxonomy** — MiniRocket + HYDRA HLS accelerators; the
   conv-engine variants (Naïve+FP, Naïve+bFP, DP-Reuse+bFP, Fused CONV+PPV) and HYDRA
   fixed-point; the pitch. Align with `PaperTexFiles/` and `paper-resubmission.md`.
   (Update repo docs; treat the `.tex` as the paper source, keep them consistent.)
2. **Improved HBM-loaded builds** — the faster HBM-loading work (spring results). What
   changed, the speedup, the numbers.
3. **FPGA-to-FPGA networking / saturation experiments** — the DECISIVE win: FPGA
   ingests inline at the CMAC (no NIC) and out-scales CPU/GPU (which are NIC-bound).
   Document the setup (cross-node cloudlab, pc160/pc161 topology, NetLayer+pktDropper,
   the loader), the bandwidth-vs-rate result, the cliff/wedge findings, and the
   in-flight bandwidth chart (see `PROMPT_network_bandwidth_chart.md`).

**Numbers integrity:** pull every quoted number from HW-validated sources
(`benchmarks.md`, the saturation `runs/`, `network_bandwidth_2026_07_04.md`,
`oct_cluster_state`), cite the run, and DO NOT fabricate or carry forward stale/known-
bad numbers (e.g. dropped MultiRocket results — see `multirocket-dropped.md`). If a
doc number can't be traced to a validated source, flag it rather than publish it.

PI context (Philip Brisk, verbatim, on why the story matters):
> "We'll use both. My take on the FCCM reviews is that we need more decisive wins…
> there is also going to be pushes for GPU results, where we are less likely to win.
> The new results with faster HBM loaded builds help. But the network experiments is
> where we can most definitively beat either a CPU or a GPU because the CPU/GPU need
> to go through a NIC and we don't."

---

## 6. Guardrails (repo-specific)

- **Protect current work.** Confirm the current branch and the active FPGA-network
  effort with the user; do not merge/delete/restructure in a way that disrupts it.
  The `saturation_harness/` tooling, `fpga-network/` sources, and the `PROMPT_*.md`
  files are live.
- CLAUDE.md rules bind: no force-push; never commit `.xclbin`/`.xo`; review diffs;
  Co-Authored-By on commits; branch before committing on a default branch.
- Ask the user to confirm the **canonical branch**, any **branch deletions** (even
  after tagging), any **file moves** that touch build paths, and any **history
  rewrite** — before executing.
- Update `memory/` + CLAUDE.md (with confirmation) if the cleanup changes repo
  structure, branch layout, or build paths.

---

## 7. First actions

1. `EnterPlanMode`. Load the superpowers skills (§0).
2. Fan out the Phase-1 discovery subagents (§2) in parallel → synthesize the **repo
   state report**.
3. Draft the cleanup+docs plan (branches to consolidate/tag/delete, junk to purge,
   `.gitignore`, structure moves, doc rewrites, numbers to correct) with explicit
   preservation accounting for every unique branch file and every precious artifact.
4. `ExitPlanMode` → get user approval. Confirm: canonical branch, deletions, moves,
   and whether history rewrite is in scope.
5. Execute in phases via subagents, verifying after each, committing small, pushing
   via `gh`/`git` (no force). Re-verify the repo still builds and the active network
   work is intact.

Deliverable: a clean, well-structured, single-source-of-truth repo — junk gone,
branches consolidated (nothing lost, everything tagged), sane `.gitignore`, and docs
that accurately tell the FCCM story with correct HBM-loading and FPGA-networking
numbers.
