---
name: gitignore-large-file-guard
description: Update and enforce .gitignore rules to prevent uploading large or transient files such as datasets, caches, model checkpoints, logs, and build artifacts. Use when repository hygiene or GitHub push safety requires blocking heavy files.
---

# Gitignore Large File Guard

## Overview
Harden `.gitignore` so data/cache/model artifacts do not enter version control, and verify no large accidental files are staged.

## Workflow
1. Inspect repository directories and identify heavy or generated paths (`data/`, `outputs/`, `logs/`, checkpoints, caches, temp files).
2. Update `.gitignore` with precise patterns that block large artifacts but keep needed source/config files trackable.
3. Preserve existing ignore rules unless they are clearly wrong.
4. Run `scripts/find_large_untracked.sh` to detect risky files not currently tracked.
5. If large files are already tracked, report exact paths and required cleanup steps.

## Rule Design Guidelines
- Prefer directory-scoped patterns (`data/**`, `outputs/**`) for generated artifacts.
- Add extension-based patterns for checkpoints and weights (`*.ckpt`, `*.pt`, `*.bin`, `*.safetensors`).
- Ignore caches/logs/temp (`.cache/`, `__pycache__/`, `*.log`, `tmp/`).
- Keep source assets intentionally tracked by adding allow-rules when needed.

## Required Output
- Updated `.gitignore`.
- Scan result summary showing files over threshold and whether they are tracked.
- Follow-up commands when history cleanup is needed.

## Resources
- Large-file scanner: `scripts/find_large_untracked.sh`
- Pattern reference: `references/common-heavy-patterns.md`
