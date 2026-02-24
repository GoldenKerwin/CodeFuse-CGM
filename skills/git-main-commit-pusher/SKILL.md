---
name: git-main-commit-pusher
description: Safely commit repository changes and push to the main branch with explicit checks for branch state, remote sync, and push readiness. Use when a user asks to finalize work by committing and pushing to origin/main.
---

# Git Main Commit Pusher

## Overview
Finalize local changes into a clean commit and push to `main` with minimal risk.

## Workflow
1. Check repository state (`git status`, current branch, remotes).
2. Confirm target branch is `main`; switch if needed.
3. Fetch remote state and rebase/pull when required before commit.
4. Review diffs for accidental binary or secret files.
5. Stage only intended files.
6. Commit with a concise, scoped message.
7. Push to `origin main`.
8. Report commit hash, changed files, and push result.

## Safety Rules
- Never use destructive reset/checkout unless explicitly requested.
- Never force-push by default.
- If rebase conflicts occur, stop and surface exact files needing manual resolution.
- If remote rejects push, provide non-destructive next steps.

## Commit Message Guidelines
- Use imperative style summary line.
- Mention scope and intent, not implementation noise.
- Keep first line focused and short.

## Required Output
- Commit hash and message.
- Files included in commit.
- Confirmation that `origin/main` received the commit.

## Resources
- Push checklist: `references/push-checklist.md`
