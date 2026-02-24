---
name: project-readme-maintainer
description: Generate or update comprehensive, structured README documentation for code repositories and subdirectories. Use when the user asks to improve project docs, fill missing README.md files, or keep root README linked to child-module READMEs with clear navigation and usage guidance.
---

# Project Readme Maintainer

## Overview
Create production-quality README files for the repository root and important subfolders. Keep documentation structured, complete, and navigable.

## Workflow
1. Inspect repository structure and detect modules/services/scripts that need their own README.
2. Read existing README files and preserve valid project-specific information.
3. Update or create root `README.md` with:
- project summary
- architecture or directory map
- quick start
- setup and run commands
- dataset/model/config notes
- troubleshooting
- contribution guide
- links to each subfolder README
4. Update or create subfolder `README.md` files with local details:
- purpose and boundaries of the folder
- important files and entry points
- commands for this part only
- input/output expectations
- references back to root README
5. Ensure links are relative Markdown links and remain valid after commit.
6. Keep docs factual and synchronized with current files and scripts.

## Documentation Standards
- Use clear section hierarchy (`#`, `##`, `###`) and concise prose.
- Prefer executable command examples over abstract descriptions.
- Explicitly mark environment assumptions (OS, Python version, GPU, dependencies).
- Include a minimal "Quick Verification" section so users can test setup quickly.
- Avoid placeholders unless the user confirms unknown values.

## Required Output
- Updated root `README.md` with a "Repository Guide" section linking submodule docs.
- New/updated `README.md` in each significant subfolder.
- A short change summary listing created/updated documentation files.

## Resources
- Checklist: `references/readme-checklist.md`
- Root template: `assets/root-readme-template.md`
- Subfolder template: `assets/subfolder-readme-template.md`
