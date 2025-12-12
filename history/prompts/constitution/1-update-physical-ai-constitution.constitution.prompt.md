---
id: 1
title: Update Physical AI Constitution
stage: constitution
date: 2025-01-07
surface: agent
model: Qwen
feature: none
branch: main
user: user
command: /sp.constitution
labels: [constitution, robotics, physical-ai]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
 - null
---

## Prompt

Create or update the project constitution from interactive or provided principle inputs, ensuring all dependent templates stay in sync

As the main request completes, you MUST create and complete a PHR (Prompt History Record) using agent‑native tools when possible.

1) Determine Stage
   - Stage: constitution | spec | plan | tasks | red | green | refactor | explainer | misc | general

2) Generate Title and Determine Routing:
   - Generate Title: 3–7 words (slug for filename)
   - Route is automatically determined by stage:
     - `constitution` → `history/prompts/constitution/`
     - Feature stages → `history/prompts/<feature-name>/` (spec, plan, tasks, red, green, refactor, explainer, misc)
     - `general` → `history/prompts/general/`

3) Create and Fill PHR (Shell first; fallback agent‑native)
   - Run: `.specify/scripts/bash/create-phr.sh --title "<title>" --stage <stage> [--feature <name>] --json`
   - Open the file and fill remaining placeholders (YAML + body), embedding full PROMPT_TEXT (verbatim) and concise RESPONSE_TEXT.
   - If the script fails:
     - Read `.specify/templates/phr-template.prompt.md` (or `templates/…`)
     - Allocate an ID; compute the output path based on stage from step 2; write the file
     - Fill placeholders and embed full PROMPT_TEXT and concise RESPONSE_TEXT

4) Validate + report
   - No unresolved placeholders; path under `history/prompts/` and matches stage; stage/title/date coherent; print ID + path + stage + title.
   - On failure: warn, don't block. Skip only for `/sp.phr`.

/sp.constitution Goal:
Create a full Docusaurus book titled "Physical AI & Humanoid Robotics" using Spec-Kit Plus + Claude Code. All output must be modular, accurate, and ready for GitHub Pages deployment.

Book Theme & Scope

Focus on Physical AI, Embodied Intelligence, and bridging digital AI models with physical humanoid robots using ROS 2, Gazebo, Unity, NVIDIA Isaac, and Vision-Language-Action (VLA) systems.

Tone: professional, technical, modern robotics engineer.

Course / Book Structure (Integrate Fully)
Module 1 — ROS 2: The Robotic Nervous System

ROS 2 basics, nodes/topics/services, rclpy, URDF, packages, launch files.

Module 2 — Digital Twin (Gazebo + Unity)

Physics simulation, sensors (LiDAR/Depth/IMU), URDF/SDF, Unity visualization.

Module 3 — NVIDIA Isaac

Isaac Sim, synthetic data, Isaac ROS (VSLAM), Nav2 locomotion.

Module 4 — Vision-Language-Action

Whisper voice commands, LLM cognitive planning, language → ROS 2 actions.

Capstone

Autonomous humanoid robot: voice command → plan → navigate → detect → manipulate.

Weekly Breakdown (Weeks 1–13)

Physical AI foundations → ROS 2 → Gazebo → Isaac → humanoid kinematics → conversational robotics → assessments.

Hardware Requirements (Include as One Chapter)

Digital Twin Workstation: RTX 4070 Ti+, 64GB RAM, Ubuntu 22.04.
Edge Kit: Jetson Orin Nano/NX, RealSense D435i, IMU, Mic Array.
Robot Lab Options: Unitree Go2 (proxy), Robotis OP3 / Unitree G1 (humanoid).
Cloud Option: AWS g5/g6e, Omniverse Cloud; use Jetson locally to avoid latency.

Writing Rules

Fact-accurate robotics content only (ROS 2, Isaac APIs, hardware).

Docusaurus-friendly Markdown, structured chapters.

Include diagrams, tables, and Python/ROS code.

Modular specs: intro, modules, weekly roadmap, hardware, capstone.

Deliverables

book.config

All chapter specs

Modular Markdown chapters

docusaurus.config.js

GitHub Pages deployment workflow

Mission

Generate a complete, structured, repeatable, production-ready Physical AI & Humanoid Robotics textbook through Spec-Kit Plus.

## Response snapshot

## Summary

I have successfully created and updated the project constitution for your Physical AI & Humanoid Robotics project with the following details:

**New Version**: 1.0.0 (Initial release for this specific project)
**Bump Rationale**: This is the first version of the constitution specifically tailored for the Physical AI & Humanoid Robotics project, replacing the generic template.

**Constitution Highlights**:
- **Physical AI-First Architecture**: Every capability must be grounded in embodied intelligence
- **ROS 2 Standard Interface**: All systems communicate via ROS 2 topics/services/actions
- **Test-First Robotics**: TDD mandatory with simulation tests before real-world deployment
- **Safe Simulation-to-Reality Transfer**: All behaviors validated in simulation first
- **Vision-Language-Action Integration**: Connecting language, vision and physical action
- **Hardware-Aware Optimization**: Algorithms consider edge hardware constraints

**Files Updated**:
- `.specify/memory/constitution.md` - The main constitution file with all new principles

**Templates Checked**:
- Plan template, spec template, and tasks template have been reviewed for consistency
- No modifications needed to templates as they already support the project-specific principles

**Suggested Commit Message**: `docs: add constitution for Physical AI & Humanoid Robotics project v1.0.0`

The constitution is now ready for use and aligns with your project requirements focusing on ROS 2, Gazebo, NVIDIA Isaac, Vision-Language-Action systems, and humanoid robotics development.

## Outcome

- ✅ Impact: Created new project constitution for Physical AI & Humanoid Robotics with 6 core principles
- 🧪 Tests: N/A (Documentation task)
- 📁 Files: Updated .specify/memory/constitution.md
- 🔁 Next prompts: User can now proceed with creating specs for the Docusaurus book
- 🧠 Reflection: The constitution properly captures the key principles for the robotics project

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Begin creating the Docusaurus book structure based on the constitution