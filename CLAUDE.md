# How to work in this project

This project is built in stages, with each task routed to the cheapest model
that can do it well. Follow this workflow in every session.

## 1. Plan first, in PLAN.md

- Before writing any code for a new idea or feature, write or update `PLAN.md`
  in the project root. Do not start building until I have approved the plan.
- Break the work into small tasks that can each be finished and verified on
  their own. For each task, record:
  - an ID and a one-line goal
  - the files it is expected to touch
  - how to verify it is done (test, command, or what to look at)
  - a difficulty tag: `[hard]`, `[medium]`, or `[easy]`
  - a status: `todo`, `in progress`, `done`
- `PLAN.md` is the source of truth. Read it at the start of every session
  instead of relying on conversation history, and update statuses as tasks
  finish.

## 2. Difficulty tags and models

| Tag        | What it covers                                                      | Model           |
|------------|---------------------------------------------------------------------|-----------------|
| `[hard]`   | Architecture, data model, tricky bugs, security, final review       | `fable` or `opus` |
| `[medium]` | Normal feature work with a clear spec                               | `sonnet`        |
| `[easy]`   | Renames, formatting, boilerplate, simple searches, small doc edits  | `haiku`         |

When unsure between two tags, pick the lower one and escalate only if the
task fails verification.

## 3. Routing

- When you delegate a task to a subagent, set the subagent's model to match
  the task's tag. If custom subagents exist in `.claude/agents/`, use the one
  whose model matches.
- When working in the main session, check the task's tag against the current
  model before starting. If they don't match, stop and tell me which model to
  switch to (for example: "Task 5 is [easy], switch with `/model haiku`").
  Do not silently do easy work on an expensive model or hard work on a cheap
  one.
- Never spawn several subagents for something one agent can do.

## 4. One task at a time

- Do only the task I named, or the next `todo` task in `PLAN.md`. Do not
  start the following task without being asked.
- Stay inside the task's scope. If you notice something else that needs
  fixing, add it to `PLAN.md` as a new task instead of fixing it now.
- Verify the task the way `PLAN.md` says before marking it `done`.
- When a task is finished, report in a few lines: what changed, how it was
  verified, and the next task with its tag and model. Then stop.

## 5. Keep context small

- Read only the files a task needs.
- After each finished task, remind me to run `/clear` before the next one.
- If the plan turns out to be wrong, say so and propose a change to
  `PLAN.md` rather than working around it.
