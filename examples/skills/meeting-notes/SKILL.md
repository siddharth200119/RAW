---
name: meeting-notes
description: Turn raw meeting transcripts or bullet dumps into structured notes with decisions and action items. Use when the user pastes a transcript, asks for minutes, or asks "what did we decide".
license: MIT
---

# Meeting notes

Turn the raw material into notes a colleague who missed the meeting can act on.

1. Read the transcript and identify attendees, topics and outcomes.
2. Write the notes using the layout in `references/template.md`.
3. List every action item with an owner. If the owner is unclear, write `owner: unassigned` instead of guessing.
4. Call the `count_words` tool on your draft and keep the notes under 300 words unless the user asks for more.

Never invent decisions that the transcript does not contain.
