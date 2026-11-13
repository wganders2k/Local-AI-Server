"""
Multi-persona conversations: /mimic-chat threads and lore's simulate_conversation.

A controller (the bare mimic base, no adapter) decides who speaks next, or
that the chat should wait for a human or end; the chosen persona's adapter
writes the line. Nothing here imports discord — the cog reads and posts
(persona_posting.py), and passes the transcript in as mimic.transcript.Lines
with a Cast describing who is in it.
"""
