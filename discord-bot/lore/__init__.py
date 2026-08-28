"""
The /lore agent: prompts, tools, the tool-calling loop, and session state.

This package must not import discord. Everything here is reachable from a test
without a gateway connection, and progress reporting goes through the
ProgressReporter protocol in lore.progress, which the cogs layer implements.
"""
