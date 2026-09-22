# Reproducible terminal recording. Run from the repository root with:
#   vhs demo.tape

Output demo.gif

Set Shell "zsh"
Set FontSize 16
Set Width 1200
Set Height 720
Set Theme "Monokai"
Set PlaybackSpeed 1.2
Set TypingSpeed 35ms

Type "clear" Enter
Sleep 500ms
Type "# Codebase Memory Vault — live feature demo" Enter
Sleep 1s

Type "source .venv/bin/activate" Enter
Sleep 1s

Type "# 1. Verify PostgreSQL, pgvector, and Gemini configuration" Enter
Type "python -m vault.cli doctor" Enter
Sleep 4s

Type "# 2. AST chunking keeps class and method context" Enter
Type "python -c \"from vault.chunker import chunk_file; c=chunk_file('auth.py', 'class TokenService:\\n    def validate(self, token):\\n        return bool(token)\\n')[0]; print(c['text'].split('\\n\\n')[0])\"" Enter
Sleep 3s

Type "# 3. Hybrid search fuses semantic and keyword ranks with RRF" Enter
Type "python -m vault.cli ask 'How are semantic and exact keyword results combined?' --limit 2" Enter
Sleep 25s

Type "# 4. Native Gemini multi-agent specialists" Enter
Type "python -c \"from vault.agents import TOOL_DISPATCH; print('  '.join(TOOL_DISPATCH))\"" Enter
Sleep 3s

Type "# Demo complete — see README.md for the full walkthrough" Enter
Sleep 2s
