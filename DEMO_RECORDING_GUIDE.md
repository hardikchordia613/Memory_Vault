# Recreating the Demo

The repository contains two demo options.

## Run the live walkthrough

Complete the README setup first, then run:

```bash
./record_demo.sh
```

This uses the real configured services. It checks system health, parses a code
sample with Tree-sitter, stores a memory through Gemini embeddings, runs the
PostgreSQL hybrid RRF query, and lists the native agent tools.

The script is safe to launch from any directory because it resolves the
repository path automatically. It never prints the Gemini API key.

## Regenerate the README GIF

Install [VHS](https://github.com/charmbracelet/vhs), ensure Docker is running,
and run this command from the repository root:

```bash
vhs demo.tape
```

VHS executes the commands in `demo.tape` and writes `demo.gif`. The recording
uses the live health, chunking, hybrid search, and agent-tool discovery paths.

Because hybrid search creates a query embedding, GIF regeneration uses a small
amount of Gemini API quota. The database must already contain at least one
memory relevant to the demo query; running `./record_demo.sh` will create one.
