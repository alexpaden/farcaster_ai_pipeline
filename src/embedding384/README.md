# Embedding384 Service

Farcaster cast text embedding service (384-dimensional vectors).

## Core Function
Continuously processes new Farcaster casts, generating 384-dimensional text embeddings for search and recommendation. Run manually with:
```bash
python -m src.embedding384.backfill
```

## Setup

1. Install service:
```bash
cp file.plist ~/Library/LaunchAgents/com.farcaster.backfill.embedding384.plist
launchctl load ~/Library/LaunchAgents/com.farcaster.backfill.embedding384.plist
```

## Commands

Start service:
```bash
launchctl start com.farcaster.backfill.embedding384
```

Stop service:
```bash
launchctl stop com.farcaster.backfill.embedding384
```

Check logs:
```bash
tail -f logs/backfill.log
tail -f logs/backfill.error.log
```

## Notes
- Runs every 5 minutes
- Logs rotate every 3 days automatically
- Requires Python environment from parent project 