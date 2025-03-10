# Embedding384 Service

Farcaster cast text embedding service (384-dimensional vectors).

## Core Function
Continuously processes new Farcaster casts, generating 384-dimensional text embeddings for search and recommendation. Run manually with:
```bash
python -m src.embedding384.backfill
```

## Setup

1. Remove existing service (if any):
```bash
# Stop and remove service
sudo launchctl bootout gui/$UID/com.farcaster.backfill.embedding384
sudo launchctl remove com.farcaster.backfill.embedding384

# Remove plist file
rm ~/Library/LaunchAgents/com.farcaster.backfill.embedding384.plist
```

2. Install service:
```bash
# Copy and set permissions
cp file.plist ~/Library/LaunchAgents/com.farcaster.backfill.embedding384.plist
chmod 644 ~/Library/LaunchAgents/com.farcaster.backfill.embedding384.plist

# Load service
sudo launchctl bootstrap gui/$UID ~/Library/LaunchAgents/com.farcaster.backfill.embedding384.plist
```

3. Verify service is running:
```bash
launchctl list | grep com.farcaster.backfill.embedding384
# Should show a PID number if running
```

## Commands

Start service:
```bash
sudo launchctl kickstart -k gui/$UID/com.farcaster.backfill.embedding384
```

Stop service:
```bash
sudo launchctl bootout gui/$UID/com.farcaster.backfill.embedding384
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