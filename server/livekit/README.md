# Call token server

Mints LiveKit room tokens for the iOS Call tab. Identity is a **username** and a **room code**. This process does not receive video or audio, and it does not change the recognition server.

Do not expose it to the public internet.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export LIVEKIT_URL=wss://YOUR_PROJECT.livekit.cloud
export LIVEKIT_API_KEY=...
export LIVEKIT_API_SECRET=...
python token_server.py
```

`POST /v1/token` with `{"room":"<code>","username":"<name>"}` returns `{url, token, room, identity}`.

`GET /health` reports whether the LiveKit env vars are set.

`python test_token_server.py` checks username rules and token minting.

Full two-phone steps are in the root README under **In-app Call tab**. Copy `.env.example` for the variable names. `.env` is gitignored.
