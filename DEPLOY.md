# Deploying to Render via GitHub

End result: two public links.

| Page | URL |
|---|---|
| Chatbot (link this from bmsit.ac.in) | `https://<your-service>.onrender.com/chatbot` |
| Admin dashboard | `https://<your-service>.onrender.com/admin` |

Render decides `<your-service>` from the service name. With `render.yaml` as
committed, the name is `bmsit-ai-agent`, so the links will be:

- `https://bmsit-ai-agent.onrender.com/chatbot`
- `https://bmsit-ai-agent.onrender.com/admin`

If that name is already taken, Render appends a suffix. The exact URL is shown
at the top of your service page after the first deploy.

---

## Before you start

Two things about this app shape the whole deployment:

**It is stateful.** The FAISS index, the page text and the delta registry all
live on local disk in `data/`. Render wipes the filesystem on every deploy and
restart, so a **persistent disk is required**. Without one you lose the entire
knowledge base each time you push. `render.yaml` mounts a 2 GB disk at
`data/`; a disk needs a paid instance type (Starter or above).

**It must run exactly one worker.** The vector index is held in process memory
and the 7 AM scheduler runs inside the web process. Two workers would each hold
a diverging copy of the index and both run the daily crawl. The start command in
`render.yaml` pins `--workers 1` on purpose. Do not raise it.

> **`/admin` has no password.** You asked for it without a login, so anyone who
> knows the URL can upload documents, trigger crawls, change the API key and
> delete the whole knowledge base. If you later want it locked down, say so and
> I will add it back.

---

## Step 1 — Push to GitHub

From the project folder:

```cmd
git init
git add .
git commit -m "BMSIT AI agent with delta ingestion"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Check before pushing that `.env` is **not** included:

```cmd
git status --short
```

`.gitignore` already excludes `.env` and `data/`. Your API keys live in `.env`,
so if it ever appears in `git status`, stop and remove it from the commit.

---

## Step 2 — Create the service on Render

1. Sign in at [render.com](https://render.com) and connect your GitHub account.
2. **New → Blueprint**.
3. Pick your repository. Render reads `render.yaml` and proposes one web service.
4. You are prompted for the two `sync: false` secrets. Paste your keys now, or
   leave them and add them under Environment afterwards.
5. Click **Apply**.
6. Choose your instance type — see *One blueprint, both tiers* below.

The first build takes a few minutes: `faiss-cpu` and `numpy` are large wheels.

---

## Step 3 — Add your API keys

Render will not read secrets from git. In the service, open **Environment** and
set:

| Key | Value |
|---|---|
| `GEMINI_API_KEY` | your key, used for chat answers |
| `GEMINI_API_KEYS` | `key1,key2,key3` — the pool used for embeddings |

Every other variable is already declared in `render.yaml`.

More keys in `GEMINI_API_KEYS` means more daily embedding quota for the first
bulk load. Get them from [Google AI Studio](https://aistudio.google.com/apikey);
separate Google accounts give separate quotas.

Save. Render restarts the service.

---

## Step 4 — Confirm it is alive

Open `https://<your-service>.onrender.com/health`. You should see:

```json
{"status":"ok","chunks":0,"vectors":0,"sources":0,...}
```

`chunks: 0` is expected. Nothing has been indexed yet.

---

## Step 5 — Load the knowledge base

Open `https://<your-service>.onrender.com/admin` and press **Scrape Now**.

Watch the status line. A full crawl of the BMSIT site takes 10–25 minutes, and
embedding is limited by your daily quota. One of two things happens:

- **Indexed** — everything finished. You are done.
- **Partially Indexed** — the daily token budget or your key quota was reached.
  Everything embedded so far is saved. Press **Scrape Now** again tomorrow (or
  after adding more keys) and it continues exactly where it stopped. Repeat
  until it says Indexed.

Then, every day at 07:00 IST, the app re-crawls and embeds **only pages whose
content changed**. That normally costs a tiny fraction of a day's quota.

---

## Step 6 — Link it from the college site

Point an "Ask BMSIT AI" button at:

```
https://<your-service>.onrender.com/chatbot
```

That page contains only the chatbot: no admin controls, mobile responsive, with
source citations under each answer.

---

## One blueprint, both tiers

There is a single `render.yaml`. It works on free and paid because it does not
declare two fields:

- **`plan` is omitted.** Render then *retains whatever compute plan the service
  currently has*, so the choice you make in the dashboard survives every future
  blueprint sync. Had it been declared, a sync would overwrite your choice.
- **`disk` is omitted.** Free instances cannot mount one, and a declared disk
  would make the blueprint invalid on the free tier.

So plan and disk are dashboard decisions, and everything else is versioned in
git. Nothing to rename, no second file, no code changes when you switch.

### To run free (testing)

Right after the blueprint creates the service, open **Settings → Instance Type**
and select **Free**. A brand-new service defaults to `0.5c-512mb`, which is a
paid plan, so do this promptly. Render retains Free from then on.

The committed env values are already sized for a 512 MB free instance: an
800-page crawl at depth 4, a 400,000-token daily ceiling and 600 chunks per run.

### To run paid (production)

1. **Settings → Instance Type → Starter** (or higher).
2. **Disks → Add Disk**:
   - Name: `bmsit-data`
   - Mount path: `/opt/render/project/src/data`
   - Size: `2` GB
3. Raise four values under **Environment** for the full site:

   | Variable | Free | Paid |
   |---|---|---|
   | `SCRAPE_MAX_PAGES` | 800 | 3000 |
   | `SCRAPE_DEPTH` | 4 | 5 |
   | `EMBED_DAILY_TOKEN_BUDGET` | 400000 | 800000, or 0 for a paid API key |
   | `EMBED_MAX_CHUNKS_PER_RUN` | 600 | 0 (unlimited) |

Adding the disk mounts fresh storage over `data/`, so the knowledge base starts
empty once. Press **Scrape Now** again; after that it persists across deploys.

> If a later blueprint sync ever detaches the disk, re-add it with the same name
> and mount path. I have not been able to verify Render's behaviour on that edge
> case, so check **Disks** after your first sync following the upgrade.

### What to expect on the free tier

1. First request after idle takes 30–60 seconds. The instance spins down after
   15 minutes of no traffic and cold-starts on the next visit.
2. **Keep the `/admin` tab open while a crawl runs.** The dashboard polls for
   status, and that traffic is what keeps the instance awake. Close the tab and
   the instance may sleep and kill the crawl part-way. Anything already embedded
   in a committed slice is saved, but on free tier it is lost at sleep anyway.
3. The 07:00 IST daily job will not fire reliably, because the process is
   usually asleep at 7 AM. Not a problem for testing; it is a real problem for
   production, and one of the reasons to move to Starter.
4. 512 MB of RAM is enough. Roughly 8,000 vectors at 768 dimensions is about
   25 MB; faiss and numpy dominate the footprint. `EMBED_MAX_CONCURRENCY` is set
   to 1 to keep peak memory low.

### A good free-tier test sequence

1. Deploy and open `/health` — expect `chunks: 0`.
2. Open `/admin`, press **Scrape Now**, leave the tab open, wait for Indexed.
3. Open `/chatbot` and ask about admissions, hostels and placements. Check that
   answers cite sources rather than saying no information is available.
4. Upload a PDF from the admin page, then confirm at `/health` that the chunk
   count **grew** rather than being replaced.
5. Press **Scrape Now** again. It should report most pages skipped as unchanged
   and embed almost nothing — that is the delta logic working.

If all five pass, switch the instance type to Starter and add the disk.

### Why production needs the paid plan

Two blockers, both structural rather than about speed: no persistent disk means
the knowledge base is destroyed on every restart, and a sleeping instance cannot
run the 07:00 crawl. At around $7.50/month for a Starter instance plus a 2 GB
disk, that is the one cost you cannot design around.

---

## Troubleshooting

**Build fails on faiss-cpu.** Confirm `PYTHON_VERSION` is `3.13.7` in the
environment. Very old Python versions have no matching wheel.

**Deploy succeeds but every page 500s.** Check the logs for
`Configuration is incomplete`. The app refuses to start with missing settings and
lists exactly which ones. Add them under Environment.

**Knowledge base is empty after a deploy.** The disk is not mounted. Confirm
under **Disks** that `bmsit-data` is mounted at
`/opt/render/project/src/data`.

**Scrape times out.** The start command needs `--timeout 600`; the default 30s
kills the request mid-crawl.

**Answers say there is no information.** Check
`https://<your-service>.onrender.com/api/admin/embeddings`. If `keys_usable` is
`0`, quota is spent. If `pending_reembed` is large, vectors are still local
fallbacks — `POST /api/admin/reembed` upgrades them once quota returns.

---

## Useful endpoints once deployed

| Endpoint | Purpose |
|---|---|
| `/health` | chunk, vector and source counts |
| `/api/admin/status` | RAG, scheduler, registry and store stats |
| `/api/admin/embeddings` | per-key usability, token budget, cache hit rate |
| `/api/admin/registry/web-bmsit` | per-page hash, vector ids, tokens |
| `POST /api/admin/scrape` | run a delta crawl now |
| `POST /api/admin/reembed` | upgrade fallback vectors |
| `POST /api/admin/reset` | wipe everything (send `{"confirm":"RESET"}`) |
