# OCI deployment

## Request path

```text
Internet (80/443) -> Caddy -> FastAPI (127.0.0.1:8000) -> Ollama (127.0.0.1:11434)
```

Only ports 80 and 443 are public. FastAPI and Ollama remain bound to localhost.

## Shared account authentication

Create or replace the shared account configuration on the server:

```bash
sudo venv/bin/python scripts/configure_auth.py
sudo ls -l /etc/mokkoji-llm.env
```

The environment file must be owned by root with mode `600`. It contains the
username, a one-way password hash, and a random session signing secret. It must
never be committed to Git.

Apply changed credentials by restarting the application:

```bash
sudo systemctl restart mokkoji-llm
```

The health endpoint stays public for monitoring. `/ask` requires a signed HTTPS
session created at `https://llm.mokkoji.site/login`.

## FastAPI service

```bash
sudo cp deploy/mokkoji-llm.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now mokkoji-llm
```

Check the service:

```bash
sudo systemctl status mokkoji-llm --no-pager
curl -sS 127.0.0.1:8000/health
```

## Caddy

```bash
sudo cp deploy/Caddyfile /etc/caddy/Caddyfile
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl restart caddy
```

Check HTTPS:

```bash
curl -sS https://llm.mokkoji.site/health
```

## Network

The OCI Security List allows inbound TCP ports 80 and 443. The instance firewall
also allows these ports before its final reject rule. Persist firewall changes
with:

```bash
sudo netfilter-persistent save
```

Do not expose ports 8000 or 11434.

## Updating the application

```bash
cd /home/ubuntu/mokkoji-llm
git pull origin main
source venv/bin/activate
python -m pip install -r requirements.txt
sudo systemctl restart mokkoji-llm
```

Application logs:

```bash
sudo journalctl -u mokkoji-llm -n 100 --no-pager
sudo journalctl -u caddy -n 100 --no-pager
```
# Notion 회의록 동기화

`mokkoji-notion-sync.service`는 `/etc/mokkoji-llm.env`의 Notion 연결정보를
읽어 회의록 원문을 `data/notion/`에 한 번 동기화하는 oneshot 서비스다.

```bash
sudo cp deploy/mokkoji-notion-sync.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start mokkoji-notion-sync
sudo journalctl -u mokkoji-notion-sync -n 50 --no-pager
```

# RAG 인덱스 생성

`mokkoji-index.service`는 `data/`의 원문을 임베딩하고, 완성된 새 인덱스만
`storage/`로 교체하는 oneshot 서비스다.

```bash
sudo cp deploy/mokkoji-index.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start mokkoji-index
sudo journalctl -u mokkoji-index -n 50 --no-pager
```
