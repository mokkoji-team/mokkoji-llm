# OCI deployment

## Request path

```text
Internet (80/443) -> Caddy -> FastAPI (127.0.0.1:8000) -> Ollama (127.0.0.1:11434)
```

Only ports 80 and 443 are public. FastAPI and Ollama remain bound to localhost.

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
