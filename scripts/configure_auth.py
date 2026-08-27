import getpass
import os
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auth import hash_password


ENV_FILE = Path("/etc/mokkoji-llm.env")

username = input("공용 계정 아이디 [mokkoji]: ").strip() or "mokkoji"
if any(character in username for character in ("\n", "\r", "=")):
    raise SystemExit("아이디에 줄바꿈이나 '='를 사용할 수 없습니다.")

password = getpass.getpass("공용 계정 비밀번호: ")
confirmation = getpass.getpass("비밀번호 확인: ")
if not password:
    raise SystemExit("비밀번호는 비워둘 수 없습니다.")
if password != confirmation:
    raise SystemExit("비밀번호가 일치하지 않습니다.")

content = "\n".join(
    (
        f"MOKKOJI_AUTH_USERNAME={username}",
        f"MOKKOJI_AUTH_PASSWORD_HASH={hash_password(password)}",
        f"MOKKOJI_SESSION_SECRET={secrets.token_hex(32)}",
        "",
    )
)

ENV_FILE.write_text(content, encoding="utf-8")
os.chmod(ENV_FILE, 0o600)
print(f"{ENV_FILE}에 공용 계정 '{username}' 설정을 저장했습니다.")
