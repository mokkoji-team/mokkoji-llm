import getpass
import hashlib
import secrets


password = getpass.getpass("공용 계정 비밀번호: ")
confirmation = getpass.getpass("비밀번호 확인: ")

if not password:
    raise SystemExit("비밀번호는 비워둘 수 없습니다.")
if password != confirmation:
    raise SystemExit("비밀번호가 일치하지 않습니다.")

salt = secrets.token_bytes(16)
password_hash = hashlib.scrypt(
    password.encode("utf-8"),
    salt=salt,
    n=2**14,
    r=8,
    p=1,
    dklen=32,
).hex()

print(f"MOKKOJI_AUTH_PASSWORD_HASH={salt.hex()}:{password_hash}")
