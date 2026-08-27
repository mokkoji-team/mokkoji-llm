import getpass

from auth import hash_password


password = getpass.getpass("공용 계정 비밀번호: ")
confirmation = getpass.getpass("비밀번호 확인: ")

if not password:
    raise SystemExit("비밀번호는 비워둘 수 없습니다.")
if password != confirmation:
    raise SystemExit("비밀번호가 일치하지 않습니다.")

print(f"MOKKOJI_AUTH_PASSWORD_HASH={hash_password(password)}")
