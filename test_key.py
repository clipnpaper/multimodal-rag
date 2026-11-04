import os
from dotenv import load_dotenv

# 1) .env 파일을 읽어서 환경변수로 등록합니다.
load_dotenv()

# 2) 환경변수에서 키를 꺼냅니다. 없으면 None이 반환됩니다.
api_key = os.getenv("UNSTRUCTURED_API_KEY")
api_url = os.getenv("UNSTRUCTURED_API_URL")

# 3) 키 전체를 출력하면 유출 위험이 있으므로 '앞 4자리만' 확인합니다.
if api_key:
    print(f"✅ 키 로딩 성공! (앞 4자리: {api_key[:4]}****)")
    print(f"✅ API URL: {api_url}")
else:
    print("❌ 키를 못 찾았습니다. 아래를 확인하세요.")
    print("   - .env 파일이 '실행하는 py 파일과 같은 폴더'에 있는가?")
    print("   - 파일명이 'env.txt'가 아니라 정확히 '.env'인가?")
    print("   - 변수명 철자가 UNSTRUCTURED_API_KEY가 맞는가?")
