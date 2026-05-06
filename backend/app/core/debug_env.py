import os
from pathlib import Path
from dotenv import load_dotenv

# Replicate the project root resolution from config.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = PROJECT_ROOT / ".env"

print(f"Project Root: {PROJECT_ROOT}")
print(f"Checking for .env at: {ENV_PATH}")
print(f"File exists: {ENV_PATH.exists()}")

load_dotenv(ENV_PATH)

cors_origins = os.getenv("CORS_ORIGINS")
print(f"CORS_ORIGINS from env: {cors_origins}")

if cors_origins:
    origins_list = [o.strip() for o in cors_origins.split(",") if o.strip()]
    print(f"Parsed Origins: {origins_list}")
else:
    print("CORS_ORIGINS not found in environment!")
