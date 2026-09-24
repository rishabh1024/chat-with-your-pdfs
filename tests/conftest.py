import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

_TEST_ENVIRONMENT = {
    "MONGODB_USER": "test-user",
    "MONGODB_PASSWORD": "test-password",
    "MONGODB_CLUSTER_ID": "test-cluster",
    "MONGO_DB_NAME": "test-database",
    "SUPABASE_KEY": "test-key",
    "SUPABASE_URL": "https://example.supabase.co",
    "SUPABASE_AUTH_URL": "https://example.supabase.co",
    "SUPABASE_DB_URL": "postgresql://test:test@localhost/test",
    "OPENROUTER_API_KEY": "test-openrouter-key",
}

for name, value in _TEST_ENVIRONMENT.items():
    os.environ.setdefault(name, value)
