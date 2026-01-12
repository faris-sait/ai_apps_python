#!/usr/bin/env python3
"""Helper script to initialize DB and verify Supabase/Postgres connection."""
import os
from db import init_db, test_connection

print("DATABASE_URL:", os.environ.get("DATABASE_URL"))
print("Initializing DB (create tables)...")
init_db()
print("Running connection test...")
ok = test_connection()
if ok:
    print("✅ DB connection OK")
else:
    print("❌ DB connection FAILED - check DATABASE_URL and network")

print("Done.")
