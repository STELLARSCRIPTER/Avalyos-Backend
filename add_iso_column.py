from app.database import engine
from sqlalchemy import text

with engine.begin() as conn:
    conn.execute(text("""
        ALTER TABLE seismic_events
        ADD COLUMN IF NOT EXISTS country_iso VARCHAR(2);
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_seismic_events_country_iso
        ON seismic_events (country_iso);
    """))
    print("Column country_iso added (or already existed).")

    result = conn.execute(text("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'seismic_events'
        ORDER BY ordinal_position;
    """))
    print("seismic_events columns:")
    for row in result:
        print(f"  {row[0]:<25} {row[1]}")
