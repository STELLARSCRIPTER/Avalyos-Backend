import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="postgres",
    user="postgres",
    password="",
)
conn.autocommit = True
cur = conn.cursor()

cur.execute("SELECT 1 FROM pg_roles WHERE rolname='avalyos';")
if cur.fetchone():
    print("User avalyos already exists — updating password")
    cur.execute("ALTER USER avalyos WITH PASSWORD 'avalyos123';")
else:
    print("Creating user avalyos")
    cur.execute("CREATE USER avalyos WITH PASSWORD 'avalyos123';")

cur.execute("SELECT 1 FROM pg_database WHERE datname='avalyos_db';")
if cur.fetchone():
    print("Database avalyos_db already exists")
else:
    print("Creating database avalyos_db")
    cur.execute("CREATE DATABASE avalyos_db OWNER avalyos;")

cur.execute("GRANT ALL PRIVILEGES ON DATABASE avalyos_db TO avalyos;")
print("Granted privileges")

cur.close()
conn.close()
print("--- DONE ---")