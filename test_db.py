import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="avalyos_db",
    user="avalyos",
    password="avalyos123",
)
cur = conn.cursor()
cur.execute("SELECT version();")
print("Connected successfully!")
print(cur.fetchone()[0])
cur.close()
conn.close()
