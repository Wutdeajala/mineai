from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)

with engine.connect() as connection:
    result = connection.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema='public';"))
    print("Connected successfully. Tables found:")
    for row in result:
        print(f" - {row[0]}")