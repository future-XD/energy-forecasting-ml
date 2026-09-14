from sqlalchemy import create_engine
# from sqlalchemy.pool import NullPool
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Fetch variables
USER = os.getenv("user")
PASSWORD = os.getenv("password")
HOST = os.getenv("host")
PORT = os.getenv("port")
DBNAME = os.getenv("dbname")

# Construct the SQLAlchemy connection string

from sqlalchemy.engine import URL


db_url = URL.create(
    drivername="postgresql+psycopg2",
    username=USER,
    password=PASSWORD,  # Raw password with special characters
    host=HOST,
    port=PORT,  # 6543 for Supabase pooler, 5432 for direct
    database=os.getenv("dbname", "postgres"),
)

engine = create_engine(db_url)
# If using Transaction Pooler or Session Pooler, we want to ensure we disable SQLAlchemy client side pooling -
# https://docs.sqlalchemy.org/en/20/core/pooling.html#switching-pool-implementations
# engine = create_engine(DATABASE_URL, poolclass=NullPool)
print(db_url)
# Test the connection
try:
    with engine.connect() as connection:
        print("Connection successful!")
except Exception as e:
    print(f"Failed to connect: {e}")
