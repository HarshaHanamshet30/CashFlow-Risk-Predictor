from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql://postgres:#3005Harsha@localhost:5433/cashflow_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
