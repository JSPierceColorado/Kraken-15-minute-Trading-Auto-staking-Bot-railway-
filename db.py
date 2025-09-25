# =============================
# db.py
# =============================
# Lightweight persistence for markets, buys, and PnL.

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, UniqueConstraint, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func
from settings import SETTINGS

Base = declarative_base()
engine = create_engine(SETTINGS.DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class SeenMarket(Base):
    __tablename__ = "seen_markets"

    id = Column(Integer, primary_key=True)
    symbol = Column(String, unique=True, index=True)
    base = Column(String, index=True)
    quote = Column(String, index=True)
    first_seen = Column(DateTime, server_default=func.now())


class Position(Base):
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True)
    base = Column(String, index=True)
    quote = Column(String, index=True)
    amount = Column(Float)             # base amount currently held (from our buys)
    avg_cost = Column(Float)           # in quote (e.g., USD per base)
    last_updated = Column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (UniqueConstraint("base", "quote", name="uix_base_quote"),)


class BuyLock(Base):
    __tablename__ = "buy_locks"

    id = Column(Integer, primary_key=True)
    base = Column(String, index=True)
    active = Column(Boolean, default=True)  # ensures only one active buy at a time per asset

    __table_args__ = (UniqueConstraint("base", name="uix_buylock_base"),)


class TradeLog(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    side = Column(String)              # BUY/SELL
    symbol = Column(String)
    base = Column(String)
    quote = Column(String)
    price = Column(Float)
    amount = Column(Float)
    notional = Column(Float)
    pnl = Column(Float, nullable=True) # for sells
    ts = Column(DateTime, server_default=func.now())


def init_db():
    Base.metadata.create_all(engine)
