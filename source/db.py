from sqlalchemy import create_engine, Column, Integer, Float, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime

Base = declarative_base()


class TerrainStats(Base):
    __tablename__ = "terrain_stats"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    prng_type = Column(Text)
    heightmap_shape = Column(Text)
    total_values = Column(Integer)

    min = Column(Float)
    max = Column(Float)
    mean = Column(Float)
    median = Column(Float)
    std = Column(Float)
    variance = Column(Float)
    range = Column(Float)

    percentile25 = Column(Float)
    percentile75 = Column(Float)
    interquartile_range = Column(Float)

    skewness = Column(Float)
    kurtosis = Column(Float)

    unique_values = Column(Integer)
    mean_gradient = Column(Float)
    max_gradient = Column(Float)

    histogram_counts = Column(Text)
    histogram_bins = Column(Text)


engine = create_engine("sqlite:///terrain.db")
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)