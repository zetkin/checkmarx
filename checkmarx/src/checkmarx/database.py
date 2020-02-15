from datetime import datetime, timedelta

from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class FormDescription(Base):

    __tablename__ = "form_description"

    id = Column(Integer, primary_key=True)
    page_height = Column(Integer)
    page_width = Column(Integer)
    qr_height = Column(Integer)
    qr_width = Column(Integer)
    qr_x_offset = Column(Integer)
    qr_y_offset = Column(Integer)
    checkbox_height = Column(Integer)
    checkbox_width = Column(Integer)


class Checkbox(Base):

    __tablename__ = "checkbox"

    id = Column(Integer, primary_key=True)
    form_description_id = ForeignKey("form_description.id")
    x = Column(Integer)
    y = Column(Integer)
    title = Column(String)


# Dependency
def get_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()


class DB:
    def __init__(self, connection_string):
        pass


# connection
engine = create_engine("postgresql://postgres:mysecretpassword@localhost:5432/db01")

# create metadata
Base.metadata.create_all(engine)

# create session
# Session = sessionmaker(bind=engine)
# session = Session()

# insert data
# tag_cool = Tag(name="cool")
# tag_car = Tag(name="car")
# tag_animal = Tag(name="animal")

# session.add_all([tag_animal, tag_car, tag_cool])
# session.commit()
