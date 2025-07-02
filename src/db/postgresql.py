from sqlmodel import SQLModel, Field, create_engine, Session, select
from sqlalchemy.dialects.postgresql import JSON
from typing import Optional, List
from sqlalchemy import Column, DateTime
from datetime import datetime
from dotenv import load_dotenv
from uuid import uuid4, UUID
import os

env_path = os.path.join(os.path.dirname(__file__), '../../.env')
load_dotenv(env_path)

psql_username = os.getenv('POSTGRES_USERNAME')
psql_password = os.getenv('POSTGRES_PASSWORD')
psql_host = os.getenv('POSTGRES_HOST')
psql_port = os.getenv('POSTGRES_PORT')
psql_database_name = os.getenv("POSTGRES_DATABASE_NAME")

DATABASE_URL = f"postgresql://{psql_username}:{psql_password}@{psql_host}:{psql_port}/{psql_database_name}"
engine = create_engine(DATABASE_URL, echo=True, future=True)


class Proctors(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True, nullable=False)
    tabSwitched: int = Field(default=0, nullable=False)
    outOfFrame: int = Field(default=0, nullable=False)
    externalMonitorDetected: bool = Field(default=False, nullable=False)
    fullScreenExited: bool = Field(default=False, nullable=False)
    createdAt: datetime = Field(nullable=False)
    updatedAt: datetime = Field(nullable=False)
    videoUrl: Optional[str] = Field(default=None, max_length=255)
    videoChunks: List[dict] = Field(sa_column=Column(JSON), default=[])
    candidatePicture: Optional[str] = Field(default=None, max_length=255)
    multiplePeople: int = Field(default=0, nullable=False)
    bannedObjects: int = Field(default=0, nullable=False)
    faceVerification: bool = Field(default=True, nullable=False)
    headPoseDetection: bool = Field(default=False, nullable=False)
    eyeTracking: bool = Field(default=False, nullable=False)
    proctored: bool = Field(default=False, nullable=False)


def extract_unproctored_record(session):
    statement = select(Proctors).where(Proctors.videoUrl != None,
                                       Proctors.candidatePicture != None,
                                       Proctors.proctored == False)
    records = session.exec(statement).first()
    return records

def update_proctored_record(session, record, updated_info):
    MULTIPLE_PEOPLE, BANNED_OBJECTS, FACE_VERIFICATION, HEADPOSE_DETECTION, EYE_TRACKING = updated_info
    record.updatedAt = datetime.utcnow()
    record.multiplePeople = MULTIPLE_PEOPLE
    record.bannedObjects = BANNED_OBJECTS
    record.faceVerification = FACE_VERIFICATION
    record.headPoseDetection = HEADPOSE_DETECTION
    record.eyeTracking = EYE_TRACKING
    record.proctored = True

    session.add(record)
    session.commit()