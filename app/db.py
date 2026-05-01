"""MongoDB connection + `cv_events` collection.

Each `/cv/frame` request appends one document. Aggregation endpoints
read recent rows. Indexes match the two query shapes:
  (meeting_id, student_id, ts) — per-student rolling window
  (meeting_id, ts)             — class-level rolling window
"""
from pymongo import MongoClient
from .config import MONGO_URI, MONGO_DB

_client = MongoClient(MONGO_URI)
db = _client[MONGO_DB]
events_col = db["cv_events"]
events_col.create_index([("meeting_id", 1), ("student_id", 1), ("ts", -1)])
events_col.create_index([("meeting_id", 1), ("ts", -1)])
