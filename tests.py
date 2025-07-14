import datetime
import shelve

from bot import ActivityType, Response, ActivityLogResponse, get_llm_response, initialize_daily_db, record_activity_db

import pytest

def test_initialize_daily_db(tmp_path):
    with shelve.open(tmp_path, writeback=True) as db:
        initialize_daily_db(db, datetime.date.today(), "user1")

        print(dict(db))

def test_record_activity_db(tmp_path):
    pts = record_activity_db(ActivityType.workout, 5, "user1", datetime.date.today(), db_path=tmp_path)
    assert pts == 5
    pts = record_activity_db(ActivityType.workout, 3, "user1", datetime.date.today(), db_path=tmp_path)
    assert pts == 1
    pts = record_activity_db(ActivityType.workout, 3, "user1", datetime.date.today(), db_path=tmp_path)
    assert pts == 0


@pytest.mark.expensive
def test_llm_client():
    output = get_llm_response("uspike", "I worked out and did PT")
    assert output == Response(activities=[
        ActivityLogResponse(
            activity_type=ActivityType.workout,
            user_id="uspike",
            points=3,
        )
    ])

@pytest.mark.expensive
def test_llm_client_silly():
    output = get_llm_response("uspike", "I sat at my desk")
    assert output == Response(activities=[])

@pytest.mark.expensive
def test_llm_client_sprints():
    output = get_llm_response("uspike", "Got some sprints done before the rain for lunch")
    assert output == Response(activities=[
        ActivityLogResponse(
            activity_type=ActivityType.workout,
            user_id="uspike",
            points=3,
        )
    ])

@pytest.mark.expensive
def test_llm_client_workout_and_watch():
    output = get_llm_response("uspike", "Lifted this morning and watched Keith’s Hex Basics and Initiations YT video as well as UBC v. Vermont semi-finals.")
    assert output == Response(activities=[
        ActivityLogResponse(
            activity_type=ActivityType.workout,
            user_id="uspike",
            points=3,
        ),
        ActivityLogResponse(
            activity_type=ActivityType.watching,
            user_id="uspike",
            points=2,
        ),
        ActivityLogResponse(
            activity_type=ActivityType.watching,
            user_id="uspike",
            points=2,
        )
    ])

@pytest.mark.expensive
def test_llm_client_film_watching_multiple():
    output = get_llm_response("maximus722", "frisbee film watching (Max and Sarah)")
    assert output == Response(activities=[
        ActivityLogResponse(
            activity_type=ActivityType.watching,
            user_id="maximus722",
            points=2,
            date=datetime.date.today(),
        ),
        ActivityLogResponse(
            activity_type=ActivityType.watching,
            user_id="sarah_guenther",
            points=2,
            date=datetime.date.today(),
        )
    ])