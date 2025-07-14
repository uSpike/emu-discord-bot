from enum import Enum
import asyncio
import logging
import datetime
import os
import sqlite3

from pydantic import BaseModel

from dotenv import load_dotenv
import openai
import discord

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

logging.getLogger("httpx").setLevel(logging.WARNING)

# Load environment variables from .env file
load_dotenv()

llm_client = openai.Client(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORGANIZATION_ID")
)

DB_FILE = "activity_log.db"


class ActivityType(Enum):
    none = "none"
    throwing = "throwing"
    workout = "workout"
    watching = "watching"
    bonding = "bonding"


max_activity_points = {
    ActivityType.workout: 6,
    #ActivityType.throwing no limit
    ActivityType.watching: 2,
    #ActivityType.bonding no limit
}

activity_points = {
    ActivityType.workout: 3,
    ActivityType.throwing: 2,  # per 15 minutes
    ActivityType.watching: 2,  # per game or media watched
    ActivityType.bonding: 2,  # per person involved
}

class ActivityLogResponse(BaseModel):
    activity_type: ActivityType
    user_id: str  # Discord username
    date: datetime.date = datetime.date.today()
    reason: str


class Response(BaseModel):
    activities: list[ActivityLogResponse]
    text_response: str | None = None


prompt_text = f"""
You are a helpful discord bot that helps users log their activities for a challenge.
The users are members of an ultimate frisbee team and log their activities to earn points for the month.
The user may provide a description of their activity, and you will determine type of activity and the
number of points they should receive for it.
You are receiving all the messages from the #challenges channel in the team's Discord server.
The activities are logged in a simple database based on your `Response` object.
Some messages may not be activities, and in that case you should respond with an empty list of activities.

# Activity Types and Points

- Workouts (`ActivityType.workout`)
  - running, biking, weightlifting, physical therapy, etc. MUFA (league) games and pickup ultimate count.
  - A message such as "PT" (physical therapy), "lifting", "workout" should be logged as a workout.
  - Dog walking/running does count as a workout.
- Throwing (`ActivityType.throwing`)
  - throwing a frisbee either alone or with others.
  - each activity should be logged for each 15 minutes of throwing.
- Watching Frisbee (`ActivityType.watching`)
  - watching ultimate frisbee games, whether on youtube/TV or in person.
  - also includes watching film, breakdowns, tutorials.
- Team Bonding (`ActivityType.bonding`)
  - Non-frisbee activities that help the team bond.  Should only apply to other humans on the team.
  - Examples: team dinners, game nights, etc.
  - Everyone involved in the activity should be recorded separately.
  - The author should @ everyone involved in the activity in the message.
  - This doesn't count for messages that are helping/sympathetic to other players in the channel.  It must be
    an activity in the real world that people are doing together.

If the message is clarifying a previous activity, you should not log it again.

# Response Format

Separate each activity into separate entries in the response `activities`.
If there are multiple people involved in the activity, each person should get points.

If the activity date is not specified in the message,
use today's date.  If the message specifies a date, such as "yesterday", "two days ago", etc., use that date instead.

Use the player's Discord username as the `user_id` field in the response.

Add a short reason for the activity / number of points in the `reason` field.

If there is no activity in the message, return an activity with `ActivityType.none` and `points` set to 0.  Set the `reason` to why you think it is not an activity.

# Additional Notes

These players like to joke around, so they may use emojis or other playful language in their messages.
If they make a joke about an activity such as petting their dog, you should not log it as a real activity.

Sometimes a user may send an abbreviated message such as "PT", "lifting", "2.8 miles" without any context.
Assume that these are valid activities and log them accordingly.

# Text response

You may return a response in the `text_response` field to reply to the user.
Do not respond to a user unless prompted to do so.
"""

previous_response_id = None  # This will be set to the ID of the previous response if needed

def get_llm_response(message: discord.Message) -> Response | None:
    """
    Call the OpenAI API to get the activity type and points based on the message.
    """
    global previous_response_id
    #previous_response_id = None

    instructions = prompt_text if previous_response_id is None else None

    message_text = f"""
    User ID: "{message.author.name}"
    Message date: {message.created_at.strftime('%Y-%m-%d %H:%M:%S')}
    Message content: "{message.content}"
    """

    response = llm_client.responses.parse(
        model="gpt-4o-mini",
        instructions=instructions,
        input=message_text,
        text_format=Response,
        previous_response_id=previous_response_id,
    )

    if response.error:
        log.error(f"Error processing message: {message}")
        log.error(f"Error: {response.error}")
        return None

    previous_response_id = response.id

    # log tokens
    log.info(f"LLM usage: {response.usage}")
    # 4o is $2.50 per million input tokens
    input_cost = response.usage.input_tokens * 2.5e-6
    log.debug(f"LLM input cost: ${input_cost:.10f}")
    # 4o is $10 per million output tokens
    output_cost = response.usage.output_tokens * 10e-6
    log.debug(f"LLM output cost: ${output_cost:.10f}")

    return response.output_parsed


def record_activity_db(activity: ActivityLogResponse, message_id: int) -> int:
    """
    Record activity in the database.  Don't allow more than the maximum points for the activity for the day.
    Return the new points allocated for the activity.
    """

    points = activity_points.get(activity.activity_type, 0)

    limit = max_activity_points.get(activity.activity_type, None)
    if limit is None:
        new_points = points
    else:
        current_points = sum(get_db_points_date_user(activity.activity_type, activity.date, activity.user_id))
        new_points = min(limit - current_points, points)
        log.info(f"Current points: {current_points}, New points: {new_points}, Limit: {limit}")

    add_db_activity(activity, new_points, message_id)
    return new_points


def initialize_db() -> None:
    """
    Initialize the database
    """
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS activities (
                id INTEGER PRIMARY KEY,
                user_id TEXT NOT NULL,
                date TEXT NOT NULL,
                message_id INTEGER NOT NULL,
                activity_type TEXT NOT NULL,
                points INTEGER NOT NULL
            )
        ''')

def get_db_points_date_user(activity: ActivityType, date: datetime.date, user_id: str) -> list[int]:
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT points FROM activities
            WHERE activity_type = ? AND date = ? AND user_id = ?
        ''', (activity.value, date.isoformat(), user_id))
        return [s[0] for s in cursor.fetchall()]

def add_db_activity(activity: ActivityLogResponse, points: int, message_id: int) -> None:
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO activities (user_id, date, message_id, activity_type, points)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            activity.user_id,
            activity.date.isoformat(),
            message_id,
            activity.activity_type.value,
            points
        ))
        conn.commit()


def get_db_message_id(message_id: int) -> int | None:
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id FROM activities WHERE message_id = ?
        ''', (message_id,))
        result = cursor.fetchone()
        return result[0] if result else None


activity_emoji_map = {
    ActivityType.workout: "🏋️",
    ActivityType.throwing: "🥏",
    ActivityType.watching: "📺",
    ActivityType.bonding: "🤝",
}
new_points_to_emoji_map = {
    0: "0️⃣",
    1: "1️⃣",
    2: "2️⃣",
    3: "3️⃣",
    4: "4️⃣",
    5: "5️⃣",
    6: "6️⃣",
}

async def print_table_of_scores() -> None:
    """
    Print a markdown table of scores for all users.
    """
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT user_id, SUM(points) FROM activities
            GROUP BY user_id
            ORDER BY SUM(points) DESC
        ''')
        rows = cursor.fetchall()

    if not rows:
        log.info("No activities found.")
        return

    content = ""

    content += "| User ID | Points |\n"
    content += "|---------|--------|\n"
    for row in rows:
        content += f"| {row[0]} | {row[1]} |\n"

    print(content)

intents = discord.Intents.default()
intents.message_content = True

bot_client = discord.Client(intents=intents)

# get messages from the #challenges channel
@bot_client.event
async def on_ready():
    log.info(f'Logged in as {bot_client.user} (ID: {bot_client.user.name})')

    channel = bot_client.get_channel(1378516451987816538) # challenges channel ID
    async for message in channel.history(limit=None, after=datetime.datetime(2025, 7, 1)):
        await parse_message(message)

@bot_client.event
async def on_message(message: discord.Message):
    response = await parse_message(message)
    if response is not None and response.text_response:
        try:
            await message.reply(response.text_response)
        except Exception as e:
            log.exception(f"An error occurred while replying to message {message.id}: {e}")


async def parse_message(message: discord.Message) -> None:
    if message.author == bot_client.user:
        return

    if message.channel.name != "challenges":
        return

    if not message.content.strip():
        return

    if get_db_message_id(message.id) is not None:
        log.info(f"!!! Activity with message ID {message.id} already exists in the database. Skipping.")
        return


    for user in message.mentions:
        mention_str = f"<@!{user.id}>" if f"<@!{user.id}>" in message.content else f"<@{user.id}>"
        message.content = message.content.replace(mention_str, f"@{user.name}")

    response = await asyncio.to_thread(
        get_llm_response,
        message
    )

    for activity in response.activities:
        if activity.activity_type == ActivityType.none:
            log.info(f">>> Skipping activity with type 'none' for user {activity.user_id}, reason: {activity.reason}")
            add_db_activity(activity, 0, message.id)
            continue

        new_points = await asyncio.to_thread(
            record_activity_db,
            activity,
            message.id,
        )
        log.info(f"*** Activity logged: {activity.user_id} {activity.date} => {activity.activity_type} + {new_points} points.  Reason: {activity.reason}")

        # add emoji reaction to the message
        try:
            activity_emoji = activity_emoji_map.get(activity.activity_type, "❓")
            #new_points_emoji = new_points_to_emoji_map.get(new_points, "❓")
            await message.add_reaction(activity_emoji)
            #await message.add_reaction(new_points_emoji)
        except Exception:
            log.exception(f"Failed to add reaction.")

    return response

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "show":
            # show stats from db
            with sqlite3.connect(DB_FILE) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM activities')
                rows = cursor.fetchall()
                for row in rows:
                    print(row)
            exit(0)
        elif sys.argv[1] == "scores":
            # print scores
            asyncio.run(print_table_of_scores())
            exit(0)

    initialize_db()

    try:
        bot_client.run(os.environ["DISCORD_BOT_TOKEN"])
    except discord.LoginFailure:
        log.exception(f"Failed to login to Discord")
    except Exception:
        log.exception(f"An error occurred")
        exit(1)
