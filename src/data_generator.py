"""Generate a synthetic mixed personal + document corpus."""
import json
import uuid
import random
from datetime import datetime, timedelta
from pathlib import Path
from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)

DIARY_TEMPLATES = [
    "Today I felt {mood}. I spent the morning {activity} and later had a conversation with {person} about {topic}.",
    "Woke up at {time} today. The weather was {weather}. I plan to {plan} this evening.",
    "Had a {quality} day at work. {event} happened during the afternoon meeting. Feeling {mood} about it.",
    "Spent some time {activity} today. It made me feel {mood}. Tomorrow I need to {plan}.",
    "Reflecting on the week: {reflection}. Next week I want to focus on {topic}.",
]

MEETING_TEMPLATES = [
    "Attendees: {attendees}. Agenda: {agenda}. Decisions: {decisions}. Action items: {actions}.",
    "Sprint review with {attendees}. Key updates: {updates}. Blockers: {blockers}. Next steps: {actions}.",
    "1:1 with {person}. Discussed {topic}. Feedback: {feedback}. Follow-up: {actions}.",
    "Team sync — {attendees}. {event} was the main highlight. We decided to {decisions}.",
]

EMAIL_TEMPLATES = [
    "Subject: {subject}. Hi {person}, I wanted to reach out regarding {topic}. {body} Let me know your thoughts. Regards, {sender}.",
    "Subject: {subject}. Dear team, Please find the updates on {topic}. {body} Best, {sender}.",
    "Subject: {subject}. Hey {person}, Quick note about {topic}. {body} Cheers, {sender}.",
]

REMINDER_TEMPLATES = [
    "Don't forget to {task} by {deadline}. Priority: {priority}.",
    "Reminder: {event} at {time} on {deadline}. Notes: {notes}.",
    "Shopping list: {items}. Pick up before {deadline}.",
    "Deadline approaching — {task} due {deadline}. Status: {status}.",
]

ARTICLE_TEMPLATES = [
    "Title: {title}. {title} is an emerging trend in {field}. Experts say that {opinion}. In practice, {practice}. Conclusion: {conclusion}.",
    "Title: {title}. Recent studies show that {finding}. This has implications for {field}. {opinion}. Practical tip: {practice}.",
    "Title: {title}. Many people struggle with {topic}. According to research, {finding}. Recommendation: {conclusion}.",
]


def _fill(template: str) -> str:
    mood = random.choice(["happy", "stressed", "excited", "tired", "optimistic", "anxious"])
    activity = random.choice(["reading", "jogging", "coding", "cooking", "meditating", "gardening"])
    person = fake.first_name()
    topic = random.choice(["career growth", "health habits", "travel plans", "family", "side projects"])
    time = f"{random.randint(5, 9)} AM"
    weather = random.choice(["sunny", "rainy", "cloudy", "windy"])
    plan = random.choice(["finish the report", "call mom", "prepare dinner", "review PRs"])
    quality = random.choice(["productive", "challenging", "relaxing", "hectic"])
    event = random.choice([
        "a surprise client request", "the demo crashed", "we hit the milestone",
        "a new hire joined", "budget approval came through"
    ])
    reflection = random.choice([
        "it was a rollercoaster", "I learned a lot", "I need better boundaries",
        "small wins compound"
    ])
    attendees = ", ".join(fake.first_name() for _ in range(random.randint(2, 5)))
    agenda = random.choice(["Q3 planning", "architecture review", "retrospective", "product roadmap"])
    decisions = random.choice([
        "migrate to the new stack", "hire two more engineers", "delay the launch by one week",
        "invest in automation"
    ])
    actions = random.choice([
        "update documentation", "schedule follow-up", "prepare budget estimate",
        "send recap email", "create JIRA tickets"
    ])
    updates = random.choice(["backend refactor 80% done", "CI pipeline green", "new design system live"])
    blockers = random.choice(["waiting for API keys", "third-party integration delayed", "none"])
    feedback = random.choice(["keep shipping", "improve communication", "great progress this quarter"])
    subject = random.choice(["Quick update", "Action required", "FYI", "Meeting notes", "Project status"])
    body = random.choice([
        "Here is the summary you requested.",
        "Things are moving faster than expected.",
        "We should sync early next week.",
        "Let me know if you need any clarifications."
    ])
    sender = fake.first_name()
    task = random.choice(["submit expense report", "book flights", "review contract", "update resume"])
    deadline = (datetime.now() + timedelta(days=random.randint(1, 14))).strftime("%Y-%m-%d")
    priority = random.choice(["high", "medium", "low"])
    event_r = random.choice(["dentist appointment", "team lunch", "parent-teacher meeting", "car service"])
    notes = random.choice(["bring insurance card", "RSVP required", "prepare slides"])
    items = ", ".join(fake.word() for _ in range(random.randint(3, 6)))
    status = random.choice(["in progress", "not started", "blocked"])
    title = fake.catch_phrase()
    field = random.choice(["AI", "healthcare", "finance", "education", "climate tech"])
    opinion = random.choice([
        "early adopters are seeing 2x efficiency gains",
        "regulation is still catching up",
        "the hype may exceed current capabilities"
    ])
    practice = random.choice([
        "start with a pilot project before scaling",
        "invest in training your team first",
        "measure outcomes rigorously"
    ])
    conclusion = random.choice([
        "the future looks promising but patience is required",
        "incremental adoption yields the best ROI",
        "staying informed is the best strategy"
    ])
    finding = random.choice([
        "sleep quality directly correlates with productivity",
        "remote teams communicate 30% more in writing",
        "micro-breaks reduce burnout significantly"
    ])

    return template.format(
        mood=mood, activity=activity, person=person, topic=topic, time=time,
        weather=weather, plan=plan, quality=quality, event=event,
        reflection=reflection, attendees=attendees, agenda=agenda,
        decisions=decisions, actions=actions, updates=updates,
        blockers=blockers, feedback=feedback, subject=subject,
        body=body, sender=sender, task=task, deadline=deadline,
        priority=priority, event_r=event_r, notes=notes, items=items,
        status=status, title=title, field=field, opinion=opinion,
        practice=practice, conclusion=conclusion, finding=finding
    )


def _make_doc(doc_type: str, template: str) -> dict:
    content = _fill(template)
    title_map = {
        "diary": f"Diary Entry — {fake.date_this_year()}",
        "meeting": f"Meeting Notes — {fake.bs().title()}",
        "email": f"Email — {fake.catch_phrase()}",
        "reminder": f"Reminder — {fake.word().title()}",
        "article": f"Article — {fake.catch_phrase()}",
    }
    return {
        "id": str(uuid.uuid4()),
        "type": doc_type,
        "title": title_map[doc_type],
        "content": content,
        "created_at": fake.date_time_this_year().isoformat(),
        "tags": [doc_type] + [fake.word() for _ in range(2)],
    }


def generate_corpus(n_per_type: int = 10) -> list[dict]:
    docs = []
    for _ in range(n_per_type):
        docs.append(_make_doc("diary", random.choice(DIARY_TEMPLATES)))
        docs.append(_make_doc("meeting", random.choice(MEETING_TEMPLATES)))
        docs.append(_make_doc("email", random.choice(EMAIL_TEMPLATES)))
        docs.append(_make_doc("reminder", random.choice(REMINDER_TEMPLATES)))
        docs.append(_make_doc("article", random.choice(ARTICLE_TEMPLATES)))
    random.shuffle(docs)
    return docs


def save_corpus(path: Path, corpus: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    from src.config import CORPUS_PATH
    corpus = generate_corpus()
    save_corpus(CORPUS_PATH, corpus)
    print(f"Generated {len(corpus)} documents → {CORPUS_PATH}")
