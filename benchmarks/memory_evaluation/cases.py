"""Original conversational probes organized by long-term memory capability."""

from __future__ import annotations

from typing import Dict, List

CATEGORIES = (
    "direct_recall",
    "composed_implicit",
    "changed_fact",
    "conflict_resolution",
    "abstention",
)


def _make_case(category: str, index: int, sessions: List[str], question: str,
               answer: str, distractors: List[str]) -> Dict[str, object]:
    options = [answer, *distractors]
    shift = index % len(options)
    options = options[shift:] + options[:shift]
    return {
        "case_id": f"{category}-{index + 1:02d}",
        "conversation_id": f"{category}-{index + 1:02d}",
        "category": category,
        "sessions": sessions,
        "question": question,
        "options": options,
        "answer": answer,
    }


def build_cases() -> List[Dict[str, object]]:
    cases: List[Dict[str, object]] = []

    direct = [
        (["User: Casey usually orders oolong tea at Harbor Cafe.",
          "User: Casey is trying a new pastry this week."],
         "What tea does Casey usually order?", "oolong tea",
         ["matcha", "rooibos", "earl grey"]),
        (["User: Morgan moved to Reno last spring.",
          "User: Morgan is still settling into the new apartment."],
         "Which city did Morgan move to?", "Reno",
         ["Boise", "Denver", "Phoenix"]),
        (["User: Riley prefers appointments in the late afternoon.",
          "Assistant: I will keep that preference in mind."],
         "What time of day works best for Riley?", "late afternoon",
         ["early morning", "midday", "late evening"]),
        (["User: Devin saves the quarterly report in the Harbor folder.",
          "User: The report is ready for review."],
         "Which folder contains Devin's report?", "Harbor folder",
         ["Maple folder", "Finance folder", "Archive folder"]),
        (["User: Jamie's locker code is 4816.",
          "User: Jamie is meeting the team after lunch."],
         "What is Jamie's locker code?", "4816",
         ["4186", "8461", "6148"]),
        (["User: Quinn named the migration project Lantern.",
          "User: Quinn plans to finish the draft this week."],
         "What name did Quinn give the project?", "Lantern",
         ["Compass", "Orchard", "Harbor"]),
        (["User: Avery's preferred delivery day is Friday.",
          "User: Avery will be away from home on Wednesday."],
         "Which delivery day does Avery prefer?", "Friday",
         ["Monday", "Wednesday", "Saturday"]),
        (["User: Rowan reserved table 17 for the group.",
          "User: The restaurant confirmed the booking."],
         "Which table did Rowan reserve?", "table 17",
         ["table 9", "table 12", "table 21"]),
    ]
    for i, row in enumerate(direct):
        cases.append(_make_case("direct_recall", i, *row))

    composed = [
        (["User: Maya is on the Orion planning team.",
          "User: The Orion planning team meets in Cedar Room."],
         "Where should I send Maya for the team meeting?", "Cedar Room",
         ["Maple Room", "Room 12", "Oak Hall"]),
        (["User: Eli manages the Atlas project.",
          "User: The Atlas project's final report is due October 18."],
         "When is the deadline for Eli's project report?", "October 18",
         ["October 8", "October 28", "November 18"]),
        (["User: Kim handles the Saturn vendor account.",
          "User: Saturn invoices are processed by Accounts Payable."],
         "Where should Kim send the vendor's invoice?", "Accounts Payable",
         ["Customer Support", "Payroll", "Legal"]),
        (["User: Jonah's train arrives at Central Station.",
          "User: The museum entrance is beside the South Gate exit."],
         "Which exit should Jonah use to reach the museum?", "South Gate",
         ["North Gate", "East Gate", "West Gate"]),
        (["User: Anya's advisor is Dr. Lee.",
          "User: Dr. Lee works in Building C."],
         "Where can Anya find her advisor?", "Building C",
         ["Building A", "Building B", "Building D"]),
        (["User: Marcus's workshop is part of Event Blue.",
          "User: Event Blue is being held at Lakeside Hall."],
         "Where should Marcus go for the workshop?", "Lakeside Hall",
         ["Riverside Hall", "North Center", "Union Hall"]),
        (["User: Priya is coordinating the Juniper release.",
          "User: Release coordinators send final files to the Launch Desk."],
         "Where should Priya send the final release files?", "Launch Desk",
         ["Design Desk", "Service Desk", "Archive Desk"]),
        (["User: Sam's mentor is Jordan.",
          "User: Jordan's office is on the fifth floor of West Tower."],
         "Which floor should Sam visit to meet the mentor?", "fifth floor",
         ["second floor", "third floor", "seventh floor"]),
    ]
    for i, row in enumerate(composed):
        cases.append(_make_case("composed_implicit", i, *row))

    changed = [
        (["User: I live in Columbus.",
          "User: Correction: I moved to Dayton last month."],
         "Which city should I use as my current home?", "Dayton",
         ["Columbus", "Cincinnati", "Toledo"]),
        (["User: My appointment is Tuesday at 2 PM.",
          "User: The appointment changed; use Thursday at 3 PM instead."],
         "What time should I put on the calendar now?", "Thursday at 3 PM",
         ["Tuesday at 2 PM", "Wednesday at 3 PM", "Friday at 2 PM"]),
        (["User: Leslie owns the Atlas rollout.",
          "User: Ownership changed yesterday; Omar owns Atlas now."],
         "Who currently owns the Atlas rollout?", "Omar",
         ["Leslie", "Priya", "Jordan"]),
        (["User: Please send updates in Spanish.",
          "User: Update: Portuguese is my preferred language now."],
         "Which language should I use for new updates?", "Portuguese",
         ["Spanish", "French", "Italian"]),
        (["User: Park in Garage A for the first visit.",
          "User: The venue changed the parking plan; use Garage B."],
         "Which garage is current for the visit?", "Garage B",
         ["Garage A", "Garage C", "Street parking"]),
        (["User: My package is going to 10 Pine Street.",
          "User: I corrected the delivery address to 38 Oak Street."],
         "What is the current delivery address?", "38 Oak Street",
         ["10 Pine Street", "18 Oak Street", "38 Pine Avenue"]),
        (["User: The review is scheduled for June 12.",
          "User: The review moved to June 19; June 12 is no longer current."],
         "What is the current review date?", "June 19",
         ["June 12", "June 16", "June 21"]),
        (["User: Use the blue draft for the presentation.",
          "User: I replaced it; the current draft is the green version."],
         "Which draft should I use now?", "green version",
         ["blue draft", "red version", "first draft"]),
    ]
    for i, row in enumerate(changed):
        cases.append(_make_case("changed_fact", i, *row))

    conflict_answer = "Ask for confirmation; the sources conflict."
    conflict = [
        (["User: The invitation says Tuesday at 2 PM.",
          "User: The calendar export says Wednesday at 2 PM. I have not confirmed which is current."],
         "Which meeting time should I rely on?", conflict_answer,
         ["Tuesday at 2 PM", "Wednesday at 2 PM", "Thursday at 2 PM"]),
        (["User: The project note names Dana as owner.",
          "User: The dashboard names Lee as owner, and neither entry is marked current."],
         "Who currently owns the project?", conflict_answer,
         ["Dana", "Lee", "The last person mentioned"]),
        (["User: One message says the shipment goes to Dock 4.",
          "User: A second message says Dock 7. No one has verified the destination."],
         "Which dock should the driver use?", conflict_answer,
         ["Dock 4", "Dock 7", "Dock 1"]),
        (["User: The draft lists a budget of $8,000.",
          "User: The finance note lists $9,000, with no approval recorded."],
         "What approved budget should I report?", conflict_answer,
         ["$8,000", "$9,000", "$8,500"]),
        (["User: The first agenda says Room 210.",
          "User: A later forwarded agenda says Room 310, but its date is unclear."],
         "Which room is confirmed for the event?", conflict_answer,
         ["Room 210", "Room 310", "Room 410"]),
        (["User: The support ticket says the service is online.",
          "User: The status page says it is degraded; no incident owner has confirmed the state."],
         "Is the service confirmed healthy?", conflict_answer,
         ["Yes, it is online", "No, it is fully down", "It is degraded"]),
        (["User: The booking message lists May 4.",
          "User: The attached itinerary lists May 5, and the booking was not rechecked."],
         "Which travel date should I use?", conflict_answer,
         ["May 4", "May 5", "May 6"]),
        (["User: One note says the client chose Standard.",
          "User: Another note says the client chose Premium; there is no signed selection."],
         "Which plan did the client confirm?", conflict_answer,
         ["Standard", "Premium", "Both plans"]),
    ]
    for i, row in enumerate(conflict):
        cases.append(_make_case("conflict_resolution", i, *row))

    missing_answer = "That detail is not stated in the conversation."
    abstain = [
        (["User: My flight to Denver leaves on Friday.",
          "User: I will take a rideshare to the airport."],
         "Which gate is the flight leaving from?", missing_answer,
         ["Gate 12", "Gate 4", "Gate 22"]),
        (["User: I moved to 38 Oak Street last week.",
          "User: The new neighborhood is close to the library."],
         "What is the apartment number?", missing_answer,
         ["Apartment 2", "Apartment 14", "Apartment 21"]),
        (["User: I start work on Monday.",
          "User: I will bring lunch from home."],
         "What time does the workday start?", missing_answer,
         ["7 AM", "8 AM", "9 AM"]),
        (["User: I made soup for dinner.",
          "User: The recipe was easy to follow."],
         "Which ingredient am I allergic to?", missing_answer,
         ["Peanuts", "Shellfish", "Sesame"]),
        (["User: The school meeting is at Fairview Elementary.",
          "User: I will arrive a few minutes early."],
         "Who is the child's teacher?", missing_answer,
         ["Ms. Carter", "Mr. Lewis", "Mrs. Patel"]),
        (["User: Our trip is to Chicago in July.",
          "User: We will visit the aquarium."],
         "Which hotel did we book?", missing_answer,
         ["Lakeside Inn", "North Park Hotel", "Union Suites"]),
        (["User: My dentist appointment is next week.",
          "User: I need to bring my insurance card."],
         "What is the dentist's name?", missing_answer,
         ["Dr. Morris", "Dr. Chen", "Dr. Rivera"]),
        (["User: I submitted the report yesterday.",
          "User: The manager said they would review it."],
         "What score did the report receive?", missing_answer,
         ["82", "91", "96"]),
    ]
    for i, row in enumerate(abstain):
        cases.append(_make_case("abstention", i, *row))

    return cases


CASES = tuple(build_cases())
