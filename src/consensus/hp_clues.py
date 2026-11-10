"""Clue bank for the hidden-profile task (`consensus.hp_data`): facts about a candidate café site.

Each entry is (value, text, keywords):

- value: the clue's hidden worth for the site, +-2 (strong) or +-1 (mild). The ground truth of a
  case is the sum over a site's clues. The values are meant to match ordinary business sense, so
  a model that reads the clues well recovers them; whether it does is checked by the full-
  information baseline (one model sees every clue).
- text: what an agent is shown, without the value.
- keywords: lower-case phrases that identify the clue when an agent mentions it in a message
  (used only by the analysis to measure which clues get shared). They are chosen so that no
  keyword of one clue appears in another clue's text (checked in tests/test_hp.py).
"""

from __future__ import annotations

CLUE_BANK: tuple[tuple[int, str, tuple[str, ...]], ...] = (
    # strong positives
    (
        2,
        "A university campus with 12,000 students is a two-minute walk away.",
        ("campus", "university", "12,000"),
    ),
    (
        2,
        "A new office tower opening next door will bring about 800 daily workers onto the block.",
        ("office tower", "800"),
    ),
    (
        2,
        "The main commuter train station is directly across the street.",
        ("train station", "commuter"),
    ),
    (
        2,
        "The landlord offers the first year rent-free on a ten-year lease.",
        ("rent-free", "rent free", "first year free"),
    ),
    (
        2,
        "Foot-traffic counts show over 4,000 people passing the door each weekday.",
        ("foot-traffic", "foot traffic", "4,000", "4000"),
    ),
    (
        2,
        "The unit already has a fully fitted commercial kitchen and espresso bar from a previous "
        "café.",
        ("fitted", "espresso bar", "commercial kitchen"),
    ),
    (2, "The nearest other café is more than a kilometre away.", ("kilometre", "km away", "1 km")),
    (
        2,
        "The city has approved a pedestrian plaza with outdoor seating right in front of the unit.",
        ("plaza",),
    ),
    (
        2,
        "A popular farmers' market sets up on the same street every Saturday.",
        ("farmers", "market"),
    ),
    (2, "A large hospital with round-the-clock shifts is next door.", ("hospital",)),
    (2, "The unit has a large sunny terrace with room for 30 outdoor seats.", ("terrace",)),
    (
        2,
        "A coworking space with 300 members in the same building has no coffee of its own.",
        ("coworking", "co-working"),
    ),
    # mild positives
    (1, "There is a bus stop right outside.", ("bus stop",)),
    (
        1,
        "The storefront has big windows and is easy to see from the main road.",
        ("big windows", "visible", "easy to see"),
    ),
    (1, "The rent is about 10% below the city average for similar units.", ("10% below",)),
    (1, "A gym on the corner opens at 6am and has a busy morning crowd.", ("gym",)),
    (1, "The street has plenty of free parking.", ("free parking",)),
    (1, "A public library across the road draws students most afternoons.", ("library",)),
    (
        1,
        "The neighbourhood has many young families moving into new apartments.",
        ("young families", "families"),
    ),
    (1, "Two nearby hotels have no breakfast service.", ("hotel", "breakfast")),
    (
        1,
        "The area is getting more popular: three new shops opened this year.",
        ("three new shops", "new shops", "getting more popular"),
    ),
    (1, "A secondary school is a few minutes away.", ("secondary school", "high school")),
    (
        1,
        "The unit sits on a corner with entrances on two streets.",
        ("corner unit", "two streets", "entrances"),
    ),
    (
        1,
        "The local council offers a small-business grant that would cover new furniture.",
        ("grant",),
    ),
    # strong negatives
    (
        -2,
        "The street will be closed for tram-line construction for most of the next two years.",
        ("tram-line", "tram line", "construction"),
    ),
    (
        -2,
        "A well-loved independent café with long queues is two doors down.",
        ("two doors down", "long queues", "queues"),
    ),
    (
        -2,
        "The landlord evicted the last two tenants and is known for sudden rent hikes.",
        ("evicted", "rent hike"),
    ),
    (-2, "The unit floods most winters, and the basement is damp.", ("flood", "damp")),
    (
        -2,
        "The street is almost empty after 6pm and at weekends.",
        ("after 6pm", "empty after", "dead after"),
    ),
    (
        -2,
        "A large chain coffee shop with a drive-through just opened across the road.",
        ("chain", "drive-through", "drive through"),
    ),
    (
        -2,
        "The building is listed, so no signage or changes to the frontage are allowed.",
        ("signage", "listed building"),
    ),
    (-2, "The rent is about 40% above the city average for similar units.", ("40% above", "40%")),
    (
        -2,
        "Health inspectors have flagged the building's drains; fixing them would cost a lot.",
        ("drains", "inspectors"),
    ),
    (
        -2,
        "The area's main employer has announced it will move out next spring.",
        ("move out", "moving out", "main employer"),
    ),
    (
        -2,
        "The unit is on the second floor with no lift and no street frontage.",
        ("second floor", "no lift", "upstairs"),
    ),
    (
        -2,
        "There have been several break-ins on the street in the past year.",
        ("break-in", "break in", "burglar"),
    ),
    # mild negatives
    (-1, "There are only a few paid parking spots nearby.", ("paid parking", "parking spots")),
    (-1, "The unit is small: about 25 square metres.", ("25 square", "25 m", "25m")),
    (
        -1,
        "Public transport is poor: the nearest stop is a fifteen-minute walk.",
        ("public transport", "fifteen-minute", "15-minute", "15 minute"),
    ),
    (-1, "The street is noisy because of a busy road.", ("noisy", "noise")),
    (-1, "The unit faces north and is dark most of the day.", ("faces north", "dark")),
    (
        -1,
        "The previous café in this unit closed after a year.",
        ("closed after", "previous café closed"),
    ),
    (-1, "The landlord wants a five-year personal guarantee.", ("personal guarantee", "guarantee")),
    (-1, "Delivery trucks may only unload before 7am.", ("unload", "deliver")),
    (-1, "The heating system is old and will need replacing within a few years.", ("heating",)),
    (-1, "A bakery a few doors away also sells coffee.", ("bakery",)),
    (
        -1,
        "The pavement outside is narrow, so outdoor seating is not possible.",
        ("pavement", "narrow"),
    ),
    (-1, "The area is mostly offices that empty out in August.", ("august",)),
)
