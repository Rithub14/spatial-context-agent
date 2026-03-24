"""Seed script — populates the landmarks table with 18 real Berlin landmarks.

Run with:
    python -m src.db.seed
"""

import logging
import sys

from sqlalchemy.orm import Session

from src.db.models import Landmark
from src.db.session import SessionLocal, create_tables

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

LANDMARKS: list[dict] = [
    {
        "name": "Brandenburg Gate",
        "description": (
            "The Brandenburg Gate is an 18th-century neoclassical monument and the "
            "most recognisable symbol of Berlin. Built between 1788 and 1791 on the "
            "orders of Prussian king Frederick William II, it served as a city gate "
            "and later became a symbol of division during the Cold War."
        ),
        "latitude": 52.5163,
        "longitude": 13.3777,
        "district": "Mitte",
        "category": "monument",
        "historical_period": "18th century / Prussian era",
        "narration_template": (
            "You're standing before the Brandenburg Gate, Berlin's most iconic symbol. "
            "Built in 1791, this neoclassical triumphal arch once marked the western "
            "entrance to the city. Napoleon marched through it in 1806, and it stood "
            "divided by the Berlin Wall for 28 years. When the Wall fell in 1989, "
            "hundreds of thousands gathered here to celebrate reunification. The "
            "Quadriga — the horse-drawn chariot on top — represents the goddess of "
            "victory returning to Berlin in peace."
        ),
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a6/Brandenburger_Tor_abends.jpg/320px-Brandenburger_Tor_abends.jpg",
    },
    {
        "name": "Reichstag Building",
        "description": (
            "The Reichstag is the historic seat of the German Parliament (Bundestag), "
            "completed in 1894. Famously wrapped by artists Christo and Jeanne-Claude "
            "in 1995, it was rebuilt with a striking glass dome by architect Norman "
            "Foster after reunification."
        ),
        "latitude": 52.5186,
        "longitude": 13.3762,
        "district": "Mitte",
        "category": "historic building",
        "historical_period": "19th century / Wilhelmine era",
        "narration_template": (
            "You're looking at the Reichstag, home of Germany's federal parliament. "
            "Completed in 1894, this building has witnessed the proclamation of the "
            "Weimar Republic, the rise of the Nazis, and the reunification of Germany. "
            "The glass dome you can see was designed by Norman Foster and symbolises "
            "transparency in democracy — visitors can walk up a spiral ramp inside "
            "and look directly down into the parliament chamber below."
        ),
        "image_url": None,
    },
    {
        "name": "East Side Gallery",
        "description": (
            "The East Side Gallery is an open-air gallery on a 1.3 km stretch of the "
            "Berlin Wall along the Spree River. It features 105 murals painted by "
            "artists from 21 countries in 1990, making it the longest remaining "
            "section of the Berlin Wall."
        ),
        "latitude": 52.5050,
        "longitude": 13.4398,
        "district": "Friedrichshain",
        "category": "memorial",
        "historical_period": "Cold War / Post-reunification",
        "narration_template": (
            "You're standing at the East Side Gallery, the longest remaining stretch "
            "of the Berlin Wall — 1.3 kilometres of concrete transformed into an "
            "open-air gallery. In 1990, artists from 21 countries painted 105 murals "
            "here, turning a symbol of oppression into a canvas of freedom. The most "
            "famous image, 'My God, Help Me to Survive This Deadly Love', shows Soviet "
            "leader Brezhnev and East German leader Honecker in a passionate kiss."
        ),
        "image_url": None,
    },
    {
        "name": "Checkpoint Charlie",
        "description": (
            "Checkpoint Charlie was the best-known Berlin Wall crossing point between "
            "East and West Berlin during the Cold War. The name was given by the Western "
            "Allies and it became an international symbol of the Cold War."
        ),
        "latitude": 52.5075,
        "longitude": 13.3904,
        "district": "Mitte",
        "category": "memorial",
        "historical_period": "Cold War (1961–1990)",
        "narration_template": (
            "You're at Checkpoint Charlie, the most famous crossing point of the "
            "Berlin Wall between East and West. From 1961 to 1990, this modest "
            "guardhouse controlled the movement of diplomats, Allied military "
            "personnel, and foreign nationals. In October 1961, Soviet and American "
            "tanks faced each other here in a tense standoff for 16 hours. Today the "
            "replica guardhouse and sandbag checkpoint remind visitors of the city's "
            "divided past."
        ),
        "image_url": None,
    },
    {
        "name": "Berlin Cathedral",
        "description": (
            "The Berlin Cathedral (Berliner Dom) is the largest Protestant church in "
            "Germany, situated on Museum Island. The current building was completed "
            "in 1905 under Kaiser Wilhelm II and houses the Hohenzollern imperial "
            "crypt with 94 sarcophagi."
        ),
        "latitude": 52.5192,
        "longitude": 13.4014,
        "district": "Mitte",
        "category": "church or cathedral",
        "historical_period": "19th–20th century / Wilhelmine era",
        "narration_template": (
            "You're gazing at the Berlin Cathedral, the most significant Protestant "
            "church in Germany. Completed in 1905 for Kaiser Wilhelm II, this "
            "Renaissance Revival masterpiece dominates Museum Island on the Spree. "
            "Beneath you lies the Hohenzollern crypt, final resting place of 94 "
            "members of the Prussian royal family. Climb the 270 steps to the dome "
            "for a breathtaking panoramic view of central Berlin."
        ),
        "image_url": None,
    },
    {
        "name": "Berlin TV Tower",
        "description": (
            "The Berliner Fernsehturm (TV Tower) at Alexanderplatz is the tallest "
            "structure in Germany at 368 metres. Built by East Germany between 1965 "
            "and 1969, it was intended as a symbol of socialist progress and remains "
            "the most visited landmark in Germany."
        ),
        "latitude": 52.5208,
        "longitude": 13.4094,
        "district": "Mitte",
        "category": "modern architecture",
        "historical_period": "Cold War / GDR era (1969)",
        "narration_template": (
            "You're standing beneath the Berlin TV Tower — the tallest structure in "
            "Germany at 368 metres. Built by East Germany in 1969 as a statement of "
            "socialist technological achievement, it dominates the Berlin skyline from "
            "almost every angle in the city. The sphere near the top houses a rotating "
            "restaurant and observation deck at 203 metres. When sunlight hits the "
            "steel sphere, it reflects a cross shape — a phenomenon the atheist East "
            "German government called 'the Pope's revenge'."
        ),
        "image_url": None,
    },
    {
        "name": "Memorial to the Murdered Jews of Europe",
        "description": (
            "The Holocaust Memorial, designed by architect Peter Eisenman, consists "
            "of 2,711 concrete slabs of varying heights arranged in a grid on a "
            "sloping field. Opened in 2005, it commemorates the six million Jewish "
            "victims of the Holocaust."
        ),
        "latitude": 52.5138,
        "longitude": 13.3783,
        "district": "Mitte",
        "category": "memorial",
        "historical_period": "Contemporary (opened 2005)",
        "narration_template": (
            "You're at the Memorial to the Murdered Jews of Europe, one of the world's "
            "most powerful Holocaust memorials. These 2,711 concrete slabs — no two "
            "the same height — were designed by Peter Eisenman to create a sense of "
            "disorientation and unease. As you walk deeper in, the slabs grow taller "
            "and the ground undulates beneath you. There are no names, no dates, no "
            "explicit message. The memorial asks you simply to feel. Beneath the "
            "surface is an information centre with the names of every known Jewish "
            "victim of the Holocaust."
        ),
        "image_url": None,
    },
    {
        "name": "Charlottenburg Palace",
        "description": (
            "Charlottenburg Palace is the largest palace in Berlin, commissioned by "
            "Queen Sophie Charlotte at the end of the 17th century. The Baroque "
            "complex includes formal gardens, the Altes Schloss, and the New Wing "
            "containing the apartments of Frederick the Great."
        ),
        "latitude": 52.5207,
        "longitude": 13.2957,
        "district": "Charlottenburg",
        "category": "historic building",
        "historical_period": "17th–18th century / Prussian Baroque",
        "narration_template": (
            "You're before Charlottenburg Palace, Berlin's largest and most splendid "
            "royal residence. Built from 1699 for Queen Sophie Charlotte, the palace "
            "is a masterpiece of Baroque architecture stretching over 500 metres. "
            "Frederick the Great added the New Wing in the 18th century, filling it "
            "with his celebrated collection of French Rococo art. The palace was "
            "largely destroyed in World War II but meticulously rebuilt — look for "
            "the golden figure of Fortuna rotating on the central tower."
        ),
        "image_url": None,
    },
    {
        "name": "Pergamon Museum",
        "description": (
            "The Pergamon Museum on Museum Island houses monumental reconstructions "
            "of ancient architecture, including the Pergamon Altar, the Ishtar Gate "
            "of Babylon, and the Market Gate of Miletus. It is one of the most visited "
            "museums in Germany."
        ),
        "latitude": 52.5212,
        "longitude": 13.3969,
        "district": "Mitte",
        "category": "museum",
        "historical_period": "Ancient world collections",
        "narration_template": (
            "You're at the Pergamon Museum, one of the world's greatest museums of "
            "ancient architecture. Inside, entire ancient structures have been "
            "reassembled at full scale — the colossal Pergamon Altar from 2nd-century "
            "BC Greece, the cobalt-blue Ishtar Gate of Babylon from 575 BC, and the "
            "Market Gate of Miletus from Roman Turkey. These weren't simply excavated — "
            "they were dismantled stone by stone and shipped to Berlin in the early "
            "20th century, making this museum as controversial as it is spectacular."
        ),
        "image_url": None,
    },
    {
        "name": "Oberbaum Bridge",
        "description": (
            "The Oberbaumbrücke is a double-deck bridge crossing the Spree, connecting "
            "the districts of Friedrichshain and Kreuzberg. The iconic red-brick "
            "bridge with its Neo-Gothic towers was built between 1894 and 1896."
        ),
        "latitude": 52.5018,
        "longitude": 13.4451,
        "district": "Friedrichshain-Kreuzberg",
        "category": "bridge",
        "historical_period": "19th century",
        "narration_template": (
            "You're standing at the Oberbaum Bridge, Berlin's most beloved bridge and "
            "the unofficial border between the rival neighbourhoods of Friedrichshain "
            "and Kreuzberg. Built in 1896, the Neo-Gothic red-brick towers make it "
            "look more like a castle than a bridge. During the Cold War it was one of "
            "the few crossing points between East and West Berlin. Today, residents "
            "from both sides meet here every year for a good-natured 'battle' throwing "
            "vegetables at each other across the bridge."
        ),
        "image_url": None,
    },
    {
        "name": "Victory Column",
        "description": (
            "The Siegessäule (Victory Column) stands 67 metres tall in the middle of "
            "the Tiergarten. Built in 1873 to commemorate Prussian victories in the "
            "wars against Denmark, Austria, and France, it was moved to its current "
            "location by the Nazis in 1939."
        ),
        "latitude": 52.5145,
        "longitude": 13.3501,
        "district": "Tiergarten",
        "category": "monument",
        "historical_period": "19th century / Prussian era",
        "narration_template": (
            "You're looking up at the Siegessäule — the Victory Column — rising 67 "
            "metres from the heart of the Tiergarten. Built in 1873 after Prussia's "
            "wars against Denmark, Austria, and France, the gilded figure at the top "
            "is Victoria, goddess of victory. Berliners affectionately call her "
            "'Goldelse' — Golden Elsie. You can climb 285 steps to the viewing "
            "platform for a panorama of the Tiergarten and the city beyond. Barack "
            "Obama chose this spot to deliver his famous Berlin speech in 2008."
        ),
        "image_url": None,
    },
    {
        "name": "Potsdamer Platz",
        "description": (
            "Potsdamer Platz was Europe's busiest traffic intersection in the 1920s, "
            "reduced to wasteland by World War II bombing and divided by the Berlin "
            "Wall. After reunification it was rebuilt as a modern urban quarter and "
            "is now a symbol of Berlin's transformation."
        ),
        "latitude": 52.5096,
        "longitude": 13.3759,
        "district": "Mitte",
        "category": "plaza or square",
        "historical_period": "20th century / Post-reunification",
        "narration_template": (
            "You're standing in Potsdamer Platz, one of the most dramatic urban "
            "transformations in modern history. In the 1920s this was Europe's "
            "busiest intersection — a roaring hub of trams, cars, and the world's "
            "first automatic traffic light. Then came the bombs of WWII and the "
            "Berlin Wall, leaving it a desolate no-man's land for 28 years. After "
            "reunification, the world's largest construction site rose from the rubble "
            "in the 1990s. The gleaming towers you see today were built in under a decade."
        ),
        "image_url": None,
    },
    {
        "name": "Gendarmenmarkt",
        "description": (
            "Gendarmenmarkt is considered Berlin's most beautiful square. It is flanked "
            "by the German Cathedral, the French Cathedral, and the Konzerthaus Berlin. "
            "The square was built in the late 17th century and takes its name from the "
            "Gens d'armes regiment that was once headquartered there."
        ),
        "latitude": 52.5138,
        "longitude": 13.3928,
        "district": "Mitte",
        "category": "plaza or square",
        "historical_period": "17th–18th century",
        "narration_template": (
            "You're in Gendarmenmarkt, widely regarded as the most beautiful square "
            "in Berlin. Three magnificent buildings frame you: the German Cathedral "
            "to your south, the French Cathedral to your north — built for the "
            "Huguenot refugees who fled France — and the Konzerthaus in the centre, "
            "designed by the great Prussian architect Karl Friedrich Schinkel in 1821. "
            "In summer the square hosts open-air concerts; in winter it becomes one of "
            "Berlin's most magical Christmas markets."
        ),
        "image_url": None,
    },
    {
        "name": "Museum Island",
        "description": (
            "Museum Island (Museumsinsel) is a UNESCO World Heritage Site and home to "
            "five world-renowned museums: the Altes Museum, the Neues Museum, the Alte "
            "Nationalgalerie, the Bode Museum, and the Pergamon Museum. The ensemble "
            "was built between 1830 and 1930."
        ),
        "latitude": 52.5196,
        "longitude": 13.3988,
        "district": "Mitte",
        "category": "museum",
        "historical_period": "19th–20th century",
        "narration_template": (
            "You're on Museum Island, a UNESCO World Heritage Site and one of the "
            "most extraordinary concentrations of art and culture on Earth. Five "
            "world-class museums occupy this small island in the Spree: the Altes "
            "Museum with Greek and Roman antiquities, the Neues Museum housing the "
            "bust of Nefertiti, the Alte Nationalgalerie with 19th-century European "
            "art, the Bode Museum, and the Pergamon Museum with its monumental "
            "ancient reconstructions. Together they hold over a million objects "
            "spanning 6,000 years of human civilisation."
        ),
        "image_url": None,
    },
    {
        "name": "Tiergarten Park",
        "description": (
            "The Tiergarten is Berlin's largest and most popular inner-city park, "
            "covering 210 hectares in the heart of the city. Originally a royal "
            "hunting ground, it was redesigned as a landscape park in the 19th "
            "century and is now a beloved green lung for Berliners."
        ),
        "latitude": 52.5145,
        "longitude": 13.3501,
        "district": "Tiergarten",
        "category": "park or garden",
        "historical_period": "17th century onwards",
        "narration_template": (
            "You're in the Tiergarten, Berlin's great central park — 210 hectares "
            "of woodland, meadows, and lakes stretching from the Brandenburg Gate "
            "to Charlottenburg. 'Tiergarten' means animal garden: it was a royal "
            "hunting ground for 200 years before being opened to the public. During "
            "the desperate winter of 1945–46, starving Berliners stripped every "
            "tree for firewood and dug up the lawns to grow vegetables. What you "
            "see today was replanted entirely from scratch in the 1950s."
        ),
        "image_url": None,
    },
    {
        "name": "Hackescher Markt",
        "description": (
            "Hackescher Markt is a lively square and S-Bahn station in Mitte, "
            "surrounded by the Hackesche Höfe — a series of interconnected Art "
            "Nouveau courtyards built in 1907. The area is a hub of independent "
            "boutiques, cafes, galleries, and nightlife."
        ),
        "latitude": 52.5228,
        "longitude": 13.4023,
        "district": "Mitte",
        "category": "market or shopping area",
        "historical_period": "Early 20th century / Art Nouveau",
        "narration_template": (
            "You're at Hackescher Markt, one of Berlin's most vibrant urban spaces. "
            "Behind the square lie the Hackesche Höfe — eight interconnected Art "
            "Nouveau courtyards built in 1907 and decorated with thousands of glazed "
            "ceramic tiles. Once the centre of Berlin's Jewish quarter, the area was "
            "devastated in WWII and left neglected in East Germany. Since the 1990s "
            "it has been reborn as a hub of independent boutiques, street food, "
            "galleries, and the best people-watching in Berlin."
        ),
        "image_url": None,
    },
    {
        "name": "Berlin Wall Memorial",
        "description": (
            "The Berlin Wall Memorial on Bernauer Strasse is the central memorial "
            "site for the Berlin Wall, preserving an authentic 60-metre section of "
            "the border strip with the 'death strip', watchtower, and escape tunnels. "
            "The open-air exhibition stretches for 1.4 kilometres."
        ),
        "latitude": 52.5352,
        "longitude": 13.3831,
        "district": "Mitte / Wedding",
        "category": "memorial",
        "historical_period": "Cold War (1961–1989)",
        "narration_template": (
            "You're at the Berlin Wall Memorial on Bernauer Strasse — the most "
            "authentic and complete remaining section of the Wall's border strip. "
            "Here you can see the full architecture of division: the inner wall, "
            "the 'death strip', the watchtower, the signal fence, and the outer wall. "
            "This street was particularly tragic — when the Wall was built overnight "
            "on 13 August 1961, residents woke to find their front doors bricked up. "
            "Some jumped from windows into West Berlin. Others dug tunnels beneath "
            "your feet. 140 people died attempting to cross the Wall."
        ),
        "image_url": None,
    },
    {
        "name": "Topography of Terror",
        "description": (
            "The Topography of Terror is a documentation centre built on the site "
            "of the former headquarters of the Gestapo and SS during the Nazi era. "
            "The outdoor and indoor exhibitions document the crimes of the Nazi "
            "terror apparatus between 1933 and 1945."
        ),
        "latitude": 52.5064,
        "longitude": 13.3831,
        "district": "Mitte",
        "category": "memorial",
        "historical_period": "Nazi era (1933–1945)",
        "narration_template": (
            "You're standing on some of the most sinister ground in Berlin. This "
            "is the Topography of Terror, built on the site where the Gestapo, SS, "
            "and Reich Security Main Office, the administrative heart of Nazi terror, "
            "had their headquarters from 1933 to 1945. From these buildings, orders "
            "went out for the persecution and murder of millions across Europe. The "
            "remaining section of Berlin Wall behind you is a reminder that this "
            "site has witnessed two totalitarian regimes. The exhibition here is "
            "unflinching, essential, and free."
        ),
        "image_url": None,
    },
]


def seed(db: Session) -> None:
    """Insert landmarks if the table is empty."""
    existing = db.query(Landmark).count()
    if existing > 0:
        logger.info("Landmarks table already has %d rows — skipping seed.", existing)
        return

    logger.info("Seeding %d Berlin landmarks...", len(LANDMARKS))
    for data in LANDMARKS:
        db.add(Landmark(**data))
    db.commit()
    logger.info("Seeding complete.")


if __name__ == "__main__":
    create_tables()
    db = SessionLocal()
    try:
        seed(db)
    except Exception as exc:
        logger.error("Seed failed: %s", exc)
        db.rollback()
        sys.exit(1)
    finally:
        db.close()
