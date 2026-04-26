#!/usr/bin/env python3
"""
Create Mt Olympus Casino & Hotel database with room rates and reservations
"""

import sqlite3
from datetime import datetime, timedelta

def create_database(db_path="gaming/mt_olympus.db"):
    """Create gaming database with sample data"""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Room Types Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS room_types (
            room_type_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            size_sqft INTEGER,
            description TEXT,
            max_occupancy INTEGER,
            base_rate REAL,
            weekend_rate REAL,
            holiday_rate REAL,
            amenities TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Sample Reservations Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reservations (
            reservation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            guest_name TEXT NOT NULL,
            room_type TEXT NOT NULL,
            check_in_date DATE NOT NULL,
            check_out_date DATE NOT NULL,
            num_guests INTEGER,
            total_cost REAL,
            status TEXT DEFAULT 'Confirmed',
            special_requests TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Packages Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS packages (
            package_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            includes TEXT,
            starting_price REAL,
            available INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Insert Room Types
    room_types = [
        ("Zeus Grand Suite", "Luxury Suite", 2800,
         "The crown jewel of Mt Olympus. Private terrace overlooking fountain, marble throughout, personal butler service",
         4, 950.00, 1200.00, 1500.00,
         "Private terrace, Fountain view, Marble bathroom, Butler service, King bed, Living room, Wet bar"),

        ("Athena Wisdom Suite", "Luxury Suite", 1500,
         "Elegant suite with study area, library, and fountain views. Perfect for business or relaxation",
         3, 520.00, 650.00, 800.00,
         "Study area, Library, Fountain view, Marble bathroom, King bed, Sofa bed"),

        ("Poseidon Ocean Suite", "Luxury Suite", 1200,
         "Water-themed décor with spa tub and garden views. Ultimate relaxation",
         2, 420.00, 520.00, 650.00,
         "Spa tub, Garden view, Water décor, King bed, Sitting area"),

        ("Apollo Sun Suite", "Luxury Suite", 900,
         "Bright and airy suite with balcony and city views",
         2, 350.00, 425.00, 550.00,
         "Private balcony, City view, King bed, Sitting area"),

        ("Olympian Deluxe Room", "Standard Room", 450,
         "Comfortable deluxe room with fountain or garden views",
         2, 199.00, 249.00, 325.00,
         "Fountain/Garden view, King or Two Queens, Marble bathroom, Mini-bar")
    ]

    cursor.executemany("""
        INSERT INTO room_types (name, category, size_sqft, description, max_occupancy,
                                base_rate, weekend_rate, holiday_rate, amenities)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, room_types)

    # Insert Sample Reservations
    today = datetime.now()
    reservations = [
        ("John Smith", "Zeus Grand Suite",
         (today + timedelta(days=3)).strftime("%Y-%m-%d"),
         (today + timedelta(days=5)).strftime("%Y-%m-%d"),
         2, 1900.00, "Confirmed", "Anniversary celebration, champagne requested"),

        ("Sarah Johnson", "Athena Wisdom Suite",
         (today + timedelta(days=7)).strftime("%Y-%m-%d"),
         (today + timedelta(days=10)).strftime("%Y-%m-%d"),
         2, 1560.00, "Confirmed", "Business trip, early check-in needed"),

        ("Michael Chen", "Olympian Deluxe Room",
         (today + timedelta(days=1)).strftime("%Y-%m-%d"),
         (today + timedelta(days=3)).strftime("%Y-%m-%d"),
         2, 398.00, "Confirmed", "Two queen beds preferred"),

        ("Emily Davis", "Poseidon Ocean Suite",
         (today + timedelta(days=14)).strftime("%Y-%m-%d"),
         (today + timedelta(days=17)).strftime("%Y-%m-%d"),
         2, 1260.00, "Confirmed", "Honeymoon package"),

        ("Robert Taylor", "Apollo Sun Suite",
         (today + timedelta(days=21)).strftime("%Y-%m-%d"),
         (today + timedelta(days=23)).strftime("%Y-%m-%d"),
         2, 700.00, "Pending", "Late checkout requested")
    ]

    cursor.executemany("""
        INSERT INTO reservations (guest_name, room_type, check_in_date, check_out_date,
                                  num_guests, total_cost, status, special_requests)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, reservations)

    # Insert Packages
    packages = [
        ("Romance Package",
         "Perfect for couples celebrating special occasions",
         "Champagne and chocolates, Couples spa treatment (50% off), Late checkout, Rose petals",
         499.00, 1),

        ("Weekend Warrior Package",
         "Make the most of your weekend getaway",
         "2 nights (Fri-Sat), $200 gaming credit, Complimentary breakfast for two",
         750.00, 1),

        ("VIP Experience",
         "The ultimate luxury experience",
         "Suite upgrade, Private casino host, Priority restaurant reservations, Premium spa access",
         1200.00, 1),

        ("Business Traveler Package",
         "Everything you need for a productive stay",
         "High-speed WiFi, Business center access, Late checkout, Complimentary breakfast",
         399.00, 1)
    ]

    cursor.executemany("""
        INSERT INTO packages (name, description, includes, starting_price, available)
        VALUES (?, ?, ?, ?, ?)
    """, packages)

    conn.commit()
    conn.close()

    print(f"✅ Database created successfully: {db_path}")
    print(f"   • Room types: {len(room_types)}")
    print(f"   • Sample reservations: {len(reservations)}")
    print(f"   • Packages: {len(packages)}")

if __name__ == "__main__":
    create_database()
