#!/usr/bin/env python3
"""Create and populate the VERO fashion recommendation SQLite database."""

import sqlite3
import json
import random
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verodata.db")

def create_tables(conn):
    c = conn.cursor()
    c.executescript("""
        DROP TABLE IF EXISTS recommendations;
        DROP TABLE IF EXISTS similar_customers;
        DROP TABLE IF EXISTS customer_segments;
        DROP TABLE IF EXISTS browsing_activity;
        DROP TABLE IF EXISTS orders;
        DROP TABLE IF EXISTS customer_preferences;
        DROP TABLE IF EXISTS products;
        DROP TABLE IF EXISTS customers;

        CREATE TABLE customers (
            customer_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            tier TEXT NOT NULL,
            member_since TEXT NOT NULL,
            loyalty_points INTEGER NOT NULL,
            phone TEXT,
            address_street TEXT,
            address_city TEXT,
            address_state TEXT,
            address_zip TEXT
        );

        CREATE TABLE customer_preferences (
            customer_id TEXT PRIMARY KEY,
            favorite_categories TEXT,
            favorite_brands TEXT DEFAULT '["VERO"]',
            price_range TEXT,
            preferred_delivery TEXT,
            notify_email INTEGER DEFAULT 1,
            notify_sms INTEGER DEFAULT 0,
            notify_push INTEGER DEFAULT 1,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );

        CREATE TABLE products (
            product_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            sale_price REAL,
            material TEXT,
            color TEXT,
            description TEXT,
            rating REAL,
            review_count INTEGER,
            in_stock INTEGER DEFAULT 1,
            stock_quantity INTEGER DEFAULT 0
        );

        CREATE TABLE orders (
            order_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            product_id TEXT NOT NULL,
            product_name TEXT,
            category TEXT,
            date TEXT NOT NULL,
            amount REAL NOT NULL,
            quantity INTEGER DEFAULT 1,
            status TEXT NOT NULL,
            rating INTEGER,
            review TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        );

        CREATE TABLE browsing_activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT NOT NULL,
            pages_viewed INTEGER,
            time_on_site TEXT,
            sessions INTEGER,
            categories_viewed TEXT,
            products_viewed TEXT,
            cart_abandonment INTEGER DEFAULT 0,
            wishlist_items TEXT,
            search_queries TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );

        CREATE TABLE customer_segments (
            customer_id TEXT PRIMARY KEY,
            segment TEXT,
            segment_id TEXT,
            engagement_score INTEGER,
            purchase_probability REAL,
            churn_risk TEXT,
            lifetime_value REAL,
            predicted_ltv REAL,
            characteristics TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );

        CREATE TABLE similar_customers (
            customer_id TEXT PRIMARY KEY,
            similar_count INTEGER,
            similarity_score REAL,
            common_purchases TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );

        CREATE TABLE recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT NOT NULL,
            product_id TEXT NOT NULL,
            ml_score REAL,
            reason TEXT,
            model_version TEXT DEFAULT 'v2.4.1',
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        );
    """)
    conn.commit()


def populate_products(conn):
    products = [
        ("PROD-1001", "Classic Leather Tote", "Handbags", 2398.00, None, "Italian Calfskin Leather", "Burgundy",
         "A timeless tote crafted from the finest Italian calfskin leather. Features hand-stitched details, suede-lined interior, and signature VERO hardware in brushed gold."),
        ("PROD-1002", "Structured Shoulder Bag", "Handbags", 3200.00, None, "Saffiano Leather", "Midnight Black",
         "Architecturally designed shoulder bag with rigid Saffiano leather construction. Internal organizational pockets and adjustable strap for versatile carrying."),
        ("PROD-1003", "Evening Clutch", "Handbags", 1850.00, None, "Silk with Crystal Embellishments", "Black",
         "An exquisite evening clutch adorned with hand-set Swarovski crystals on pure silk. Includes detachable gold chain strap and mirror-lined interior."),
        ("PROD-1004", "Crossbody Messenger", "Handbags", 2650.00, None, "Pebbled Leather", "Cognac",
         "Luxurious crossbody messenger in rich pebbled leather. Adjustable strap, multiple compartments, and the iconic VERO clasp closure."),
        ("PROD-1005", "Weekend Travel Bag", "Handbags", 4200.00, None, "Canvas and Leather Trim", "Navy/Tan",
         "The quintessential weekend companion. Premium canvas body with full-grain leather trim, brass hardware, and spacious lined interior."),
        ("PROD-1006", "Designer Wallet", "Handbags", 1150.00, None, "Embossed Calfskin", "Black",
         "Slim continental wallet in signature VERO embossed calfskin. Features 12 card slots, zip coin pocket, and two bill compartments."),
        ("PROD-1007", "Wool Tailored Coat", "Outerwear", 1899.00, None, "100% Virgin Wool", "Storm Slate",
         "Impeccably tailored coat in premium virgin wool. Features notched lapel, double-breasted closure, and satin-lined interior for effortless elegance."),
        ("PROD-1008", "Cashmere Overcoat", "Outerwear", 4800.00, None, "100% Mongolian Cashmere", "Camel",
         "The ultimate luxury overcoat in the world's finest Mongolian cashmere. Single-breasted design with horn buttons and hand-finished seams."),
        ("PROD-1009", "Shearling Jacket", "Outerwear", 5500.00, None, "Genuine Shearling", "Cognac",
         "Statement shearling jacket with buttery-soft exterior and natural wool interior. Asymmetric zip closure and adjustable belt for a flattering silhouette."),
        ("PROD-1010", "Quilted Puffer Coat", "Outerwear", 2400.00, None, "Technical Fabric with Down Fill", "Black",
         "Performance meets luxury in this quilted puffer with premium goose down fill. Water-resistant technical fabric with matte finish and concealed hood."),
        ("PROD-1011", "Cashmere Sweater", "Knitwear", 2198.00, None, "100% Pure Cashmere", "Creme",
         "Cloud-soft crew neck sweater in 2-ply pure cashmere. Relaxed fit with ribbed trim and a subtle VERO logo at the hem."),
        ("PROD-1012", "Merino Turtleneck", "Knitwear", 3328.00, None, "Extra-Fine Merino Wool", "Ivory",
         "Refined turtleneck in ultra-fine 15.5 micron merino wool. Slim fit with seamless construction for unparalleled comfort and sophistication."),
        ("PROD-1013", "Cashmere Cardigan", "Knitwear", 2800.00, None, "4-Ply Cashmere", "Grey",
         "Sumptuous open-front cardigan in heavyweight 4-ply cashmere. Features patch pockets, ribbed cuffs, and a relaxed oversized silhouette."),
        ("PROD-1014", "Cable Knit Sweater", "Knitwear", 1950.00, None, "Cashmere Blend", "Cream",
         "Traditional cable knit reimagined in a luxurious cashmere-silk blend. Chunky texture with a modern relaxed fit."),
        ("PROD-1015", "Turtleneck Dress", "Knitwear", 3100.00, None, "Merino Wool", "Black",
         "Sleek turtleneck midi dress in fine-gauge merino wool. Body-skimming silhouette with side slits and ribbed trim details."),
        ("PROD-1016", "Silk Slip Dress", "Dresses", 3498.00, None, "100% Mulberry Silk", "Midnight",
         "Ethereal slip dress in the finest grade-6A mulberry silk. Bias-cut with adjustable spaghetti straps and French seam finishing."),
        ("PROD-1017", "Evening Gown", "Dresses", 6800.00, None, "Silk Georgette with Beading", "Black",
         "Red-carpet worthy evening gown with hand-beaded bodice in silk georgette. Floor-length with a dramatic train and open back detail."),
        ("PROD-1018", "Midi Wrap Dress", "Dresses", 2400.00, None, "Crepe de Chine", "Burgundy",
         "Flattering wrap dress in fluid crepe de chine. Self-tie waist, flutter sleeves, and a midi-length hem that moves beautifully."),
        ("PROD-1019", "Cocktail Dress", "Dresses", 3200.00, None, "Silk Taffeta", "Midnight Blue",
         "Show-stopping cocktail dress in structured silk taffeta. Features a sculpted bodice, flared skirt, and hidden pockets."),
        ("PROD-1020", "Silk Blouse", "Tops", 1678.00, None, "100% Silk Crepe", "Emerald",
         "Vibrant silk crepe blouse with a relaxed pussy-bow neckline. Mother-of-pearl buttons and French cuffs add refined detail."),
        ("PROD-1021", "Satin Camisole", "Tops", 1200.00, None, "Silk Satin", "Champagne",
         "Delicate camisole in luminous silk satin. Features lace-trimmed neckline, adjustable straps, and a relaxed fit perfect for layering."),
        ("PROD-1022", "Cashmere Shell", "Tops", 1850.00, None, "Lightweight Cashmere", "Ivory",
         "Versatile sleeveless shell in featherweight cashmere. Clean lines and a boat neckline make this a wardrobe essential."),
        ("PROD-1023", "Tailored Trousers", "Bottoms", 1450.00, None, "Wool Gabardine", "Black",
         "Precision-cut trousers in Italian wool gabardine. High waist, straight leg, and invisible zip closure for a flawless silhouette."),
        ("PROD-1024", "Pleated Midi Skirt", "Bottoms", 1680.00, None, "Silk Taffeta", "Navy",
         "Graceful pleated midi skirt in crisp silk taffeta. All-around knife pleats with a fitted waistband and silk lining."),
        ("PROD-1025", "Leather Pants", "Bottoms", 3400.00, None, "Lambskin Leather", "Black",
         "Sleek straight-leg pants in butter-soft lambskin leather. Fully lined with concealed zip and hook closure."),
        ("PROD-1026", "Leather Pumps", "Footwear", 1250.00, None, "Italian Calfskin", "Black",
         "Timeless pointed-toe pumps in polished Italian calfskin. 85mm stiletto heel with leather sole and cushioned insole."),
        ("PROD-1027", "Ankle Boots", "Footwear", 1580.00, None, "Suede with Leather Sole", "Taupe",
         "Refined ankle boots in velvety suede with a stacked 70mm heel. Inside zip closure and Goodyear-welted leather sole."),
        ("PROD-1028", "Designer Sneakers", "Footwear", 1100.00, None, "Leather and Suede", "White/Gold",
         "Elevated sneakers combining smooth leather and suede panels with gold-tone hardware. Italian-made with memory foam insole."),
        ("PROD-1029", "Over-the-Knee Boots", "Footwear", 2200.00, None, "Stretch Suede", "Black",
         "Dramatic over-the-knee boots in stretch suede for a second-skin fit. 60mm block heel and inner zip with suede pull tab."),
        ("PROD-1030", "Gold Chain Necklace", "Accessories", 2800.00, None, "18K Gold Plated", "Gold",
         "Bold chain necklace with interlocking VERO monogram links. 18K gold plating over sterling silver with lobster clasp closure."),
        ("PROD-1031", "Pearl Earrings", "Accessories", 1400.00, None, "Freshwater Pearls", "White",
         "Elegant drop earrings featuring AAA-grade freshwater pearls suspended from 18K gold-plated posts. Hypoallergenic and lightweight."),
        ("PROD-1032", "Leather Belt", "Accessories", 890.00, None, "Italian Calfskin", "Cognac",
         "Classic belt in smooth Italian calfskin with the signature VERO buckle in antiqued gold. Available in multiple sizes."),
        ("PROD-1033", "Cashmere Scarf", "Accessories", 798.00, None, "100% Cashmere", "Navy Charcoal",
         "Generously sized scarf in ultra-soft cashmere with a subtle herringbone pattern. Hand-rolled edges and VERO logo tag."),
        ("PROD-1034", "Designer Sunglasses", "Accessories", 1768.00, None, "Cellulose Acetate", "White",
         "Oversized cat-eye sunglasses in premium cellulose acetate. UV400 protection with gradient lenses and gold-tone VERO temples."),
        ("PROD-1035", "Silk Scarf", "Accessories", 895.00, None, "100% Silk Twill", "Various",
         "Hand-rolled silk twill scarf with exclusive VERO botanical print. Can be worn as a headscarf, neck tie, or bag accessory."),
    ]

    random.seed(42)
    rows = []
    for p in products:
        pid, name, cat, price, sale, mat, color, desc = p
        rating = round(random.uniform(4.5, 5.0), 1)
        review_count = random.randint(200, 5000)
        in_stock = 1
        stock_qty = random.randint(5, 90)
        # A few items on sale
        if pid in ("PROD-1007", "PROD-1014", "PROD-1021", "PROD-1033"):
            sale = round(price * random.uniform(0.75, 0.85), 2)
        rows.append((pid, name, cat, price, sale, mat, color, desc, rating, review_count, in_stock, stock_qty))

    conn.executemany(
        "INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    return {r[0]: (r[1], r[2], r[3]) for r in rows}  # id -> (name, category, price)


def populate_customers(conn):
    customers = [
        ("CUST-10001", "Isabelle Laurent", "isabelle.laurent@gmail.com", "vip", "2019-03-15", 12450,
         "(503) 555-0142", "2847 NW Thurman St", "Portland", "OR", "97210"),
        ("CUST-10002", "Sophia Nakamura", "sophia.nakamura@outlook.com", "premium", "2021-06-22", 3870,
         "(206) 555-0198", "1523 Pike Place", "Seattle", "WA", "98101"),
        ("CUST-10003", "Elena Vasquez", "elena.vasquez@icloud.com", "premium", "2020-11-08", 4210,
         "(415) 555-0267", "891 Valencia St", "San Francisco", "CA", "94110"),
        ("CUST-10004", "Charlotte Whitfield", "c.whitfield@proton.me", "vip", "2018-01-10", 14820,
         "(212) 555-0334", "740 Park Avenue Apt 12B", "New York", "NY", "10021"),
        ("CUST-10005", "Amara Okafor", "amara.okafor@gmail.com", "standard", "2023-09-01", 980,
         "(512) 555-0411", "4521 South Lamar Blvd", "Austin", "TX", "78745"),
        ("CUST-10006", "Margaux Delacroix", "margaux.d@yahoo.com", "vip", "2019-07-20", 11300,
         "(312) 555-0578", "1200 N Lake Shore Dr Unit 34", "Chicago", "IL", "60610"),
        ("CUST-10007", "Priya Sharma", "priya.sharma@gmail.com", "premium", "2021-02-14", 3540,
         "(310) 555-0623", "8834 Melrose Ave", "Los Angeles", "CA", "90069"),
        ("CUST-10008", "Olivia Chen", "olivia.chen@hotmail.com", "standard", "2024-01-18", 720,
         "(720) 555-0789", "3201 Blake St", "Denver", "CO", "80205"),
        ("CUST-10009", "Victoria St. James", "victoria.stjames@gmail.com", "vip", "2017-11-05", 9850,
         "(305) 555-0856", "500 Brickell Key Dr PH1", "Miami", "FL", "33131"),
        ("CUST-10010", "Natasha Volkov", "natasha.volkov@proton.me", "premium", "2020-08-30", 4680,
         "(617) 555-0934", "72 Beacon St", "Boston", "MA", "02108"),
        ("CUST-10011", "Grace Kim", "grace.kim@gmail.com", "standard", "2023-04-12", 1450,
         "(503) 555-1042", "1890 SE Hawthorne Blvd", "Portland", "OR", "97214"),
        ("CUST-10012", "Diana Rosetti", "diana.rosetti@icloud.com", "premium", "2021-10-03", 2890,
         "(619) 555-1167", "7402 La Jolla Blvd", "San Diego", "CA", "92037"),
        ("CUST-10013", "Harper Blackwell", "harper.blackwell@outlook.com", "vip", "2018-05-28", 13200,
         "(202) 555-1245", "3100 Massachusetts Ave NW", "Washington", "DC", "20008"),
        ("CUST-10014", "Luna Martinez", "luna.martinez@gmail.com", "standard", "2024-03-07", 540,
         "(602) 555-1398", "2150 E Camelback Rd", "Phoenix", "AZ", "85016"),
        ("CUST-10015", "Camille Dubois", "camille.dubois@yahoo.com", "premium", "2022-01-19", 3150,
         "(615) 555-1456", "1808 West End Ave", "Nashville", "TN", "37203"),
    ]

    conn.executemany(
        "INSERT INTO customers VALUES (?,?,?,?,?,?,?,?,?,?,?)", customers
    )
    conn.commit()
    return {c[0]: {"name": c[1], "tier": c[3]} for c in customers}


def populate_preferences(conn):
    prefs = [
        ("CUST-10001", '["Handbags","Outerwear","Dresses"]', '["VERO"]', "ultra-luxury", "white-glove", 1, 1, 1),
        ("CUST-10002", '["Knitwear","Tops","Accessories"]', '["VERO"]', "premium", "express", 1, 0, 1),
        ("CUST-10003", '["Dresses","Footwear","Accessories"]', '["VERO"]', "premium", "express", 1, 1, 0),
        ("CUST-10004", '["Dresses","Handbags","Footwear","Accessories"]', '["VERO"]', "ultra-luxury", "white-glove", 1, 1, 1),
        ("CUST-10005", '["Tops","Bottoms","Accessories"]', '["VERO"]', "premium", "express", 1, 0, 0),
        ("CUST-10006", '["Outerwear","Knitwear","Handbags"]', '["VERO"]', "ultra-luxury", "white-glove", 1, 1, 1),
        ("CUST-10007", '["Dresses","Tops","Footwear"]', '["VERO"]', "premium", "express", 1, 0, 1),
        ("CUST-10008", '["Knitwear","Accessories","Bottoms"]', '["VERO"]', "premium", "express", 0, 0, 1),
        ("CUST-10009", '["Dresses","Handbags","Accessories","Outerwear"]', '["VERO"]', "ultra-luxury", "white-glove", 1, 1, 1),
        ("CUST-10010", '["Outerwear","Knitwear","Tops"]', '["VERO"]', "premium", "express", 1, 1, 0),
        ("CUST-10011", '["Footwear","Accessories","Tops"]', '["VERO"]', "premium", "express", 1, 0, 1),
        ("CUST-10012", '["Handbags","Dresses","Bottoms"]', '["VERO"]', "premium", "express", 1, 0, 0),
        ("CUST-10013", '["Outerwear","Dresses","Handbags","Knitwear"]', '["VERO"]', "ultra-luxury", "white-glove", 1, 1, 1),
        ("CUST-10014", '["Tops","Accessories","Footwear"]', '["VERO"]', "premium", "express", 0, 0, 1),
        ("CUST-10015", '["Knitwear","Dresses","Accessories"]', '["VERO"]', "premium", "express", 1, 1, 0),
    ]
    conn.executemany(
        "INSERT INTO customer_preferences VALUES (?,?,?,?,?,?,?,?)", prefs
    )
    conn.commit()


def populate_orders(conn, product_map):
    random.seed(123)

    reviews_positive = [
        "Absolutely stunning quality. Worth every penny.",
        "The craftsmanship is exceptional. A true luxury piece.",
        "Beautiful design and impeccable finish. VERO never disappoints.",
        "Exceeded my expectations. The material is divine.",
        "A masterpiece of design. I receive compliments every time I wear it.",
        "Superb quality and elegant packaging. A wonderful experience.",
        "The attention to detail is remarkable. True artisan work.",
        "Luxurious feel and perfect fit. My new favorite piece.",
        "Gorgeous color and incredible softness. Obsessed!",
        "Five stars without hesitation. VERO defines luxury.",
        "Perfect addition to my wardrobe. The quality speaks for itself.",
        "Impeccable stitching and premium materials. A timeless investment.",
        "The leather smell alone tells you this is high quality.",
        "Elegant, sophisticated, and beautifully made.",
        "I own several VERO pieces and this is among the best.",
    ]

    # Define orders per customer: (customer_id, list of (product_id, date, status))
    order_specs = {
        # VIP: 8-12 orders each
        "CUST-10001": [
            ("PROD-1001", "2024-02-14"), ("PROD-1008", "2024-04-20"), ("PROD-1016", "2024-06-10"),
            ("PROD-1005", "2024-08-15"), ("PROD-1030", "2024-10-01"), ("PROD-1002", "2024-12-20"),
            ("PROD-1017", "2025-02-28"), ("PROD-1007", "2025-05-15"), ("PROD-1033", "2025-07-22"),
            ("PROD-1026", "2025-09-18"),
        ],
        "CUST-10004": [
            ("PROD-1017", "2024-01-20"), ("PROD-1002", "2024-03-08"), ("PROD-1019", "2024-05-12"),
            ("PROD-1026", "2024-06-30"), ("PROD-1031", "2024-08-22"), ("PROD-1004", "2024-10-15"),
            ("PROD-1016", "2025-01-05"), ("PROD-1029", "2025-03-18"), ("PROD-1035", "2025-06-10"),
            ("PROD-1023", "2025-08-04"), ("PROD-1009", "2025-10-20"), ("PROD-1030", "2025-11-30"),
        ],
        "CUST-10006": [
            ("PROD-1008", "2024-01-15"), ("PROD-1013", "2024-03-22"), ("PROD-1001", "2024-05-30"),
            ("PROD-1011", "2024-07-18"), ("PROD-1010", "2024-09-05"), ("PROD-1005", "2024-11-12"),
            ("PROD-1014", "2025-01-28"), ("PROD-1033", "2025-04-15"), ("PROD-1007", "2025-07-01"),
            ("PROD-1006", "2025-09-22"),
        ],
        "CUST-10009": [
            ("PROD-1016", "2024-02-05"), ("PROD-1003", "2024-04-12"), ("PROD-1034", "2024-06-20"),
            ("PROD-1017", "2024-08-08"), ("PROD-1030", "2024-09-30"), ("PROD-1008", "2024-11-25"),
            ("PROD-1019", "2025-02-14"), ("PROD-1001", "2025-04-30"), ("PROD-1035", "2025-07-08"),
            ("PROD-1009", "2025-10-01"), ("PROD-1025", "2025-11-15"),
        ],
        "CUST-10013": [
            ("PROD-1009", "2024-01-30"), ("PROD-1007", "2024-03-15"), ("PROD-1017", "2024-05-20"),
            ("PROD-1002", "2024-07-10"), ("PROD-1013", "2024-09-01"), ("PROD-1016", "2024-11-08"),
            ("PROD-1010", "2025-01-12"), ("PROD-1011", "2025-03-25"), ("PROD-1005", "2025-06-18"),
            ("PROD-1030", "2025-08-30"), ("PROD-1018", "2025-11-05"),
        ],
        # Premium: 4-7 orders each
        "CUST-10002": [
            ("PROD-1011", "2024-03-10"), ("PROD-1020", "2024-06-25"), ("PROD-1033", "2024-09-14"),
            ("PROD-1022", "2025-01-20"), ("PROD-1035", "2025-05-08"), ("PROD-1012", "2025-08-15"),
        ],
        "CUST-10003": [
            ("PROD-1018", "2024-02-28"), ("PROD-1026", "2024-05-15"), ("PROD-1034", "2024-08-20"),
            ("PROD-1031", "2024-11-30"), ("PROD-1028", "2025-03-10"), ("PROD-1019", "2025-07-22"),
            ("PROD-1035", "2025-10-18"),
        ],
        "CUST-10007": [
            ("PROD-1016", "2024-04-08"), ("PROD-1020", "2024-07-15"), ("PROD-1027", "2024-10-22"),
            ("PROD-1021", "2025-02-05"), ("PROD-1018", "2025-06-30"),
        ],
        "CUST-10010": [
            ("PROD-1007", "2024-03-20"), ("PROD-1012", "2024-06-12"), ("PROD-1022", "2024-09-28"),
            ("PROD-1010", "2025-01-15"), ("PROD-1014", "2025-05-20"), ("PROD-1008", "2025-09-10"),
        ],
        "CUST-10012": [
            ("PROD-1004", "2024-04-15"), ("PROD-1018", "2024-07-28"), ("PROD-1024", "2024-11-05"),
            ("PROD-1023", "2025-02-18"), ("PROD-1001", "2025-06-25"),
        ],
        "CUST-10015": [
            ("PROD-1014", "2024-05-02"), ("PROD-1033", "2024-08-18"), ("PROD-1015", "2024-11-22"),
            ("PROD-1035", "2025-03-08"), ("PROD-1011", "2025-07-15"), ("PROD-1031", "2025-10-30"),
        ],
        # Standard: 2-4 orders each
        "CUST-10005": [
            ("PROD-1020", "2024-10-05"), ("PROD-1032", "2025-01-22"), ("PROD-1023", "2025-06-10"),
        ],
        "CUST-10008": [
            ("PROD-1014", "2024-06-15"), ("PROD-1033", "2024-10-20"), ("PROD-1032", "2025-04-12"),
        ],
        "CUST-10011": [
            ("PROD-1028", "2024-07-08"), ("PROD-1035", "2024-11-15"), ("PROD-1026", "2025-03-28"),
            ("PROD-1031", "2025-08-05"),
        ],
        "CUST-10014": [
            ("PROD-1021", "2024-08-12"), ("PROD-1034", "2025-02-25"),
        ],
    }

    order_num = 1000
    all_orders = []
    for cust_id, specs in order_specs.items():
        for i, (prod_id, date) in enumerate(specs):
            order_num += 1
            name, cat, price = product_map[prod_id]
            qty = 1
            amount = price * qty

            # Determine status based on date
            if date <= "2025-08-01":
                status = "delivered"
            elif date <= "2025-10-15":
                status = "shipped"
            else:
                status = "processing"

            rating = None
            review = None
            if status == "delivered":
                rating = random.choice([4, 5, 5, 5, 5])
                review = random.choice(reviews_positive)

            oid = f"ORD-{date[:4]}-{order_num}"
            all_orders.append((oid, cust_id, prod_id, name, cat, date, amount, qty, status, rating, review))

    conn.executemany(
        "INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?)", all_orders
    )
    conn.commit()
    return all_orders


def populate_browsing(conn):
    random.seed(456)
    categories = ["Handbags", "Outerwear", "Knitwear", "Dresses", "Tops", "Bottoms", "Footwear", "Accessories"]

    browsing = []
    tier_config = {
        "vip": {"pages": (45, 120), "time": (25, 90), "sessions": (15, 40), "cats": 6, "prods": 10, "cart": 0, "wish": 6, "queries": 5},
        "premium": {"pages": (25, 70), "time": (15, 55), "sessions": (8, 25), "cats": 4, "prods": 7, "cart": 1, "wish": 4, "queries": 3},
        "standard": {"pages": (10, 35), "time": (8, 30), "sessions": (3, 12), "cats": 3, "prods": 4, "cart": 2, "wish": 2, "queries": 2},
    }

    customer_tiers = {
        "CUST-10001": "vip", "CUST-10002": "premium", "CUST-10003": "premium",
        "CUST-10004": "vip", "CUST-10005": "standard", "CUST-10006": "vip",
        "CUST-10007": "premium", "CUST-10008": "standard", "CUST-10009": "vip",
        "CUST-10010": "premium", "CUST-10011": "standard", "CUST-10012": "premium",
        "CUST-10013": "vip", "CUST-10014": "standard", "CUST-10015": "premium",
    }

    search_terms = [
        "cashmere sweater", "silk dress", "leather tote", "evening gown",
        "winter coat", "pearl earrings", "ankle boots", "designer wallet",
        "wool coat", "gold necklace", "cocktail dress", "silk blouse",
        "puffer coat", "leather pants", "cashmere scarf", "crossbody bag",
    ]

    all_prods = [f"PROD-{1001+i}" for i in range(35)]

    for cust_id, tier in customer_tiers.items():
        cfg = tier_config[tier]
        pages = random.randint(*cfg["pages"])
        mins = random.randint(*cfg["time"])
        time_str = f"{mins} min"
        sessions = random.randint(*cfg["sessions"])
        cats = json.dumps(random.sample(categories, cfg["cats"]))
        prods = json.dumps(random.sample(all_prods, cfg["prods"]))
        cart = random.randint(0, cfg["cart"])
        wish = json.dumps(random.sample(all_prods, cfg["wish"]))
        queries = json.dumps(random.sample(search_terms, cfg["queries"]))

        browsing.append((cust_id, pages, time_str, sessions, cats, prods, cart, wish, queries))

    conn.executemany(
        "INSERT INTO browsing_activity (customer_id, pages_viewed, time_on_site, sessions, categories_viewed, products_viewed, cart_abandonment, wishlist_items, search_queries) VALUES (?,?,?,?,?,?,?,?,?)",
        browsing
    )
    conn.commit()


def populate_segments(conn):
    random.seed(789)

    segments = {
        "vip": [
            ("Ultra-Luxury Connoisseur", "SEG-ULC"),
            ("Fashion Elite", "SEG-FE"),
        ],
        "premium": [
            ("Style-Forward Professional", "SEG-SFP"),
            ("Luxury Enthusiast", "SEG-LE"),
        ],
        "standard": [
            ("Emerging Luxury Shopper", "SEG-ELS"),
            ("Trend-Conscious Buyer", "SEG-TCB"),
        ],
    }

    customer_tiers = {
        "CUST-10001": "vip", "CUST-10002": "premium", "CUST-10003": "premium",
        "CUST-10004": "vip", "CUST-10005": "standard", "CUST-10006": "vip",
        "CUST-10007": "premium", "CUST-10008": "standard", "CUST-10009": "vip",
        "CUST-10010": "premium", "CUST-10011": "standard", "CUST-10012": "premium",
        "CUST-10013": "vip", "CUST-10014": "standard", "CUST-10015": "premium",
    }

    characteristics_pool = {
        "vip": [
            "Frequent high-value purchaser", "Brand loyalist", "Early adopter of new collections",
            "Prefers exclusive items", "Engages with personal styling services",
            "Attends VIP events", "Multi-category shopper", "Gift buyer",
        ],
        "premium": [
            "Regular seasonal buyer", "Quality-conscious", "Follows fashion trends",
            "Responsive to curated recommendations", "Values craftsmanship",
            "Growing purchase frequency", "Brand-aware",
        ],
        "standard": [
            "Occasion-driven buyer", "Price-aware luxury shopper", "Exploring brand offerings",
            "Responds to promotions", "Potential for tier upgrade",
            "Selective purchaser", "Digital-first shopper",
        ],
    }

    rows = []
    for cust_id, tier in customer_tiers.items():
        seg_name, seg_id = random.choice(segments[tier])
        if tier == "vip":
            engagement = random.randint(82, 98)
            purchase_prob = round(random.uniform(0.80, 0.95), 2)
            churn_risk = "low"
            ltv = round(random.uniform(35000, 85000), 2)
            predicted_ltv = round(ltv * random.uniform(1.1, 1.4), 2)
            chars = random.sample(characteristics_pool[tier], 4)
        elif tier == "premium":
            engagement = random.randint(60, 82)
            purchase_prob = round(random.uniform(0.55, 0.80), 2)
            churn_risk = random.choice(["low", "low", "medium"])
            ltv = round(random.uniform(12000, 35000), 2)
            predicted_ltv = round(ltv * random.uniform(1.05, 1.3), 2)
            chars = random.sample(characteristics_pool[tier], 3)
        else:
            engagement = random.randint(35, 60)
            purchase_prob = round(random.uniform(0.30, 0.55), 2)
            churn_risk = random.choice(["medium", "medium", "high"])
            ltv = round(random.uniform(3000, 12000), 2)
            predicted_ltv = round(ltv * random.uniform(1.0, 1.2), 2)
            chars = random.sample(characteristics_pool[tier], 3)

        rows.append((cust_id, seg_name, seg_id, engagement, purchase_prob, churn_risk, ltv, predicted_ltv, json.dumps(chars)))

    conn.executemany(
        "INSERT INTO customer_segments VALUES (?,?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()


def populate_similar_customers(conn):
    random.seed(321)
    common_items_pool = [
        "Cashmere Sweater", "Silk Slip Dress", "Classic Leather Tote",
        "Evening Gown", "Wool Tailored Coat", "Leather Pumps",
        "Gold Chain Necklace", "Cashmere Overcoat", "Designer Sunglasses",
        "Crossbody Messenger", "Merino Turtleneck", "Cocktail Dress",
    ]

    rows = []
    for i in range(1, 16):
        cust_id = f"CUST-{10000+i}"
        sim_count = random.randint(200, 900)
        sim_score = round(random.uniform(0.75, 0.95), 2)
        common = json.dumps(random.sample(common_items_pool, random.randint(3, 6)))
        rows.append((cust_id, sim_count, sim_score, common))

    conn.executemany(
        "INSERT INTO similar_customers VALUES (?,?,?,?)", rows
    )
    conn.commit()


def populate_recommendations(conn):
    random.seed(654)

    # Personalized recommendations per customer based on their preferences
    rec_map = {
        "CUST-10001": [
            ("PROD-1004", 0.94, "Based on your love of premium handbags and your Burgundy Tote purchase"),
            ("PROD-1009", 0.91, "Top-rated outerwear matching your luxury preference"),
            ("PROD-1019", 0.89, "Complements your Evening Gown and Silk Slip Dress collection"),
            ("PROD-1003", 0.87, "Perfect evening accessory to pair with your recent dress purchases"),
            ("PROD-1031", 0.84, "Trending among VIP customers with similar taste profiles"),
            ("PROD-1025", 0.81, "Luxury leather piece recommended by our style AI"),
        ],
        "CUST-10002": [
            ("PROD-1013", 0.92, "Premium cashmere to complement your Cashmere Sweater"),
            ("PROD-1014", 0.88, "Cable knit style popular among knitwear enthusiasts"),
            ("PROD-1021", 0.85, "Versatile silk top that pairs with your existing wardrobe"),
            ("PROD-1031", 0.83, "Accessory trending with customers who love knitwear"),
            ("PROD-1015", 0.80, "Knitwear dress that bridges your favorite categories"),
        ],
        "CUST-10003": [
            ("PROD-1016", 0.93, "Silk dress matching your preference for elegant eveningwear"),
            ("PROD-1029", 0.90, "Statement footwear that complements your dress collection"),
            ("PROD-1030", 0.87, "Gold necklace frequently bought with cocktail dresses"),
            ("PROD-1027", 0.84, "Ankle boots popular among style-forward professionals"),
            ("PROD-1017", 0.82, "Evening gown recommended based on your browsing history"),
            ("PROD-1032", 0.79, "Leather accessory to complete your look"),
        ],
        "CUST-10004": [
            ("PROD-1008", 0.95, "Luxury cashmere outerwear for your ultra-luxury wardrobe"),
            ("PROD-1003", 0.92, "Evening clutch to complement your gown collection"),
            ("PROD-1025", 0.89, "Leather pants trending among Fashion Elite customers"),
            ("PROD-1011", 0.86, "Cashmere sweater popular with customers who share your taste"),
            ("PROD-1034", 0.83, "Designer sunglasses frequently paired with your style profile"),
            ("PROD-1018", 0.81, "Wrap dress in your preferred Burgundy colorway"),
            ("PROD-1010", 0.78, "Luxury puffer coat for your outerwear collection"),
        ],
        "CUST-10005": [
            ("PROD-1021", 0.85, "Satin camisole that pairs beautifully with your Tailored Trousers"),
            ("PROD-1035", 0.82, "Silk scarf accessory at an accessible price point"),
            ("PROD-1024", 0.79, "Pleated skirt to expand your bottoms collection"),
            ("PROD-1033", 0.76, "Cashmere scarf trending among similar shoppers"),
            ("PROD-1028", 0.73, "Designer sneakers popular with trend-conscious buyers"),
        ],
        "CUST-10006": [
            ("PROD-1009", 0.93, "Premium shearling to elevate your outerwear collection"),
            ("PROD-1012", 0.90, "Fine merino turtleneck complementing your knitwear pieces"),
            ("PROD-1002", 0.88, "Structured bag matching your handbag preferences"),
            ("PROD-1015", 0.85, "Knitwear dress bridging two of your favorite categories"),
            ("PROD-1030", 0.82, "Gold statement piece popular among Ultra-Luxury Connoisseurs"),
            ("PROD-1003", 0.79, "Evening clutch for your accessory collection"),
        ],
        "CUST-10007": [
            ("PROD-1019", 0.91, "Cocktail dress matching your love of eveningwear"),
            ("PROD-1022", 0.88, "Cashmere shell perfect for layering with your silk pieces"),
            ("PROD-1026", 0.85, "Classic pumps to complete your outfit collection"),
            ("PROD-1029", 0.82, "Over-the-knee boots trending this season"),
            ("PROD-1017", 0.79, "Evening gown recommended based on your style profile"),
        ],
        "CUST-10008": [
            ("PROD-1011", 0.84, "Cashmere sweater complementing your Cable Knit purchase"),
            ("PROD-1035", 0.81, "Silk scarf accessory popular among similar shoppers"),
            ("PROD-1024", 0.78, "Pleated skirt to expand your bottoms collection"),
            ("PROD-1031", 0.75, "Pearl earrings frequently bought with cashmere pieces"),
            ("PROD-1023", 0.72, "Tailored trousers matching your style preferences"),
        ],
        "CUST-10009": [
            ("PROD-1002", 0.94, "Structured bag to complement your Classic Leather Tote"),
            ("PROD-1018", 0.91, "Wrap dress in a rich colorway you tend to prefer"),
            ("PROD-1031", 0.88, "Pearl earrings popular among your customer segment"),
            ("PROD-1010", 0.85, "Luxury puffer coat for transitional seasons"),
            ("PROD-1004", 0.82, "Crossbody messenger highly rated by VIP customers"),
            ("PROD-1011", 0.80, "Cashmere sweater for casual luxury moments"),
            ("PROD-1015", 0.77, "Turtleneck dress trending among Fashion Elite"),
        ],
        "CUST-10010": [
            ("PROD-1009", 0.90, "Shearling jacket to elevate your outerwear wardrobe"),
            ("PROD-1013", 0.87, "Cashmere cardigan complementing your turtleneck"),
            ("PROD-1011", 0.84, "Pure cashmere sweater matching your knitwear taste"),
            ("PROD-1020", 0.81, "Silk blouse popular among style-forward professionals"),
            ("PROD-1015", 0.78, "Merino dress bridging knitwear and fashion"),
            ("PROD-1005", 0.75, "Weekend travel bag trending among similar customers"),
        ],
        "CUST-10011": [
            ("PROD-1027", 0.86, "Ankle boots to complement your sneaker collection"),
            ("PROD-1034", 0.83, "Designer sunglasses popular with your customer segment"),
            ("PROD-1032", 0.80, "Leather belt accessory trending this season"),
            ("PROD-1020", 0.77, "Silk blouse to expand your tops collection"),
            ("PROD-1029", 0.74, "Over-the-knee boots for a statement look"),
        ],
        "CUST-10012": [
            ("PROD-1002", 0.89, "Structured bag matching your handbag preference"),
            ("PROD-1016", 0.86, "Silk slip dress to expand your dress wardrobe"),
            ("PROD-1025", 0.83, "Leather pants trending among luxury enthusiasts"),
            ("PROD-1005", 0.80, "Weekend travel bag for the discerning traveler"),
            ("PROD-1019", 0.77, "Cocktail dress popular among similar customers"),
            ("PROD-1030", 0.74, "Gold necklace to accessorize your outfits"),
        ],
        "CUST-10013": [
            ("PROD-1008", 0.93, "Cashmere overcoat to complement your shearling jacket"),
            ("PROD-1019", 0.90, "Cocktail dress matching your eveningwear collection"),
            ("PROD-1004", 0.87, "Crossbody messenger popular among Fashion Elite"),
            ("PROD-1012", 0.84, "Merino turtleneck complementing your cashmere pieces"),
            ("PROD-1031", 0.82, "Pearl earrings trending among VIP customers"),
            ("PROD-1015", 0.79, "Turtleneck dress bridging knitwear and dresses"),
            ("PROD-1025", 0.76, "Leather pants for a bold statement look"),
            ("PROD-1034", 0.73, "Designer sunglasses to complete your accessories"),
        ],
        "CUST-10014": [
            ("PROD-1020", 0.83, "Silk blouse complementing your Satin Camisole"),
            ("PROD-1033", 0.80, "Cashmere scarf at an accessible luxury price point"),
            ("PROD-1028", 0.77, "Designer sneakers trending among similar shoppers"),
            ("PROD-1032", 0.74, "Leather belt to accessorize your wardrobe"),
            ("PROD-1026", 0.71, "Classic pumps popular among trend-conscious buyers"),
        ],
        "CUST-10015": [
            ("PROD-1013", 0.91, "Cashmere cardigan matching your knitwear love"),
            ("PROD-1016", 0.88, "Silk slip dress for your dress collection"),
            ("PROD-1012", 0.85, "Merino turtleneck complementing your cashmere pieces"),
            ("PROD-1030", 0.82, "Gold necklace trending among luxury enthusiasts"),
            ("PROD-1034", 0.79, "Designer sunglasses for a polished look"),
            ("PROD-1018", 0.76, "Wrap dress popular among your customer segment"),
        ],
    }

    rows = []
    for cust_id, recs in rec_map.items():
        for prod_id, score, reason in recs:
            rows.append((cust_id, prod_id, score, reason, "v2.4.1"))

    conn.executemany(
        "INSERT INTO recommendations (customer_id, product_id, ml_score, reason, model_version) VALUES (?,?,?,?,?)",
        rows
    )
    conn.commit()


def main():
    # Remove existing DB if present
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    print(f"Creating database at {DB_PATH}...")
    create_tables(conn)
    print("  Tables created.")

    product_map = populate_products(conn)
    print(f"  Inserted {len(product_map)} products.")

    customer_info = populate_customers(conn)
    print(f"  Inserted {len(customer_info)} customers.")

    populate_preferences(conn)
    print("  Inserted customer preferences.")

    orders = populate_orders(conn, product_map)
    print(f"  Inserted {len(orders)} orders.")

    populate_browsing(conn)
    print("  Inserted browsing activity.")

    populate_segments(conn)
    print("  Inserted customer segments.")

    populate_similar_customers(conn)
    print("  Inserted similar customers.")

    populate_recommendations(conn)
    print("  Inserted recommendations.")

    # Print summary
    print("\n--- Database Summary ---")
    for table in ["customers", "customer_preferences", "products", "orders",
                   "browsing_activity", "customer_segments", "similar_customers", "recommendations"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count} rows")

    conn.close()
    file_size = os.path.getsize(DB_PATH)
    print(f"\nDatabase file size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    print("Done!")


if __name__ == "__main__":
    main()
