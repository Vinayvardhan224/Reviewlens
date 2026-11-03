"""
Script to generate standard data/products.csv and data/reviews.csv
conforming to the ReviewLens Blueprint specifications with explicit prepared_demo provenance.
"""

import json
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PRODUCTS = [
    {
        "product_id": "samsung-galaxy-s24-fe",
        "product_name": "Samsung Galaxy S24 FE 5G (8GB RAM, 128GB)",
        "brand": "Samsung",
        "category": "Smartphones",
        "price": 39999.0,
        "rating": 4.5,
        "review_count": 79752,
        "description": "Premium Galaxy AI flagship experience featuring 6.7-inch Dynamic AMOLED 2X 120Hz display, Exynos 2400e processor with enlarged vapor chamber, 50MP triple camera system, and 7 years of promised OS updates.",
        "amazon_url": "https://www.amazon.in/s?k=Samsung+Galaxy+S24+FE",
        "image_url": "https://images.unsplash.com/photo-1610945265064-0e34e5519bbf?auto=format&fit=crop&w=600&q=80",
        "currency": "INR",
        "specifications": json.dumps({
            "Processor": "Samsung Exynos 2400e Deca-Core (4nm)",
            "Display": "6.7-inch Dynamic AMOLED 2X, 120Hz FHD+ (1900 nits peak)",
            "Camera": "Triple: 50MP Primary (OIS) + 12MP Ultra-Wide + 8MP 3x Telephoto",
            "Front Camera": "10MP Wide (f/2.4)",
            "Battery": "4700 mAh with 25W fast charging & 15W wireless",
            "OS & Support": "Android 14 (One UI 6.1) with 7 Years OS Updates",
            "Build": "Armor Aluminum frame, Gorilla Glass Victus+ back, IP68 water resistance",
            "Weight": "213g"
        }),
        "availability": "In Stock",
        "data_type": "prepared_demo"
    },
    {
        "product_id": "apple-iphone-15-pro-max",
        "product_name": "Apple iPhone 15 Pro Max (256 GB, Natural Titanium)",
        "brand": "Apple",
        "category": "Smartphones",
        "price": 159900.0,
        "rating": 4.6,
        "review_count": 3189,
        "description": "Forged in aerospace-grade Titanium with Apple A17 Pro 3nm chip, customizable Action Button, 48MP main camera with 5x optical tetraprism zoom, and universal USB-C 3.0 connector.",
        "amazon_url": "https://www.amazon.in/s?k=iPhone+15+Pro+Max",
        "image_url": "https://images.unsplash.com/photo-1695048133142-1a20484d2569?auto=format&fit=crop&w=600&q=80",
        "currency": "INR",
        "specifications": json.dumps({
            "Processor": "Apple A17 Pro (3nm Hexa-Core with Hardware Ray Tracing)",
            "Display": "6.7-inch Super Retina XDR OLED, 120Hz ProMotion (2000 nits)",
            "Camera": "Triple: 48MP Main (Sensor-Shift OIS) + 12MP Ultra-Wide + 12MP 5x Tetraprism Zoom",
            "Front Camera": "12MP TrueDepth with Autofocus",
            "Battery": "4441 mAh with 20W wired & 15W MagSafe wireless",
            "OS & Support": "iOS 17 (Dynamic Island, 5+ Years OS updates)",
            "Build": "Grade 5 Titanium frame, Ceramic Shield glass, IP68 (6m depth)",
            "Weight": "221g"
        }),
        "availability": "In Stock",
        "data_type": "prepared_demo"
    },
    {
        "product_id": "samsung-galaxy-s25-ultra",
        "product_name": "Samsung Galaxy S25 Ultra 5G (12GB RAM, 256GB)",
        "brand": "Samsung",
        "category": "Smartphones",
        "price": 129999.0,
        "rating": 4.8,
        "review_count": 12450,
        "description": "Apex Android super-flagship featuring Snapdragon 8 Elite 3nm architecture, 200MP Quad Camera with dual telephotos, Gorilla Armor anti-reflective glass, and built-in S-Pen.",
        "amazon_url": "https://www.amazon.in/s?k=Samsung+Galaxy+S25+Ultra",
        "image_url": "https://images.unsplash.com/photo-1580910051074-3eb694886505?auto=format&fit=crop&w=600&q=80",
        "currency": "INR",
        "specifications": json.dumps({
            "Processor": "Qualcomm Snapdragon 8 Elite for Galaxy (3nm)",
            "Display": "6.86-inch Dynamic AMOLED 2X, LTPO 120Hz QHD+ (2600 nits)",
            "Camera": "Quad: 200MP (Wide OIS) + 50MP UW + 50MP 5x Periscope + 10MP 3x Telephoto",
            "Front Camera": "12MP Dual Pixel AF",
            "Battery": "5000 mAh with 45W fast charging & 15W wireless",
            "OS & Support": "Android 15 (One UI 7) with 7 Years OS Updates",
            "Build": "Grade 5 Titanium chassis, Gorilla Armor anti-reflective, IP68, S-Pen",
            "Weight": "218g"
        }),
        "availability": "In Stock",
        "data_type": "prepared_demo"
    },
    {
        "product_id": "oneplus-12",
        "product_name": "OnePlus 12 5G (12GB RAM, 256GB, Silky Black)",
        "brand": "OnePlus",
        "category": "Smartphones",
        "price": 64999.0,
        "rating": 4.4,
        "review_count": 8920,
        "description": "Speed powerhouse featuring Qualcomm Snapdragon 8 Gen 3, 4th Gen Hasselblad camera with 64MP periscope telephoto, 5400 mAh battery, and ultra-fast 100W SUPERVOOC charging.",
        "amazon_url": "https://www.amazon.in/s?k=OnePlus+12",
        "image_url": "https://images.unsplash.com/photo-1592899677977-9c10ca588bbd?auto=format&fit=crop&w=600&q=80",
        "currency": "INR",
        "specifications": json.dumps({
            "Processor": "Qualcomm Snapdragon 8 Gen 3 (4nm)",
            "Display": "6.82-inch ProXDR 2K OLED, 120Hz LTPO (4500 nits peak)",
            "Camera": "Triple Hasselblad: 50MP Sony LYT-808 + 48MP UW + 64MP 3x Periscope",
            "Front Camera": "32MP Sony IMX615",
            "Battery": "5400 mAh with 100W wired & 50W AIRVOOC wireless",
            "OS & Support": "OxygenOS 14 (4 Years OS Updates)",
            "Build": "Glass back, aluminum frame, IP65 dust and water resistance",
            "Weight": "220g"
        }),
        "availability": "In Stock",
        "data_type": "prepared_demo"
    },
    {
        "product_id": "google-pixel-8-pro",
        "product_name": "Google Pixel 8 Pro (12GB RAM, 128GB, Bay Blue)",
        "brand": "Google",
        "category": "Smartphones",
        "price": 79999.0,
        "rating": 4.3,
        "review_count": 5120,
        "description": "Engineered by Google with Tensor G3 processor, computational photography with Magic Editor and Best Take, Super Actua display, and 7 years of direct Pixel Feature Drops.",
        "amazon_url": "https://www.amazon.in/s?k=Google+Pixel+8+Pro",
        "image_url": "https://images.unsplash.com/photo-1598327105666-5b89351aff97?auto=format&fit=crop&w=600&q=80",
        "currency": "INR",
        "specifications": json.dumps({
            "Processor": "Google Tensor G3 with Titan M2 security coprocessor",
            "Display": "6.7-inch Super Actua LTPO OLED 120Hz (2400 nits peak)",
            "Camera": "Triple: 50MP Octa PD Main + 48MP Quad PD UW + 48MP 5x Telephoto",
            "Front Camera": "10.5MP Dual PD with autofocus",
            "Battery": "5050 mAh with 30W wired & 23W wireless",
            "OS & Support": "Clean Android 14 with 7 Years direct OS & Security Drops",
            "Build": "Polished aluminum frame, matte Victus 2 back, IP68",
            "Weight": "213g"
        }),
        "availability": "In Stock",
        "data_type": "prepared_demo"
    },
    {
        "product_id": "sony-wh1000xm5",
        "product_name": "Sony WH-1000XM5 Wireless Noise Cancelling Headphones",
        "brand": "Sony",
        "category": "Audio",
        "price": 29990.0,
        "rating": 4.6,
        "review_count": 14200,
        "description": "Industry-leading active noise cancellation with 8 microphones and Auto NC Optimizer, LDAC High-Resolution Audio, crystal clear hands-free calls, and 30-hour battery life.",
        "amazon_url": "https://www.amazon.in/s?k=Sony+WH-1000XM5",
        "image_url": "https://images.unsplash.com/photo-1546435770-a3e426bf472b?auto=format&fit=crop&w=600&q=80",
        "currency": "INR",
        "specifications": json.dumps({
            "Driver Size": "30mm precision-engineered carbon fiber composite",
            "Noise Cancellation": "Dual Processors (V1 + QN1) with 8 microphones",
            "Battery Life": "30 hours with ANC On (3 min quick charge gives 3 hours)",
            "Bluetooth Codecs": "LDAC, AAC, SBC with multipoint connection",
            "Weight": "250g ultra-lightweight ergonomic design",
            "Microphones": "4 beamforming mics with AI noise reduction"
        }),
        "availability": "In Stock",
        "data_type": "prepared_demo"
    },
    {
        "product_id": "apple-airpods-pro-2",
        "product_name": "Apple AirPods Pro (2nd Gen) with MagSafe Case (USB-C)",
        "brand": "Apple",
        "category": "Audio",
        "price": 24900.0,
        "rating": 4.7,
        "review_count": 28400,
        "description": "Apple-designed H2 chip delivers up to 2x more Active Noise Cancellation, Adaptive Audio, Personalized Spatial Audio with dynamic head tracking, and dust/water-resistant case.",
        "amazon_url": "https://www.amazon.in/s?k=AirPods+Pro+2+USB-C",
        "image_url": "https://images.unsplash.com/photo-1600294037681-c80b4cb5b434?auto=format&fit=crop&w=600&q=80",
        "currency": "INR",
        "specifications": json.dumps({
            "Processor": "Apple H2 headphone chip in earbuds, U1 in MagSafe Case",
            "ANC": "Up to 2x more Active Noise Cancellation + Transparency Mode",
            "Battery Life": "6 hours listening (30 hours total with charging case)",
            "Charging": "USB-C, MagSafe, Apple Watch charger, or Qi certified",
            "Resistance": "IP54 dust, sweat, and water resistance for buds & case",
            "Features": "Adaptive Audio, Conversation Awareness, Spatial Audio"
        }),
        "availability": "In Stock",
        "data_type": "prepared_demo"
    },
    {
        "product_id": "bose-qc-ultra-headphones",
        "product_name": "Bose QuietComfort Ultra Wireless Noise Cancelling Headphones",
        "brand": "Bose",
        "category": "Audio",
        "price": 35900.0,
        "rating": 4.5,
        "review_count": 6700,
        "description": "World-class noise cancellation with breakthrough Bose Immersive Audio, luxurious plush protein leather cushions, CustomTune technology, and up to 24 hours of playback.",
        "amazon_url": "https://www.amazon.in/s?k=Bose+QuietComfort+Ultra",
        "image_url": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?auto=format&fit=crop&w=600&q=80",
        "currency": "INR",
        "specifications": json.dumps({
            "Audio Tech": "Bose Immersive Audio spatial sound + CustomTune acoustic calibration",
            "Noise Cancellation": "Quiet Mode, Aware Mode, and Immersion Mode",
            "Battery Life": "24 hours (18 hours with Immersive Audio)",
            "Build": "Cast aluminum arms, protein leather earcups, collapsible fold",
            "Weight": "252g",
            "Codecs": "aptX Adaptive, AAC, SBC"
        }),
        "availability": "In Stock",
        "data_type": "prepared_demo"
    },
    {
        "product_id": "apple-macbook-air-m3",
        "product_name": "Apple MacBook Air 15-inch M3 Chip (16GB RAM, 512GB SSD)",
        "brand": "Apple",
        "category": "Laptops",
        "price": 134900.0,
        "rating": 4.8,
        "review_count": 4890,
        "description": "Ultra-thin aluminum laptop powered by Apple M3 8-core CPU / 10-core GPU, stunning 15.3-inch Liquid Retina display, dual external display support, and all-day 18-hour battery.",
        "amazon_url": "https://www.amazon.in/s?k=MacBook+Air+15+M3",
        "image_url": "https://images.unsplash.com/photo-1517336714731-489689fd1ca8?auto=format&fit=crop&w=600&q=80",
        "currency": "INR",
        "specifications": json.dumps({
            "Processor": "Apple M3 (8-core CPU, 10-core GPU, 16-core Neural Engine)",
            "Display": "15.3-inch Liquid Retina IPS (2880 x 1864, 500 nits, True Tone)",
            "Memory & SSD": "16GB Unified Memory, 512GB High-Speed SSD",
            "Battery Life": "Up to 18 hours Apple TV app movie playback (66.5Wh)",
            "Weight & Profile": "1.51 kg | 11.5mm thin fanless silent aluminum unibody",
            "Ports": "MagSafe 3, two Thunderbolt / USB 4 ports, 3.5mm headphone jack"
        }),
        "availability": "In Stock",
        "data_type": "prepared_demo"
    },
    {
        "product_id": "asus-rog-zephyrus-g14",
        "product_name": "ASUS ROG Zephyrus G14 (2024) OLED Gaming Laptop",
        "brand": "ASUS",
        "category": "Laptops",
        "price": 174990.0,
        "rating": 4.6,
        "review_count": 2180,
        "description": "Precision-crafted gaming laptop with AMD Ryzen 9 8945HS, NVIDIA GeForce RTX 4070 GPU, vibrant 3K 120Hz ROG Nebula OLED display, and Slash Lighting aluminum CNC chassis.",
        "amazon_url": "https://www.amazon.in/s?k=ROG+Zephyrus+G14+2024",
        "image_url": "https://images.unsplash.com/photo-1603302576837-37561b2e2302?auto=format&fit=crop&w=600&q=80",
        "currency": "INR",
        "specifications": json.dumps({
            "Processor": "AMD Ryzen 9 8945HS (8 Cores / 16 Threads, up to 5.2 GHz with Ryzen AI)",
            "Graphics": "NVIDIA GeForce RTX 4070 (8GB GDDR6, 90W TGP)",
            "Display": "14.0-inch 3K (2880 x 1800) OLED 120Hz 0.2ms, 100% DCI-P3 G-Sync",
            "RAM & SSD": "32GB LPDDR5X-6400 RAM, 1TB PCIe 4.0 NVMe SSD",
            "Weight & Build": "1.50 kg | All-aluminum CNC chassis with Slash Lighting",
            "Cooling": "ROG Intelligent Cooling with tri-fan technology and liquid metal"
        }),
        "availability": "In Stock",
        "data_type": "prepared_demo"
    },
    {
        "product_id": "sony-alpha-7-iv",
        "product_name": "Sony Alpha 7 IV Full-Frame Mirrorless Camera (Body Only)",
        "brand": "Sony",
        "category": "Cameras",
        "price": 214990.0,
        "rating": 4.8,
        "review_count": 1820,
        "description": "Hybrid powerhouse with 33MP full-frame Exmor R CMOS sensor, BIONZ XR processing engine, 4K 60p 10-bit 4:2:2 recording, real-time eye autofocus, and 5-axis optical image stabilization.",
        "amazon_url": "https://www.amazon.in/s?k=Sony+Alpha+7+IV",
        "image_url": "https://images.unsplash.com/photo-1516035069371-29a1b244cc32?auto=format&fit=crop&w=600&q=80",
        "currency": "INR",
        "specifications": json.dumps({
            "Sensor": "33.0 Megapixel Full-Frame back-illuminated Exmor R CMOS",
            "Image Processor": "BIONZ XR image processing engine (8x faster)",
            "Autofocus": "759 phase-detection AF points with Real-time Eye AF for Humans/Animals/Birds",
            "Video": "4K 60p 10-bit 4:2:2 internal recording, S-Cinetone, S-Log3",
            "Stabilization": "5-axis optical in-body image stabilization (up to 5.5 stops)",
            "Viewfinder": "3.68 million dot Quad-VGA OLED electronic viewfinder"
        }),
        "availability": "In Stock",
        "data_type": "prepared_demo"
    },
    {
        "product_id": "apple-watch-series-9",
        "product_name": "Apple Watch Series 9 GPS (45mm Starlight Aluminum)",
        "brand": "Apple",
        "category": "Wearables",
        "price": 41900.0,
        "rating": 4.6,
        "review_count": 9400,
        "description": "Powerful health and fitness tracker with S9 SiP chip, double tap gesture interaction, 2000-nit Always-On Retina display, on-device Siri, and comprehensive ECG & blood oxygen metrics.",
        "amazon_url": "https://www.amazon.in/s?k=Apple+Watch+Series+9",
        "image_url": "https://images.unsplash.com/photo-1508685096489-7aacd43bd3b1?auto=format&fit=crop&w=600&q=80",
        "currency": "INR",
        "specifications": json.dumps({
            "Chipset": "Apple S9 SiP 64-bit dual-core with 4-core Neural Engine",
            "Display": "Always-On Retina OLED, edge-to-edge, up to 2000 nits peak",
            "Health Sensors": "ECG app, optical heart rate, blood oxygen, temperature sensing",
            "Battery": "18 hours regular use (36 hours in Low Power Mode) with fast charging",
            "Interaction": "Double Tap magic gesture, on-device Siri processing",
            "Durability": "Crack-resistant front crystal, IP6X dust-resistant, 50m water resistant"
        }),
        "availability": "In Stock",
        "data_type": "prepared_demo"
    }
]

# Rich review repository covering all 8 aspects for each product with authentic-style text
REVIEWS = []

# Template generator helper to populate varied reviews per product
REVIEWS_TEMPLATES = {
    "samsung-galaxy-s24-fe": [
        ("The Dynamic AMOLED display is gorgeous and super smooth at 120Hz. Battery life easily lasts me a full day with 6 to 7 hours of screen-on-time. For ₹39,999, getting 7 years of Android updates is unreal value!", 5.0, True),
        ("Camera photos from the 50MP main sensor are sharp with vibrant Samsung colors. Night mode is solid. However, the 10MP selfie camera is a bit grainy in low light.", 4.0, True),
        ("Performance is great for daily multitasking and light gaming. The phone stays relatively cool thanks to the enlarged vapor chamber. But 25W charging feels slow in 2024, and no charger in box is annoying.", 4.0, True),
        ("Solid build quality with Gorilla Glass Victus+ back and aluminum rails. Feels premium in hand. The speakers are loud with clear stereo separation.", 5.0, True),
        ("Average battery charging speeds. Takes over 75 minutes to charge from 0 to 100%. Otherwise, software One UI is clean and full of useful AI features.", 3.0, True),
        ("Unbeatable value for money. Why spend 1.5 lakhs on an Ultra when this S24 FE handles everything flawlessly? Display is bright even under scorching outdoor sunlight.", 5.0, True),
        ("Very pleased with the cameras and overall speed. Phone doesn't lag at all. Only drawback is the telephoto zoom is limited to 3x optical, but portrait shots look stunning.", 4.0, True),
        ("Battery drains a little fast while recording 4K video or heavy navigation. Still gets me through a typical workday. Software support for 7 years gives huge peace of mind.", 4.0, True),
        ("Great phone overall, but customer support at the service center was slow when I asked about screen protector replacement. Phone hardware itself is 5 stars.", 4.0, False),
        ("Extremely happy with the purchase. Best mid-range flagship under 40k. Smooth screen, crisp cameras, solid battery, and excellent Samsung ecosystem integration.", 5.0, True)
    ],
    "apple-iphone-15-pro-max": [
        ("The Titanium finish feels so much lighter and comfortable than previous stainless steel models. Apple A17 Pro runs every heavy game like Resident Evil 4 with console-quality graphics.", 5.0, True),
        ("The 5x optical tetraprism camera is a game changer for concert videos and distant portraits. Video quality in 4K ProRes is simply unmatched by any Android phone.", 5.0, True),
        ("Display is breathtaking with 2000 nits outdoor peak brightness. ProMotion 120Hz is buttery smooth. Finally having USB-C makes file transfers to external SSDs instantaneous.", 5.0, True),
        ("Extremely expensive at ₹1,59,900. While the hardware is top notch, the phone gets noticeably warm during 30 minutes of heavy gaming or fast charging. Battery life is good but charging is still slow.", 3.0, True),
        ("Battery life lasts almost 1.5 days on normal usage. The Action button is handy for quickly toggling silent mode or opening camera. Sound quality from stereo speakers is rich and deep.", 5.0, True),
        ("Camera lens reflections and flare at night are still present. For this astronomical price tag, Apple should have solved the lens ghosting. Still, day shots and video are best in class.", 4.0, True),
        ("Build quality is superb, feels ultra-durable. iOS 17 is fluid with zero stutter. Face ID works in pitch darkness. You pay a heavy premium, but it is undeniably an elite device.", 5.0, True),
        ("Slow 20W charging takes nearly 2 hours to top up completely. At 1.6 lakhs, they don't even include a charging brick. Value for money is quite poor, though performance is exceptional.", 3.0, True),
        ("Superb display and lightning-fast processor. Dynamic Island has grown on me with live delivery updates and music controls. Customer care and Apple store experience was top notch.", 5.0, True),
        ("Pure luxury. 5x zoom captures crisp details across the stadium. Battery easily survives heavy travel photography. Worth it if budget is not a constraint.", 5.0, True)
    ],
    "samsung-galaxy-s25-ultra": [
        ("The Snapdragon 8 Elite is an absolute monster. Zero frame drops in Genshin Impact. And the Gorilla Armor anti-reflective screen completely eliminates ambient glare. Revolutionary display!", 5.0, True),
        ("200MP camera produces unbelievable detail. Having both 3x and 5x optical telephotos gives incredible versatility. Battery easily lasts 1.5 days on heavy use.", 5.0, True),
        ("The built-in S-Pen is so practical for signing PDFs, annotating screenshots, and taking quick notes. Titanium design with flat sides feels much better to hold than the curved screens.", 5.0, True),
        ("Amazing device, but the boxy corners make it uncomfortable in small pockets. It is also very expensive at ₹1.3 lakh. 45W charging is decent but competitors offer 100W.", 4.0, True),
        ("Speakers tuned by AKG are loud and punchy. Software One UI 7 feels fast and refined with smart Galaxy AI translations. Truly the undisputed king of Android flagships.", 5.0, True),
        ("Camera shutter lag is finally eliminated. Photos of moving pets and kids come out tack sharp. Battery charging from 0 to 65% in 30 minutes is very acceptable.", 5.0, True),
        ("Heavy and bulky to use with one hand. Screen and cameras are unbelievable, but price is quite steep. Best phone on the market if you don't mind the size.", 4.0, True),
        ("Outstanding battery life, easily 9 hours of screen on time. Display brightness in direct sunlight is unmatched. S-Pen air actions work like magic for presentations.", 5.0, True)
    ],
    "oneplus-12": [
        ("100W SUPERVOOC charging is mind-blowing! Goes from 10% to 100% in just 26 minutes. 5400 mAh battery easily lasts almost two full days of regular use.", 5.0, True),
        ("Snapdragon 8 Gen 3 delivers silky smooth 120fps gaming. The Hasselblad periscope camera takes stunning portraits with natural bokeh and true-to-life skin tones.", 5.0, True),
        ("Amazing value for ₹64,999 compared to Samsung and Apple flagships. Beautiful 2K display with curved edges. Only downside is IP65 rating instead of IP68.", 4.0, True),
        ("Curved display leads to occasional accidental palm touches while typing. OxygenOS is fast but has some minor ColorOS bloatware notifications that need disabling.", 3.0, True),
        ("Great audio quality with Dolby Atmos. Haptic feedback is crisp and precise. Outstanding performance and battery for the price tag.", 5.0, True),
        ("Fastest charging phone I have ever owned. Camera is surprisingly good, especially the 3x periscope zoom. Great alternative to overpriced flagships.", 5.0, True)
    ],
    "google-pixel-8-pro": [
        ("The AI photo editing tools (Magic Editor and Best Take) are wizardry. Point-and-shoot camera consistency is the best in the industry. Skin tones look 100% natural.", 5.0, True),
        ("Clean stock Android experience with zero bloatware and 7 years of promised OS updates. The Super Actua display is vivid and very bright outdoors.", 5.0, True),
        ("Tensor G3 runs warm during extended 4K 60fps video recording or outdoor gaming. Modem signal in underground metro could be stronger.", 3.0, True),
        ("Battery life is average, lasts about a full day (5 hours SOT). Charging speed is moderate at 30W. But the camera and AI software make up for it.", 4.0, True),
        ("Love the matte glass back and camera visor design. Voice typing with assistant is unmatched in accuracy. Highly recommend for pure Android lovers.", 5.0, True)
    ],
    "sony-wh1000xm5": [
        ("The active noise cancellation is pure magic. Completely silences airplane cabin drone, metro rumble, and cafe chatter. Soundstage is wide with rich bass.", 5.0, True),
        ("Extremely comfortable for 8+ hour work sessions. 30 hours of battery life means I charge it only once every two weeks. Microphone quality on Zoom calls is crystal clear.", 5.0, True),
        ("The redesign does not fold as compactly as the older XM4, making the travel case a bit bulky. Touch gestures take some getting used to.", 4.0, True),
        ("LDAC codec with high-res audio tracks sounds phenomenal. Multi-point connection switches seamlessly between my MacBook and phone.", 5.0, True),
        ("A bit expensive at ₹29,990, but worth every single rupee if you travel often or work in a noisy office environment.", 5.0, True)
    ],
    "apple-airpods-pro-2": [
        ("Noise cancellation on these tiny earbuds is shockingly good. Adaptive Audio smoothly blends transparency mode when someone talks to you.", 5.0, True),
        ("Spatial Audio with head tracking makes watching movies on Apple TV or iPad feel like a cinema. Sound is balanced with tight punchy bass.", 5.0, True),
        ("USB-C case with speaker chirp for Find My is super convenient. Battery easily lasts 6 hours on a single charge and case charges rapidly.", 5.0, True),
        ("Fit is comfortable with the XS/S/M/L silicone tips. Touch swipe volume controls on the stems work reliably without reaching for your phone.", 5.0, True),
        ("Expensive for wireless earbuds, and spatial audio features are limited if you are an Android user. For Apple users, it is a no-brainer 10/10.", 4.0, True)
    ],
    "apple-macbook-air-m3": [
        ("The fanless M3 design is completely silent yet blasts through 4K video editing in Final Cut Pro without breaking a sweat. Battery life is genuinely 15-18 hours!", 5.0, True),
        ("Stunning 15.3-inch Liquid Retina display. Huge glass trackpad is the best in any laptop. MagSafe charging frees up both Thunderbolt ports.", 5.0, True),
        ("16GB RAM is fluid for heavy multitasking with 40+ Chrome tabs and Xcode running. Solid aluminum unibody build feels like it will last a decade.", 5.0, True),
        ("Base model only has 2 USB-C ports on the left side, wish it had at least one port on the right side. Screen doesn't support 120Hz ProMotion.", 4.0, True),
        ("Best everyday laptop on the planet. Ultra-slim profile, brilliant keyboard, amazing quad speakers, and unmatched battery endurance.", 5.0, True)
    ],
    "asus-rog-zephyrus-g14": [
        ("The 3K 120Hz OLED screen is jaw-dropping! Inky blacks, HDR colors, and zero ghosting. RTX 4070 runs Cyberpunk 2077 and Black Myth Wukong beautifully.", 5.0, True),
        ("All-aluminum CNC chassis with Slash Lighting looks modern and sleek without being overly gamer-ish. Weighs only 1.5kg, so easy to carry to college.", 5.0, True),
        ("Fans get noticeably loud in Turbo mode under full GPU load, which is expected for such a thin gaming machine. Battery life for normal browsing is around 6-7 hours.", 4.0, True),
        ("Keyboard key travel is comfortable and the glass trackpad is enormous. Speakers with Dolby Atmos are the best I have heard on any Windows laptop.", 5.0, True)
    ],
    "sony-alpha-7-iv": [
        ("The 33MP sensor hits the sweet spot for commercial photography and 4K 60p video. Real-time eye tracking autofocus locks onto subjects instantly.", 5.0, True),
        ("Dynamic range and low-light ISO performance are stellar. 10-bit 4:2:2 S-Log3 gives immense flexibility during DaVinci Resolve color grading.", 5.0, True),
        ("4K 60p has a 1.5x Super35 crop factor, which requires wider lenses. Also screen resolution could be slightly sharper. Otherwise an unbeatable hybrid camera.", 4.0, True),
        ("Incredible build ergonomics and weather sealing. Dual card slots provide crucial redundancy for wedding shoots. Battery life with Z-batteries is solid.", 5.0, True)
    ],
    "apple-watch-series-9": [
        ("The Double Tap gesture is surprisingly useful when carrying groceries or holding a coffee. Display is easily readable under blinding sunlight.", 5.0, True),
        ("Health sensors (ECG, Heart Rate, Sleep Tracking) are clinically accurate. Workout tracking and GPS mapping are seamless.", 5.0, True),
        ("Battery life remains strictly 1 day (18-24 hours). You must charge it daily before sleep. Otherwise, it is the best smartwatch companion for iPhone.", 4.0, True),
        ("Fluid animations with watchOS 10 and S9 chip. On-device Siri answers requests instantly without requiring phone connectivity.", 5.0, True)
    ],
    "bose-qc-ultra-headphones": [
        ("World-class noise cancellation. It cancels deep aircraft drone even better than Sony XM5. The earcups are super soft and fit over ears like a cloud.", 5.0, True),
        ("Bose Immersive Audio provides a wide, natural soundstage for acoustic tracks and live recordings. CustomTune adapts to ear shape automatically.", 5.0, True),
        ("App connectivity can sometimes take 5 seconds to recognize headphones on Bluetooth startup. Price is slightly steep at ₹35,900.", 4.0, True),
        ("Collapsible folding design makes it more compact in travel bags than the XM5. Call clarity in windy outdoor conditions is surprisingly crisp.", 5.0, True)
    ]
}

review_counter = 1
for pid, review_list in REVIEWS_TEMPLATES.items():
    for text, rating, verified in review_list:
        REVIEWS.append({
            "review_id": f"REV-{pid[:6].upper()}-{review_counter:04d}",
            "product_id": pid,
            "review_text": text,
            "review_rating": float(rating),
            "review_date": f"2024-0{min(9, max(1, (review_counter % 9) + 1))}-{10 + (review_counter % 18):02d}",
            "verified_purchase": verified,
            "source": "prepared_demo",
            "data_type": "prepared_demo"
        })
        review_counter += 1

# Save products.csv
df_products = pd.DataFrame(PRODUCTS)
df_products.to_csv(DATA_DIR / "products.csv", index=False, encoding="utf-8")
print(f"Saved {len(df_products)} products to {DATA_DIR / 'products.csv'}")

# Save reviews.csv
df_reviews = pd.DataFrame(REVIEWS)
df_reviews.to_csv(DATA_DIR / "reviews.csv", index=False, encoding="utf-8")
print(f"Saved {len(df_reviews)} reviews to {DATA_DIR / 'reviews.csv'}")
