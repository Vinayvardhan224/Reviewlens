"""
Script to generate standard data/products.csv and data/reviews.csv
conforming to the ReviewLens Implementation Review specifications.
Includes category-specific reviews for Smartphones, Laptops, Cameras, and Headphones.
"""

import json
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PRODUCTS = [
    # ------------------ SMARTPHONES ------------------
    {
        "product_id": "samsung-galaxy-s24-fe",
        "product_name": "Samsung Galaxy S24 FE 5G (8GB RAM, 128GB)",
        "brand": "Samsung",
        "category": "Smartphones",
        "price": 39999.0,
        "original_price": 59999.0,
        "rating": 4.5,
        "review_count": 79752,
        "description": "Value flagship featuring 6.7-inch Dynamic AMOLED 2X 120Hz display, Exynos 2400e processor with enlarged vapor chamber, 50MP triple camera system, and 7 years of promised OS updates.",
        "amazon_url": "https://www.amazon.in/s?k=Samsung+Galaxy+S24+FE",
        "product_url": "https://www.amazon.in/s?k=Samsung+Galaxy+S24+FE",
        "image_url": "https://images.unsplash.com/photo-1610945265064-0e34e5519bbf?auto=format&fit=crop&w=600&q=80",
        "currency": "INR",
        "specifications": json.dumps({
            "Processor": "Samsung Exynos 2400e Deca-Core (4nm)",
            "Display": "6.7-inch Dynamic AMOLED 2X, 120Hz FHD+ (1900 nits peak)",
            "Camera": "Triple: 50MP Main (OIS) + 12MP Ultra-Wide + 8MP 3x Telephoto",
            "Front Camera": "10MP Wide (f/2.4)",
            "Battery": "4700 mAh with 25W fast charging & 15W wireless",
            "Charging": "25W wired fast charging (takes 75 mins to full)",
            "OS & Support": "Android 14 (One UI 6.1) with 7 Years OS Updates",
            "Build": "Armor Aluminum frame, Gorilla Glass Victus+ back, IP68",
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
        "original_price": 179900.0,
        "rating": 4.6,
        "review_count": 3189,
        "description": "Aerospace-grade Titanium with Apple A17 Pro 3nm chip, customizable Action Button, 48MP main camera with 5x optical tetraprism zoom, and universal USB-C 3.0 connector.",
        "amazon_url": "https://www.amazon.in/s?k=iPhone+15+Pro+Max",
        "product_url": "https://www.amazon.in/s?k=iPhone+15+Pro+Max",
        "image_url": "https://images.unsplash.com/photo-1695048133142-1a20484d2569?auto=format&fit=crop&w=600&q=80",
        "currency": "INR",
        "specifications": json.dumps({
            "Processor": "Apple A17 Pro (3nm Hexa-Core with Hardware Ray Tracing)",
            "Display": "6.7-inch Super Retina XDR OLED, 120Hz ProMotion (2000 nits)",
            "Camera": "Triple: 48MP Main (Sensor-Shift OIS) + 12MP Ultra-Wide + 12MP 5x Tetraprism Zoom",
            "Front Camera": "12MP TrueDepth with Autofocus",
            "Battery": "4441 mAh with 20W wired & 15W MagSafe wireless",
            "Charging": "20W wired charging (takes 1h 45m to full), MagSafe wireless",
            "OS & Support": "iOS 17 (Dynamic Island, 5+ Years OS updates)",
            "Build": "Grade 5 Titanium frame, Ceramic Shield glass, IP68",
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
        "original_price": 144999.0,
        "rating": 4.8,
        "review_count": 12450,
        "description": "Apex Android flagship featuring Snapdragon 8 Elite 3nm architecture, 200MP Quad Camera with dual telephotos, Gorilla Armor anti-reflective glass, and built-in S-Pen.",
        "amazon_url": "https://www.amazon.in/s?k=Samsung+Galaxy+S25+Ultra",
        "product_url": "https://www.amazon.in/s?k=Samsung+Galaxy+S25+Ultra",
        "image_url": "https://images.unsplash.com/photo-1580910051074-3eb694886505?auto=format&fit=crop&w=600&q=80",
        "currency": "INR",
        "specifications": json.dumps({
            "Processor": "Qualcomm Snapdragon 8 Elite for Galaxy (3nm)",
            "Display": "6.86-inch Dynamic AMOLED 2X, LTPO 120Hz QHD+ (2600 nits)",
            "Camera": "Quad: 200MP (Wide OIS) + 50MP UW + 50MP 5x Periscope + 10MP 3x Telephoto",
            "Front Camera": "12MP Dual Pixel AF",
            "Battery": "5000 mAh with 45W fast charging & 15W wireless",
            "Charging": "45W fast charging (65% in 30 mins)",
            "OS & Support": "Android 15 (One UI 7) with 7 Years OS Updates",
            "Build": "Grade 5 Titanium chassis, Gorilla Armor glass, IP68, S-Pen",
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
        "original_price": 69999.0,
        "rating": 4.4,
        "review_count": 8920,
        "description": "Performance powerhouse featuring Qualcomm Snapdragon 8 Gen 3, 4th Gen Hasselblad camera with 64MP periscope telephoto, 5400 mAh battery, and ultra-fast 100W SUPERVOOC charging.",
        "amazon_url": "https://www.amazon.in/s?k=OnePlus+12",
        "product_url": "https://www.amazon.in/s?k=OnePlus+12",
        "image_url": "https://images.unsplash.com/photo-1592899677977-9c10ca588bbd?auto=format&fit=crop&w=600&q=80",
        "currency": "INR",
        "specifications": json.dumps({
            "Processor": "Qualcomm Snapdragon 8 Gen 3 (4nm)",
            "Display": "6.82-inch ProXDR 2K OLED, 120Hz LTPO (4500 nits peak)",
            "Camera": "Triple Hasselblad: 50MP Sony LYT-808 + 48MP UW + 64MP 3x Periscope",
            "Front Camera": "32MP Sony IMX615",
            "Battery": "5400 mAh with 100W wired & 50W AIRVOOC wireless",
            "Charging": "100W SUPERVOOC charging (0 to 100% in 26 mins)",
            "OS & Support": "OxygenOS 14 (4 Years OS Updates)",
            "Build": "Glass back, aluminum frame, IP65 dust and water resistance",
            "Weight": "220g"
        }),
        "availability": "In Stock",
        "data_type": "prepared_demo"
    },

    # ------------------ LAPTOPS ------------------
    {
        "product_id": "apple-macbook-air-m3",
        "product_name": "Apple MacBook Air 15-inch M3 Chip (16GB RAM, 512GB SSD)",
        "brand": "Apple",
        "category": "Laptops",
        "price": 134900.0,
        "original_price": 154900.0,
        "rating": 4.8,
        "review_count": 4890,
        "description": "Ultra-thin laptop powered by Apple M3 8-core CPU / 10-core GPU, 15.3-inch Liquid Retina display, dual external display support, and silent fanless cooling.",
        "amazon_url": "https://www.amazon.in/s?k=MacBook+Air+15+M3",
        "product_url": "https://www.amazon.in/s?k=MacBook+Air+15+M3",
        "image_url": "https://images.unsplash.com/photo-1517336714731-489689fd1ca8?auto=format&fit=crop&w=600&q=80",
        "currency": "INR",
        "specifications": json.dumps({
            "Processor": "Apple M3 (8-core CPU, 16-core Neural Engine)",
            "RAM": "16GB Unified Memory (high-bandwidth 100GB/s)",
            "GPU": "10-core GPU with hardware-accelerated ray tracing",
            "Keyboard": "Magic Keyboard with Touch ID and ambient light sensor",
            "Cooling": "Fanless passive cooling (100% completely silent)",
            "Display": "15.3-inch Liquid Retina IPS (2880 x 1864, 500 nits)",
            "Battery Life": "Up to 18 hours Apple TV playback",
            "Weight": "1.51 kg"
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
        "original_price": 199990.0,
        "rating": 4.6,
        "review_count": 2180,
        "description": "Precision gaming laptop with AMD Ryzen 9 8945HS, NVIDIA GeForce RTX 4070 GPU, vibrant 3K 120Hz OLED display, and Slash Lighting aluminum CNC chassis.",
        "amazon_url": "https://www.amazon.in/s?k=ROG+Zephyrus+G14+2024",
        "product_url": "https://www.amazon.in/s?k=ROG+Zephyrus+G14+2024",
        "image_url": "https://images.unsplash.com/photo-1603302576837-37561b2e2302?auto=format&fit=crop&w=600&q=80",
        "currency": "INR",
        "specifications": json.dumps({
            "Processor": "AMD Ryzen 9 8945HS (8 Cores / 16 Threads, up to 5.2 GHz with Ryzen AI)",
            "RAM": "32GB LPDDR5X-6400 RAM (dual channel)",
            "GPU": "NVIDIA GeForce RTX 4070 Laptop GPU (8GB GDDR6, 90W TGP)",
            "Keyboard": "Single-zone RGB keyboard with 1.7mm key travel",
            "Cooling": "ROG Intelligent Cooling with tri-fan technology and liquid metal thermal compound",
            "Display": "14.0-inch 3K (2880 x 1800) OLED 120Hz 0.2ms, G-Sync",
            "Weight": "1.50 kg"
        }),
        "availability": "In Stock",
        "data_type": "prepared_demo"
    },
    {
        "product_id": "dell-xps-13-plus",
        "product_name": "Dell XPS 13 Plus 9320 (Intel Core i7-1360P, 16GB, 1TB)",
        "brand": "Dell",
        "category": "Laptops",
        "price": 159990.0,
        "original_price": 179990.0,
        "rating": 4.1,
        "review_count": 1420,
        "description": "Minimalist premium laptop with capacitive touch function row, seamless zero-lattice glass trackpad, and vivid 3.5K OLED touch screen.",
        "amazon_url": "https://www.amazon.in/s?k=Dell+XPS+13+Plus",
        "product_url": "https://www.amazon.in/s?k=Dell+XPS+13+Plus",
        "image_url": "https://images.unsplash.com/photo-1593642632823-8f785ba67e45?auto=format&fit=crop&w=600&q=80",
        "currency": "INR",
        "specifications": json.dumps({
            "Processor": "13th Gen Intel Core i7-1360P (12 Cores, up to 5.0 GHz)",
            "RAM": "16GB LPDDR5 6000MHz",
            "GPU": "Intel Iris Xe Graphics",
            "Keyboard": "Zero-lattice edge-to-edge keyboard with capacitive touch function bar",
            "Cooling": "Dual fan cooling system with heat pipes",
            "Display": "13.4-inch 3.5K (3456 x 2160) OLED InfinityEdge Touch",
            "Weight": "1.26 kg"
        }),
        "availability": "In Stock",
        "data_type": "prepared_demo"
    },

    # ------------------ HEADPHONES ------------------
    {
        "product_id": "sony-wh1000xm5",
        "product_name": "Sony WH-1000XM5 Wireless Noise Cancelling Headphones",
        "brand": "Sony",
        "category": "Headphones",
        "price": 29990.0,
        "original_price": 34990.0,
        "rating": 4.6,
        "review_count": 14200,
        "description": "Industry-leading active noise cancellation with 8 microphones, LDAC High-Resolution Audio, crystal clear hands-free calls, and 30-hour battery life.",
        "amazon_url": "https://www.amazon.in/s?k=Sony+WH-1000XM5",
        "product_url": "https://www.amazon.in/s?k=Sony+WH-1000XM5",
        "image_url": "https://images.unsplash.com/photo-1546435770-a3e426bf472b?auto=format&fit=crop&w=600&q=80",
        "currency": "INR",
        "specifications": json.dumps({
            "Sound Quality": "30mm carbon fiber driver with LDAC High-Res Audio & DSEE Extreme",
            "Battery": "30 hours playback with ANC on (3-min charge gives 3 hours)",
            "Comfort": "Soft fit synthetic leather with stepless sliding headband (250g)",
            "Noise Cancellation": "Auto NC Optimizer with dual processors (V1 + QN1) and 8 mics",
            "Connectivity": "Bluetooth 5.2, multipoint connection for 2 devices, Google Fast Pair",
            "Weight": "250g"
        }),
        "availability": "In Stock",
        "data_type": "prepared_demo"
    },
    {
        "product_id": "bose-qc-ultra-headphones",
        "product_name": "Bose QuietComfort Ultra Wireless Noise Cancelling Headphones",
        "brand": "Bose",
        "category": "Headphones",
        "price": 35900.0,
        "original_price": 39900.0,
        "rating": 4.5,
        "review_count": 6700,
        "description": "World-class noise cancellation with breakthrough Bose Immersive Audio, luxurious plush protein leather cushions, CustomTune technology, and up to 24 hours of playback.",
        "amazon_url": "https://www.amazon.in/s?k=Bose+QuietComfort+Ultra",
        "product_url": "https://www.amazon.in/s?k=Bose+QuietComfort+Ultra",
        "image_url": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?auto=format&fit=crop&w=600&q=80",
        "currency": "INR",
        "specifications": json.dumps({
            "Sound Quality": "Bose Immersive Audio spatial sound + CustomTune acoustic calibration",
            "Battery": "24 hours battery life (18 hours with Immersive Audio)",
            "Comfort": "Plush protein leather ear cushions and lightweight headband (252g)",
            "Noise Cancellation": "Quiet Mode, Aware Mode with ActiveSense, and Immersion Mode",
            "Connectivity": "Bluetooth 5.3 with Snapdragon Sound (aptX Adaptive) & multipoint",
            "Weight": "252g"
        }),
        "availability": "In Stock",
        "data_type": "prepared_demo"
    },
    {
        "product_id": "apple-airpods-pro-2",
        "product_name": "Apple AirPods Pro (2nd Gen) with MagSafe Case (USB-C)",
        "brand": "Apple",
        "category": "Headphones",
        "price": 24900.0,
        "original_price": 26900.0,
        "rating": 4.7,
        "review_count": 28400,
        "description": "Apple-designed H2 chip delivers up to 2x more Active Noise Cancellation, Adaptive Audio, Personalized Spatial Audio with dynamic head tracking, and USB-C case.",
        "amazon_url": "https://www.amazon.in/s?k=AirPods+Pro+2+USB-C",
        "product_url": "https://www.amazon.in/s?k=AirPods+Pro+2+USB-C",
        "image_url": "https://images.unsplash.com/photo-1600294037681-c80b4cb5b434?auto=format&fit=crop&w=600&q=80",
        "currency": "INR",
        "specifications": json.dumps({
            "Sound Quality": "Custom high-excursion Apple driver with Personalized Spatial Audio",
            "Battery": "6 hours listening per charge (30 hours total with MagSafe Case)",
            "Comfort": "Four sizes of silicone tips (XS, S, M, L) for secure acoustic seal",
            "Noise Cancellation": "Up to 2x more Active Noise Cancellation + Adaptive Audio",
            "Connectivity": "Bluetooth 5.3, instantaneous Apple device auto-switching",
            "Weight": "5.3g per earbud | 50.8g case"
        }),
        "availability": "In Stock",
        "data_type": "prepared_demo"
    },

    # ------------------ CAMERAS ------------------
    {
        "product_id": "sony-alpha-7-iv",
        "product_name": "Sony Alpha 7 IV Full-Frame Mirrorless Camera (Body Only)",
        "brand": "Sony",
        "category": "Cameras",
        "price": 214990.0,
        "original_price": 242990.0,
        "rating": 4.8,
        "review_count": 1820,
        "description": "Hybrid powerhouse with 33MP full-frame Exmor R CMOS sensor, BIONZ XR processing engine, 4K 60p 10-bit 4:2:2 recording, real-time eye autofocus, and 5-axis IBIS.",
        "amazon_url": "https://www.amazon.in/s?k=Sony+Alpha+7+IV",
        "product_url": "https://www.amazon.in/s?k=Sony+Alpha+7+IV",
        "image_url": "https://images.unsplash.com/photo-1516035069371-29a1b244cc32?auto=format&fit=crop&w=600&q=80",
        "currency": "INR",
        "specifications": json.dumps({
            "Image Quality": "33.0 Megapixel Full-Frame back-illuminated Exmor R CMOS sensor",
            "Autofocus": "759-point phase-detection AF with real-time Eye AF for human, animal, bird",
            "Lens Quality": "Sony E-mount compatible with over 70 native full-frame lenses",
            "Low-Light Performance": "ISO range 100-51200 (expandable to 50-204800) with 15 stops dynamic range",
            "Video Recording": "4K 60p (Super 35) & 4K 30p 10-bit 4:2:2 internal with S-Cinetone / S-Log3",
            "Stabilization": "5-axis optical in-body image stabilization (5.5-step advantage)",
            "Weight": "658g"
        }),
        "availability": "In Stock",
        "data_type": "prepared_demo"
    },
    {
        "product_id": "canon-eos-r6-mark-ii",
        "product_name": "Canon EOS R6 Mark II Mirrorless Camera (Body Only)",
        "brand": "Canon",
        "category": "Cameras",
        "price": 219990.0,
        "original_price": 249990.0,
        "rating": 4.7,
        "review_count": 1450,
        "description": "High-speed hybrid camera with 24.2MP CMOS sensor, blazing 40 fps electronic shutter, 6K oversampled uncropped 4K 60p video, and Dual Pixel CMOS AF II.",
        "amazon_url": "https://www.amazon.in/s?k=Canon+EOS+R6+Mark+II",
        "product_url": "https://www.amazon.in/s?k=Canon+EOS+R6+Mark+II",
        "image_url": "https://images.unsplash.com/photo-1502920917128-1aa500764cbd?auto=format&fit=crop&w=600&q=80",
        "currency": "INR",
        "specifications": json.dumps({
            "Image Quality": "24.2 Megapixel Full-Frame CMOS sensor with DIGIC X processor",
            "Autofocus": "Dual Pixel CMOS AF II with AI deep learning subject detection",
            "Lens Quality": "Canon RF mount with ultra-sharp L-series professional optics",
            "Low-Light Performance": "Native ISO 102,400 with autofocus sensitivity down to -6.5 EV",
            "Video Recording": "Uncropped 4K 60p oversampled from 6K, Canon Log 3 & HDR PQ",
            "Stabilization": "In-body image stabilizer with up to 8.0 stops coordinated control",
            "Weight": "670g"
        }),
        "availability": "In Stock",
        "data_type": "prepared_demo"
    }
]

# Targeted category-specific reviews covering the exact aspects from Section 4.1:
# - Smartphones: Camera, Battery, Display, Charging, Performance
# - Laptops: Processor, RAM, GPU, Keyboard, Cooling
# - Headphones: Sound quality, Battery, Comfort, Noise cancellation, Connectivity
# - Cameras: Image quality, Autofocus, Lens quality, Low-light performance, Video recording

CATEGORY_REVIEWS = {
    # ---------------- SMARTPHONES ----------------
    "samsung-galaxy-s24-fe": [
        ("The Dynamic AMOLED display is gorgeous and super smooth with vibrant colors. Battery life easily lasts me a full day with 6 to 7 hours screen-on-time.", 5.0, True),
        ("The 50MP camera takes crisp photos in daylight. Portrait shots look very natural, but front selfie camera is a bit grainy indoors.", 4.0, True),
        ("Performance on the Exynos 2400e is snappy with no lag during everyday multitasking and light gaming.", 4.0, True),
        ("Charging speed is disappointing. The 25W charging takes over 75 minutes to charge from zero to full, and there is no charger in the box.", 3.0, True),
        ("Display brightness outdoors under scorching direct sun is surprisingly clear. Solid camera versatility with the 3x optical telephoto lens.", 5.0, True),
        ("Battery drains rather quickly when using 5G navigation and recording 4K video. Slow charging makes topping up tedious.", 3.0, True),
        ("Amazing performance for the price. Phone stays relatively cool under regular loads and the screen is butter smooth.", 5.0, True),
        ("The main camera optical image stabilization produces blur-free pictures of moving subjects. Great camera quality.", 5.0, True)
    ],
    "apple-iphone-15-pro-max": [
        ("The 5x optical tetraprism camera is fantastic for capturing sharp distant subjects and concerts. Camera video recording in 4K ProRes is the best in the industry.", 5.0, True),
        ("Super Retina XDR display with 2000 nits peak brightness is breathtaking. ProMotion 120Hz display makes scrolling incredibly fluid.", 5.0, True),
        ("Performance with the A17 Pro chip is blazing fast. Console-grade gaming runs with high frame rates, though performance thermals warm up during intense sessions.", 5.0, True),
        ("Battery life lasts a solid 1.5 days on normal usage. But 20W charging takes almost two full hours to reach 100%, which is very slow charging for a flagship.", 3.0, True),
        ("Camera night mode portraits look stunning with accurate skin tones. Camera shutter lag is completely nonexistent.", 5.0, True),
        ("Display is crisp and bright outdoors. Battery endurance is stellar, easily surviving heavy travel days.", 5.0, True),
        ("Charging is quite slow at only 20W. You really notice how long it takes to charge compared to Android rivals.", 3.0, True),
        ("Unmatched performance and video camera quality. The action button and Titanium design make it very comfortable to hold.", 5.0, True)
    ],
    "samsung-galaxy-s25-ultra": [
        ("Snapdragon 8 Elite delivers monster performance. Zero stutters or dropped frames in the most demanding games. Incredible performance power.", 5.0, True),
        ("The 200MP camera captures insane levels of fine detail. Having dual optical telephoto cameras (3x and 5x) makes this the ultimate camera phone.", 5.0, True),
        ("Display with Gorilla Armor anti-reflective glass completely eliminates glare. The QHD+ Dynamic AMOLED display is hands-down the best screen on the market.", 5.0, True),
        ("Battery life is fantastic, consistently giving 8 to 9 hours of screen on time. 45W fast charging tops up the 5000 mAh battery to 65% in half an hour.", 5.0, True),
        ("Camera shutter is instantaneous and low-light nightography captures clean, noise-free shots without motion blur.", 5.0, True),
        ("Huge display is bright and beautiful. Fast 45W charging is reliable, and battery easily powers through intensive workdays.", 5.0, True)
    ],
    "oneplus-12": [
        ("The 100W SUPERVOOC charging is mind-blowing! Charges the huge 5400 mAh battery from 10% to 100% in just 26 minutes. Best charging speed ever.", 5.0, True),
        ("Snapdragon 8 Gen 3 performance is buttery smooth. Gaming at 120fps is effortless with great thermal stability.", 5.0, True),
        ("The 2K ProXDR display is intensely bright and responsive. Curved screen edges look sleek though occasional accidental palm touches happen.", 4.0, True),
        ("Hasselblad camera system produces punchy colors and natural portrait bokeh. The 3x periscope camera zoom is surprisingly sharp.", 5.0, True),
        ("Battery endurance easily reaches nearly two full days. Combined with blazing 100W charging, you never have battery anxiety.", 5.0, True),
        ("Camera photos in low light are decent, but not quite matching iPhone or Ultra camera consistency.", 4.0, True)
    ],

    # ---------------- LAPTOPS ----------------
    "apple-macbook-air-m3": [
        ("The M3 processor is blazing fast and handles heavy multitasking and 4K Final Cut video editing with ease.", 5.0, True),
        ("Passive fanless cooling means the laptop is 100% completely silent, yet the chassis cooling keeps it remarkably cool during office workloads.", 5.0, True),
        ("Magic keyboard typing experience is crisp with comfortable key travel. The 16GB RAM ensures zero swapping or memory bottleneck.", 5.0, True),
        ("The integrated 10-core GPU handles photo editing and light gaming nicely, but it is not a dedicated gaming GPU for triple-A titles.", 4.0, True),
        ("Stunning Liquid Retina display and unbelievable 16-hour battery life. Best processor and memory efficiency of any ultraportable.", 5.0, True),
        ("Keyboard backlighting is clean and comfortable for late-night coding. 16GB RAM handles 40+ browser tabs and Docker without lag.", 5.0, True)
    ],
    "asus-rog-zephyrus-g14": [
        ("The NVIDIA RTX 4070 GPU runs Cyberpunk 2077, Black Myth Wukong, and Forza Horizon at high framerates with ray tracing enabled.", 5.0, True),
        ("AMD Ryzen 9 processor is a beast for compiling code and rendering 3D scenes. Outstanding processor benchmarks.", 5.0, True),
        ("32GB high-speed RAM means seamless video editing and gaming without any memory slowdowns.", 5.0, True),
        ("The cooling system with dual fans and liquid metal keeps GPU temperatures under control, but cooling fan noise gets very loud in Turbo gaming mode.", 4.0, True),
        ("Keyboard has generous 1.7mm key travel and feels great for long typing and gaming sessions. RGB keyboard lighting looks sleek.", 5.0, True),
        ("Under heavy GPU load, the cooling fans ramp up loudly like a jet engine. But gaming performance and the 3K OLED screen are spectacular.", 4.0, True)
    ],
    "dell-xps-13-plus": [
        ("The Intel Core i7 processor handles productivity spreadsheets and office software smoothly.", 4.0, True),
        ("Zero-lattice keyboard looks futuristic, but the flat keyboard layout leads to more typos than traditional tactile keyboards.", 3.0, True),
        ("Cooling fans kick in frequently and the bottom chassis gets hot during sustained CPU loads. Cooling could definitely be better.", 3.0, True),
        ("16GB RAM is fast for everyday tasks, though integrated Intel Iris Xe GPU struggles with modern 3D gaming.", 3.0, True),
        ("OLED display is sharp and vibrant, but thermal cooling issues and capacitive function keys frustrate power users.", 3.0, True)
    ],

    # ---------------- HEADPHONES ----------------
    "sony-wh1000xm5": [
        ("The active noise cancellation is pure magic. It completely silences metro hum, cafe chatter, and airplane cabin noise.", 5.0, True),
        ("Sound quality is balanced with rich, deep bass and clear vocal separation when listening to high-res LDAC audio tracks.", 5.0, True),
        ("Extremely comfortable soft-fit ear cushions. I can wear these headphones for 8 hours of work with zero ear fatigue or comfort issues.", 5.0, True),
        ("Battery life lasts an incredible 30 hours with ANC enabled. Quick charging gives 3 hours of playback in just 3 minutes.", 5.0, True),
        ("Bluetooth connectivity is rock solid. The multipoint connection switches seamlessly between my laptop and smartphone without drops.", 5.0, True),
        ("Noise cancellation microphone clarity on phone calls in windy streets is remarkably clear and isolated.", 5.0, True),
        ("Earcups feel comfortable, but the headband padding is slightly narrow. Still, sound quality and noise cancellation are unbeatable.", 4.0, True)
    ],
    "bose-qc-ultra-headphones": [
        ("Noise cancellation is the best in the world. It blocks out low-frequency aircraft rumble even more effectively than the Sony XM5.", 5.0, True),
        ("Plush protein leather cushions provide cloud-like comfort. The clamping force is gentle yet secure, making comfort a 10/10.", 5.0, True),
        ("Bose Immersive Audio delivers spacious spatial sound quality with natural instrument placement and crisp highs.", 5.0, True),
        ("Battery life provides 24 hours of listening. However, with Immersive Audio enabled, battery life drops to around 18 hours.", 4.0, True),
        ("Bluetooth 5.3 connectivity pairs quickly, though the Bose mobile app occasionally takes a few seconds to connect.", 4.0, True),
        ("Exceptional comfort and class-leading noise cancellation. Sound quality has noticeably improved with tight, controlled bass.", 5.0, True)
    ],
    "apple-airpods-pro-2": [
        ("Noise cancellation on these compact earbuds is mind-blowing. Adaptive Audio seamlessly blends transparency when someone speaks to you.", 5.0, True),
        ("Sound quality with Personalized Spatial Audio makes watching movies and Dolby Atmos music feel immersive and wide.", 5.0, True),
        ("Earbuds are super comfortable with the silicone tips, staying securely in my ears during long running workouts without comfort irritation.", 5.0, True),
        ("Battery life gives a reliable 6 hours on the buds and up to 30 hours with the MagSafe case. USB-C charging is very convenient.", 5.0, True),
        ("Instantaneous connectivity and seamless automatic switching across iPhone, iPad, and Mac without any connection drops.", 5.0, True),
        ("Outstanding noise cancellation and pocketable comfort. Microphone call quality in loud environments is clear.", 5.0, True)
    ],

    # ---------------- CAMERAS ----------------
    "sony-alpha-7-iv": [
        ("The 33MP sensor delivers extraordinary image quality with superb dynamic range and lifelike color gradations in RAW files.", 5.0, True),
        ("Real-time Eye autofocus is instantaneous and sticky. The autofocus locks onto moving birds, pets, and portrait subjects without hunting.", 5.0, True),
        ("Low-light performance at ISO 6400 and 12800 is impressively clean with minimal noise. Shadow recovery in post-production is remarkable.", 5.0, True),
        ("Video recording in 10-bit 4:2:2 with S-Cinetone produces cinematic color out of the box. Only downside is 4K 60p has a 1.5x crop factor.", 4.0, True),
        ("Massive Sony E-mount lens quality ecosystem allows using sharp G Master and Sigma prime lenses with stellar optical sharpness.", 5.0, True),
        ("Autofocus tracking in video recording mode never loses the subject. Exceptional image quality for commercial photography.", 5.0, True),
        ("Low-light performance allows shooting nighttime cityscapes handheld thanks to the 5-axis in-body image stabilization.", 5.0, True)
    ],
    "canon-eos-r6-mark-ii": [
        ("Dual Pixel autofocus II is wizardry! It detects aircraft, vehicles, animals, and human eyes across the entire frame with zero autofocus hunting.", 5.0, True),
        ("Image quality from the 24MP sensor produces legendary Canon skin tones and deep contrast in both RAW and JPEG formats.", 5.0, True),
        ("Uncropped 6K oversampled 4K 60p video recording is razor-sharp with no thermal overheating limits during typical video shoots.", 5.0, True),
        ("Low-light performance is sensational. Low-light focusing works down to -6.5 EV in near pitch-black darkness with clean high-ISO files.", 5.0, True),
        ("RF lens quality is top-tier with incredible corner-to-corner optical resolving power, though native RF lenses are quite expensive.", 4.0, True),
        ("Blazing 40 fps electronic burst shooting captures fast sports action effortlessly. Outstanding autofocus and video recording.", 5.0, True)
    ]
}

REVIEWS = []
rev_id_counter = 1

for pid, rev_list in CATEGORY_REVIEWS.items():
    for text, rating, verified in rev_list:
        REVIEWS.append({
            "review_id": f"REV-{pid[:6].upper()}-{rev_id_counter:04d}",
            "product_id": pid,
            "review_text": text,
            "review_rating": float(rating),
            "review_date": f"2024-0{min(9, max(1, (rev_id_counter % 9) + 1))}-{10 + (rev_id_counter % 18):02d}",
            "verified_purchase": verified,
            "source": "prepared_dataset",
            "data_type": "prepared_demo"
        })
        rev_id_counter += 1

# Save products.csv
df_products = pd.DataFrame(PRODUCTS)
df_products.to_csv(DATA_DIR / "products.csv", index=False, encoding="utf-8")
print(f"Saved {len(df_products)} products to {DATA_DIR / 'products.csv'}")

# Save reviews.csv
df_reviews = pd.DataFrame(REVIEWS)
df_reviews.to_csv(DATA_DIR / "reviews.csv", index=False, encoding="utf-8")
print(f"Saved {len(df_reviews)} reviews to {DATA_DIR / 'reviews.csv'}")
