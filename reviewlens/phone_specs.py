"""
Comprehensive Phone Specifications, Benchmarks, and Review Intelligence Database.
Modeled after 91mobiles / GSMArena spec comparisons, augmented with ReviewLens NLP sentiment.
"""

from typing import Dict, Any, List

PHONE_DATABASE: Dict[str, Dict[str, Any]] = {
    "Samsung Galaxy S24 FE": {
        "brand": "Samsung",
        "model": "Galaxy S24 FE",
        "price_inr": "₹39,999",
        "price_usd": "$649",
        "spec_score": 90,
        "user_rating": 4.5,
        "total_ratings": 79752,
        "expert_rating": "7.9 / 10",
        "image_url": "https://img.icons8.com/color/144/samsung.png",
        "quick_verdict": "Value flagship with 7 years of OS updates, Dynamic AMOLED 2X, and solid 4,700 mAh battery at 1/4th the price of ultra-flagships.",
        "pros": [
            "Appealing, minimalistic Galaxy design",
            "Balanced Exynos 2400e performance with Vapor Chamber cooling",
            "7 full years of Android OS & Security updates",
            "Dynamic AMOLED 2X 120Hz display with Gorilla Glass Victus+",
            "Excellent value for money (₹39,999)"
        ],
        "cons": [
            "No wall charger included in box",
            "Average 10MP selfie camera compared to Pro Max",
            "3x Optical Zoom limited vs 5x periscope",
            "25W charging is modest compared to Chinese rivals"
        ],
        "nlp_aspect_scores": {
            "Display & Visuals": 78.0,
            "Battery & Power": 65.0,
            "Performance & Speed": 62.0,
            "Build Quality & Durability": 80.0,
            "Price & Value for Money": 92.0,
            "Sound & Audio": 70.0,
            "Customer Support & Warranty": 75.0,
        },
        "specs": {
            "Performance": {
                "Chipset": "Samsung Exynos 2400e (4nm)",
                "CPU Architecture": "Deca-Core (10 cores up to 3.1 GHz)",
                "GPU Graphics": "Xclipse 940 (AMD RDNA3 based)",
                "RAM": "8 GB LPDDR5X",
                "Cooling System": "Enlarged Vapor Chamber Cooling",
                "Geekbench Multi-Core": "6,407",
                "AnTuTu Score": "1,625,873",
                "Gaming Thermals (30 min)": "+20.6% heating (Stays cool)"
            },
            "Display": {
                "Screen Size": "6.7 inches (17.02 cm)",
                "Display Technology": "Dynamic AMOLED 2X",
                "Resolution": "1080 x 2340 pixels (FHD+)",
                "Refresh Rate": "120 Hz Adaptive",
                "Peak Brightness": "1,900 nits",
                "Pixel Density": "385 ppi",
                "Screen Protection": "Corning Gorilla Glass Victus+"
            },
            "Cameras": {
                "Rear Camera System": "Triple: 50 MP (Wide, OIS) + 12 MP (Ultra-Wide) + 8 MP (3x Telephoto OIS)",
                "Front Camera": "10 MP Wide (f/2.4)",
                "Optical Zoom": "3x Optical Zoom, 30x Digital Zoom",
                "Video Recording": "8K @ 30 fps, 4K @ 60 fps",
                "Camera Features": "ProVisual AI Engine, Nightography, Dual OIS"
            },
            "Battery & Charging": {
                "Battery Capacity": "4,700 mAh Li-ion",
                "Wired Charging": "25W Fast Charging (50% in 30 mins)",
                "Wireless Charging": "15W Fast Wireless",
                "Reverse Wireless Charging": "Yes (Wireless PowerShare)",
                "Battery Life Benchmark": "11h 20m (PCMark continuous)"
            },
            "Build & Connectivity": {
                "Weight & Thickness": "213 grams | 8.0 mm slim",
                "Build Materials": "Armor Aluminum frame, Gorilla Glass back",
                "Water & Dust Resistance": "IP68 (up to 1.5m for 30 mins)",
                "Operating System": "Android 14 with One UI 6.1 (7 Years OS Updates)",
                "Network & SIM": "5G Dual SIM (Nano-SIM + eSIM, 18 5G bands)",
                "Audio": "Stereo Speakers with Dolby Atmos"
            }
        }
    },
    "Apple iPhone 15 Pro Max": {
        "brand": "Apple",
        "model": "iPhone 15 Pro Max",
        "price_inr": "₹159,900",
        "price_usd": "$1,199",
        "spec_score": 91,
        "user_rating": 4.6,
        "total_ratings": 3189,
        "expert_rating": "8.4 / 10",
        "image_url": "https://img.icons8.com/color/144/mac-os.png",
        "quick_verdict": "Ultimate premium performance with 3nm Apple A17 Pro, 5x Tetraprism Telephoto zoom, aerospace Titanium frame, and universal USB-C.",
        "pros": [
            "Aerospace-grade Titanium frame (lighter & durable)",
            "Class-leading Apple A17 Pro 3nm chipset with hardware Ray Tracing",
            "5x Optical Zoom Tetraprism telephoto camera",
            "ProRes 4K 60fps video recording directly to external SSD",
            "Super Retina XDR with 2,000 nits outdoor peak brightness"
        ],
        "cons": [
            "Very high price tag (₹1,59,900)",
            "Higher thermal heating under sustained 3D gaming (+43.2%)",
            "20W charging speed takes over 1 hr 45 mins to full",
            "No charger in box"
        ],
        "nlp_aspect_scores": {
            "Display & Visuals": 88.0,
            "Battery & Power": 68.0,
            "Performance & Speed": 90.0,
            "Build Quality & Durability": 86.0,
            "Price & Value for Money": 35.0,
            "Sound & Audio": 82.0,
            "Customer Support & Warranty": 80.0,
        },
        "specs": {
            "Performance": {
                "Chipset": "Apple A17 Pro (3nm)",
                "CPU Architecture": "Hexa-Core (6 cores up to 3.78 GHz)",
                "GPU Graphics": "Apple 6-Core GPU with Hardware Ray Tracing",
                "RAM": "8 GB LPDDR5",
                "Cooling System": "Titanium chassis thermal dissipation",
                "Geekbench Multi-Core": "7,193",
                "AnTuTu Score": "1,545,034",
                "Gaming Thermals (30 min)": "+43.2% heating (Runs warm)"
            },
            "Display": {
                "Screen Size": "6.7 inches (17.02 cm)",
                "Display Technology": "Super Retina XDR OLED (ProMotion)",
                "Resolution": "1290 x 2796 pixels (FHD+)",
                "Refresh Rate": "120 Hz ProMotion",
                "Peak Brightness": "2,000 nits outdoor",
                "Pixel Density": "460 ppi",
                "Screen Protection": "Ceramic Shield front glass"
            },
            "Cameras": {
                "Rear Camera System": "Triple: 48 MP (Wide, Sensor-Shift OIS) + 12 MP (Ultra-Wide) + 12 MP (5x Tetraprism Telephoto OIS)",
                "Front Camera": "12 MP TrueDepth with Autofocus (f/1.9)",
                "Optical Zoom": "5x Optical Zoom, 25x Digital Zoom",
                "Video Recording": "4K @ 60 fps ProRes, Action Mode, Spatial Video",
                "Camera Features": "Photonic Engine, Smart HDR 5, Night Mode Portraits"
            },
            "Battery & Charging": {
                "Battery Capacity": "4,441 mAh Li-ion",
                "Wired Charging": "20W Fast Charging (50% in 30 mins)",
                "Wireless Charging": "15W MagSafe / Qi2 Wireless",
                "Reverse Wireless Charging": "No (4.5W wired reverse charging via Type-C)",
                "Battery Life Benchmark": "Up to 29 hrs video playback"
            },
            "Build & Connectivity": {
                "Weight & Thickness": "221 grams | 8.25 mm",
                "Build Materials": "Grade 5 Titanium band, matte textured glass back",
                "Water & Dust Resistance": "IP68 (up to 6m for 30 mins)",
                "Operating System": "iOS 17 with Dynamic Island & Action Button (5+ Yrs Updates)",
                "Network & SIM": "5G Dual SIM (Nano-SIM + eSIM, 22 5G bands)",
                "Audio": "Stereo Speakers with Spatial Audio & Dolby Atmos"
            }
        }
    },
    "Samsung Galaxy S25 Ultra": {
        "brand": "Samsung",
        "model": "Galaxy S25 Ultra",
        "price_inr": "₹129,999",
        "price_usd": "$1,299",
        "spec_score": 96,
        "user_rating": 4.8,
        "total_ratings": 12450,
        "expert_rating": "9.2 / 10",
        "image_url": "https://img.icons8.com/color/144/samsung.png",
        "quick_verdict": "The apex Android super-flagship featuring Snapdragon 8 Elite 3nm, 200MP Quad camera with dual telephotos, built-in S-Pen, and Gorilla Armor anti-reflective glass.",
        "pros": [
            "Snapdragon 8 Elite (3nm) with massive Geekbench & gaming scores",
            "Gorilla Armor glass reduces screen glare by 75%",
            "200 MP camera + 5x periscope telephoto + 3x portrait telephoto",
            "Built-in S-Pen stylus with Bluetooth air gestures",
            "7 full years of Android & Galaxy AI updates"
        ],
        "cons": [
            "Large boxy frame requires two-handed use",
            "Expensive (₹1,29,999)",
            "45W charging remains slower than 100W competitors"
        ],
        "nlp_aspect_scores": {
            "Display & Visuals": 95.0,
            "Battery & Power": 85.0,
            "Performance & Speed": 96.0,
            "Build Quality & Durability": 90.0,
            "Price & Value for Money": 65.0,
            "Sound & Audio": 85.0,
            "Customer Support & Warranty": 82.0,
        },
        "specs": {
            "Performance": {
                "Chipset": "Qualcomm Snapdragon 8 Elite for Galaxy (3nm)",
                "CPU Architecture": "Oryon Octa-Core up to 4.47 GHz",
                "GPU Graphics": "Adreno 830",
                "RAM": "12 GB / 16 GB LPDDR5X",
                "Cooling System": "Dual Vapor Chamber Cooling",
                "Geekbench Multi-Core": "9,850",
                "AnTuTu Score": "2,850,000",
                "Gaming Thermals (30 min)": "+18.2% heating (Advanced cooling)"
            },
            "Display": {
                "Screen Size": "6.86 inches Dynamic AMOLED 2X",
                "Display Technology": "LTPO 1-120Hz Anti-Reflective",
                "Resolution": "1440 x 3120 pixels (QHD+)",
                "Refresh Rate": "120 Hz Adaptive",
                "Peak Brightness": "2,600 nits",
                "Pixel Density": "505 ppi",
                "Screen Protection": "Corning Gorilla Armor"
            },
            "Cameras": {
                "Rear Camera System": "Quad: 200 MP (Wide) + 50 MP (Ultra-Wide) + 50 MP (5x Periscope OIS) + 10 MP (3x Telephoto)",
                "Front Camera": "12 MP Dual Pixel AF",
                "Optical Zoom": "3x and 5x Optical Zoom, 100x Space Zoom",
                "Video Recording": "8K @ 30 fps, 4K @ 120 fps",
                "Camera Features": "ProVisual AI, Expert RAW 24-bit, Astro Hyperlapse"
            },
            "Battery & Charging": {
                "Battery Capacity": "5,000 mAh Li-ion",
                "Wired Charging": "45W Super Fast Charging 2.0 (65% in 30 mins)",
                "Wireless Charging": "15W Fast Wireless Charging 2.0",
                "Reverse Wireless Charging": "Yes (Wireless PowerShare 4.5W)",
                "Battery Life Benchmark": "14h 30m continuous web/gaming"
            },
            "Build & Connectivity": {
                "Weight & Thickness": "218 grams | 8.2 mm flat edges",
                "Build Materials": "Grade 5 Titanium frame with S-Pen built-in",
                "Water & Dust Resistance": "IP68 (1.5m for 30 mins)",
                "Operating System": "Android 15 with One UI 7 (7 Years OS Updates)",
                "Network & SIM": "5G Dual SIM, Wi-Fi 7, Bluetooth 5.4, UWB",
                "Audio": "Stereo Speakers tuned by AKG with Dolby Atmos"
            }
        }
    }
}
