# ReviewLens: 3-Minute Live Evaluation & Viva Demonstration Script

A structured, timed walkthrough designed for academic evaluators, viva panels, and project demonstrations.

---

### Phase 1: Authentication Gate & Identity Architecture (0:00 - 0:30)
1. **Launch Screen**: Open `http://localhost:8501` (or deployed URL).
   - *Presenter Explains*: "ReviewLens enforces a strict authentication gate before any heavy dataset loading to eliminate memory leaks and multi-render flashing."
2. **Instant Demo Mode**: Click **⚡ Enter Instant Demo Mode →** (or sign in as `demo@reviewlens.com` / `Password@123`).
   - *Presenter Explains*: "Notice the amber **DEMO MODE ACTIVE** banner at the top. For security, demo OTP bypasses (`123456`) are strictly restricted to designated demo accounts and new registrations; real accounts cannot be compromised via mock codes."
   - *Technical Note*: "Password storage uses OWASP-standard PBKDF2-HMAC-SHA256 with 600,000 iterations and per-user cryptographic salts."

---

### Phase 2: Live Review Diagnostic Lab & NLP Decomposition (0:30 - 1:15)
1. **Navigate**: In the sidebar, select **🧪 Live Review Analyzer**.
2. **Input Review**: Paste or select a complex consumer review:
   > *"The vibrant 120Hz OLED display is extraordinary and colors pop, but battery life is terrible and dies after 3 hours of video playback. Do not buy this!"*
3. **Analyze**: Click **Analyze Review Now →**.
4. **Key Features to Highlight**:
   - **Real-Time Word Highlighting**: Green badges for positive tokens (*"vibrant"*, *"extraordinary"*), red badges for negative tokens (*"terrible"*, *"dies"*).
   - **Aspect Decomposition**: ReviewLens automatically isolates `Display` (+0.88 Positive) and `Battery` (-0.72 Negative) using spaCy syntactic dependency relations rather than hardcoded global rules.
   - **Authenticity Meter**: Displays Spam Probability Score and authenticity flags.
   - **Context-Aware AI Reply**: Automatically generates a managerial response acknowledging the display praise while offering diagnostic assistance for battery drain.

---

### Phase 3: Smart Shopping Catalogue & Aspect Duels (1:15 - 2:00)
1. **Navigate**: Click **🏠 Home / Discovery**.
2. **Review Intelligence Bento**:
   - Observe the **Net Sentiment Score (NSS)** metric tile:
     $$ \text{NSS} = \frac{\text{Positive} - \text{Negative}}{\text{Total}} \times 100 $$
   - Highlight the **Aspect Sentiment Breakdown** chart with transparent glass styling.
   - Show the **Unsupervised LDA Topic Discovery** tile with interactive keyword clusters.
3. **Product Duel**:
   - Add two smartphones or laptops to the **⚔️ Compare Tray**.
   - Click **⚔️ Product Comparison**.
   - Show the multi-axial **Aspect Radar Chart** directly contrasting Battery, Display, Camera, and Build Quality between competitors.

---

### Phase 4: Ground-Truth Model Evaluation & Academic Transparency (2:00 - 2:30)
1. **Navigate**: Select **📈 Model Evaluation & Benchmarks**.
2. **Key Evidence to Point Out to the Evaluator**:
   - **Three-Way Benchmark**: Compare VADER Lexicon (Zero-shot, <1ms) vs Classical Machine Learning (TF-IDF + Logistic Regression, ~2ms) vs Published DistilBERT Transformer (~25ms).
   - **SemEval-2014 Gold ABSA**: Show 89.7% aspect recall on the official Laptop gold standard.
   - **Ott et al. Deceptive Spam F1**: Show 90.7% F1 on supervised deception benchmark.
   - **Transparent Limitations Tile**: Show Section 9 detailing domain shift caveats, neutral class sensitivity, and small demonstration corpus constraints.

---

### Phase 5: Progressive Web App & Offline Installation (2:30 - 3:00)
1. **Appearance Toggle**: Open sidebar > **🎨 Appearance & Performance** > Toggle **Reduce visual effects** to show instant rendering mode.
2. **Dedicated Install App Page**:
   - **Navigate**: Sign in, open **📲 Install App** from the sidebar's **More** group.
   - *Presenter Explains*: "ReviewLens provides a dedicated, glassmorphic installation hub for Android, iOS, and Desktop with zero app store downloads."
   - Point out the 3 platform tabs (Android Chrome, iOS Safari Share, and Desktop address-bar install), the copyable app link with QR code generator, and the **🚀 Open installable version** button linking directly to the standalone PWA shell.
   - Review the "Good to know" operational caveats (network requirement, container wake-up time, Google sign-in).
3. **PWA Standalone App**:
   - Show the PWA shell running from GitHub Pages or locally (`pwa/index.html`).
   - Demonstrate the browser install prompt ("Install ReviewLens").
   - Disconnect Wi-Fi / simulate offline in DevTools to display the custom branded `pwa/offline.html` service worker fallback.
4. **Conclusion**: Conclude with test suite evidence (`python -m unittest discover tests` passing with 165+ unit tests).
