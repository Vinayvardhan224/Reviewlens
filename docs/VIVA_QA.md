# ReviewLens: 25 High-Probability Viva Defense Questions & Answers

An exhaustive academic and architectural preparation guide for examiners, professors, and technical review committees.

---

### 1. Why use VADER instead of immediately fine-tuning a Deep Transformer (like BERT or RoBERTa)?
**Answer**: VADER is a rule-based valence lexicon optimized specifically for social and consumer expressions. It incurs virtually zero inference latency (<1ms on standard CPU), requires no costly GPU training pipelines, and is inherently explainable down to exact sentiment modifiers, capitalization, and punctuation rules. Furthermore, VADER serves as a strong zero-shot baseline before justifying deep models on edge devices.

### 2. What are the key limitations of VADER in consumer electronics?
**Answer**: VADER struggles with:
1. **Neutral Class Boundary**: Text stating plain technical specifications (e.g., *"The laptop has 16GB RAM and two ports"*) often lands in the $[-0.05, +0.05]$ deadband, giving low neutral F1 (~0.33).
2. **Domain-Specific Lexicon Drift**: Slang like *"this phone is a beast"* or technical jargon like *"fan noise"* are not natively weighted unless domain modifiers are added.
3. **Complex Sarcasm**: Sarcastic remarks containing literal positive adjectives (*"Brilliant battery, lasts 20 minutes"*) will trick lexicon summations.

### 3. How does ReviewLens calculate the Net Sentiment Score (NSS)?
**Answer**:
$$ \text{NSS} = \frac{\text{Positive Reviews} - \text{Negative Reviews}}{\text{Total Reviews}} \times 100 $$
NSS produces a normalized scalar between -100 and +100. When an aspect has 0 mentions, ReviewLens outputs `None` rather than 0.0 to prevent misleading consumers into thinking an unmentioned aspect is neutral.

### 4. How does your Aspect-Based Sentiment Analysis (ABSA) work without deep neural token classification?
**Answer**: ReviewLens uses a hybrid approach:
1. Category-specific seed taxonomies for common hardware components (Camera, Battery, Display, etc.).
2. **Syntactic Dependency Parsing via spaCy**: Traverses grammatical tree relations (`amod`, `acomp`, `nsubj`, `attr`) to connect descriptive adjectives to target noun chunks, isolating fine-grained sentiment at the clause level.

### 5. Why do you use Latent Dirichlet Allocation (LDA) for topic modeling?
**Answer**: LDA is a generative probabilistic model that assumes documents are mixtures of topics, and topics are mixtures of words. It allows unsupervised discovery of latent conversational themes across hundreds of reviews without requiring manual labelling.

### 6. What TF-IDF representation was used in the supervised benchmark?
**Answer**: Sublinear term-frequency scaling ($1 + \log(\text{tf})$), unigram through trigram lexical combinations (1-3 ngrams), with minimum document frequency `min_df=2` and vocabulary capped at 5,000 features.

### 7. What classifier was paired with TF-IDF, and why?
**Answer**: Logistic Regression with $L_2$ regularization ($C=2.0$). Linear models with sparse n-gram matrices provide fast, highly interpretable coefficients while achieving competitive macro F1 on consumer texts.

### 8. What is the Ott et al. benchmark, and why is there a domain shift caveat?
**Answer**: Ott et al. (ACL 2011) is the gold standard benchmark for deceptive opinion spam, comprising 1,600 truthful and deceptive reviews of 20 Chicago hotels gathered via Mechanical Turk. Because deceptive hotel reviews heavily feature imaginative narratives of amenities and family stays, applying this model directly to technical electronics creates a domain shift that slightly lowers precision.

### 9. How does ReviewLens detect fake or spam reviews in production?
**Answer**: A two-layer defense:
1. **Rule-Based Heuristic Filter**: Checks excessive promotional vocabulary, review brevity, all-caps ratio, and URL links.
2. **Supervised ML Model**: An ExtraTrees ensemble trained on Ott et al. n-gram and behavioral features, outputting calibrated spam probabilities.

### 10. Why was the Stanford Sentiment Treebank (SST-2) included in evaluation?
**Answer**: SST-2 is part of the official GLUE benchmark. Evaluating on 872 independent test sentences provides an un-cherrypicked academic cross-validation proving our pipeline performs consistently against standard NLP literature.

### 11. Why does DistilBERT appear in the SST-2 table with a "[Ref]" citation?
**Answer**: DistilBERT's 92.7% accuracy is an official published reference metric from GLUE literature. In the spirit of scientific transparency, we explicitly label it as a literature benchmark rather than claiming local in-memory execution on lightweight Streamlit Cloud containers.

### 12. What was your SemEval-2014 Task 4 result?
**Answer**: On the official SemEval-2014 Laptop gold benchmark, our syntactic dependency aspect parser achieved **89.7% recall** in capturing consumer aspects, with **54.5% precision** and **67.8% F1**. The moderate precision is expected because syntactic parsing extracts all qualifying noun-adjective pairs, some of which are generic product references.

### 13. What is the purpose of Qualitative Error Analysis?
**Answer**: Metric numbers alone hide systematic linguistic failure modes. We cataloged 30 actual failure cases into structured categories: `boundary_neutral`, `negation_scope`, `sarcasm`, `lexical_ambiguity`, and `domain_shift`.

### 14. What is litotes, and why does it break sentiment analyzers?
**Answer**: Litotes is affirmation expressed by the negative of the contrary (e.g., *"not bad"*, *"not unreliable"*). Simple bag-of-words classifiers detect the word *"not"* and register a negative polarity, whereas the human speaker intended a positive or acceptable rating.

### 15. How does ReviewLens ensure secure password storage?
**Answer**: It implements OWASP-recommended **PBKDF2-HMAC-SHA256** with **600,000 iterations** and a unique cryptographically generated 16-byte salt per user. Plaintext passwords are never stored or kept in session memory.

### 16. How is user database persistence made atomic and thread-safe?
**Answer**: Database writes use `tempfile.NamedTemporaryFile` within the target directory, followed by `os.fsync()` and `os.replace()` under a module-level `threading.Lock()`. This guarantees atomic file replacement on disk and eliminates corruption under concurrent requests.

### 17. How does the Email OTP verification work?
**Answer**: Generates a 6-digit cryptographic code via `secrets.randbelow()`, stores a salted SHA-256 hash in `st.session_state` alongside expiration and attempt limits, and dispatches via Gmail SMTP SSL on port 465. Verification employs `hmac.compare_digest()` to eliminate timing attack vectors.

### 18. What rate-limiting and security controls protect the OTP service?
**Answer**:
1. 60-second cooldown between resends.
2. 5 dispatches per target per hour persisted to disk in `data/otp_rate_limits.json`.
3. 100 global dispatches per 24 hours to prevent Gmail API exhaustion.
4. Account lockout of 10 minutes after 5 consecutive failed login attempts.

### 19. What was the critical Demo Mode safety vulnerability you fixed?
**Answer**: In sandbox demo mode, an OTP bypass code (`123456`) was supported. If left unrestricted, an attacker could abuse this code to reset the password of any registered user. We restricted `123456` bypass strictly to `DEMO_ACCOUNTS` and brand-new signups, completely blocking password resets for existing accounts.

### 20. Why did you restructure the Streamlit application execution order?
**Answer**: To resolve the "double login" and page flashing bug on Streamlit Cloud. Streamlit runs top-to-bottom on every interaction. By placing the authentication check and `st.stop()` *before* any catalog CSV parsing or chart generation, unauthorized sessions halt immediately without loading unnecessary datasets.

### 21. Why did you delete the `pages/` directory?
**Answer**: Streamlit's native multi-page router auto-discovers scripts inside `pages/` and executes their state machines concurrently with the main router in `app.py`. Deleting `pages/` centralized all routing inside `modules/navigation.py`, eliminating ghost render loops.

### 22. What is the architecture of the Progressive Web Application (PWA)?
**Answer**:
- `pwa/index.html`: Fast HTML5 shell hosting the app viewport.
- `pwa/sw.js`: Service worker caching static icon assets and intercepting network errors.
- `pwa/manifest.webmanifest`: Defines app icon assets (split into `any` and `maskable` purposes), standalone display mode, and dark purple theme color `#07051A`.
- `pwa/offline.html`: Branded offline fallback screen shown when connectivity drops.

### 23. What does your automated pre-flight deployment script check?
**Answer**: `scripts/check_deployment.py` automatically checks:
1. `pwa/config.js` has a valid HTTPS production URL.
2. Web manifest is valid JSON with existing icon files.
3. Separate `any` and `maskable` icon purposes.
4. Consistency between `requirements.txt` and `requirements-lock.txt`.
5. `.streamlit/secrets.toml` is untracked by git.
6. Tracked files contain zero leaked API keys or private certificates.

### 24. What happens on Streamlit Cloud when an ephemeral instance restarts?
**Answer**: Streamlit Cloud instances operate on containerized ephemeral storage. When containers cycle, newly written files in `data/users.json` revert to repository git state. We explicitly documented this in the System Status panel with an Ephemeral Storage Notice. For production scalability, the storage layer can be swapped to Supabase or AWS RDS with zero changes to the `modules/auth.py` interface.

### 25. How do you summarize the overarching academic contribution of ReviewLens?
**Answer**: ReviewLens demonstrates that a thoughtfully engineered hybrid NLP architecture—uniting high-speed rule-based lexicons, syntactic dependency trees, linear machine learning baselines, and transparent error analysis—can deliver actionable, explainable consumer intelligence at a fraction of the carbon and financial cost of unconstrained deep learning pipelines.
