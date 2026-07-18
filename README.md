# A.R.I.A-Accessibility-Routing-Intelligent-Administration-FIFA-World-Cup-2026
FIFA World Cup 2026 • Smart Stadiums & Tournament Operations
A.R.I.A is an AI-powered chat copilot and real-time operations dashboard built specifically for fans with accessibility needs attending FIFA World Cup 2026 matches across the USA, Canada, and Mexico. A fan declares their language and access needs once, then asks anything in plain language—such as "quietest gate for a sensory-sensitive kid?", "wheelchair route from Gate C to my section?", or "¿dónde está la sala de lactancia?"—and receives a concise, screen-reader-friendly answer grounded securely in a structured venue dataset.

The system runs in two parallel modes to ensure high availability:

Live Mode: Uses the Google GenAI SDK (gemini-2.5-flash by default) driving an intelligent function-calling loop over localized stadium data tools.

Offline Mode: A fully deterministic keyword and intent router that strips text normalization and applies regex matching to local data tables. It requires zero API keys or network connection, serving as a robust fallback if the network drops.

🚀 Key Features
1. Fan Assistant Chat Interface
Accessibility-First Design: Fully compliant with semantic landmarks, custom screen-reader focus handling, and strict aria-live attributes to announce live-streaming responses clearly.

Multilingual Core: Full native and semantic routing for English, Spanish, French, Arabic, and Hindi, complete with automatic Right-to-Left (RTL) text layout switching for Arabic.

Context-Aware Routing: The AI evaluates a unified profile context (Selected Venue + Declared Need + Live Feeds) so a sensory need guides users away from loud gates, while a mobility need reroutes users dynamically if an elevator goes down.

2. Live Operations Admin Dashboard
Real-Time Sim-Feed: Secure operational overview showing live gate traffic metrics, seating congestion maps, and critical resource statuses.

Outage Flagging: Administrative control panels to log dynamic infrastructure disruptions (e.g., broken escalators, elevator outages) that instantly update the Fan Assistant routing logic.

🛠️ Project Architecture & Stack
'''Plaintext
├── app/                  # Backend Application Core (Python / FastAPI)
│   ├── main.py           # Application Entry point, Security Middleware & API Router
│   ├── assistant.py      # Live Gemini AI Logic & Function-Calling Loop
│   ├── offline.py        # Deterministic Regex Keyword Fallback Engine
│   └── tools.py          # Venue Data Filters, Live-Ops Simulators, & Route Planners
└── static/               # Frontend Assets (100% Zero-Framework Vanilla Stack)
    ├── index.html        # Highly semantic structural layout (WCAG Compliant)
    ├── styles.css        # Clean layout styling with opaque, high-contrast UI panels
    └── app.js            # Secure token handling, Fetch API calls, & WebGL interface wrapper'''
Frontend: Standard HTML5, Vanilla CSS3, and Vanilla JavaScript. Built completely free of heavy frameworks or external CDNs to maximize loading speeds and stability over congested stadium cellular networks.

Backend: FastAPI (Python 3.12+).

AI Engine: Google GenAI SDK providing active runtime tooling adjustments based on structural mock inputs.

Security & Controls: Custom token-bucket rate-limiting middleware, strict automated Content Security Policies (CSP), frame-options protection, and HTTP header hardening.

💻 Getting Started Locally
Prerequisites
Python 3.12 or newer installed on your machine.

Installation & Launch
Clone the project and enter the working directory:

'''Bash
cd aria-stadium-platform
Set up a clean virtual environment:'''

Mac/Linux:

'''Bash
python3 -m venv .venv
source .venv/bin/activate'''
Windows:

'''DOS
python -m venv .venv
.venv\Scripts\activate'''
Install the dependencies:

'''Bash
pip install -r requirements.txt'''
Configure Environment Variables:
Create a .env file in the root directory:

'''Code snippet
GEMINI_API_KEY="your_optional_gemini_api_key"
ADMIN_USERNAME="admin"
ADMIN_PASSWORD="your_secure_dashboard_password"
(Note: Omitting the GEMINI_API_KEY will safely run the entire chat interface using the deterministic local Offline Engine).'''

Fire up the development server:

'''Bash
uvicorn app.main:app --reload'''
View the platform:

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser to launch the Fan Assistant interface.

Navigate to your designated dashboard route to sign in to the Admin Console using your configuration credentials.

🔒 Security Operations
Zero-Leak Principles: The system environment files are hard-blocked from tracking. All sensitive keys are evaluated out of system memory exclusively, and internal authentication tokens are signed securely using backend middleware dependencies.

UI Stacking Insulation: Floating operational parameters run completely encapsulated inside opaque structural layouts (background: rgba(25, 25, 25, 0.95)), preventing text readability conflicts or visual overlap issues over embedded core rendering windows.

Strict Input Sanitization: Core inputs are constrained via strict validation schemas (1-2000 character restrictions, explicit historical conversation sequence maximums) protecting endpoints directly from malicious buffer attacks or prompt injection models.

🏆 This project was built for the Virtual Wars Hackathon by Hack2Skill.
