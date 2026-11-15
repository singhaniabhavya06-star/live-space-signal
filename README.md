# Live Space Signal — Streamlit AI Prototype

## What is included
- Visitor dashboard with green / yellow / red occupancy statuses
- Password-protected admin panel to update occupancy and add locations
- AI Space Assistant (OpenAI API when configured; local fallback otherwise)
- Random Forest next-hour crowd forecast using synthetic demonstration data
- Management insights, analytics charts, update history and CSV export
- SQLite database created automatically

## A. Run it on your computer (easiest reliable first test)

1. Install Python 3.10+ from https://www.python.org/downloads/
2. Extract this ZIP file.
3. Open Command Prompt / Terminal in the extracted `Live_Space_Signal_Streamlit` folder.
4. Install packages:

   `pip install -r requirements.txt`

5. Start the app:

   `streamlit run app.py`

6. It opens in your browser, normally at `http://localhost:8501`.

Default demo admin password: `signal123`.

## B. Access from your phone on the same Wi-Fi

Run:

`streamlit run app.py --server.address 0.0.0.0`

Find your computer's local IP address and open `http://YOUR-COMPUTER-IP:8501` on your phone. Both devices must be on the same Wi-Fi. This is not public internet hosting.

## C. Publish it online so your professor can open a link

1. Create/sign in to GitHub: https://github.com/
2. Create a repository named `live-space-signal`.
3. Upload `app.py` and `requirements.txt` to the repository root. README is optional.
4. Open Streamlit Community Cloud: https://share.streamlit.io/
5. Sign in with GitHub and choose **Create app**.
6. Select your repository, branch, and `app.py` as the main file.
7. In the app's **Advanced settings / Secrets**, add TOML like:

   ```
   ADMIN_PASSWORD = "choose-a-strong-password"
   OPENAI_API_KEY = "your-api-key"
   OPENAI_MODEL = "gpt-5.5"
   ```

   The OpenAI settings are optional. Do not commit API keys or passwords to GitHub.
8. Deploy. Streamlit Community Cloud will provide a public URL you can share.

## D. Enable the OpenAI assistant (optional)

Without a key, the app still answers common questions using a local rule-based fallback. To enable generative AI, add `OPENAI_API_KEY` to Streamlit Cloud Secrets as above, or set it as an environment variable locally.

API usage can incur separate charges. Keep the key private.

## Important limitations
- Seeded stall names and occupancy values are synthetic, not actual Migsun Ultimo data.
- Staff/admin manually enter occupancy; there are no sensors or automatic people counters.
- Status thresholds: Available <50%, Limited 50% to <80%, Full >=80%.
- The Random Forest forecast uses synthetic evening-peak data. It is a demonstration of the ML workflow, not a validated prediction of actual market footfall.
- The local fallback assistant is rule-based, not generative AI. OpenAI integration is optional.
- The app uses SQLite. Hosted app filesystems may be reset during restarts/redeployments; use a managed persistent database such as Supabase PostgreSQL before relying on long-term online data storage.
- For credible ML, collect timestamped real occupancy observations for several weeks, use chronological train/test splits, and report MAE/RMSE against a baseline.
- This is an academic prototype, not a production service. Add stronger authentication, role-based access, secure persistent storage, HTTPS and audit controls before real deployment.
