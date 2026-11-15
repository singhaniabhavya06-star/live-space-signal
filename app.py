import os
import sqlite3
import math
import random
from datetime import datetime

import pandas as pd
import streamlit as st

try:
    from sklearn.ensemble import RandomForestRegressor
except ImportError:
    RandomForestRegressor = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ---------- CONFIGURATION ----------

def get_secret(name, default=""):
    """Safely read Streamlit secrets or environment variables."""
    try:
        return st.secrets.get(name, os.environ.get(name, default))
    except Exception:
        return os.environ.get(name, default)


DB = os.environ.get(
    "DATABASE_PATH",
    os.path.join(os.path.dirname(__file__), "live_space_signal.db")
)

st.set_page_config(
    page_title="Live Space Signal",
    page_icon="🟢",
    layout="wide"
)

STATUS_COLOR = {
    "Available": "#3ea67c",
    "Limited": "#e0a82e",
    "Full": "#e5575a"
}

STATUS_ICON = {
    "Available": "🟢",
    "Limited": "🟡",
    "Full": "🔴"
}


# ---------- DATABASE FUNCTIONS ----------

def conn():
    c = sqlite3.connect(DB, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def status_for(occupied, capacity):
    ratio = occupied / capacity if capacity else 0

    if ratio >= 0.8:
        return "Full"
    elif ratio >= 0.5:
        return "Limited"
    return "Available"


def init():
    with conn() as c:

        c.execute("""
            CREATE TABLE IF NOT EXISTS locations(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                category TEXT NOT NULL,
                capacity INTEGER NOT NULL,
                occupied INTEGER NOT NULL,
                status TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS history(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                location_id INTEGER,
                status TEXT,
                occupied INTEGER,
                capacity INTEGER,
                recorded_at TEXT
            )
        """)

        if c.execute(
            "SELECT COUNT(*) FROM locations"
        ).fetchone()[0] == 0:

            now = datetime.now().astimezone().isoformat(
                timespec="minutes"
            )

            seed = [
                ("Stall A", "Food Stall", 20, 5),
                ("Stall B", "Food Stall", 20, 13),
                ("Stall C", "Food Stall", 20, 20),
                ("Stall D", "Food Stall", 16, 4),
                ("Seating Zone A", "Seating Area", 40, 27),
                ("Seating Zone B", "Seating Area", 30, 8)
            ]

            for name, category, capacity, occupied in seed:

                status = status_for(occupied, capacity)

                cur = c.execute(
                    """
                    INSERT INTO locations
                    (name, category, capacity, occupied, status, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        name,
                        category,
                        capacity,
                        occupied,
                        status,
                        now
                    )
                )

                c.execute(
                    """
                    INSERT INTO history
                    (location_id, status, occupied, capacity, recorded_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        cur.lastrowid,
                        status,
                        occupied,
                        capacity,
                        now
                    )
                )


def get_locations():
    with conn() as c:
        return pd.DataFrame([
            dict(row)
            for row in c.execute(
                "SELECT * FROM locations ORDER BY category, name"
            )
        ])


def get_history():
    with conn() as c:
        return pd.DataFrame([
            dict(row)
            for row in c.execute("""
                SELECT h.*, l.name
                FROM history h
                JOIN locations l ON l.id = h.location_id
                ORDER BY h.id DESC
            """)
        ])


def update(locid, capacity, occupied):

    status = status_for(occupied, capacity)

    now = datetime.now().astimezone().isoformat(
        timespec="minutes"
    )

    with conn() as c:

        c.execute(
            """
            UPDATE locations
            SET capacity=?, occupied=?, status=?, updated_at=?
            WHERE id=?
            """,
            (
                capacity,
                occupied,
                status,
                now,
                locid
            )
        )

        c.execute(
            """
            INSERT INTO history
            (location_id, status, occupied, capacity, recorded_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                locid,
                status,
                occupied,
                capacity,
                now
            )
        )

    return status


def add_location(name, category, capacity):

    now = datetime.now().astimezone().isoformat(
        timespec="minutes"
    )

    with conn() as c:

        cur = c.execute(
            """
            INSERT INTO locations
            (name, category, capacity, occupied, status, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                category,
                capacity,
                0,
                "Available",
                now
            )
        )

        c.execute(
            """
            INSERT INTO history
            (location_id, status, occupied, capacity, recorded_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                cur.lastrowid,
                "Available",
                0,
                capacity,
                now
            )
        )


init()


# ---------- AI ASSISTANT ----------

def local_answer(question, df):

    q = question.lower()

    if df.empty:
        return "There are no locations in the system yet."

    if any(word in q for word in ["seat", "seating", "sit"]):

        locations = df[
            (df.category == "Seating Area") &
            (df.status != "Full")
        ].copy()

    else:
        locations = df[df.status != "Full"].copy()

    keywords = [
        "least",
        "available",
        "recommend",
        "where",
        "alternative",
        "crowded",
        "seat",
        "suggest"
    ]

    if any(word in q for word in keywords):

        if locations.empty:
            return (
                "All suitable participating locations are currently "
                "marked Full. Please check again shortly."
            )

        locations["pct"] = (
            locations.occupied / locations.capacity
        ) * 100

        locations = locations.sort_values(
            "pct"
        ).head(3)

        recommendations = []

        for i, (_, row) in enumerate(
            locations.iterrows(),
            1
        ):
            recommendations.append(
                f"{i}. **{row['name']}** — "
                f"{row['pct']:.0f}% occupied "
                f"({row['status']})"
            )

        return (
            "Based on the latest reported availability:\n\n"
            + "\n\n".join(recommendations)
            + "\n\nThese are occupancy levels, not guaranteed waiting times."
        )

    full = df[df.status == "Full"].name.tolist()

    if any(word in q for word in ["full", "busy", "crowd"]):

        return (
            "Currently marked Full: "
            + (", ".join(full) if full else "none")
            + "."
        )

    return (
        "Ask me for the least-crowded stall, available seating, "
        "an alternative to a full location, or current crowd levels."
    )


def assistant(question, df):

    key = get_secret("OPENAI_API_KEY", "")

    if key and OpenAI:

        try:

            client = OpenAI(api_key=key)

            data = df[
                [
                    "name",
                    "category",
                    "capacity",
                    "occupied",
                    "status",
                    "updated_at"
                ]
            ].to_dict("records")

            response = client.responses.create(

                model=get_secret(
                    "OPENAI_MODEL",
                    "gpt-5.5"
                ),

                instructions=(
                    "You are Live Space Signal's visitor assistant. "
                    "Use only supplied location data. Recommend low-occupancy "
                    "suitable options. Never invent locations, wait times "
                    "or sensor readings. Mention updates are manual and "
                    "may be stale. Be concise."
                ),

                input=(
                    f"Current location data: {data}\n"
                    f"Visitor question: {question}"
                ),

                max_output_tokens=220
            )

            return response.output_text, "OpenAI API"

        except Exception as exc:

            st.session_state["ai_error"] = str(exc)[:240]

    return local_answer(question, df), "Local fallback"


# ---------- CROWD FORECAST MODEL ----------

def synthetic_training():

    rng = random.Random(42)
    rows = []

    for location_id in range(1, 21):

        for day in range(7):

            for hour in range(10, 23):

                evening = math.exp(
                    -((hour - 19) / 2.2) ** 2
                )

                lunch = 0.42 * math.exp(
                    -((hour - 13) / 1.8) ** 2
                )

                value = (
                    (0.10 + 0.58 * evening + lunch)
                    * (0.82 + location_id * 0.012)
                    * (1 if day < 5 else 1.10)
                    + rng.uniform(-0.07, 0.07)
                )

                rows.append((
                    location_id,
                    hour,
                    day,
                    max(0.02, min(0.98, value))
                ))

    return pd.DataFrame(rows)


@st.cache_resource
def model():

    if RandomForestRegressor is None:
        return None

    data = synthetic_training()

    regressor = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        min_samples_leaf=2
    )

    regressor.fit(
        data[[0, 1, 2]],
        data[3]
    )

    return regressor


def forecasts(df):

    regressor = model()

    hour = (datetime.now().hour + 1) % 24

    rows = []

    for _, row in df.iterrows():

        if regressor:

            value = float(
                regressor.predict(
                    pd.DataFrame([[
                        int(row.id),
                        hour,
                        datetime.now().weekday()
                    ]])
                )[0]
            )

        else:

            value = row.occupied / row.capacity

        value = max(0, min(1, value))

        rows.append({

            "Location": row["name"],

            "Category": row.category,

            "Predicted occupancy": round(value * 100),

            "Crowd level": (
                "High" if value >= 0.8
                else "Moderate" if value >= 0.5
                else "Low"
            ),

            "Forecast hour": f"{hour:02d}:00"
        })

    return pd.DataFrame(rows)


# ---------- UI STYLING ----------

st.markdown("""
<style>

.block-container {
    max-width: 1250px;
    padding-top: 1.5rem;
}

.hero {
    background: linear-gradient(120deg, #153d32, #2f7a5f);
    padding: 2.2rem 2rem;
    border-radius: 20px;
    color: #fff;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 24px rgba(0,0,0,.35);
}

.hero .eyebrow {
    color: #a9dcc5;
    letter-spacing: .14em;
    font-size: .78rem;
    font-weight: 600;
    margin: 0 0 .4rem;
}

.hero h1 {
    color: #fff;
    font-size: 2.4rem;
    line-height: 1.15;
    margin: 0 0 .6rem;
    padding: 0;
}

.hero p.sub {
    color: #d7ebe0;
    font-size: 1.05rem;
    margin: 0;
}

div[data-testid="stMetric"] {
    background: #1f2228;
    border: 1px solid #2f343c;
    border-left: 4px solid #3ea67c;
    padding: 16px 18px;
    border-radius: 14px;
}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] * {
    color: #9aa4b2 !important;
    font-weight: 600;
}

div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] * {
    color: #f2f2f2 !important;
    font-weight: 700;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 16px;
}

.pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: .78rem;
    font-weight: 700;
    letter-spacing: .04em;
    color: #101214;
}

.cat {
    color: #8b95a3;
    font-size: .75rem;
    letter-spacing: .12em;
    font-weight: 600;
}

.spaces {
    color: #aab3bf;
    font-size: .85rem;
}

section[data-testid="stSidebar"] {
    border-right: 1px solid #2f343c;
}

</style>
""", unsafe_allow_html=True)


# ---------- NAVIGATION ----------

st.sidebar.title("🟢 Live Space Signal")



page = st.sidebar.radio(
    "Navigate",
    [
        "Visitor Dashboard",
        "Admin Panel",
        "AI & Crowd Forecast",
        "Analytics",
        "3D Food Court",
        "3D Digital Signage System",
        "3D Staff Workstation"
    ]
)



st.sidebar.caption(
    "Academic prototype · Manual occupancy updates"
)

st.sidebar.caption(
    "Storage: SQLite demo database; cloud restarts may reset data. "
    "Use managed storage for persistent deployment."
)

df = get_locations()


# ---------- VISITOR DASHBOARD ----------

if page == "Visitor Dashboard":

    st.markdown(
        '<div class="hero">'
        '<p class="eyebrow">MIGSUN ULTIMO · VISITOR INFORMATION</p>'
        '<h1>Find your space. Skip the guesswork.</h1>'
        '<p class="sub">Check participating food stalls and seating areas '
        'before deciding where to go.</p>'
        '</div>',
        unsafe_allow_html=True
    )

    counts = df.status.value_counts()

    a, b, c, d = st.columns(4)

    a.metric("📍 Locations", len(df))

    b.metric(
        "🟢 Available",
        int(counts.get("Available", 0))
    )

    c.metric(
        "🟡 Limited",
        int(counts.get("Limited", 0))
    )

    d.metric(
        "🔴 Full",
        int(counts.get("Full", 0))
    )

    st.subheader("Current availability")

    x, y, z = st.columns([2, 1, 1])

    term = x.text_input(
        "Search locations",
        placeholder="Type a stall or seating zone name…"
    )

    filt = y.selectbox(
        "Filter status",
        ["All", "Available", "Limited", "Full"]
    )

    cat_f = z.selectbox(
        "Category",
        ["All", "Food Stall", "Seating Area"]
    )

    view = df.copy()

    if term:
        view = view[
            view.name.str.contains(
                term,
                case=False,
                na=False
            )
        ]

    if filt != "All":
        view = view[view.status == filt]

    if cat_f != "All":
        view = view[view.category == cat_f]

    if view.empty:
        st.info("No locations match your filters.")

    cards = st.columns(3)

    for i, (_, row) in enumerate(view.iterrows()):

        pct = round(
            row.occupied / row.capacity * 100
        )

        with cards[i % 3]:

            with st.container(border=True):

                st.markdown(
                    f'<div class="cat">{row.category.upper()}</div>',
                    unsafe_allow_html=True
                )

                st.markdown("### " + row["name"])

                st.markdown(
                    f'<span class="pill" '
                    f'style="background:{STATUS_COLOR[row.status]}">'
                    f'{STATUS_ICON[row.status]} {row.status.upper()}'
                    f'</span>',
                    unsafe_allow_html=True
                )

                st.progress(
                    pct / 100,
                    text=f"{pct}% occupied"
                )

                st.markdown(
                    f'<div class="spaces">'
                    f'<b>{row.capacity - row.occupied}</b> spaces available · '
                    f'{row.occupied}/{row.capacity} occupied'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.caption(
                    "Updated "
                    + pd.to_datetime(row.updated_at).strftime(
                        "%d %b, %I:%M %p"
                    )
                )

    st.divider()

    st.subheader("🤖 AI Space Assistant")

    st.write(
        "Ask for the least-crowded stall, available seating, or an alternative."
    )

    if "messages" not in st.session_state:

        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! Ask me where to find available food stalls "
                    "or seating."
                )
            }
        ]

    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input(
        "e.g. Where can I find the least crowded seating?"
    )

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        answer, mode = assistant(question, df)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        with st.chat_message("assistant"):

            st.markdown(answer)

            st.caption("Mode: " + mode)

        if st.session_state.get("ai_error"):

            st.caption(
                "OpenAI connection issue; local fallback used: "
                + st.session_state["ai_error"]
            )

            st.session_state.pop("ai_error", None)

    st.info(
        "Available: below 50% occupied · "
        "Limited: 50–79% · Full: 80% or above."
    )


# ---------- ADMIN PANEL ----------

elif page == "Admin Panel":

    st.title("⚙️ Management Admin Panel")

    password = st.text_input(
        "Admin password",
        type="password"
    )

    expected = get_secret(
        "ADMIN_PASSWORD",
        "signal123"
    )

    if expected == "signal123":

        st.warning(
            "Demo password is active. Before public presentation, "
            "set ADMIN_PASSWORD in Streamlit → App settings → Secrets."
        )

    if password != expected:

        st.warning(
            "Enter the admin password to access update controls."
        )

        st.stop()

    st.write(
        "Update capacity and occupied seats. Status is calculated automatically."
    )

    for _, row in df.iterrows():

        with st.container(border=True):

            a, b, c, d = st.columns([2, 1, 1, 1])

            a.markdown(
                f"**{row['name']}**  \n"
                f"{row.category} · "
                f"{STATUS_ICON[row.status]} {row.status}"
            )

            capacity = b.number_input(
                "Capacity",
                1,
                10000,
                int(row.capacity),
                key=f"cap{row.id}"
            )

            occupied = c.number_input(
                "Occupied",
                0,
                10000,
                int(row.occupied),
                key=f"occ{row.id}"
            )

            if d.button(
                "Save",
                key=f"save{row.id}",
                use_container_width=True
            ):

                if occupied > capacity:

                    st.error(
                        "Occupied count cannot exceed capacity."
                    )

                else:

                    st.success(
                        f"Updated to "
                        f"{update(int(row.id), int(capacity), int(occupied))}"
                    )

                    st.rerun()

    # ---------- CROWD SIMULATOR ----------

    st.divider()

    st.subheader(
        "🎛️ Crowd simulation — demonstration only"
    )

    st.caption(
        "Use this control during your faculty demo to show how occupancy "
        "status and analytics respond. This changes the stored demo data "
        "and creates a history record."
    )

    sim_location = (
        st.selectbox(
            "Choose a location to simulate",
            df["name"].tolist(),
            key="sim_location"
        )
        if not df.empty
        else None
    )

    if sim_location:

        sim_row = df[
            df.name == sim_location
        ].iloc[0]

        sim_pct = st.slider(
            "Simulated occupancy",
            0,
            100,
            int(round(
                sim_row.occupied / sim_row.capacity * 100
            )),
            5,
            key="sim_pct"
        )

        if st.button(
            "Apply simulation",
            type="primary",
            use_container_width=True
        ):

            simulated_occupied = min(
                int(sim_row.capacity),
                round(int(sim_row.capacity) * sim_pct / 100)
            )

            new_status = update(
                int(sim_row.id),
                int(sim_row.capacity),
                simulated_occupied
            )

            st.success(
                f"Demo update applied: {sim_location} "
                f"is now {new_status} at {sim_pct}% occupancy."
            )

            st.rerun()

        if st.button(
            "Reset this location to original demo value",
            use_container_width=True
        ):

            baseline = {
                "Stall A": 5,
                "Stall B": 13,
                "Stall C": 20,
                "Stall D": 4,
                "Seating Zone A": 27,
                "Seating Zone B": 8
            }

            reset_occupied = min(
                int(sim_row.capacity),
                baseline.get(sim_location, 0)
            )

            new_status = update(
                int(sim_row.id),
                int(sim_row.capacity),
                reset_occupied
            )

            st.success(
                f"{sim_location} reset to "
                f"{reset_occupied}/{int(sim_row.capacity)} "
                f"({new_status})."
            )

            st.rerun()

    # ---------- ADD LOCATION ----------

    st.divider()

    st.subheader("Add a location")

    with st.form("add"):

        name = st.text_input("Location name")

        category = st.selectbox(
            "Category",
            ["Food Stall", "Seating Area"]
        )

        capacity = st.number_input(
            "Capacity",
            1,
            10000,
            20
        )

        go = st.form_submit_button(
            "Add location"
        )

        if go:

            try:

                if not name.strip():

                    st.error("Enter a location name.")

                else:

                    add_location(
                        name.strip(),
                        category,
                        int(capacity)
                    )

                    st.success("Location added.")

                    st.rerun()

            except sqlite3.IntegrityError:

                st.error(
                    "That location already exists."
                )


# ---------- AI & CROWD FORECAST ----------

elif page == "AI & Crowd Forecast":

    st.title("🧠 AI & Machine Learning")

    st.write(
        "Recommendations use current status data. "
        "Forecasts demonstrate a Random Forest workflow."
    )

    fc = forecasts(df)

    st.subheader(
        "Predicted occupancy · next hour"
    )

    st.dataframe(
        fc,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Predicted occupancy":
                st.column_config.ProgressColumn(
                    "Predicted occupancy",
                    format="%d%%",
                    min_value=0,
                    max_value=100
                )
        }
    )

    st.warning(
        "Forecast model is trained on synthetic evening-peak patterns, "
        "not actual Migsun Ultimo observations. Treat it as a demonstration only."
    )

    if not df.empty:

        busy = (
            df.assign(
                pct=df.occupied / df.capacity * 100
            )
            .sort_values(
                "pct",
                ascending=False
            )
            .head(2)
        )

        st.subheader(
            "Management insights"
        )

        st.write(
            f"- {int((df.status == 'Available').sum())} "
            f"of {len(df)} locations are currently marked Available."
        )

        st.write(
            "- Highest occupancy: "
            + ", ".join(
                f"{row['name']} ({row.pct:.0f}%)"
                for _, row in busy.iterrows()
            )
            + "."
        )

        st.write(
            "- Management action: verify high-occupancy locations "
            "during peak hours and encourage prompt staff updates."
        )


# ---------- ANALYTICS ----------

elif page == "Analytics":

    st.title(
        "📊 Business & Crowd Analytics"
    )

    hist = get_history()

    if not df.empty:

        dfx = df.assign(
            occupancy_pct=df.occupied / df.capacity * 100
        )

        a, b, c = st.columns(3)

        a.metric(
            "Average occupancy",
            f"{dfx.occupancy_pct.mean():.1f}%"
        )

        b.metric(
            "Highest occupancy",
            f"{dfx.occupancy_pct.max():.1f}%"
        )

        c.metric(
            "Locations tracked",
            len(df)
        )

        st.subheader(
            "Current status distribution"
        )

        st.bar_chart(
            df.status.value_counts().reindex(
                ["Available", "Limited", "Full"],
                fill_value=0
            ),
            color="#3ea67c"
        )

        st.subheader(
            "Occupancy by location"
        )

        st.bar_chart(
            dfx.set_index("name").occupancy_pct.sort_values(
                ascending=False
            ),
            color="#3ea67c"
        )

    st.subheader(
        "Status update history"
    )

    if hist.empty:

        st.info(
            "No updates recorded."
        )

    else:

        st.dataframe(
            hist[
                [
                    "recorded_at",
                    "name",
                    "status",
                    "occupied",
                    "capacity"
                ]
            ].head(100),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download history CSV",
            hist.to_csv(index=False).encode(),
            "live_space_signal_history.csv",
            "text/csv"
        )


elif page == "3D Food Court":
    st.title("🏙️ Live Space Signal — 3D Food Court")
    st.caption(
        "Conceptual 3D reconstruction based on site photographs. "
        "The internal layout is illustrative, not an architectural survey."
    )

    if df.empty:
        st.warning("No locations are available to display.")
    else:
        import streamlit.components.v1 as components

        locations_json = df[
            ["name", "category", "capacity", "occupied", "status"]
        ].to_json(orient="records")

        html = r"""
        <div id="lss-scene" style="width:100%;height:700px;background:#dce5e3;border-radius:16px;overflow:hidden;position:relative">
        <div id="lss-label" style="position:absolute;top:12px;left:12px;z-index:2;background:#ffffffed;padding:10px 14px;border-radius:10px;font:14px Arial;color:#172b27">
        <b>LIVE SPACE SIGNAL · 3D PLAZA</b><br>
        <span style="color:#38845e">● Available</span>
        <span style="color:#c89525">● Limited</span>
        <span style="color:#d94b4b">● Full</span>
        </div>
        </div>

        <script type="importmap">
        {
          "imports": {
            "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
          }
        }
        </script>

        <script type="module">
        import * as THREE from "three";
        import { OrbitControls } from "three/addons/controls/OrbitControls.js";

        const locations = __DATA__;
        const root = document.getElementById("lss-scene");

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xdce5e3);

        const camera = new THREE.PerspectiveCamera(
          45, root.clientWidth / root.clientHeight, 0.1, 1000
        );
        camera.position.set(24, 23, 29);

        const renderer = new THREE.WebGLRenderer({antialias:true});
        renderer.setSize(root.clientWidth, root.clientHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        root.appendChild(renderer.domElement);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.target.set(0, 1, 0);
        controls.maxPolarAngle = Math.PI / 2.05;

        scene.add(new THREE.HemisphereLight(0xffffff, 0x75816e, 2));

        const sun = new THREE.DirectionalLight(0xffffff, 2.2);
        sun.position.set(-12, 25, 15);
        sun.castShadow = true;
        scene.add(sun);

        function material(color, roughness=0.85) {
          return new THREE.MeshStandardMaterial({color, roughness});
        }

        function block(x,y,z,w,h,d,color) {
          const mesh = new THREE.Mesh(
            new THREE.BoxGeometry(w,h,d), material(color)
          );
          mesh.position.set(x,y+h/2,z);
          mesh.castShadow = true;
          mesh.receiveShadow = true;
          scene.add(mesh);
          return mesh;
        }

        // Paved plaza
        block(0,-0.25,0,34,0.5,27,0xb9b9ae);

        // Paving grid
        for(let x=-16; x<=16; x+=2) {
          block(x,0.005,0,0.035,0.01,26,0x999d94);
        }
        for(let z=-12; z<=12; z+=2) {
          block(0,0.006,z,33,0.01,0.035,0x999d94);
        }

        // Commercial shopfronts at the rear
        const shopColors = [0x76685b,0x424b4b,0x806b55,0x565d62];
        for(let i=0;i<4;i++) {
          const x=-12+i*8;
          block(x,0,-10,7.5,3.6,2.2,shopColors[i]);
          block(x,1.8,-8.82,6.6,1.25,0.08,0x283c42);
          block(x,3.65,-10,7.7,0.15,2.4,0x4a4b49);
          block(x,0.3,-8.78,6.5,0.18,0.12,0xc49b6c);
        }

        // Upper residential building backdrop
        block(0,3.7,-12.2,33,7,1.2,0xc5b69c);
        for(let i=0;i<11;i++) {
          const x=-15+i*3;
          block(x,5.2,-11.55,1.2,1.8,0.08,0x8da8a9);
          block(x,8,-11.55,1.2,1.8,0.08,0x8da8a9);
        }

        // Shop signage boards
        function label(text,x,y,z) {
          const canvas=document.createElement("canvas");
          canvas.width=512; canvas.height=128;
          const ctx=canvas.getContext("2d");
          ctx.fillStyle="#263b38";
          ctx.fillRect(0,0,512,128);
          ctx.fillStyle="#ffffff";
          ctx.font="bold 38px Arial";
          ctx.textAlign="center";
          ctx.fillText(text,256,78);
          const texture=new THREE.CanvasTexture(canvas);
          const sprite=new THREE.Sprite(
            new THREE.SpriteMaterial({map:texture})
          );
          sprite.position.set(x,y,z);
          sprite.scale.set(4.2,1.05,1);
          scene.add(sprite);
        }

        const stalls=locations.filter(x=>x.category==="Food Stall");
        stalls.forEach((item,i)=>{
          const x=-12+i*8;
          label(item.name,x,2.65,-8.65);
        });

        // Seating tables and chairs
        function table(x,z) {
          block(x,0.85,z,1.7,0.16,1.7,0x80664d);
          block(x,0.35,z,0.14,0.9,0.14,0x51483d);
          for(const dx of [-1.25,1.25]) {
            block(x+dx,0.45,z,0.65,0.15,0.65,0x536f64);
            block(x+dx,0.05,z,0.1,0.7,0.1,0x51483d);
          }
          for(const dz of [-1.25,1.25]) {
            block(x,0.45,z+dz,0.65,0.15,0.65,0x536f64);
            block(x,0.05,z+dz,0.1,0.7,0.1,0x51483d);
          }
        }

        for(let i=0;i<3;i++) {
          table(-10+i*4,2);
          table(-10+i*4,7);
        }

        // Palm trees and landscaping
        function tree(x,z) {
          block(x,0.2,z,0.45,3,0.45,0x806044);
          for(let i=0;i<7;i++) {
            const angle=i*Math.PI*2/7;
            const leaf=new THREE.Mesh(
              new THREE.ConeGeometry(0.55,3,5),
              material(0x34764c)
            );
            leaf.position.set(
              x+Math.cos(angle)*1.0,3.6,z+Math.sin(angle)*1.0
            );
            leaf.rotation.z=-Math.PI/2;
            leaf.rotation.y=-angle;
            scene.add(leaf);
          }
          block(x,0.1,z,2.4,0.2,2.4,0x77796b);
        }

        [-15,-5,5,15].forEach(x=>tree(x,-4));
        [-15,15].forEach(x=>tree(x,10));

        // Occupancy indicators
        const colors={
          "Available":0x31a56d,
          "Limited":0xe2aa35,
          "Full":0xe34f4f
        };

        const positions=[
          [-12,-6],[-4,-6],[4,-6],[12,-6],
          [-5,11],[5,11]
        ];

        locations.forEach((item,i)=>{
          const p=positions[i] || [0,0];
          const color=colors[item.status] || 0x777777;
          const marker=new THREE.Mesh(
            new THREE.CylinderGeometry(1.15,1.15,0.16,40),
            new THREE.MeshStandardMaterial({
              color:color,emissive:color,emissiveIntensity:0.18
            })
          );
          marker.position.set(p[0],0.12,p[1]);
          scene.add(marker);

          const pct=Math.round(item.occupied/item.capacity*100);
          
const textSprite=document.createElement("canvas");
textSprite.width=640;
textSprite.height=160;

const ctx=textSprite.getContext("2d");

// Dark label background
ctx.fillStyle="#202b30";
ctx.beginPath();
ctx.roundRect(10,10,620,140,18);
ctx.fill();

// Stall or seating name
ctx.textAlign="center";
ctx.textBaseline="middle";
ctx.fillStyle="#ffffff";
ctx.font="bold 40px Arial";
ctx.fillText(item.name,320,52);

// Occupancy status colour
const statusColor = {
  "Available":"#42d991",
  "Limited":"#ffc247",
  "Full":"#ff6262"
};

ctx.fillStyle=statusColor[item.status] || "#ffffff";
ctx.font="bold 34px Arial";
ctx.fillText(
  item.status+" · "+pct+"% occupied",
  320,108
);

          const sprite=new THREE.Sprite(
            new THREE.SpriteMaterial({
              map:new THREE.CanvasTexture(textSprite),
              transparent:true
            })
          );
          sprite.position.set(p[0],2.0,p[1]);
          sprite.scale.set(4.5,1.15,1);
          scene.add(sprite);
        });

        // Plaza boundary
        block(0,0,-13.2,34,0.5,0.5,0x626d69);
        block(0,0,13.2,34,0.5,0.5,0x626d69);
        block(-16.7,0,0,0.5,0.5,26,0x626d69);
        block(16.7,0,0,0.5,0.5,26,0x626d69);

        function animate() {
          requestAnimationFrame(animate);
          controls.update();
          renderer.render(scene,camera);
        }
        animate();

        window.addEventListener("resize",()=>{
          camera.aspect=root.clientWidth/root.clientHeight;
          camera.updateProjectionMatrix();
          renderer.setSize(root.clientWidth,root.clientHeight);
        });
        </script>
        </div>
        """

        html = html.replace("__DATA__", locations_json)

        components.html(
            html,
            height=720,
            scrolling=False
        )

        st.caption(
            "Occupancy indicators use the latest manually entered app data. "
            "The 3D layout is illustrative."
        )

elif page == "3D Digital Signage System":
    st.title("🖥️ Live Space Signal – 3D Digital Signage")
    st.caption(
        "Interactive 3D kiosk with a visitor display driven by current occupancy data."
    )

    if df.empty:
        st.warning("No location data is available.")
    else:
        import json
        import streamlit.components.v1 as components

        signage_data = df[
            ["name", "category", "capacity", "occupied", "status"]
        ].copy()

        signage_data["percentage"] = (
            signage_data["occupied"] / signage_data["capacity"] * 100
        ).round().astype(int)

        signage_json = json.dumps(
            signage_data.to_dict(orient="records")
        )

        html = r"""
        <div id="signage-scene"
     style="width:100%;height:720px;border-radius:16px;
            overflow:hidden;background:#111827;">
</div>

<div id="signage-error"
     style="color:white;background:#7f1d1d;padding:12px;
            margin-top:8px;border-radius:8px;font:14px Arial;">
    Loading 3D engine...
</div>

<script>
window.addEventListener("error", function(event) {
    const box = document.getElementById("signage-error");
    if (box) box.textContent = "JavaScript error: " + event.message;
});

window.addEventListener("unhandledrejection", function(event) {
    const box = document.getElementById("signage-error");
    if (box) {
        box.textContent =
            "Module loading error: " +
            (event.reason?.message || String(event.reason));
    }
});
</script>

        <script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }
}
</script>

<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

        const data = __SIGNAGE_DATA__;

        const root = document.getElementById("signage-scene");

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x111827);

        const camera = new THREE.PerspectiveCamera(
            42, root.clientWidth / root.clientHeight, 0.1, 100
        );
        camera.position.set(4, 3.2, 5.5);

        const renderer = new THREE.WebGLRenderer({antialias:true});
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.setSize(root.clientWidth, root.clientHeight);
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        root.appendChild(renderer.domElement);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.target.set(0, 1.6, 0);

        scene.add(new THREE.HemisphereLight(0xffffff, 0x334155, 2));

        const light = new THREE.DirectionalLight(0xffffff, 3);
        light.position.set(4, 8, 5);
        light.castShadow = true;
        scene.add(light);

        const floor = new THREE.Mesh(
            new THREE.BoxGeometry(8, 0.2, 5),
            new THREE.MeshStandardMaterial({color:0x374151})
        );
        floor.position.y = -0.12;
        floor.receiveShadow = true;
        scene.add(floor);

        function box(w,h,d,color,x,y,z,metalness=0.25) {
            const mesh = new THREE.Mesh(
                new THREE.BoxGeometry(w,h,d),
                new THREE.MeshStandardMaterial({
                    color,
                    metalness,
                    roughness:0.45
                })
            );
            mesh.position.set(x,y,z);
            mesh.castShadow = true;
            mesh.receiveShadow = true;
            scene.add(mesh);
            return mesh;
        }

        // Machine base and cabinet
        box(2.4,0.22,1.35,0x64748b,0,0.12,0);
        box(2.05,2.8,1.05,0x263449,0,1.62,0);
        box(2.12,0.12,1.12,0x94a3b8,0,3.05,0);

        // Front display frame
        box(1.85,1.65,0.12,0x020617,0,2.05,0.56);

        // Screen canvas
        const screenCanvas = document.createElement("canvas");
        screenCanvas.width = 900;
        screenCanvas.height = 760;
        const ctx = screenCanvas.getContext("2d");

        function drawScreen() {
            ctx.fillStyle = "#f8fafc";
            ctx.fillRect(0,0,900,760);

            ctx.fillStyle = "#ea580c";
            ctx.fillRect(0,0,900,105);

            ctx.fillStyle = "#ffffff";
            ctx.font = "bold 42px Arial";
            ctx.fillText("LIVE SPACE SIGNAL",35,67);
            ctx.fillStyle = "#0f172a";
            ctx.font = "bold 34px Arial";
            ctx.fillText("VISITOR DASHBOARD",35,155);

            ctx.font = "24px Arial";
            ctx.fillStyle = "#475569";
            ctx.fillText("Live food court availability",35,195);

            let y = 245;

            data.slice(0,5).forEach(item => {
                const color =
                    item.status === "Available" ? "#16a34a" :
                    item.status === "Limited" ? "#d97706" : "#dc2626";

                ctx.fillStyle = "#e2e8f0";
                ctx.fillRect(30,y,840,85);

                ctx.fillStyle = "#0f172a";
                ctx.font = "bold 27px Arial";
                ctx.fillText(item.name,50,y+35);

                ctx.fillStyle = color;
                ctx.font = "bold 25px Arial";
                ctx.fillText(
                    item.percentage + "% occupied",50,y+68
                );

                ctx.textAlign = "right";
                ctx.fillText(item.status,840,y+48);
                ctx.textAlign = "left";

                y += 98;
            });

            ctx.fillStyle = "#334155";
            ctx.font = "20px Arial";
            ctx.fillText("Information based on latest staff update",35,735);
        }

        drawScreen();

        const texture = new THREE.CanvasTexture(screenCanvas);
        texture.colorSpace = THREE.SRGBColorSpace;

        const screen = new THREE.Mesh(
            new THREE.PlaneGeometry(1.68,1.42),
            new THREE.MeshBasicMaterial({map:texture})
        );
        screen.position.set(0,2.05,0.63);
        scene.add(screen);

        // Small camera / sensor above the display
        box(0.18,0.12,0.08,0x020617,0,2.98,0.59);

        // Side ventilation slots
        for (let i=0;i<4;i++) {
            box(0.28,0.045,0.025,0x0f172a,1.04,1.3+i*0.16,0.2);
        }

        // Floor shadow platform
        const platform = new THREE.Mesh(
            new THREE.BoxGeometry(3.2,0.08,2),
            new THREE.MeshStandardMaterial({color:0x475569})
        );
        platform.position.set(0,-0.02,0);
        scene.add(platform);

        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene,camera);
        }
        animate();

        window.addEventListener("resize",()=>{
            camera.aspect = root.clientWidth/root.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(root.clientWidth,root.clientHeight);
        });
        </script>
        """

        html = html.replace(
            "__SIGNAGE_DATA__",
            signage_json
        )

        components.html(
            html,
            height=740,
            scrolling=False
        )

        st.caption(
            "The visitor screen reflects the latest occupancy data entered "
            "in the admin panel. The 3D kiosk is a conceptual prototype."
        )
# ---------- 3D STAFF WORKSTATION ----------

elif page == "3D Staff Workstation":

    import json
    import streamlit.components.v1 as components

    st.title("🖥️ Live Space Signal — Staff Workstation")

    st.caption(
        "Staff occupancy-entry station · Manual updates · SQLite database"
    )

    st.subheader("3D Staff Counter")

    html = """
    <div id="staff-scene"
         style="width:100%;height:580px;background:#111827;
                border-radius:16px;overflow:hidden;">
    </div>

    <script type="module">
    import * as THREE from
    "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";

    import { OrbitControls } from
    "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js";

    const root = document.getElementById("staff-scene");

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111827);

    const camera = new THREE.PerspectiveCamera(
        45, root.clientWidth/root.clientHeight, 0.1, 100
    );

    camera.position.set(5,4,7);

    const renderer = new THREE.WebGLRenderer({antialias:true});
    renderer.setSize(root.clientWidth,root.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
    renderer.shadowMap.enabled = true;
    root.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera,renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0,1.2,0);

    scene.add(new THREE.HemisphereLight(0xffffff,0x334155,2));

    const light = new THREE.DirectionalLight(0xffffff,3);
    light.position.set(4,8,5);
    light.castShadow = true;
    scene.add(light);

    function box(w,h,d,color,x,y,z) {
        const mesh = new THREE.Mesh(
            new THREE.BoxGeometry(w,h,d),
            new THREE.MeshStandardMaterial({
                color:color,
                roughness:0.55
            })
        );
        mesh.position.set(x,y,z);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        scene.add(mesh);
        return mesh;
    }

    // Floor and workstation platform
    box(12,0.2,8,0x374151,0,-0.2,0);
    box(5,0.15,2.5,0x64748b,0,0,0);

    // Staff counter
    box(4,1.1,1.2,0x475569,0,0.65,-1.3);
    box(4.1,0.12,1.3,0x94a3b8,0,1.25,-1.3);

    // Monitor
    box(1.9,1.25,0.12,0x020617,0,2.05,-1.5);
    box(0.12,0.6,0.12,0x94a3b8,0,1.45,-1.5);
    box(0.8,0.08,0.5,0x64748b,0,1.35,-1.5);

    // Keyboard
    box(1.3,0.07,0.45,0x111827,0,1.38,-0.7);

    // Mouse
    box(0.2,0.08,0.3,0x0f172a,0.95,1.38,-0.7);

    // Staff chair
    box(0.9,0.15,0.9,0x334155,0,0.8,1.1);
    box(0.9,1.1,0.15,0x334155,0,1.35,1.5);
    box(0.12,0.8,0.12,0x94a3b8,0,0.3,1.1);

    // Side equipment cabinet
    box(0.9,1.6,0.8,0x263449,-2.5,0.8,-1.3);

    // Monitor screen
    const canvas = document.createElement("canvas");
    canvas.width = 800;
    canvas.height = 500;

    const ctx = canvas.getContext("2d");

    ctx.fillStyle = "#f8fafc";
    ctx.fillRect(0,0,800,500);

    ctx.fillStyle = "#166534";
    ctx.fillRect(0,0,800,90);

    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 38px Arial";
    ctx.fillText("LIVE SPACE SIGNAL",30,58);

    ctx.fillStyle = "#0f172a";
    ctx.font = "bold 32px Arial";
    ctx.fillText("STAFF CONTROL PANEL",30,155);

    ctx.font = "24px Arial";
    ctx.fillStyle = "#475569";
    ctx.fillText("Manual occupancy reporting station",30,205);

    ctx.fillStyle = "#16a34a";
    ctx.fillRect(30,260,740,90);

    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 28px Arial";
    ctx.fillText("UPDATE STALL / SEATING AVAILABILITY",55,315);

    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;

    const screen = new THREE.Mesh(
        new THREE.PlaneGeometry(1.75,1.05),
        new THREE.MeshBasicMaterial({map:texture})
    );

    screen.position.set(0,2.05,-1.425);
    scene.add(screen);

    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene,camera);
    }

    animate();

    window.addEventListener("resize",()=>{
        camera.aspect = root.clientWidth/root.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(root.clientWidth,root.clientHeight);
    });
    </script>
    """

    components.html(
        html,
        height=600,
        scrolling=False
    )

    st.divider()

    st.subheader("📝 Staff Occupancy Entry")

    st.write(
        "Select a location, enter the current occupied count, "
        "and submit the update."
    )

    if not df.empty:

        with st.form("staff_occupancy_form"):

            location_name = st.selectbox(
                "Select stall or seating area",
                df["name"].tolist()
            )

            selected = df[
                df.name == location_name
            ].iloc[0]

            st.info(
                f"Current occupancy: {int(selected.occupied)} / "
                f"{int(selected.capacity)}"
            )

            occupied_count = st.number_input(
                "Current occupied spaces",
                min_value=0,
                max_value=int(selected.capacity),
                value=int(selected.occupied),
                step=1
            )

            staff_name = st.text_input(
                "Staff member name"
            )

            submitted = st.form_submit_button(
                "Submit occupancy update",
                type="primary",
                use_container_width=True
            )

            if submitted:

                if not staff_name.strip():

                    st.error(
                        "Enter the staff member name."
                    )

                else:

                    new_status = update(
                        int(selected.id),
                        int(selected.capacity),
                        int(occupied_count)
                    )

                    st.success(
                        f"Update saved. {location_name} is now "
                        f"{new_status}."
                    )

                    st.rerun()

        st.divider()

        st.subheader("Current location status")

        st.dataframe(
            df[
                [
                    "name",
                    "category",
                    "occupied",
                    "capacity",
                    "status",
                    "updated_at"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning("No locations are available.")
# ---------- FOOTER ----------

st.sidebar.divider()

st.sidebar.caption(
    "Live Space Signal · Student prototype"
)
