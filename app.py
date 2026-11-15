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


# DATABASE FUNCTIONS

def conn():
    c = sqlite3.connect(DB, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def status_for(o, c):
    r = o / c if c else 0
    return "Full" if r >= .8 else (
        "Limited" if r >= .5 else "Available"
    )


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

            for n, k, cap, occ in seed:
                s = status_for(occ, cap)

                cur = c.execute(
                    """INSERT INTO locations
                    (name,category,capacity,occupied,status,updated_at)
                    VALUES(?,?,?,?,?,?)""",
                    (n, k, cap, occ, s, now)
                )

                c.execute(
                    """INSERT INTO history
                    (location_id,status,occupied,capacity,recorded_at)
                    VALUES(?,?,?,?,?)""",
                    (cur.lastrowid, s, occ, cap, now)
                )


def get_locations():
    with conn() as c:
        return pd.DataFrame([
            dict(r) for r in c.execute(
                "SELECT * FROM locations ORDER BY category,name"
            )
        ])


def get_history():
    with conn() as c:
        return pd.DataFrame([
            dict(r) for r in c.execute("""
                SELECT h.*,l.name
                FROM history h
                JOIN locations l ON l.id=h.location_id
                ORDER BY h.id DESC
            """)
        ])


def update(locid, cap, occ):
    s = status_for(occ, cap)
    now = datetime.now().astimezone().isoformat(
        timespec="minutes"
    )

    with conn() as c:
        c.execute(
            """UPDATE locations
            SET capacity=?,occupied=?,status=?,updated_at=?
            WHERE id=?""",
            (cap, occ, s, now, locid)
        )

        c.execute(
            """INSERT INTO history
            (location_id,status,occupied,capacity,recorded_at)
            VALUES(?,?,?,?,?)""",
            (locid, s, occ, cap, now)
        )

    return s


def add_location(name, category, cap):
    now = datetime.now().astimezone().isoformat(
        timespec="minutes"
    )

    with conn() as c:
        cur = c.execute(
            """INSERT INTO locations
            (name,category,capacity,occupied,status,updated_at)
            VALUES(?,?,?,?,?,?)""",
            (name, category, cap, 0, "Available", now)
        )

        c.execute(
            """INSERT INTO history
            (location_id,status,occupied,capacity,recorded_at)
            VALUES(?,?,?,?,?)""",
            (cur.lastrowid, "Available", 0, cap, now)
        )


init()


# AI ASSISTANT

def local_answer(q, df):
    q = q.lower()

    if df.empty:
        return "There are no locations in the system yet."

    if any(w in q for w in ["seat", "seating", "sit"]):
        x = df[
            (df.category == "Seating Area") &
            (df.status != "Full")
        ].copy()
    else:
        x = df[df.status != "Full"].copy()

    if any(w in q for w in [
        "least", "available", "recommend",
        "where", "alternative", "crowded",
        "seat", "suggest"
    ]):

        if x.empty:
            return (
                "All suitable participating locations are "
                "currently marked Full. Please check again shortly."
            )

        x["pct"] = x.occupied / x.capacity * 100
        x = x.sort_values("pct").head(3)

        return (
            "Based on the latest reported availability:\n\n" +
            "\n\n".join(
                f"{i}. **{r['name']}** — "
                f"{r['pct']:.0f}% occupied ({r['status']})"
                for i, (_, r) in enumerate(x.iterrows(), 1)
            ) +
            "\n\nThese are occupancy levels, not guaranteed waiting times."
        )

    full = df[df.status == "Full"].name.tolist()

    if any(w in q for w in ["full", "busy", "crowd"]):
        return (
            "Currently marked Full: " +
            (", ".join(full) if full else "none") + "."
        )

    return (
        "Ask me for the least-crowded stall, available seating, "
        "an alternative to a full location, or current crowd levels."
    )


def assistant(q, df):

    key = st.secrets.get(
        "OPENAI_API_KEY",
        os.environ.get("OPENAI_API_KEY", "")
    )

    if key and OpenAI:

        try:
            client = OpenAI(api_key=key)

            data = df[
                [
                    "name", "category", "capacity",
                    "occupied", "status", "updated_at"
                ]
            ].to_dict("records")

            res = client.responses.create(
                model=st.secrets.get(
                    "OPENAI_MODEL",
                    os.environ.get("OPENAI_MODEL", "gpt-5.5")
                ),

                instructions=(
                    "You are Live Space Signal's visitor assistant. "
                    "Use only supplied location data. Recommend "
                    "low-occupancy suitable options. Never invent "
                    "locations, wait times or sensor readings. "
                    "Mention updates are manual and may be stale. "
                    "Be concise."
                ),

                input=f"Current location data: {data}\nVisitor question: {q}",
                max_output_tokens=220
            )

            return res.output_text, "OpenAI API"

        except Exception as exc:
            st.session_state["ai_error"] = str(exc)[:240]

    return local_answer(q, df), "Local fallback"


# RANDOM FOREST FORECASTING

def synthetic_training():
    rng = random.Random(42)
    rows = []

    for lid in range(1, 21):
        for day in range(7):
            for hour in range(10, 23):

                eve = math.exp(-((hour - 19) / 2.2) ** 2)

                lunch = .42 * math.exp(
                    -((hour - 13) / 1.8) ** 2
                )

                v = (
                    (.10 + .58 * eve + lunch)
                    * (.82 + lid * .012)
                    * (1 if day < 5 else 1.10)
                    + rng.uniform(-.07, .07)
                )

                rows.append(
                    (lid, hour, day, max(.02, min(.98, v)))
                )

    return pd.DataFrame(rows)


@st.cache_resource
def model():

    if RandomForestRegressor is None:
        return None

    d = synthetic_training()

    m = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        min_samples_leaf=2
    )

    m.fit(d[[0, 1, 2]], d[3])

    return m


def forecasts(df):

    m = model()
    hour = (datetime.now().hour + 1) % 24
    rows = []

    for _, r in df.iterrows():

        v = (
            float(
                m.predict(
                    pd.DataFrame([[
                        int(r.id),
                        hour,
                        datetime.now().weekday()
                    ]])
                )[0]
            )
            if m else r.occupied / r.capacity
        )

        v = max(0, min(1, v))

        rows.append({
            "Location": r["name"],
            "Category": r.category,
            "Predicted occupancy": round(v * 100),
            "Crowd level": (
                "High" if v >= .8
                else "Moderate" if v >= .5
                else "Low"
            ),
            "Forecast hour": f"{hour:02d}:00"
        })

    return pd.DataFrame(rows)


# USER INTERFACE DESIGN

st.markdown("""
<style>

.block-container {
    max-width: 1250px;
    padding-top: 1.5rem;
}

.hero {
    background: linear-gradient(120deg,#153d32,#2f7a5f);
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


# SIDEBAR NAVIGATION

st.sidebar.title("🟢 Live Space Signal")

page = st.sidebar.radio(
    "Navigate",
    [
        "Visitor Dashboard",
        "Admin Panel",
        "AI & Crowd Forecast",
        "Analytics"
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


# VISITOR DASHBOARD

if page == "Visitor Dashboard":

    st.markdown(
        '<div class="hero">'
        '<p class="eyebrow">MIGSUN ULTIMO · VISITOR INFORMATION</p>'
        '<h1>Find your space. Skip the guesswork.</h1>'
        '<p class="sub">Check participating food stalls and seating areas '
        'before deciding where to go.</p></div>',
        unsafe_allow_html=True
    )

    counts = df.status.value_counts()

    a, b, c, d = st.columns(4)

    a.metric("📍 Locations", len(df))
    b.metric("🟢 Available", int(counts.get("Available", 0)))
    c.metric("🟡 Limited", int(counts.get("Limited", 0)))
    d.metric("🔴 Full", int(counts.get("Full", 0)))

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
                term, case=False, na=False
            )
        ]

    if filt != "All":
        view = view[view.status == filt]

    if cat_f != "All":
        view = view[view.category == cat_f]

    if view.empty:
        st.info("No locations match your filters.")

    cards = st.columns(3)

    for i, (_, r) in enumerate(view.iterrows()):

        pct = round(r.occupied / r.capacity * 100)

        with cards[i % 3]:

            with st.container(border=True):

                st.markdown(
                    f'<div class="cat">{r.category.upper()}</div>',
                    unsafe_allow_html=True
                )

                st.markdown("### " + r["name"])

                st.markdown(
                    f'<span class="pill" '
                    f'style="background:{STATUS_COLOR[r.status]}">'
                    f'{STATUS_ICON[r.status]} {r.status.upper()}</span>',
                    unsafe_allow_html=True
                )

                st.progress(
                    pct / 100,
                    text=f"{pct}% occupied"
                )

                st.markdown(
                    f'<div class="spaces"><b>'
                    f'{r.capacity - r.occupied}</b> spaces available · '
                    f'{r.occupied}/{r.capacity} occupied</div>',
                    unsafe_allow_html=True
                )

                st.caption(
                    "Updated " +
                    pd.to_datetime(r.updated_at).strftime(
                        "%d %b, %I:%M %p"
                    )
                )

    st.divider()

    st.subheader("🤖 AI Space Assistant")

    st.write(
        "Ask for the least-crowded stall, available seating, "
        "or an alternative."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! Ask me where to find available "
                    "food stalls or seating."
                )
            }
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    q = st.chat_input(
        "e.g. Where can I find the least crowded seating?"
    )

    if q:

        st.session_state.messages.append(
            {"role": "user", "content": q}
        )

        with st.chat_message("user"):
            st.markdown(q)

        answer, mode = assistant(q, df)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        with st.chat_message("assistant"):
            st.markdown(answer)
            st.caption("Mode: " + mode)

        if st.session_state.get("ai_error"):

            st.caption(
                "OpenAI connection issue; local fallback used: " +
                st.session_state["ai_error"]
            )

            st.session_state.pop("ai_error", None)

    st.info(
        "Available: below 50% occupied · "
        "Limited: 50–79% · Full: 80% or above."
    )


# ADMIN PANEL

elif page == "Admin Panel":

    st.title("⚙️ Management Admin Panel")

    password = st.text_input(
        "Admin password",
        type="password"
    )

    expected = st.secrets.get(
        "ADMIN_PASSWORD",
        os.environ.get("ADMIN_PASSWORD", "signal123")
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
        "Update capacity and occupied seats. "
        "Status is calculated automatically."
    )

    for _, r in df.iterrows():

        with st.container(border=True):

            a, b, c, d = st.columns([2, 1, 1, 1])

            a.markdown(
                f"**{r['name']}**  \n"
                f"{r.category} · {STATUS_ICON[r.status]} {r.status}"
            )

            cap = b.number_input(
                "Capacity",
                1,
                10000,
                int(r.capacity),
                key=f"cap{r.id}"
            )

            occ = c.number_input(
                "Occupied",
                0,
                10000,
                int(r.occupied),
                key=f"occ{r.id}"
            )

            if d.button(
                "Save",
                key=f"save{r.id}",
                use_container_width=True
            ):

                if occ > cap:
                    st.error(
                        "Occupied count cannot exceed capacity."
                    )

                else:
                    st.success(
                        f"Updated to {update(int(r.id), int(cap), int(occ))}"
                    )
                    st.rerun()

    # CROWD SIMULATION

    st.divider()

    st.subheader(
        "🎛️ Crowd simulation — demonstration only"
    )

    st.caption(
        "Use this control during your faculty demo to show how "
        "occupancy status and analytics respond. This changes "
        "the stored demo data and creates a history record."
    )

    sim_location = (
        st.selectbox(
            "Choose a location to simulate",
            df["name"].tolist(),
            key="sim_location"
        )
        if not df.empty else None
    )

    if sim_location:

        sim_row = df[df.name == sim_location].iloc[0]

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

            reset_occ = min(
                int(sim_row.capacity),
                baseline.get(sim_location, 0)
            )

            new_status = update(
                int(sim_row.id),
                int(sim_row.capacity),
                reset_occ
            )

            st.success(
                f"{sim_location} reset to "
                f"{reset_occ}/{int(sim_row.capacity)} "
                f"({new_status})."
            )

            st.rerun()

    # ADD LOCATION

    st.divider()

    st.subheader("Add a location")

    with st.form("add"):

        name = st.text_input("Location name")

        cat = st.selectbox(
            "Category",
            ["Food Stall", "Seating Area"]
        )

        cap = st.number_input(
            "Capacity",
            1,
            10000,
            20
        )

        go = st.form_submit_button("Add location")

        if go:

            try:

                if not name.strip():
                    st.error("Enter a location name.")

                else:
                    add_location(
                        name.strip(),
                        cat,
                        int(cap)
                    )

                    st.success("Location added.")
                    st.rerun()

            except sqlite3.IntegrityError:
                st.error("That location already exists.")


# AI AND CROWD FORECAST

elif page == "AI & Crowd Forecast":

    st.title("🧠 AI & Machine Learning")

    st.write(
        "Recommendations use current status data. "
        "Forecasts demonstrate a Random Forest workflow."
    )

    fc = forecasts(df)

    st.subheader("Predicted occupancy · next hour")

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
        "Forecast model is trained on synthetic evening-peak "
        "patterns, not actual Migsun Ultimo observations. "
        "Treat it as a demonstration only."
    )

    if not df.empty:

        busy = (
            df.assign(
                pct=df.occupied / df.capacity * 100
            )
            .sort_values("pct", ascending=False)
            .head(2)
        )

        st.subheader("Management insights")

        st.write(
            f"- {int((df.status == 'Available').sum())} "
            f"of {len(df)} locations are currently marked Available."
        )

        st.write(
            "- Highest occupancy: " +
            ", ".join(
                f"{r['name']} ({r.pct:.0f}%)"
                for _, r in busy.iterrows()
            ) + "."
        )

        st.write(
            "- Management action: verify high-occupancy "
            "locations during peak hours and encourage "
            "prompt staff updates."
        )


# ANALYTICS DASHBOARD

elif page == "Analytics":

    st.title("📊 Business & Crowd Analytics")

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

        st.subheader("Current status distribution")

        st.bar_chart(
            df.status.value_counts().reindex(
                ["Available", "Limited", "Full"],
                fill_value=0
            ),
            color="#3ea67c"
        )

        st.subheader("Occupancy by location")

        st.bar_chart(
            dfx.set_index("name")
            .occupancy_pct
            .sort_values(ascending=False),
            color="#3ea67c"
        )

    st.subheader("Status update history")

    if hist.empty:

        st.info("No updates recorded.")

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


# FOOTER

st.sidebar.divider()

st.sidebar.caption(
    "Live Space Signal · Student prototype"
    )import os
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


# DATABASE FUNCTIONS

def conn():
    c = sqlite3.connect(DB, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def status_for(o, c):
    r = o / c if c else 0
    return "Full" if r >= .8 else (
        "Limited" if r >= .5 else "Available"
    )


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

            for n, k, cap, occ in seed:
                s = status_for(occ, cap)

                cur = c.execute(
                    """INSERT INTO locations
                    (name,category,capacity,occupied,status,updated_at)
                    VALUES(?,?,?,?,?,?)""",
                    (n, k, cap, occ, s, now)
                )

                c.execute(
                    """INSERT INTO history
                    (location_id,status,occupied,capacity,recorded_at)
                    VALUES(?,?,?,?,?)""",
                    (cur.lastrowid, s, occ, cap, now)
                )


def get_locations():
    with conn() as c:
        return pd.DataFrame([
            dict(r) for r in c.execute(
                "SELECT * FROM locations ORDER BY category,name"
            )
        ])


def get_history():
    with conn() as c:
        return pd.DataFrame([
            dict(r) for r in c.execute("""
                SELECT h.*,l.name
                FROM history h
                JOIN locations l ON l.id=h.location_id
                ORDER BY h.id DESC
            """)
        ])


def update(locid, cap, occ):
    s = status_for(occ, cap)
    now = datetime.now().astimezone().isoformat(
        timespec="minutes"
    )

    with conn() as c:
        c.execute(
            """UPDATE locations
            SET capacity=?,occupied=?,status=?,updated_at=?
            WHERE id=?""",
            (cap, occ, s, now, locid)
        )

        c.execute(
            """INSERT INTO history
            (location_id,status,occupied,capacity,recorded_at)
            VALUES(?,?,?,?,?)""",
            (locid, s, occ, cap, now)
        )

    return s


def add_location(name, category, cap):
    now = datetime.now().astimezone().isoformat(
        timespec="minutes"
    )

    with conn() as c:
        cur = c.execute(
            """INSERT INTO locations
            (name,category,capacity,occupied,status,updated_at)
            VALUES(?,?,?,?,?,?)""",
            (name, category, cap, 0, "Available", now)
        )

        c.execute(
            """INSERT INTO history
            (location_id,status,occupied,capacity,recorded_at)
            VALUES(?,?,?,?,?)""",
            (cur.lastrowid, "Available", 0, cap, now)
        )


init()


# AI ASSISTANT

def local_answer(q, df):
    q = q.lower()

    if df.empty:
        return "There are no locations in the system yet."

    if any(w in q for w in ["seat", "seating", "sit"]):
        x = df[
            (df.category == "Seating Area") &
            (df.status != "Full")
        ].copy()
    else:
        x = df[df.status != "Full"].copy()

    if any(w in q for w in [
        "least", "available", "recommend",
        "where", "alternative", "crowded",
        "seat", "suggest"
    ]):

        if x.empty:
            return (
                "All suitable participating locations are "
                "currently marked Full. Please check again shortly."
            )

        x["pct"] = x.occupied / x.capacity * 100
        x = x.sort_values("pct").head(3)

        return (
            "Based on the latest reported availability:\n\n" +
            "\n\n".join(
                f"{i}. **{r['name']}** — "
                f"{r['pct']:.0f}% occupied ({r['status']})"
                for i, (_, r) in enumerate(x.iterrows(), 1)
            ) +
            "\n\nThese are occupancy levels, not guaranteed waiting times."
        )

    full = df[df.status == "Full"].name.tolist()

    if any(w in q for w in ["full", "busy", "crowd"]):
        return (
            "Currently marked Full: " +
            (", ".join(full) if full else "none") + "."
        )

    return (
        "Ask me for the least-crowded stall, available seating, "
        "an alternative to a full location, or current crowd levels."
    )


def assistant(q, df):

    key = st.secrets.get(
        "OPENAI_API_KEY",
        os.environ.get("OPENAI_API_KEY", "")
    )

    if key and OpenAI:

        try:
            client = OpenAI(api_key=key)

            data = df[
                [
                    "name", "category", "capacity",
                    "occupied", "status", "updated_at"
                ]
            ].to_dict("records")

            res = client.responses.create(
                model=st.secrets.get(
                    "OPENAI_MODEL",
                    os.environ.get("OPENAI_MODEL", "gpt-5.5")
                ),

                instructions=(
                    "You are Live Space Signal's visitor assistant. "
                    "Use only supplied location data. Recommend "
                    "low-occupancy suitable options. Never invent "
                    "locations, wait times or sensor readings. "
                    "Mention updates are manual and may be stale. "
                    "Be concise."
                ),

                input=f"Current location data: {data}\nVisitor question: {q}",
                max_output_tokens=220
            )

            return res.output_text, "OpenAI API"

        except Exception as exc:
            st.session_state["ai_error"] = str(exc)[:240]

    return local_answer(q, df), "Local fallback"


# RANDOM FOREST FORECASTING

def synthetic_training():
    rng = random.Random(42)
    rows = []

    for lid in range(1, 21):
        for day in range(7):
            for hour in range(10, 23):

                eve = math.exp(-((hour - 19) / 2.2) ** 2)

                lunch = .42 * math.exp(
                    -((hour - 13) / 1.8) ** 2
                )

                v = (
                    (.10 + .58 * eve + lunch)
                    * (.82 + lid * .012)
                    * (1 if day < 5 else 1.10)
                    + rng.uniform(-.07, .07)
                )

                rows.append(
                    (lid, hour, day, max(.02, min(.98, v)))
                )

    return pd.DataFrame(rows)


@st.cache_resource
def model():

    if RandomForestRegressor is None:
        return None

    d = synthetic_training()

    m = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        min_samples_leaf=2
    )

    m.fit(d[[0, 1, 2]], d[3])

    return m


def forecasts(df):

    m = model()
    hour = (datetime.now().hour + 1) % 24
    rows = []

    for _, r in df.iterrows():

        v = (
            float(
                m.predict(
                    pd.DataFrame([[
                        int(r.id),
                        hour,
                        datetime.now().weekday()
                    ]])
                )[0]
            )
            if m else r.occupied / r.capacity
        )

        v = max(0, min(1, v))

        rows.append({
            "Location": r["name"],
            "Category": r.category,
            "Predicted occupancy": round(v * 100),
            "Crowd level": (
                "High" if v >= .8
                else "Moderate" if v >= .5
                else "Low"
            ),
            "Forecast hour": f"{hour:02d}:00"
        })

    return pd.DataFrame(rows)


# USER INTERFACE DESIGN

st.markdown("""
<style>

.block-container {
    max-width: 1250px;
    padding-top: 1.5rem;
}

.hero {
    background: linear-gradient(120deg,#153d32,#2f7a5f);
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


# SIDEBAR NAVIGATION

st.sidebar.title("🟢 Live Space Signal")

page = st.sidebar.radio(
    "Navigate",
    [
        "Visitor Dashboard",
        "Admin Panel",
        "AI & Crowd Forecast",
        "Analytics"
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


# VISITOR DASHBOARD

if page == "Visitor Dashboard":

    st.markdown(
        '<div class="hero">'
        '<p class="eyebrow">MIGSUN ULTIMO · VISITOR INFORMATION</p>'
        '<h1>Find your space. Skip the guesswork.</h1>'
        '<p class="sub">Check participating food stalls and seating areas '
        'before deciding where to go.</p></div>',
        unsafe_allow_html=True
    )

    counts = df.status.value_counts()

    a, b, c, d = st.columns(4)

    a.metric("📍 Locations", len(df))
    b.metric("🟢 Available", int(counts.get("Available", 0)))
    c.metric("🟡 Limited", int(counts.get("Limited", 0)))
    d.metric("🔴 Full", int(counts.get("Full", 0)))

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
                term, case=False, na=False
            )
        ]

    if filt != "All":
        view = view[view.status == filt]

    if cat_f != "All":
        view = view[view.category == cat_f]

    if view.empty:
        st.info("No locations match your filters.")

    cards = st.columns(3)

    for i, (_, r) in enumerate(view.iterrows()):

        pct = round(r.occupied / r.capacity * 100)

        with cards[i % 3]:

            with st.container(border=True):

                st.markdown(
                    f'<div class="cat">{r.category.upper()}</div>',
                    unsafe_allow_html=True
                )

                st.markdown("### " + r["name"])

                st.markdown(
                    f'<span class="pill" '
                    f'style="background:{STATUS_COLOR[r.status]}">'
                    f'{STATUS_ICON[r.status]} {r.status.upper()}</span>',
                    unsafe_allow_html=True
                )

                st.progress(
                    pct / 100,
                    text=f"{pct}% occupied"
                )

                st.markdown(
                    f'<div class="spaces"><b>'
                    f'{r.capacity - r.occupied}</b> spaces available · '
                    f'{r.occupied}/{r.capacity} occupied</div>',
                    unsafe_allow_html=True
                )

                st.caption(
                    "Updated " +
                    pd.to_datetime(r.updated_at).strftime(
                        "%d %b, %I:%M %p"
                    )
                )

    st.divider()

    st.subheader("🤖 AI Space Assistant")

    st.write(
        "Ask for the least-crowded stall, available seating, "
        "or an alternative."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! Ask me where to find available "
                    "food stalls or seating."
                )
            }
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    q = st.chat_input(
        "e.g. Where can I find the least crowded seating?"
    )

    if q:

        st.session_state.messages.append(
            {"role": "user", "content": q}
        )

        with st.chat_message("user"):
            st.markdown(q)

        answer, mode = assistant(q, df)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        with st.chat_message("assistant"):
            st.markdown(answer)
            st.caption("Mode: " + mode)

        if st.session_state.get("ai_error"):

            st.caption(
                "OpenAI connection issue; local fallback used: " +
                st.session_state["ai_error"]
            )

            st.session_state.pop("ai_error", None)

    st.info(
        "Available: below 50% occupied · "
        "Limited: 50–79% · Full: 80% or above."
    )


# ADMIN PANEL

elif page == "Admin Panel":

    st.title("⚙️ Management Admin Panel")

    password = st.text_input(
        "Admin password",
        type="password"
    )

    expected = st.secrets.get(
        "ADMIN_PASSWORD",
        os.environ.get("ADMIN_PASSWORD", "signal123")
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
        "Update capacity and occupied seats. "
        "Status is calculated automatically."
    )

    for _, r in df.iterrows():

        with st.container(border=True):

            a, b, c, d = st.columns([2, 1, 1, 1])

            a.markdown(
                f"**{r['name']}**  \n"
                f"{r.category} · {STATUS_ICON[r.status]} {r.status}"
            )

            cap = b.number_input(
                "Capacity",
                1,
                10000,
                int(r.capacity),
                key=f"cap{r.id}"
            )

            occ = c.number_input(
                "Occupied",
                0,
                10000,
                int(r.occupied),
                key=f"occ{r.id}"
            )

            if d.button(
                "Save",
                key=f"save{r.id}",
                use_container_width=True
            ):

                if occ > cap:
                    st.error(
                        "Occupied count cannot exceed capacity."
                    )

                else:
                    st.success(
                        f"Updated to {update(int(r.id), int(cap), int(occ))}"
                    )
                    st.rerun()

    # CROWD SIMULATION

    st.divider()

    st.subheader(
        "🎛️ Crowd simulation — demonstration only"
    )

    st.caption(
        "Use this control during your faculty demo to show how "
        "occupancy status and analytics respond. This changes "
        "the stored demo data and creates a history record."
    )

    sim_location = (
        st.selectbox(
            "Choose a location to simulate",
            df["name"].tolist(),
            key="sim_location"
        )
        if not df.empty else None
    )

    if sim_location:

        sim_row = df[df.name == sim_location].iloc[0]

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

            reset_occ = min(
                int(sim_row.capacity),
                baseline.get(sim_location, 0)
            )

            new_status = update(
                int(sim_row.id),
                int(sim_row.capacity),
                reset_occ
            )

            st.success(
                f"{sim_location} reset to "
                f"{reset_occ}/{int(sim_row.capacity)} "
                f"({new_status})."
            )

            st.rerun()

    # ADD LOCATION

    st.divider()

    st.subheader("Add a location")

    with st.form("add"):

        name = st.text_input("Location name")

        cat = st.selectbox(
            "Category",
            ["Food Stall", "Seating Area"]
        )

        cap = st.number_input(
            "Capacity",
            1,
            10000,
            20
        )

        go = st.form_submit_button("Add location")

        if go:

            try:

                if not name.strip():
                    st.error("Enter a location name.")

                else:
                    add_location(
                        name.strip(),
                        cat,
                        int(cap)
                    )

                    st.success("Location added.")
                    st.rerun()

            except sqlite3.IntegrityError:
                st.error("That location already exists.")


# AI AND CROWD FORECAST

elif page == "AI & Crowd Forecast":

    st.title("🧠 AI & Machine Learning")

    st.write(
        "Recommendations use current status data. "
        "Forecasts demonstrate a Random Forest workflow."
    )

    fc = forecasts(df)

    st.subheader("Predicted occupancy · next hour")

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
        "Forecast model is trained on synthetic evening-peak "
        "patterns, not actual Migsun Ultimo observations. "
        "Treat it as a demonstration only."
    )

    if not df.empty:

        busy = (
            df.assign(
                pct=df.occupied / df.capacity * 100
            )
            .sort_values("pct", ascending=False)
            .head(2)
        )

        st.subheader("Management insights")

        st.write(
            f"- {int((df.status == 'Available').sum())} "
            f"of {len(df)} locations are currently marked Available."
        )

        st.write(
            "- Highest occupancy: " +
            ", ".join(
                f"{r['name']} ({r.pct:.0f}%)"
                for _, r in busy.iterrows()
            ) + "."
        )

        st.write(
            "- Management action: verify high-occupancy "
            "locations during peak hours and encourage "
            "prompt staff updates."
        )


# ANALYTICS DASHBOARD

elif page == "Analytics":

    st.title("📊 Business & Crowd Analytics")

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

        st.subheader("Current status distribution")

        st.bar_chart(
            df.status.value_counts().reindex(
                ["Available", "Limited", "Full"],
                fill_value=0
            ),
            color="#3ea67c"
        )

        st.subheader("Occupancy by location")

        st.bar_chart(
            dfx.set_index("name")
            .occupancy_pct
            .sort_values(ascending=False),
            color="#3ea67c"
        )

    st.subheader("Status update history")

    if hist.empty:

        st.info("No updates recorded.")

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


# FOOTER

st.sidebar.divider()

st.sidebar.caption(
    "Live Space Signal · Student prototype"
)
