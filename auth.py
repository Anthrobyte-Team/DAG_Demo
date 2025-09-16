import bcrypt
import streamlit as st
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import certifi
from sqlalchemy import create_engine, text
import re
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
import smtplib

# ---- Load URI from secrets and build engine (cached) ----
try:
    MYSQL_URI = st.secrets["mysql"]["uri"]
except Exception:
    st.error("Database URI missing in secrets. Add [mysql].uri in .streamlit/secrets.toml")
    raise

@st.cache_resource
def get_engine():
    try:
        engine = create_engine(MYSQL_URI, pool_pre_ping=True, pool_size=10, max_overflow=20, connect_args={"ssl": {
        "ca": certifi.where()
    }})
        logging.info("SQLAlchemy engine created (auth.py).")
        return engine
    except Exception as e:
        logging.error(f"Failed to create SQLAlchemy engine in auth.py: {e}")
        return None

engine = get_engine()

def _ensure_engine():
    if engine is None:
        st.error("DB engine not initialized. Check secrets/URI/SSL.")
        st.stop()

# ---------- password helpers ----------
def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def check_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False
    
def send_custom_email(sender_email, sender_password, receiver_email, subject, summary, smtp_server="smtp.gmail.com", smtp_port=587, logger=None):
    """
    Send an email using Gmail SMTP.
    """
    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg["Subject"] = subject
        msg.attach(MIMEText(summary, "plain"))

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())

        if logger:
            logger.info(f"Email sent successfully to {receiver_email}")
        return {"status": "success", "message": "Email sent successfully"}
    except Exception as e:
        if logger:
            logger.error(f"Failed to send email: {e}")
        return {"status": "error", "message": f"Failed to send email: {e}"}

# ---------- user CRUD ----------
def create_user(*, username: str, role: str, password_plain: str,
                phone_number: str | None, email: str, status: str = "active") -> int:
    _ensure_engine()
    pw_hash = hash_password(password_plain)
    with engine.begin() as conn:
        result = conn.execute(
            text("""
                INSERT INTO user_login (username, role, password_hash, status, phone_number, email, password)
                VALUES (:username, :role, :password_hash, :status, :phone_number, :email, :password_plain)
            """),
            {
                "username": username,
                "role": role,
                "password_hash": pw_hash,
                "status": status,
                "phone_number": phone_number,
                "email": email,
                "password_plain": password_plain,  # WARNING: plaintext!
            }
        )
        try:
            new_id = result.inserted_primary_key[0]
        except Exception:
            new_id = None
        return new_id

def fetch_user_by_login(login: str):
    _ensure_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT id, username, email, phone_number, role, status, password_hash, password
                  FROM user_login
                 WHERE status='active' AND (username=:login OR email=:login)
                 LIMIT 1
            """),
            {"login": login}
        )
        row = result.mappings().first()
        return row

def fetch_user_by_email(email: str):
    _ensure_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT id, username, email, phone_number, role, status, password_hash, password
                  FROM user_login
                 WHERE status='active' AND email=:email
                 LIMIT 1
            """),
            {"email": email}
        )
        return result.mappings().first()

# ---------- Streamlit UI gate ----------
def login_gate():
    """Call this at the top of app.py. Stops page if not authed."""
    if st.session_state.get("auth_ok"):
        return  # already authed

    # --- Use Streamlit columns for split layout ---
    left, right = st.columns([1.3, 1])

    with left:
        st.markdown(
            """
            <div style='font-size:2.2rem; font-weight:800; color:#232323; margin-top:7.0rem; margin-bottom:0.7em; line-height:1.1; font-family:Montserrat,Segoe UI,sans-serif;'>
                We Don’t Just Implement AI.<br>We Guide Transformation.
            </div>
            <div style='font-size:1.15rem; color:#232323cc; margin-bottom:2.5em; font-weight:500; font-family:Montserrat,Segoe UI,sans-serif;'>
                In a world of AI consultants, dashboards, and automation vendors,<br>
                Anthrobyte offers something rarer.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        with st.container():
            st.markdown(
                "<div style='text-align:center; margin-bottom:1.2em;'>"
                "<span style='color:#232323; font-weight:800; font-size:2rem;'>Sign in to your account</span><br>"
                "<span style='color:#232323cc; font-size:1.05rem;'>Empower your AI journey with Anthrobyte</span>"
                "</div>",
                unsafe_allow_html=True
            )

            with st.form("login_form", clear_on_submit=False):
                login_input = st.text_input("Email or Username")
                pw_input    = st.text_input("Password", type="password")
                submitted   = st.form_submit_button("Login")

            if submitted:
                user = fetch_user_by_login((login_input or "").strip())
                if user and check_password(pw_input or "", user["password_hash"]):
                    st.session_state["auth_ok"] = True
                    st.session_state["user"] = {
                        "id": user["id"],
                        "username": user["username"],
                        "email": user["email"],
                        "role": user["role"],
                    }
                    st.success("✅ Successfully logged in!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials or account disabled.")

            with st.expander("Forgot password?", expanded=False):
                with st.form("forgot_pw_form", clear_on_submit=False):
                    reset_email = st.text_input("Enter your account email", key="fp_email")
                    fp_submit   = st.form_submit_button("Email me")

                if fp_submit:
                    email = (reset_email or "").strip()
                    if not EMAIL_RE.match(email):
                        st.error("Please enter a valid email address.")
                    else:
                        user = fetch_user_by_email(email)
                        generic_msg = "If this email is registered, you'll receive an email shortly."
                        if not user:
                            st.info(generic_msg)
                        else:
                            plain = (user.get("password") or "").strip()
                            if not plain:
                                st.error("We couldn't retrieve a password for this account. Please contact support or use reset.")
                            else:
                                try:
                                    try:
                                        SENDER = st.secrets["email"]["sender"]
                                        APP_PW = st.secrets["email"]["app_password"]
                                    except Exception:
                                        st.error("Email sender credentials missing. Add [email].sender and [email].app_password in secrets.")
                                        st.stop()

                                    subject = "Your account password"
                                    body = (
                                        f"Hello {user['username']},\n\n"
                                        f"Per your request, here is your password: {plain}\n\n"
                                        "For your security, consider changing it after logging in.\n\n"
                                        "- Team"
                                    )
                                    mail_status = send_custom_email(SENDER, APP_PW, email, subject, body, logger=logging)
                                    if mail_status.get("status") == "success":
                                        st.success(generic_msg)
                                    else:
                                        logging.error(f"Mail error: {mail_status.get('message')}")
                                        st.error("We couldn't send the email right now. Please try again later.")
                                except Exception:
                                    logging.exception("Forgot-password email failed.")
                                    st.error("Something went wrong. Please try again later.")


    st.stop()  # prevent the rest of the app from rendering

def logout_sidebar():
    with st.sidebar:
        if st.session_state.get("auth_ok"):
            u = st.session_state.get("user", {})
            st.markdown(f"**Logged in as:** {u.get('username','?')} ({u.get('role','?')})")
            if st.button("Logout"):
                for k in ("auth_ok", "user"):
                    st.session_state.pop(k, None)
                st.rerun()
