from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    send_file,
    make_response,
    abort,
    jsonify,
)
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    login_required,
    logout_user,
    current_user,
)
from flask import g
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from wtforms import BooleanField, RadioField, StringField, PasswordField, SubmitField, FileField
from wtforms.validators import AnyOf, DataRequired, Email, Length, InputRequired

import os
import datetime
import json
import mimetypes
import queue
import re
import shutil
import subprocess
import sys
import threading
import traceback
import uuid
from datetime import datetime
from html import escape, unescape
from urllib.parse import quote, urlparse

from stress_highlight import SentenceRecognizer


app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-secret-change-me")

# Protect all state-changing endpoints from CSRF. Forms use ``hidden_tag``
# so the token rides along; AJAX/fetch callers must send the token in the
# ``X-CSRFToken`` header (see templates).
csrf = CSRFProtect(app)

# Ensure ``app.logger`` shows INFO-level records under Flask's default
# dev server (otherwise the auth diagnostics added below would silently
# go to a NullHandler when running ``python app.py``).
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
app.logger.setLevel(logging.INFO)


def _flash_form_errors(form, action_label):
    """Flash every WTForms validation error and mirror them to the log.

    Without this, ``form.validate_on_submit()`` returning ``False`` just
    re-renders the page with no message, so users (and operators) never
    learn that e.g. the password was too short or the email malformed.
    """
    if not form.errors:
        return
    parts = []
    for field_name, messages in form.errors.items():
        field = getattr(form, field_name, None)
        label = getattr(getattr(field, "label", None), "text", field_name)
        for message in messages:
            parts.append(f"{label}: {message}")
    summary = " | ".join(parts)
    app.logger.warning("%s validation failed: %s", action_label, summary)
    flash(f"Please fix the following before {action_label.lower()}: {summary}", "danger")

app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", "500")) * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
UPLOAD_DIR = os.path.join(STORAGE_DIR, "uploads")
TEMP_DIR = os.path.join(STORAGE_DIR, "tmp")
AUDIO_EXTENSIONS = {"wav", "mp3", "ogg", "flac", "m4a", "aac"}
VIDEO_EXTENSIONS = {"mp4", "webm", "mov", "mkv", "avi", "m4v"}
ALLOWED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
BROWSER_VIDEO_TYPE = "video/mp4"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# SQLite shim that mimics the small subset of flask_mysqldb's API used here.
# This avoids rewriting every existing SQL query while still using a local
# zero-config SQLite database file.
# ---------------------------------------------------------------------------
DB_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "app.db")


class _SQLiteCursorShim:
    """Wrap a sqlite3 cursor so existing MySQL-style queries keep working.

    - Translates ``%s`` placeholders to ``?``.
    - Emulates ``SELECT LAST_INSERT_ID()`` by returning the last INSERT's
      rowid captured on this cursor instance.
    """

    def __init__(self, cursor):
        self._c = cursor
        self._last_insert_id = 0
        self._return_last_id = False

    def execute(self, query, params=None):
        stripped = query.strip().upper()
        if stripped.startswith("SELECT LAST_INSERT_ID()"):
            self._return_last_id = True
            return self
        self._return_last_id = False
        q = query.replace("%s", "?")
        if params is None:
            self._c.execute(q)
        else:
            self._c.execute(q, params)
        if stripped.startswith("INSERT"):
            self._last_insert_id = self._c.lastrowid
        return self

    def fetchone(self):
        if self._return_last_id:
            self._return_last_id = False
            return (self._last_insert_id,)
        return self._c.fetchone()

    def fetchall(self):
        return self._c.fetchall()

    def close(self):
        self._c.close()

    @property
    def lastrowid(self):
        return self._c.lastrowid

    @property
    def rowcount(self):
        return self._c.rowcount


class _DBConnectionProxy:
    """Per-request SQLite connection accessor."""

    def _conn(self):
        if "db" not in g:
            conn = sqlite3.connect(DB_PATH)
            conn.execute("PRAGMA foreign_keys = ON")
            g.db = conn
        return g.db

    def cursor(self):
        return _SQLiteCursorShim(self._conn().cursor())

    def commit(self):
        self._conn().commit()

    def rollback(self):
        self._conn().rollback()


class _MysqlShim:
    @property
    def connection(self):
        return _DBConnectionProxy()


mysql = _MysqlShim()


@app.teardown_appcontext
def _close_db(exception):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    """Create tables on first run."""
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                username       TEXT UNIQUE NOT NULL,
                email          TEXT UNIQUE NOT NULL,
                password_hash  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audiofiles (
                audio_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id         INTEGER NOT NULL,
                file_name       TEXT,
                file_type       TEXT,
                file_size       INTEGER,
                upload_datetime TEXT,
                audio_content   BLOB,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );
            CREATE TABLE IF NOT EXISTS vttfiles (
                vtt_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                audio_id    INTEGER NOT NULL,
                file_name   TEXT,
                vtt_content TEXT,
                created_at  TEXT,
                FOREIGN KEY (audio_id) REFERENCES audiofiles(audio_id)
            );
            CREATE TABLE IF NOT EXISTS jsonfiles (
                json_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                audio_id     INTEGER NOT NULL,
                file_name    TEXT,
                json_content TEXT,
                created_at   TEXT,
                FOREIGN KEY (audio_id) REFERENCES audiofiles(audio_id)
            );
            CREATE TABLE IF NOT EXISTS processing_jobs (
                job_id       TEXT PRIMARY KEY,
                user_id      INTEGER NOT NULL,
                audio_id     INTEGER,
                status       TEXT,
                progress     INTEGER,
                message      TEXT,
                error        TEXT,
                created_at   TEXT,
                updated_at   TEXT,
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                FOREIGN KEY (audio_id) REFERENCES audiofiles(audio_id)
            );
            """
        )
        existing_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(audiofiles)").fetchall()
        }
        if "file_path" not in existing_columns:
            conn.execute("ALTER TABLE audiofiles ADD COLUMN file_path TEXT")
        if "media_kind" not in existing_columns:
            conn.execute("ALTER TABLE audiofiles ADD COLUMN media_kind TEXT DEFAULT 'audio'")
        if "stored_file_name" not in existing_columns:
            conn.execute("ALTER TABLE audiofiles ADD COLUMN stored_file_name TEXT")
        conn.commit()
    finally:
        conn.close()


def mark_interrupted_processing_jobs():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            UPDATE processing_jobs
            SET status = ?, progress = ?, message = ?, error = ?, updated_at = ?
            WHERE status IN (?, ?, ?)
            """,
            (
                "failed",
                100,
                "Processing interrupted. Please upload the file again.",
                "The server stopped while this job was processing.",
                datetime.now(),
                "uploaded",
                "queued",
                "processing",
            ),
        )
        conn.commit()
    finally:
        conn.close()


init_db()
mark_interrupted_processing_jobs()

# Set up Flask-Login
login_manager = LoginManager(app)
login_manager.login_view = "login"


# User model for Flask-Login
class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username


@login_manager.user_loader
def load_user(user_id):
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT * FROM users WHERE user_id = %s", (user_id,))
    user_data = cursor.fetchone()
    if user_data:
        return User(user_data[0], user_data[1])
    return None


# Registration form
class RegistrationForm(FlaskForm):
    username = StringField(
        "Username", validators=[DataRequired(), Length(min=4, max=25)]
    )
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired(), Length(min=6)])
    submit = SubmitField("Register")


# Login form
class LoginForm(FlaskForm):
    username = StringField("Username", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Sign In")


SENSITIVITY_CHOICES = [
    ("off", "Off"),
    ("sensitive", "Sensitive"),
    ("ultra", "Ultra-sensitive"),
]
SENSITIVITY_VALUES = [value for value, _ in SENSITIVITY_CHOICES]


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    sensitivity = RadioField(
        "Sensitivity",
        choices=SENSITIVITY_CHOICES,
        default="off",
        validators=[AnyOf(SENSITIVITY_VALUES)],
    )
    submit = SubmitField("Upload File")


# Registration route
@app.route("/register", methods=["GET", "POST"])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        username = form.username.data
        email = form.email.data
        password_hash = generate_password_hash(form.password.data)

        # Check if the user already exists
        cursor = mysql.connection.cursor()
        cursor.execute(
            "SELECT * FROM users WHERE username = %s OR email = %s", (username, email)
        )
        existing_user = cursor.fetchone()

        if existing_user:
            app.logger.warning(
                "Register rejected: duplicate username=%r or email=%r", username, email
            )
            flash("Username or email already exists", "danger")
        else:
            # Insert into the database
            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
                (username, email, password_hash),
            )
            mysql.connection.commit()
            app.logger.info("Register success: username=%r", username)
            flash("Registration successful! You can now log in.", "success")
            return redirect(url_for("login"))
    elif request.method == "POST":
        _flash_form_errors(form, "Registration")

    return render_template("register.html", form=form)


@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data

        # Retrieve the user from the database
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user_data = cursor.fetchone()

        if user_data and check_password_hash(user_data[3], password):
            user = User(user_data[0], user_data[1])
            login_user(user)
            app.logger.info("Login success: username=%r", username)
            flash("Login successful!", "success")
            # Check if a 'next' parameter exists, and redirect accordingly
            target = _safe_next_url(request.args.get("next"))
            return redirect(target or url_for("upload"))
        else:
            reason = "unknown user" if not user_data else "bad password"
            app.logger.warning("Login failed: username=%r reason=%s", username, reason)
            flash("Invalid username or password", "danger")
    elif request.method == "POST":
        _flash_form_errors(form, "Login")

    return render_template("login.html", form=form)


# Logout
@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("home"))


# Route to check authentication
@app.route("/check_auth")
def check_auth():
    if current_user.is_authenticated:
        # If user is logged in, redirect to the upload page
        return redirect(url_for("home"))
    else:
        # If user is not logged in, redirect to the login page
        # Use 'next' to specify the intended action after login
        return redirect(url_for("login", next=url_for("check_auth")))


# Route for the home page
@app.route("/")
def home():
    return render_template("home.html")


def _safe_next_url(next_page):
    """Return ``next_page`` only when it points back to this host.

    Rejects scheme-relative URLs like ``//evil.com/foo`` (which start with
    ``/`` but redirect off-site) and absolute URLs whose ``netloc`` differs
    from the current request. Returns ``None`` when the value is missing or
    unsafe so callers can fall back to a default route.
    """
    if not next_page:
        return None
    parsed = urlparse(next_page)
    if parsed.scheme or parsed.netloc:
        # Absolute or scheme-relative URL. Only accept if same host.
        request_host = urlparse(request.host_url).netloc
        if parsed.netloc and parsed.netloc != request_host:
            return None
        if not parsed.netloc:
            # ``//something`` style — definitely off-site.
            return None
    if not next_page.startswith("/"):
        return None
    if next_page.startswith("//"):
        return None
    return next_page


def get_file_extension(filename):
    if not filename or "." not in filename:
        return ""
    return filename.rsplit(".", 1)[1].lower()


def get_media_kind(filename):
    extension = get_file_extension(filename)
    if extension in VIDEO_EXTENSIONS:
        return "video"
    return "audio"


def allowed_file(filename):
    return get_file_extension(filename) in ALLOWED_EXTENSIONS


# Regex describing every inline tag we let through the VTT sanitizer.
# Supports:
#   <u>, </u>, <b>, </b>, <i>, </i>
#   <c.classname>, <c.cls1.cls2>, </c>            (WebVTT cue spans, e.g. <c.color-FF0000>)
# Anything else (e.g. <script>) is escaped on the way out.
_ALLOWED_VTT_TAG_RE = re.compile(
    r"""<
        /?                                          # optional closing slash
        (?:
            u | b | i                               # simple inline tags
          | c (?: \. [A-Za-z0-9_\-]+ )*             # <c> or <c.dot.separated.classes>
        )
        \s*
        >
    """,
    re.VERBOSE,
)


def sanitize_vtt_content(content):
    """Make ``content`` safe to round-trip through HTML rendering.

    Allowed inline tags (<u>, <b>, <i> and any <c.classname>) are preserved
    verbatim while every other ``<...>`` is HTML-escaped. The function also
    decodes any pre-escaped entities first, which transparently repairs
    legacy DB rows whose tags were double-escaped by an older sanitizer
    (e.g. ``&lt;i&gt;`` -> ``<i>``).
    """
    if not content:
        return ""

    # Normalize newlines and revive a few characters the editor may have
    # left double-escaped (``-->`` markers + any leftover entities).
    text = (
        content.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("--&gt;", "-->")
    )
    text = unescape(text)

    # Stash every allowed tag behind a NUL-delimited placeholder so the
    # subsequent html.escape() pass leaves them intact.
    captured_tags = []

    def _capture(match):
        captured_tags.append(match.group(0))
        return f"\x00CC_TAG_{len(captured_tags) - 1}\x00"

    text = _ALLOWED_VTT_TAG_RE.sub(_capture, text)
    text = escape(text, quote=False)

    def _restore(match):
        return captured_tags[int(match.group(1))]

    text = re.sub(r"\x00CC_TAG_(\d+)\x00", _restore, text)
    # html.escape turns the VTT timestamp arrow "-->" into "--&gt;" because
    # the trailing ">" is a stand-alone character. Restore it so cues stay
    # parseable by browsers and downloadable VTT files remain valid.
    return text.replace("--&gt;", "-->")


def resolve_media_path(file_path):
    if not file_path:
        return None
    if os.path.isabs(file_path):
        return os.path.abspath(file_path)
    return os.path.abspath(os.path.join(BASE_DIR, file_path))


def is_path_inside(path, directory):
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(directory)]) == os.path.abspath(directory)
    except ValueError:
        return False


def relative_to_base(path):
    return os.path.relpath(os.path.abspath(path), BASE_DIR)


def update_job(job_id, status, progress, message, audio_id=None, error=None):
    now = datetime.now()
    conn = sqlite3.connect(DB_PATH)
    try:
        params = [status, progress, message, error, now, job_id]
        query = (
            "UPDATE processing_jobs SET status = ?, progress = ?, message = ?, "
            "error = ?, updated_at = ?"
        )
        if audio_id is not None:
            query += ", audio_id = ?"
            params = [status, progress, message, error, now, audio_id, job_id]
        query += " WHERE job_id = ?"
        conn.execute(query, params)
        conn.commit()
    finally:
        conn.close()


def create_job(job_id, user_id):
    now = datetime.now()
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT INTO processing_jobs (job_id, user_id, status, progress, message, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (job_id, user_id, "uploaded", 0, "Upload received", now, now),
        )
        conn.commit()
    finally:
        conn.close()


def extract_audio_from_video(video_path, audio_path):
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("FFmpeg is required to process video files. Please install FFmpeg and make sure it is available in PATH.")
    subprocess.run(
        [
            ffmpeg_path,
            "-y",
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            audio_path,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def convert_video_for_browser(source_path, output_path):
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("FFmpeg is required to prepare video playback. Please install FFmpeg and make sure it is available in PATH.")
    subprocess.run(
        [
            ffmpeg_path,
            "-y",
            "-i",
            source_path,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            output_path,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def update_media_file_for_playback(audio_id, file_path, file_type, stored_file_name, file_name=None):
    conn = sqlite3.connect(DB_PATH)
    try:
        if file_name is None:
            conn.execute(
                "UPDATE audiofiles SET file_path = ?, file_type = ?, stored_file_name = ? WHERE audio_id = ?",
                (relative_to_base(file_path), file_type, stored_file_name, audio_id),
            )
        else:
            conn.execute(
                "UPDATE audiofiles SET file_path = ?, file_type = ?, stored_file_name = ?, file_name = ? WHERE audio_id = ?",
                (relative_to_base(file_path), file_type, stored_file_name, file_name, audio_id),
            )
        conn.commit()
    finally:
        conn.close()


def run_transcription_subprocess(audio_path, sensitivity="off"):
    script_path = os.path.join(BASE_DIR, "sentence_recognition.py")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    # ``encoding="utf-8"`` is critical on Windows where ``text=True`` would
    # otherwise decode subprocess output with the system code page (e.g.
    # ``cp874``) and crash on any non-ASCII byte in stderr.
    cmd = [sys.executable, script_path, audio_path]
    tier = sensitivity if sensitivity in SENSITIVITY_VALUES else "off"
    if tier != "off":
        cmd.extend(["--sensitivity", tier])
    result = subprocess.run(
        cmd,
        cwd=BASE_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        if len(detail) > 1200:
            detail = detail[-1200:]
        raise RuntimeError(f"Transcription subprocess exited with code {result.returncode}. {detail}")
    return os.path.splitext(audio_path)[0] + ".json"


@app.route("/audio/<int:audio_id>", methods=["GET"])
@app.route("/media/<int:audio_id>", methods=["GET"])
@login_required
def get_audio(audio_id):
    cursor = mysql.connection.cursor()
    try:
        cursor.execute(
            "SELECT file_type, audio_content, file_path, file_name FROM audiofiles WHERE audio_id = %s AND user_id = %s",
            (audio_id, current_user.id),
        )
        result = cursor.fetchone()
    finally:
        cursor.close()

    if not result:
        return abort(404, "Media file not found")

    file_type, audio_content, file_path, file_name = result
    resolved_path = resolve_media_path(file_path)
    media_mimetype = mimetypes.guess_type(resolved_path or file_name or "")[0] or file_type or "application/octet-stream"

    if resolved_path and is_path_inside(resolved_path, UPLOAD_DIR) and os.path.isfile(resolved_path):
        response = send_file(
            resolved_path,
            mimetype=media_mimetype,
            as_attachment=False,
            download_name=file_name,
            conditional=True,
        )
        response.headers["Accept-Ranges"] = "bytes"
        return response

    if audio_content is None:
        return abort(404, "Media file content not found")

    response = make_response(audio_content)
    response.headers["Content-Type"] = media_mimetype
    response.headers["Content-Disposition"] = f"inline; filename={file_name or f'media_{audio_id}'}"
    response.headers["Accept-Ranges"] = "bytes"
    return response


@app.route("/edit/<int:audio_id>", methods=["GET"])
@login_required
def edit(audio_id):
    cursor = mysql.connection.cursor()
    try:
        cursor.execute(
            "SELECT v.vtt_content, COALESCE(a.media_kind, 'audio'), a.file_type FROM vttfiles v INNER JOIN audiofiles a ON v.audio_id = a.audio_id WHERE v.audio_id = %s AND a.user_id = %s",
            (audio_id, current_user.id),
        )
        result = cursor.fetchone()

        if not result:
            flash(
                "VTT file not found or you do not have permission to edit it.", "danger"
            )
            return redirect(url_for("your_files"))

        vtt_content = sanitize_vtt_content(result[0])
        media_kind = result[1]
        media_type = result[2]
        
        # แก้ไข vtt_content ให้ WEBVTT อยู่ชิดซ้าย
        if vtt_content:
            lines = vtt_content.split('\n')
            # หาบรรทัดที่มีคำว่า WEBVTT
            for i, line in enumerate(lines):
                if 'WEBVTT' in line:
                    # แทนที่บรรทัดนั้นด้วย WEBVTT ที่ไม่มี whitespace
                    lines[i] = 'WEBVTT'
                    break
            # รวมกลับเป็น string
            vtt_content = '\n'.join(lines)

        return render_template(
            "edit.html",
            audio_id=audio_id,
            vtt_content=vtt_content,
            media_kind=media_kind,
            media_type=media_type,
        )

    except Exception as e:
        flash(f"An error occurred: {str(e)}", "danger")
        return redirect(url_for("your_files"))
    finally:
        cursor.close()


@app.route("/save_vtt/<int:audio_id>", methods=["POST"])
@login_required
def save_vtt(audio_id):
    # Accept JSON bodies (interactive Save buttons) AND form-encoded bodies
    # (auto-save via ``navigator.sendBeacon`` with FormData -- the only way
    # to carry the CSRF token during ``pagehide`` since sendBeacon cannot
    # set custom request headers).
    if request.is_json:
        edited_content = (request.get_json(silent=True) or {}).get("content", "")
    elif "content" in request.form:
        edited_content = request.form.get("content", "")
    else:
        raw_body = request.get_data(as_text=True)
        try:
            edited_content = json.loads(raw_body).get("content", "")
        except Exception:
            edited_content = raw_body

    cursor = None
    try:
        edited_content = sanitize_vtt_content(edited_content)

        cursor = mysql.connection.cursor()
        cursor.execute(
            "UPDATE vttfiles SET vtt_content = %s WHERE audio_id = %s AND audio_id IN (SELECT audio_id FROM audiofiles WHERE user_id = %s)",
            (edited_content, audio_id, current_user.id),
        )
        mysql.connection.commit()
        # Note: do NOT call ``flash`` here. AJAX clients ignore the response
        # body's flash messages, but Flask still queues them in the session
        # and they leak onto the next full page render.
        return jsonify({"message": "VTT content saved successfully."}), 200

    except Exception as e:
        print(f"Error saving VTT file: {e}")
        return jsonify({"message": "Failed to save VTT file. Please try again."}), 500

    finally:
        if cursor:
            cursor.close()


@app.route("/download_vtt/<int:audio_id>", methods=["GET"])
@login_required
def download_vtt(audio_id):
    cursor = mysql.connection.cursor()
    try:
        # ดึงข้อมูล VTT content และชื่อไฟล์ VTT ที่อาจมีการเปลี่ยนแปลงแล้ว
        cursor.execute(
            "SELECT v.vtt_content, v.file_name, a.file_name as original_audio_filename FROM vttfiles v "
            "INNER JOIN audiofiles a ON v.audio_id = a.audio_id "
            "WHERE v.audio_id = %s AND a.user_id = %s", 
            (audio_id, current_user.id)
        )
        result = cursor.fetchone()

        if not result:
            flash("VTT file not found.", "danger")
            return redirect(url_for("your_files"))

        vtt_content, vtt_filename, original_audio_filename = result
        vtt_content = sanitize_vtt_content(vtt_content)
        
        # ใช้ชื่อไฟล์ VTT ที่อยู่ในฐานข้อมูล (ซึ่งอาจมีการ rename ไปแล้ว)
        # ถ้าไม่มีหรือเป็นชื่อเริ่มต้น ให้ใช้ชื่อเดียวกับไฟล์เสียงแทน
        if not vtt_filename or vtt_filename == "stress_closed_caption.vtt":
            base_filename = os.path.splitext(original_audio_filename)[0]
            download_filename = f"{base_filename}.vtt"
        else:
            download_filename = vtt_filename
        
        # แก้ไข format ของ VTT content
        if vtt_content:
            lines = vtt_content.splitlines()
            new_lines = []
            webvtt_line = None
            
            # แยก WEBVTT ออกมาก่อน
            for line in lines:
                if 'WEBVTT' in line:
                    webvtt_line = 'WEBVTT'
                else:
                    new_lines.append(line.rstrip())
            
            # สร้าง content ใหม่โดยเริ่มด้วย WEBVTT
            final_content = ['WEBVTT']  # เริ่มด้วย WEBVTT
            final_content.extend(new_lines)  # เพิ่มเนื้อหาที่เหลือ
            
            # รวมกลับเป็น string
            vtt_content = sanitize_vtt_content('\n'.join(final_content))

        response = make_response(vtt_content)
        # ``filename=`` only handles ASCII reliably; ``filename*=UTF-8''...``
        # (RFC 6266) covers non-ASCII names. Quoting the ASCII fallback
        # protects names that contain spaces or special characters.
        ascii_fallback = download_filename.encode("ascii", "replace").decode("ascii")
        encoded_name = quote(download_filename, safe="")
        response.headers["Content-Disposition"] = (
            f'attachment; filename="{ascii_fallback}"; filename*=UTF-8\'\'{encoded_name}'
        )
        response.headers["Content-Type"] = "text/vtt; charset=utf-8"
        return response
    
    except Exception as e:
        flash(f"An error occurred: {str(e)}", "danger")
        return redirect(url_for("your_files"))
    
    finally:
        cursor.close()


@app.route("/rename_vtt/<int:audio_id>", methods=["POST"])
@login_required
def rename_vtt(audio_id):
    new_filename = request.json.get("new_filename")
    
    # ตรวจสอบว่าชื่อไฟล์ใหม่ถูกต้อง
    if not new_filename or not new_filename.strip():
        return jsonify({"success": False, "message": "Filename cannot be empty"}), 400
    
    # เพิ่มนามสกุล .vtt ถ้าไม่มี
    if not new_filename.lower().endswith('.vtt'):
        new_filename += '.vtt'
    
    # ทำให้ชื่อไฟล์ปลอดภัย
    new_filename = secure_filename(new_filename)
    
    cursor = mysql.connection.cursor()
    try:
        # อัปเดตชื่อไฟล์ในฐานข้อมูล
        cursor.execute(
            "UPDATE vttfiles SET file_name = %s WHERE audio_id = %s AND audio_id IN "
            "(SELECT audio_id FROM audiofiles WHERE user_id = %s)",
            (new_filename, audio_id, current_user.id)
        )
        mysql.connection.commit()
        
        return jsonify({"success": True, "message": "File renamed successfully", "new_filename": new_filename}), 200
    
    except Exception as e:
        print(f"Error renaming VTT file: {e}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500
    
    finally:
        cursor.close()


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return redirect('https://ca-t.psu.ac.th/')


@app.route('/delete-audio/<int:audio_id>', methods=['DELETE'])
@login_required
def delete_audio(audio_id):
    media_path = None
    try:
        cursor = mysql.connection.cursor()

        cursor.execute(
            "SELECT file_path FROM audiofiles WHERE audio_id = %s AND user_id = %s",
            (audio_id, current_user.id),
        )
        result = cursor.fetchone()
        if not result:
            return jsonify({'error': 'Media file not found'}), 404
        media_path = resolve_media_path(result[0])
        
        cursor.execute("DELETE FROM vttfiles WHERE audio_id = %s AND audio_id IN (SELECT audio_id FROM audiofiles WHERE user_id = %s)", 
                      (audio_id, current_user.id))
        
        cursor.execute("DELETE FROM jsonfiles WHERE audio_id = %s AND audio_id IN (SELECT audio_id FROM audiofiles WHERE user_id = %s)", 
                      (audio_id, current_user.id))

        cursor.execute("DELETE FROM processing_jobs WHERE audio_id = %s AND user_id = %s", 
                      (audio_id, current_user.id))
        
        cursor.execute("DELETE FROM audiofiles WHERE audio_id = %s AND user_id = %s", 
                      (audio_id, current_user.id))
        
        mysql.connection.commit()
        if media_path and is_path_inside(media_path, UPLOAD_DIR) and os.path.isfile(media_path):
            os.remove(media_path)
        return jsonify({'success': True, 'message': 'Media file deleted successfully'}), 200
        
    except Exception as e:
        print(f"Error deleting media file: {e}")
        return jsonify({'error': str(e)}), 500
        
    finally:
        cursor.close()


@app.route("/get_vtt_filename/<int:audio_id>", methods=["GET"])
@login_required
def get_vtt_filename(audio_id):
    cursor = mysql.connection.cursor()
    try:
        # ดึงข้อมูลชื่อไฟล์ VTT
        cursor.execute(
            "SELECT v.file_name, a.file_name as audio_filename FROM vttfiles v "
            "INNER JOIN audiofiles a ON v.audio_id = a.audio_id "
            "WHERE v.audio_id = %s AND a.user_id = %s", 
            (audio_id, current_user.id)
        )
        result = cursor.fetchone()

        if not result:
            return jsonify({"success": False, "message": "File not found"}), 404

        vtt_filename, audio_filename = result

        # If the stored name is missing or the legacy default, derive one
        # from the audio filename. We do *not* persist this; GET endpoints
        # must not mutate state. ``rename_vtt`` is the dedicated writer.
        if not vtt_filename or vtt_filename == "stress_closed_caption.vtt":
            base_filename = os.path.splitext(audio_filename)[0]
            vtt_filename = f"{base_filename}.vtt"

        return jsonify({
            "success": True,
            "filename": vtt_filename,
            "original_audio": audio_filename,
        }), 200
    
    except Exception as e:
        print(f"Error fetching VTT filename: {e}")
        return jsonify({"success": False, "message": str(e)}), 500
    
    finally:
        cursor.close()


def _delete_existing_caption_rows(audio_id):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("DELETE FROM vttfiles WHERE audio_id = ?", (audio_id,))
        conn.execute("DELETE FROM jsonfiles WHERE audio_id = ?", (audio_id,))
        conn.commit()
    finally:
        conn.close()


def insert_audio_file_to_db(
    user_id,
    file_name,
    file_type,
    file_size,
    upload_date,
    audio_content=None,
    file_path=None,
    media_kind="audio",
    stored_file_name=None,
):
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO audiofiles (
                user_id, file_name, file_type, file_size, upload_datetime,
                audio_content, file_path, media_kind, stored_file_name
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                file_name,
                file_type,
                file_size,
                upload_date,
                audio_content,
                file_path,
                media_kind,
                stored_file_name,
            ),
        )
        conn.commit()
        return cursor.lastrowid
    except Exception as e:
        print("Error inserting media file into database:", e)
        raise
    finally:
        if conn:
            conn.close()


def insert_json_file_to_db(audio_id, json_file_name, json_content, created_at):
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO jsonfiles (audio_id, file_name, json_content, created_at) VALUES (?, ?, ?, ?)",
            (audio_id, json_file_name, json.dumps(json_content), created_at),
        )
        conn.commit()
    except Exception as e:
        print("Error inserting JSON into database:", e)
        raise
    finally:
        if conn:
            conn.close()


def insert_vtt_file_to_db(audio_id, vtt_file_name, vtt_content, created_at):
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO vttfiles (audio_id, file_name, vtt_content, created_at) VALUES (?, ?, ?, ?)",
            (audio_id, vtt_file_name, vtt_content, created_at),
        )
        conn.commit()
    except Exception as e:
        print("Error inserting VTT file into database:", e)
        raise
    finally:
        if conn:
            conn.close()


def process_media_job(job_id, user_id, audio_id, media_path, media_kind, original_file_name, *, skip_video_conversion=False, sensitivity="off"):
    job_temp_dir = os.path.join(TEMP_DIR, job_id)
    os.makedirs(job_temp_dir, exist_ok=True)
    try:
        update_job(job_id, "processing", 10, "Preparing media file", audio_id=audio_id)
        extension = get_file_extension(original_file_name)

        if media_kind == "video":
            if skip_video_conversion:
                # Retry path: the file at ``media_path`` is already the
                # browser-ready MP4, just re-use it for audio extraction.
                update_job(job_id, "processing", 30, "Extracting audio from video", audio_id=audio_id)
                audio_input_path = os.path.join(job_temp_dir, "extracted_audio.wav")
                extract_audio_from_video(media_path, audio_input_path)
            else:
                update_job(job_id, "processing", 15, "Preparing video playback", audio_id=audio_id)
                browser_file_name = f"{os.path.splitext(os.path.basename(media_path))[0]}_browser.mp4"
                browser_video_path = os.path.join(os.path.dirname(media_path), browser_file_name)
                convert_video_for_browser(media_path, browser_video_path)
                # Keep the displayed file_name consistent with the actual
                # browser-served content type so download_name and the
                # MIME type agree (e.g. ``lecture.mov`` -> ``lecture.mp4``).
                display_base = os.path.splitext(secure_filename(original_file_name))[0]
                display_name = f"{display_base}.mp4"
                update_media_file_for_playback(
                    audio_id,
                    browser_video_path,
                    BROWSER_VIDEO_TYPE,
                    browser_file_name,
                    file_name=display_name,
                )
                if os.path.abspath(media_path) != os.path.abspath(browser_video_path):
                    try:
                        os.remove(media_path)
                    except OSError:
                        pass
                media_path = browser_video_path
                update_job(job_id, "processing", 30, "Extracting audio from video", audio_id=audio_id)
                audio_input_path = os.path.join(job_temp_dir, "extracted_audio.wav")
                extract_audio_from_video(media_path, audio_input_path)
        else:
            audio_input_path = os.path.join(job_temp_dir, f"source.{extension or 'wav'}")
            shutil.copy2(media_path, audio_input_path)

        model_name = os.environ.get("WHISPER_MODEL", "large-v3")
        tier = sensitivity if sensitivity in SENSITIVITY_VALUES else "off"
        if tier == "ultra":
            mode_label = " (ultra-sensitive)"
        elif tier == "sensitive":
            mode_label = " (sensitive)"
        else:
            mode_label = ""
        update_job(job_id, "processing", 45, f"Transcribing speech with Faster Whisper {model_name}{mode_label}", audio_id=audio_id)
        json_path = run_transcription_subprocess(audio_input_path, sensitivity=tier)
        if not json_path or not os.path.isfile(json_path):
            raise RuntimeError("Transcription failed before a JSON file was generated.")

        with open(json_path, "r", encoding="utf-8") as json_file:
            json_content = json.load(json_file)

        update_job(job_id, "processing", 80, "Generating closed captions", audio_id=audio_id)
        base_filename = os.path.splitext(secure_filename(original_file_name))[0]
        vtt_file_name = f"{base_filename}.vtt"
        vtt_path = os.path.join(job_temp_dir, vtt_file_name)
        recognizer = SentenceRecognizer(audio_input_path, json_path)
        recognizer.generate_vtt(vtt_path)

        if not os.path.isfile(vtt_path):
            raise RuntimeError("Caption generation failed before a VTT file was generated.")

        with open(vtt_path, "r", encoding="utf-8") as vtt_file:
            vtt_content = sanitize_vtt_content(vtt_file.read().lstrip())

        created_at = datetime.now()
        # Replace any previous JSON/VTT rows for this audio so retries do not
        # leave stale duplicates around (the schema allows multiple rows but
        # download_vtt and edit only ever read the first one).
        _delete_existing_caption_rows(audio_id)
        insert_json_file_to_db(audio_id, os.path.basename(json_path), json_content, created_at)
        insert_vtt_file_to_db(audio_id, vtt_file_name, vtt_content, created_at)
        update_job(job_id, "ready", 100, "Caption file is ready", audio_id=audio_id)

    except Exception as err:
        print("Processing error:", err)
        traceback.print_exc()
        update_job(job_id, "failed", 100, "Processing failed", audio_id=audio_id, error=str(err))
    finally:
        shutil.rmtree(job_temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Single-worker processing queue.
#
# Uploads no longer spawn a thread each; instead they enqueue the job here and
# return immediately. ONE dedicated worker thread drains the queue and runs
# ``process_media_job`` strictly one at a time. This keeps memory/GPU usage
# bounded (faster-whisper loads one model at a time) and gives every waiting
# user a deterministic queue position.
#
# DEPLOYMENT NOTE: this is an in-process queue, so the app MUST run as a
# SINGLE process (e.g. ``waitress-serve`` with threads, or ``gunicorn -w 1``).
# Running multiple worker processes would give each its own queue + worker.
# ---------------------------------------------------------------------------
_job_queue = queue.Queue()


def _job_worker():
    while True:
        task = _job_queue.get()
        try:
            process_media_job(*task["args"], **task["kwargs"])
        except Exception as err:  # never let one bad job kill the worker
            print("Job worker error:", err)
            traceback.print_exc()
            try:
                update_job(
                    task.get("job_id"),
                    "failed",
                    100,
                    "Processing failed",
                    error=str(err),
                )
            except Exception:
                pass
        finally:
            _job_queue.task_done()


def enqueue_job(job_id, args, kwargs=None):
    """Add a job to the processing queue (FIFO, one-at-a-time)."""
    _job_queue.put({"job_id": job_id, "args": args, "kwargs": kwargs or {}})


_job_worker_thread = threading.Thread(target=_job_worker, name="job-worker", daemon=True)
_job_worker_thread.start()


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    form = UploadFileForm()
    submitted = False

    if request.method == "POST" and form.validate_on_submit():
        file = form.file.data

        if file and allowed_file(file.filename):
            try:
                job_id = uuid.uuid4().hex
                file_name = secure_filename(file.filename)
                stored_file_name = f"{job_id}_{file_name}"
                user_upload_dir = os.path.join(UPLOAD_DIR, str(current_user.id))
                os.makedirs(user_upload_dir, exist_ok=True)
                file_path = os.path.join(user_upload_dir, stored_file_name)
                file.save(file_path)

                guessed_type = mimetypes.guess_type(file_name)[0]
                file_type = guessed_type or file.mimetype or file.content_type or "application/octet-stream"
                file_size = os.path.getsize(file_path)
                upload_date = datetime.now()
                media_kind = get_media_kind(file_name)
                create_job(job_id, current_user.id)
                audio_id = insert_audio_file_to_db(
                    current_user.id,
                    file_name,
                    file_type,
                    file_size,
                    upload_date,
                    None,
                    relative_to_base(file_path),
                    media_kind,
                    stored_file_name,
                )
                update_job(job_id, "queued", 5, "File uploaded. Waiting in queue", audio_id=audio_id)
                sensitivity = form.sensitivity.data if form.sensitivity.data in SENSITIVITY_VALUES else "off"
                enqueue_job(
                    job_id,
                    args=(job_id, current_user.id, audio_id, file_path, media_kind, file_name),
                    kwargs={"sensitivity": sensitivity},
                )
                submitted = True
                if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                    return jsonify({"job_id": job_id, "audio_id": audio_id}), 202
                flash("File uploaded. Caption processing has been queued.", "success")
                return redirect(url_for("your_files"))

            except Exception as err:
                print("Error:", err)
                traceback.print_exc()
                session["message"] = (
                    "An error occurred while processing the request. Please try again."
                )
                if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                    return jsonify({"error": session["message"]}), 500
                return redirect(url_for("upload"))

        else:
            session["message"] = (
                "Only audio or video files are allowed."
            )
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify({"error": session["message"]}), 400
            return redirect(url_for("upload"))

    if request.method == "POST" and request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({"error": "Upload form validation failed."}), 400

    return render_template(
        "upload.html", form=form, submitted=submitted,
        max_upload_bytes=app.config["MAX_CONTENT_LENGTH"],
        allowed_extensions=sorted(ALLOWED_EXTENSIONS),
    )


@app.route("/retry_job/<int:audio_id>", methods=["POST"])
@login_required
def retry_job(audio_id):
    cursor = mysql.connection.cursor()
    try:
        cursor.execute(
            "SELECT file_path, file_name, file_type, COALESCE(media_kind, 'audio') FROM audiofiles WHERE audio_id = %s AND user_id = %s",
            (audio_id, current_user.id),
        )
        row = cursor.fetchone()
    finally:
        cursor.close()

    if not row:
        return jsonify({"error": "Media not found"}), 404
    file_path, file_name, file_type, media_kind = row
    media_path = resolve_media_path(file_path)
    if not media_path or not is_path_inside(media_path, UPLOAD_DIR) or not os.path.isfile(media_path):
        return jsonify({"error": "Media file is missing on disk"}), 404

    # If the persisted file is already a browser-ready MP4 (the result of a
    # previous conversion), don't transcode again.
    skip_video_conversion = (media_kind == "video" and file_type == BROWSER_VIDEO_TYPE)

    job_id = uuid.uuid4().hex
    create_job(job_id, current_user.id)
    update_job(job_id, "queued", 5, "Retry queued", audio_id=audio_id)
    enqueue_job(
        job_id,
        args=(job_id, current_user.id, audio_id, media_path, media_kind, file_name),
        kwargs={"skip_video_conversion": skip_video_conversion},
    )
    return jsonify({"job_id": job_id, "audio_id": audio_id}), 202


@app.route("/job_status/<job_id>", methods=["GET"])
@login_required
def job_status(job_id):
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT status, progress, message, error, audio_id, created_at FROM processing_jobs WHERE job_id = ? AND user_id = ?",
            (job_id, current_user.id),
        )
        result = cursor.fetchone()
        if not result:
            return jsonify({"error": "Job not found"}), 404

        status, progress, message, error, audio_id, created_at = result
        response = {
            "status": status,
            "progress": progress,
            "message": message,
            "error": error,
            "audio_id": audio_id,
        }

        # Queue visibility: while a job waits, tell the user how many jobs are
        # ahead of it so they get a deterministic "you are Nth in line".
        # ``queue_ahead`` counts the one currently processing (if any) plus all
        # queued jobs created before this one. ``queue_position`` is 1-based
        # (1 == next up). Processing happens strictly one-at-a-time, so this is
        # an accurate ordering.
        if status in ("queued", "uploaded"):
            ahead_processing = conn.execute(
                "SELECT COUNT(*) FROM processing_jobs WHERE status = 'processing'"
            ).fetchone()[0]
            ahead_queued = conn.execute(
                "SELECT COUNT(*) FROM processing_jobs "
                "WHERE status IN ('queued', 'uploaded') AND created_at < ? AND job_id != ?",
                (created_at, job_id),
            ).fetchone()[0]
            ahead = ahead_processing + ahead_queued
            response["queue_ahead"] = ahead
            response["queue_position"] = ahead + 1

        if status == "ready" and audio_id:
            response["redirect_url"] = url_for("edit", audio_id=audio_id)
        return jsonify(response)
    finally:
        conn.close()


@app.route("/your_files", methods=["GET"])
@login_required
def your_files():
    page = request.args.get('page', 1, type=int)
    per_page = 6  # Number of items per page
    
    cursor = mysql.connection.cursor()
    
    # Get all files but ordered
    cursor.execute(
        """SELECT a.audio_id, a.user_id, a.file_name, a.file_type, a.file_size, a.upload_datetime,
                  COALESCE(a.media_kind, 'audio') AS media_kind,
                  COALESCE(p.status, CASE WHEN v.vtt_id IS NOT NULL THEN 'ready' ELSE 'unknown' END) AS processing_status,
                  COALESCE(p.progress, CASE WHEN v.vtt_id IS NOT NULL THEN 100 ELSE 0 END) AS processing_progress,
                  COALESCE(p.message, '') AS processing_message
           FROM audiofiles a
           LEFT JOIN processing_jobs p ON p.audio_id = a.audio_id
           LEFT JOIN vttfiles v ON v.audio_id = a.audio_id
           WHERE a.user_id = %s 
           ORDER BY a.upload_datetime DESC""",
        (current_user.id,)
    )
    audio_files = cursor.fetchall()

    if not audio_files:
        flash("No media files found.", "warning")

    return render_template(
        "your_files.html", 
        audio_files=audio_files,
        page=page,
        per_page=per_page
    )


# =====================================================
# Redesigned UI routes (templates/new/*).
# These mirror the original endpoints but render the
# new templates and link within the /new/* namespace.
# All AJAX endpoints (save_vtt, download_vtt, rename_vtt,
# delete-audio, get_audio, job_status, get_vtt_filename)
# remain shared with the legacy UI.
# =====================================================


@app.route("/new/")
def new_home():
    return render_template("new/home.html")


@app.route("/new/about")
def new_about():
    return render_template("new/about.html")


@app.route("/new/register", methods=["GET", "POST"])
def new_register():
    form = RegistrationForm()
    if form.validate_on_submit():
        username = form.username.data
        email = form.email.data
        password_hash = generate_password_hash(form.password.data)

        cursor = mysql.connection.cursor()
        cursor.execute(
            "SELECT * FROM users WHERE username = %s OR email = %s", (username, email)
        )
        existing_user = cursor.fetchone()

        if existing_user:
            app.logger.warning(
                "Register rejected: duplicate username=%r or email=%r", username, email
            )
            flash("Username or email already exists", "danger")
        else:
            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
                (username, email, password_hash),
            )
            mysql.connection.commit()
            app.logger.info("Register success: username=%r", username)
            flash("Registration successful! You can now log in.", "success")
            return redirect(url_for("new_login"))
    elif request.method == "POST":
        _flash_form_errors(form, "Registration")

    return render_template("new/register.html", form=form)


@app.route("/new/login", methods=["GET", "POST"])
def new_login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user_data = cursor.fetchone()

        if user_data and check_password_hash(user_data[3], password):
            user = User(user_data[0], user_data[1])
            login_user(user)
            app.logger.info("Login success: username=%r", username)
            flash("Login successful!", "success")
            target = _safe_next_url(request.args.get("next"))
            return redirect(target or url_for("new_upload"))
        reason = "unknown user" if not user_data else "bad password"
        app.logger.warning("Login failed: username=%r reason=%s", username, reason)
        flash("Invalid username or password", "danger")
    elif request.method == "POST":
        _flash_form_errors(form, "Login")

    return render_template("new/login.html", form=form)


@app.route("/new/logout")
@login_required
def new_logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("new_home"))


@app.route("/new/upload", methods=["GET", "POST"])
@login_required
def new_upload():
    form = UploadFileForm()
    submitted = False

    if request.method == "POST" and form.validate_on_submit():
        file = form.file.data

        if file and allowed_file(file.filename):
            try:
                job_id = uuid.uuid4().hex
                file_name = secure_filename(file.filename)
                stored_file_name = f"{job_id}_{file_name}"
                user_upload_dir = os.path.join(UPLOAD_DIR, str(current_user.id))
                os.makedirs(user_upload_dir, exist_ok=True)
                file_path = os.path.join(user_upload_dir, stored_file_name)
                file.save(file_path)

                guessed_type = mimetypes.guess_type(file_name)[0]
                file_type = guessed_type or file.mimetype or file.content_type or "application/octet-stream"
                file_size = os.path.getsize(file_path)
                upload_date = datetime.now()
                media_kind = get_media_kind(file_name)
                create_job(job_id, current_user.id)
                audio_id = insert_audio_file_to_db(
                    current_user.id,
                    file_name,
                    file_type,
                    file_size,
                    upload_date,
                    None,
                    relative_to_base(file_path),
                    media_kind,
                    stored_file_name,
                )
                update_job(job_id, "queued", 5, "File uploaded. Waiting in queue", audio_id=audio_id)
                sensitivity = form.sensitivity.data if form.sensitivity.data in SENSITIVITY_VALUES else "off"
                enqueue_job(
                    job_id,
                    args=(job_id, current_user.id, audio_id, file_path, media_kind, file_name),
                    kwargs={"sensitivity": sensitivity},
                )
                submitted = True
                if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                    return jsonify({"job_id": job_id, "audio_id": audio_id}), 202
                flash("File uploaded. Caption processing has been queued.", "success")
                return redirect(url_for("new_your_files"))

            except Exception as err:
                print("Error:", err)
                traceback.print_exc()
                session["message"] = (
                    "An error occurred while processing the request. Please try again."
                )
                if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                    return jsonify({"error": session["message"]}), 500
                return redirect(url_for("new_upload"))

        else:
            session["message"] = "Only audio or video files are allowed."
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify({"error": session["message"]}), 400
            return redirect(url_for("new_upload"))

    if request.method == "POST" and request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({"error": "Upload form validation failed."}), 400

    return render_template(
        "new/upload.html", form=form, submitted=submitted,
        max_upload_bytes=app.config["MAX_CONTENT_LENGTH"],
        allowed_extensions=sorted(ALLOWED_EXTENSIONS),
    )


@app.route("/new/your_files", methods=["GET"])
@login_required
def new_your_files():
    page = request.args.get('page', 1, type=int)
    per_page = 5

    cursor = mysql.connection.cursor()
    cursor.execute(
        """SELECT a.audio_id, a.user_id, a.file_name, a.file_type, a.file_size, a.upload_datetime,
                  COALESCE(a.media_kind, 'audio') AS media_kind,
                  COALESCE(p.status, CASE WHEN v.vtt_id IS NOT NULL THEN 'ready' ELSE 'unknown' END) AS processing_status,
                  COALESCE(p.progress, CASE WHEN v.vtt_id IS NOT NULL THEN 100 ELSE 0 END) AS processing_progress,
                  COALESCE(p.message, '') AS processing_message
           FROM audiofiles a
           LEFT JOIN processing_jobs p ON p.audio_id = a.audio_id
           LEFT JOIN vttfiles v ON v.audio_id = a.audio_id
           WHERE a.user_id = %s
           ORDER BY a.upload_datetime DESC""",
        (current_user.id,)
    )
    audio_files = cursor.fetchall()

    return render_template(
        "new/your_files.html",
        audio_files=audio_files,
        page=page,
        per_page=per_page,
    )


@app.route("/new/edit/<int:audio_id>", methods=["GET"])
@login_required
def new_edit(audio_id):
    cursor = mysql.connection.cursor()
    try:
        cursor.execute(
            "SELECT v.vtt_content, COALESCE(a.media_kind, 'audio'), a.file_type FROM vttfiles v INNER JOIN audiofiles a ON v.audio_id = a.audio_id WHERE v.audio_id = %s AND a.user_id = %s",
            (audio_id, current_user.id),
        )
        result = cursor.fetchone()

        if not result:
            flash(
                "VTT file not found or you do not have permission to edit it.", "danger"
            )
            return redirect(url_for("new_your_files"))

        vtt_content = sanitize_vtt_content(result[0])
        media_kind = result[1]
        media_type = result[2]

        if vtt_content:
            lines = vtt_content.split('\n')
            for i, line in enumerate(lines):
                if 'WEBVTT' in line:
                    lines[i] = 'WEBVTT'
                    break
            vtt_content = '\n'.join(lines)

        return render_template(
            "new/edit.html",
            audio_id=audio_id,
            vtt_content=vtt_content,
            media_kind=media_kind,
            media_type=media_type,
        )
    except Exception as e:
        flash(f"An error occurred: {str(e)}", "danger")
        return redirect(url_for("new_your_files"))
    finally:
        cursor.close()


if __name__ == "__main__":
    # use_reloader=False prevents loading the Whisper model twice in dev mode.
    debug_mode = os.environ.get("FLASK_DEBUG", "0").lower() in ("1", "true", "yes", "on")
    app.run(debug=debug_mode, use_reloader=False)
