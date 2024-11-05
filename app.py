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
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, FileField
from wtforms.validators import DataRequired, Email, Length, InputRequired

import os
import datetime

from sentence_recognition import SentenceRecognition

sentence_recognizer = SentenceRecognition()
from stress_highlight import SentenceRecognizer


app = Flask(__name__)
app.secret_key = os.urandom(24)

app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB

# Set up MySQL connection
app.config["MYSQL_USER"] = "root"
app.config["MYSQL_PASSWORD"] = ""
app.config["MYSQL_DB"] = "flask_users"
app.config["MYSQL_HOST"] = "localhost"

mysql = MySQL(app)

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


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
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
            flash("Username or email already exists", "danger")
        else:
            # Insert into the database
            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
                (username, email, password_hash),
            )
            mysql.connection.commit()
            flash("Registration successful! You can now log in.", "success")
            return redirect(url_for("login"))

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
            flash("Login successful!", "success")
            # Check if a 'next' parameter exists, and redirect accordingly
            next_page = request.args.get("next")
            if not next_page or next_page.startswith("/"):  # Ensure valid URL
                return redirect(next_page or url_for("upload"))
            else:
                flash("Invalid redirect URL.", "danger")
                return redirect(url_for("upload"))
        else:
            flash("Invalid username or password", "danger")

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


@app.route("/audio/<int:audio_id>", methods=["GET"])
def get_audio(audio_id):
    cursor = mysql.connection.cursor()
    cursor.execute(
        "SELECT file_type, audio_content FROM audiofiles WHERE audio_id = %s",
        (audio_id,),
    )
    result = cursor.fetchone()

    if not result:
        return abort(404, "Audio file not found")

    file_type, audio_content = result
    response = make_response(audio_content)
    response.headers["Content-Type"] = file_type
    response.headers["Content-Disposition"] = f"inline; filename=audio_{audio_id}"
    return response


def allowed_file(filename):
    allowed_extensions = {"wav", "mp3", "ogg", "flac"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


@app.route("/edit/<int:audio_id>", methods=["GET"])
@login_required
def edit(audio_id):
    cursor = mysql.connection.cursor()
    try:
        cursor.execute(
            "SELECT v.vtt_content FROM vttfiles v INNER JOIN audiofiles a ON v.audio_id = a.audio_id WHERE v.audio_id = %s AND a.user_id = %s",
            (audio_id, current_user.id),
        )
        result = cursor.fetchone()

        if not result:
            flash(
                "VTT file not found or you do not have permission to edit it.", "danger"
            )
            return redirect(url_for("your_files"))

        vtt_content = result[0]  # Assuming vtt_content is the first column returned

        return render_template("edit.html", audio_id=audio_id, vtt_content=vtt_content)

    except Exception as e:
        flash(f"An error occurred: {str(e)}", "danger")
        return redirect(url_for("your_files"))
    finally:
        cursor.close()


@app.route("/save_vtt/<int:audio_id>", methods=["POST"])
@login_required
def save_vtt(audio_id):
    edited_content = request.json.get("content")  # Retrieve JSON content correctly

    try:
        print(
            f"Edited content received: {edited_content}"
        )  # Add this line for debugging

        # Replace any encoded arrows with the correct characters
        edited_content = edited_content.replace("--&gt;", "-->")

        cursor = mysql.connection.cursor()
        cursor.execute(
            "UPDATE vttfiles SET vtt_content = %s WHERE audio_id = %s AND audio_id IN (SELECT audio_id FROM audiofiles WHERE user_id = %s)",
            (edited_content, audio_id, current_user.id),
        )
        mysql.connection.commit()
        flash("VTT content saved successfully.", "success")
        return jsonify({"message": "VTT content saved successfully."}), 200

    except Exception as e:
        print(f"Error saving VTT file: {e}")
        flash("Failed to save VTT file. Please try again.", "danger")
        return jsonify({"message": "Failed to save VTT file. Please try again."}), 500

    finally:
        cursor.close()


@app.route("/download_vtt/<int:audio_id>", methods=["GET"])
@login_required
def download_vtt(audio_id):
    cursor = mysql.connection.cursor()
    cursor.execute(
        "SELECT TRIM(vtt_content) FROM vttfiles WHERE audio_id = %s", (audio_id,)
    )
    result = cursor.fetchone()

    if not result:
        flash("VTT file not found.", "danger")
        return redirect(url_for("your_files"))

    vtt_content = result[0]

    response = make_response(vtt_content)
    response.headers["Content-Disposition"] = (
        f"attachment; filename=edited_vtt_file.vtt"
    )
    response.headers["Content-Type"] = "text/vtt"
    return response


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return redirect('https://ca-t.psu.ac.th/')


@app.route('/delete-audio/<int:audio_id>', methods=['DELETE'])
@login_required
def delete_audio(audio_id):
    try:
        cursor = mysql.connection.cursor()
        
        # First delete related VTT files
        cursor.execute("DELETE FROM vttfiles WHERE audio_id = %s AND audio_id IN (SELECT audio_id FROM audiofiles WHERE user_id = %s)", 
                      (audio_id, current_user.id))
        
        # Then delete related JSON files
        cursor.execute("DELETE FROM jsonfiles WHERE audio_id = %s AND audio_id IN (SELECT audio_id FROM audiofiles WHERE user_id = %s)", 
                      (audio_id, current_user.id))
        
        # Finally delete the audio file
        cursor.execute("DELETE FROM audiofiles WHERE audio_id = %s AND user_id = %s", 
                      (audio_id, current_user.id))
        
        mysql.connection.commit()
        return jsonify({'success': True, 'message': 'Audio file deleted successfully'}), 200
        
    except Exception as e:
        print(f"Error deleting audio file: {e}")
        return jsonify({'error': str(e)}), 500
        
    finally:
        cursor.close()





def insert_audio_file_to_db(
    user_id, file_name, file_type, file_size, upload_date, audio_content
):
    try:
        cursor = mysql.connection.cursor()

        # Insert into the audiofiles table
        cursor.execute(
            "INSERT INTO audiofiles (user_id, file_name, file_type, file_size, upload_datetime, audio_content) VALUES (%s, %s, %s, %s, %s, %s)",
            (user_id, file_name, file_type, file_size, upload_date, audio_content),
        )
        mysql.connection.commit()

        # Retrieve the last inserted audio_id
        cursor.execute("SELECT LAST_INSERT_ID()")
        audio_id = cursor.fetchone()[0]
        return audio_id
    except Exception as e:
        print("Error inserting audio file into database:", e)
        raise


import json
from datetime import datetime


def insert_json_file_to_db(audio_file_name, json_file_name, json_content, created_at):
    try:
        cursor = mysql.connection.cursor()

        # Retrieve the audio_id based on the audio file name
        cursor.execute(
            "SELECT audio_id FROM audiofiles WHERE file_name = %s", (audio_file_name,)
        )
        audio_id = cursor.fetchone()

        # Debugging: Print audio file name
        print("Audio file name:", audio_file_name)

        if audio_id:
            audio_id = audio_id[0]

            # Debugging: Print audio ID
            print("Audio ID:", audio_id)

            # Insert into the jsonfiles table
            cursor.execute(
                "INSERT INTO jsonfiles (audio_id, file_name, json_content, created_at) VALUES (%s, %s, %s, %s)",
                (audio_id, json_file_name, json.dumps(json_content), created_at),
            )
            mysql.connection.commit()
        else:
            raise ValueError("Associated audio file not found.")
    except Exception as e:
        print("Error inserting JSON into database:", e)
        raise


def insert_vtt_file_to_db(audio_id, vtt_file_name, vtt_content, created_at):
    try:
        cursor = mysql.connection.cursor()

        # Insert into the vttfiles table
        cursor.execute(
            "INSERT INTO vttfiles (audio_id, file_name, vtt_content, created_at) VALUES (%s, %s, %s, %s)",
            (audio_id, vtt_file_name, vtt_content, created_at),
        )
        mysql.connection.commit()
    except Exception as e:
        print("Error inserting VTT file into database:", e)
        raise


# Define the directory where your app.py is located
base_dir = os.path.dirname(os.path.abspath(__file__))


# Function to delete files recursively in a directory
def delete_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            # Check file extension and delete if it matches json, vtt, srt, tsv, txt
            if file.endswith((".json", ".vtt", ".srt", ".tsv", ".txt")):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")


# Function to delete audio files specifically
def delete_audio_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            # Check if file is an audio file (you can adjust the condition based on your audio file types)
            if file.endswith((".mp3", ".wav", ".ogg")):
                os.remove(file_path)
                print(f"Deleted audio file: {file_path}")


import traceback  # Add this import


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    form = UploadFileForm()
    submitted = False

    if form.validate_on_submit():
        file = form.file.data

        if file and allowed_file(file.filename):
            main_directory = os.path.abspath(os.path.dirname(__file__))
            file_path = os.path.join(main_directory, secure_filename(file.filename))
            file.save(file_path)

            file_name = secure_filename(file.filename)
            file_type = file.content_type
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)
            upload_date = datetime.now()
            json_file_name = os.path.splitext(file_name)[0] + ".json"

            file.seek(0)
            audio_content = file.read()

            try:
                # Recognize audio | output = JSON, etc.
                sentence_recognizer.recognize(file_name)

                # Read the generated JSON file
                with open(
                    os.path.join(main_directory, json_file_name), "r"
                ) as json_file:
                    json_content = json.load(json_file)

                # Highlight stress word | output = VTT
                vtt_file_name = "stress_closed_caption.vtt"
                audio_file = file_name

                recognizer = SentenceRecognizer(audio_file, json_file_name)
                recognizer.generate_vtt(vtt_file_name)

                # Read the VTT file content
                with open(vtt_file_name, "r") as vtt_file:
                    vtt_content = vtt_file.read()

                # Insert audio file into DB and get audio_id
                audio_id = insert_audio_file_to_db(
                    current_user.id,
                    file_name,
                    file_type,
                    file_size,
                    upload_date,
                    audio_content,
                )

                # Insert JSON file into DB
                insert_json_file_to_db(
                    file_name, json_file_name, json_content, upload_date
                )

                # Insert VTT file into DB
                insert_vtt_file_to_db(audio_id, vtt_file_name, vtt_content, upload_date)

                # Clear files
                delete_files(base_dir)  # Delete json, vtt, srt, tsv, txt files
                delete_audio_files(base_dir)  # Delete audio files
                submitted = True
                return redirect(url_for("edit", audio_id=audio_id))

            except Exception as err:
                print("Error:", err)
                traceback.print_exc()  # This will print the full traceback to the console
                session["message"] = (
                    "An error occurred while processing the request. Please try again."
                )
                return redirect(url_for("upload"))

        else:
            session["message"] = (
                "Only audio files are allowed (extensions: .wav, .mp3, .ogg, .flac)"
            )
            return redirect(url_for("upload"))

    return render_template("upload.html", form=form, submitted=submitted)


@app.route("/your_files", methods=["GET"])
@login_required
def your_files():
    page = request.args.get('page', 1, type=int)
    per_page = 6  # Number of items per page
    
    cursor = mysql.connection.cursor()
    
    # Get all files but ordered
    cursor.execute(
        """SELECT audio_id, user_id, file_name, file_type, file_size, upload_datetime 
           FROM audiofiles 
           WHERE user_id = %s 
           ORDER BY upload_datetime DESC""",
        (current_user.id,)
    )
    audio_files = cursor.fetchall()

    if not audio_files:
        flash("No audio files found.", "warning")

    return render_template(
        "your_files.html", 
        audio_files=audio_files,
        page=page,
        per_page=per_page
    )


if __name__ == "__main__":
    app.run(debug=True)
