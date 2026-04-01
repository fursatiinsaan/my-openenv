TASKS = sorted([
    {
        "id": 0,
        "title": "Customer Identity Login Service",
        "domain": "backend",
        "difficulty": "hard",
        "objective": "Review a customer authentication flow that mixes login, session creation, and audit logging.",
        "tags": ["security", "auth", "database"],
        "code": """
import sqlite3
import time


class AuditLogger:
    def __init__(self, path):
        self.path = path

    def log(self, event_type, message):
        handle = open(self.path, "a")
        handle.write(f"{time.time()}::{event_type}::{message}\\n")


class SessionStore:
    def __init__(self):
        self.active_sessions = {}

    def create_session(self, user_id):
        token = f"session-{user_id}-{int(time.time())}"
        self.active_sessions[token] = {"user_id": user_id, "created_at": time.time()}
        return token


class UserService:
    def __init__(self):
        self.conn = sqlite3.connect("users.db")
        self.logger = AuditLogger("auth.log")
        self.sessions = SessionStore()

    def normalize_username(self, username):
        return username.strip()

    def login(self, username, password, ip_address):
        username = self.normalize_username(username)
        query = (
            f"SELECT id, role FROM users "
            f"WHERE username='{username}' AND password='{password}'"
        )
        cursor = self.conn.cursor()
        row = cursor.execute(query).fetchone()
        self.logger.log("login_attempt", f"{ip_address}:{username}:{password}")

        if row:
            token = self.sessions.create_session(row[0])
            self.logger.log("login_success", f"user={username}, token={token}")
            return {"ok": True, "role": row[1], "session_token": token}

        return {"ok": False, "message": "invalid credentials"}

    def get_user_data(self, user_id):
        cursor = self.conn.cursor()
        profile = cursor.execute("SELECT * FROM users WHERE id=" + str(user_id))
        return profile.fetchall()

    def close(self):
        pass
""",
        "issues": [
            {
                "label": "sql_injection",
                "keywords": ["sql", "injection", "query"],
                "aliases": ["unsafe query construction", "string-formatted sql"],
                "weight": 2.5,
            },
            {
                "label": "plaintext_password_logging",
                "keywords": ["password", "logging", "plain"],
                "aliases": ["sensitive credentials in logs", "password exposure in audit logs"],
                "weight": 2.0,
            },
            {
                "label": "connection_not_closed",
                "keywords": ["close", "connection"],
                "aliases": ["database connection leak"],
                "weight": 1.5,
            },
            {
                "label": "no_input_validation",
                "keywords": ["validation", "sanitize"],
                "aliases": ["missing input validation", "unvalidated user input"],
                "weight": 1.5,
            },
        ],
    },
    {
        "id": 9,
        "title": "CSV Import Preview Tool",
        "domain": "data",
        "difficulty": "easy",
        "objective": "Inspect a small CSV preview utility used before datasets are uploaded into a larger pipeline.",
        "tags": ["files", "data-ingest", "basics"],
        "code": """
def preview_rows(path):
    handle = open(path)
    lines = handle.readlines()

    rows = []
    for line in lines[:5]:
        rows.append(line.strip().split(","))

    return rows


def summarize_preview(path):
    rows = preview_rows(path)
    return {"preview_count": len(rows), "first_row": rows[0] if rows else []}
""",
        "issues": [
            {
                "label": "file_not_closed",
                "keywords": ["close", "resource"],
                "aliases": ["open file leak", "file handle leak"],
                "weight": 1.2,
            },
        ],
    },
    {
        "id": 7,
        "title": "Enterprise Dataset Export Service",
        "domain": "data-platform",
        "difficulty": "extreme",
        "objective": "Analyze a service that exports enterprise datasets and records who downloaded what.",
        "tags": ["privacy", "exports", "authorization"],
        "code": """
import csv


class ExportService:
    def __init__(self, audit_path):
        self.audit_path = audit_path

    def export_rows(self, rows, user_email, include_internal_notes=False):
        with open("dataset_export.csv", "w") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "text", "label", "notes"])
            for row in rows:
                notes = row.get("internal_notes", "") if include_internal_notes else ""
                writer.writerow([row["id"], row["text"], row["label"], notes])

        self.log_download(user_email, len(rows))
        return "dataset_export.csv"

    def log_download(self, user_email, count):
        with open(self.audit_path, "a") as handle:
            handle.write(f"{user_email} downloaded {count} rows\\n")


def export_for_customer(service, rows, requester):
    if requester.get("tier") == "enterprise":
        return service.export_rows(rows, requester["email"], include_internal_notes=True)
    return service.export_rows(rows, requester["email"])
""",
        "issues": [
            {
                "label": "hardcoded_export_path",
                "keywords": ["path", "dataset_export"],
                "aliases": ["fixed export filename", "hardcoded csv path"],
                "weight": 1.2,
            },
            {
                "label": "pii_logging",
                "keywords": ["email", "pii", "logging"],
                "aliases": ["personal data in logs", "sensitive user data logging"],
                "weight": 1.8,
            },
            {
                "label": "missing_access_control",
                "keywords": ["access", "authorization", "permission"],
                "aliases": ["authorization missing", "no permission checks"],
                "weight": 2.2,
            },
        ],
    },
    {
        "id": 2,
        "title": "Experiment Training Pipeline",
        "domain": "ml",
        "difficulty": "very hard",
        "objective": "Review an internal training pipeline that loads CSV-like samples and performs a custom gradient loop.",
        "tags": ["training", "ml", "reliability"],
        "code": """
import numpy as np


class ExperimentTracker:
    def __init__(self):
        self.history = []

    def log_epoch(self, epoch, loss):
        self.history.append({"epoch": epoch, "loss": loss})


class Model:
    def __init__(self):
        self.weights = None
        self.loss_history = []
        self.tracker = ExperimentTracker()

    def initialize(self, width):
        self.weights = np.random.rand(width)

    def train(self, data):
        self.initialize(len(data[0]))

        for epoch in range(1000):
            total_loss = 0
            for row in data:
                features = np.array(row)
                prediction = np.dot(self.weights, features)
                error = prediction - row[-1]
                total_loss += error ** 2
                self.weights -= 0.01 * error * features

            self.loss_history.append(total_loss)
            self.tracker.log_epoch(epoch, total_loss)

    def predict(self, x):
        return np.dot(self.weights, x)


def load_data(path):
    handle = open(path)
    rows = handle.readlines()
    return [list(map(float, row.split(","))) for row in rows]


def summarize_training(model):
    return {
        "epochs": len(model.loss_history),
        "final_loss": model.loss_history[-1],
        "first_loss": model.loss_history[0],
    }


def run_training(path):
    dataset = load_data(path)
    model = Model()
    model.train(dataset)
    return summarize_training(model)
""",
        "issues": [
            {
                "label": "no_data_validation",
                "keywords": ["validation", "check"],
                "aliases": ["missing dataset validation", "unsafe training input assumptions"],
                "weight": 1.5,
            },
            {
                "label": "file_not_closed",
                "keywords": ["close", "resource"],
                "aliases": ["open file leak", "file handle leak"],
                "weight": 1.5,
            },
            {
                "label": "inefficient_training",
                "keywords": ["loop", "epoch"],
                "aliases": ["costly nested training loop", "unoptimized epoch loop"],
                "weight": 1.2,
            },
            {
                "label": "no_exception_handling",
                "keywords": ["exception", "try"],
                "aliases": ["missing error handling", "no failure handling"],
                "weight": 1.5,
            },
        ],
    },
    {
        "id": 6,
        "title": "GPU Training Job Orchestrator",
        "domain": "orchestration",
        "difficulty": "extreme",
        "objective": "Review a job scheduler that builds shell commands from user-supplied training configs.",
        "tags": ["jobs", "shell", "platform"],
        "code": """
import json
import subprocess


def load_config(config_text):
    return json.loads(config_text)


def build_command(config):
    command = "python train.py"
    command += " --epochs " + str(config.get("epochs", 1))
    command += " --lr " + str(config.get("lr", 0.001))
    command += " --dataset " + config["dataset"]
    command += " --output " + config.get("output_dir", "/tmp/run")
    command += " --run-name " + config.get("run_name", "nightly")
    return command


def submit_job(config_text):
    config = load_config(config_text)
    command = build_command(config)
    return subprocess.check_output(command, shell=True).decode()


def queue_job(config_text, owner):
    payload = load_config(config_text)
    payload["owner"] = owner
    return submit_job(json.dumps(payload))
""",
        "issues": [
            {
                "label": "shell_injection",
                "keywords": ["shell", "injection", "subprocess"],
                "aliases": ["command injection", "unsafe subprocess shell true"],
                "weight": 2.5,
            },
            {
                "label": "missing_config_validation",
                "keywords": ["validation", "config"],
                "aliases": ["unvalidated training config", "missing schema validation"],
                "weight": 1.5,
            },
            {
                "label": "hardcoded_output_path",
                "keywords": ["path", "output"],
                "aliases": ["fixed output directory", "hardcoded /tmp path"],
                "weight": 1.2,
            },
        ],
    },
    {
        "id": 5,
        "title": "Online Feature Store Sync",
        "domain": "ml-platform",
        "difficulty": "extreme",
        "objective": "Inspect a feature-store sync utility that reads remote feature vectors and caches them locally.",
        "tags": ["networking", "cache", "feature-store"],
        "code": """
import json
import requests


class FeatureStore:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.cache = {}
        self.audit_events = []

    def record_event(self, event_type, payload):
        self.audit_events.append({"event_type": event_type, "payload": payload})

    def load_user(self, user_id):
        if user_id in self.cache:
            self.record_event("cache_hit", user_id)
            return self.cache[user_id]

        response = requests.get(self.endpoint + "/features/" + user_id, verify=False)
        payload = response.json()
        self.cache[user_id] = payload
        self.record_event("cache_miss", {"user_id": user_id, "size": len(str(payload))})
        return payload

    def save_snapshot(self, path):
        with open(path, "w") as handle:
            handle.write(json.dumps({"cache": self.cache, "events": self.audit_events}))


def refresh_all(store, ids):
    results = []
    for user_id in ids:
        results.append(store.load_user(user_id))
    return results
""",
        "issues": [
            {
                "label": "tls_verification_disabled",
                "keywords": ["verify", "tls", "ssl"],
                "aliases": ["certificate verification disabled", "requests verify false"],
                "weight": 2.0,
            },
            {
                "label": "no_timeout_on_network_call",
                "keywords": ["timeout", "requests"],
                "aliases": ["missing request timeout", "unbounded network wait"],
                "weight": 1.5,
            },
            {
                "label": "unbounded_cache_growth",
                "keywords": ["cache", "memory"],
                "aliases": ["cache can grow forever", "no cache eviction"],
                "weight": 1.2,
            },
        ],
    },
    {
        "id": 3,
        "title": "Partner File Upload Gateway",
        "domain": "web",
        "difficulty": "very hard",
        "objective": "Analyze a Flask upload endpoint used by external partners to send batch files and run support commands.",
        "tags": ["uploads", "api", "security"],
        "code": """
from flask import Flask, request
import os

app = Flask(__name__)


def build_destination(team_name, filename):
    return "uploads/" + team_name + "/" + filename


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    filename = file.filename
    team_name = request.form.get("team", "default")
    upload_path = build_destination(team_name, filename)

    if not os.path.exists("uploads/" + team_name):
        os.makedirs("uploads/" + team_name)

    file.save(upload_path)
    return {"saved_to": upload_path, "size": len(file.read())}


@app.route("/preview")
def preview():
    path = request.args.get("path")
    return open(path).read()


@app.route("/exec")
def run():
    cmd = request.args.get("cmd")
    output = os.popen(cmd).read()
    return {"output": output}
""",
        "issues": [
            {
                "label": "file_upload_vulnerability",
                "keywords": ["upload", "validation"],
                "aliases": ["unsafe file upload", "missing file type validation"],
                "weight": 2.0,
            },
            {
                "label": "path_traversal",
                "keywords": ["filename", "path"],
                "aliases": ["directory traversal", "unsafe path join"],
                "weight": 2.0,
            },
            {
                "label": "remote_code_execution",
                "keywords": ["exec", "command"],
                "aliases": ["arbitrary command execution", "unsafe os.popen"],
                "weight": 3.0,
            },
        ],
    },
    {
        "id": 8,
        "title": "Profile Settings API",
        "domain": "backend",
        "difficulty": "easy",
        "objective": "Review a lightweight profile update handler used by an internal settings page.",
        "tags": ["profiles", "validation", "pii"],
        "code": """
def update_profile(user, payload):
    profile = {
        "display_name": payload.get("display_name", ""),
        "email": payload.get("email", ""),
    }

    if payload.get("timezone"):
        profile["timezone"] = payload["timezone"]

    print("updating profile for", user["email"])
    return profile


def save_profile(store, user, payload):
    profile = update_profile(user, payload)
    store[user["id"]] = profile
    return {"saved": True, "profile": profile}
""",
        "issues": [
            {
                "label": "pii_logging",
                "keywords": ["email", "logging", "pii"],
                "aliases": ["personal data in logs", "sensitive user data logging"],
                "weight": 1.4,
            },
            {
                "label": "missing_email_validation",
                "keywords": ["validation", "email"],
                "aliases": ["unvalidated email input", "missing input validation"],
                "weight": 1.1,
            },
        ],
    },
    {
        "id": 1,
        "title": "Production Log Retention Worker",
        "domain": "systems",
        "difficulty": "hard",
        "objective": "Inspect a retention worker that parses production logs, archives incident files, and deletes local artifacts.",
        "tags": ["ops", "logs", "shell"],
        "code": """
import os
from datetime import datetime


def parse_line(line):
    parts = line.strip().split("|")
    if len(parts) < 3:
        return {"level": "UNKNOWN", "service": "unknown", "message": line.strip()}
    return {"level": parts[0], "service": parts[1], "message": parts[2]}


def process_logs(file_path, archive_dir):
    handle = open(file_path)
    lines = handle.readlines()
    summary = {"errors": 0, "warnings": 0, "services": set()}

    for index in range(len(lines)):
        entry = parse_line(lines[index])
        summary["services"].add(entry["service"])

        if entry["level"] == "ERROR":
            print(datetime.utcnow(), entry["service"], entry["message"])
            summary["errors"] += 1
        if entry["level"] == "WARN":
            summary["warnings"] += 1

    if summary["errors"] > 100:
        os.system("cp " + file_path + " " + archive_dir)

    summary["services"] = sorted(summary["services"])
    return summary


def delete_temp_copy(path):
    os.system("rm " + path)


def run_retention(file_path, archive_dir, cleanup_path):
    report = process_logs(file_path, archive_dir)
    if report["errors"] == 0:
        delete_temp_copy(cleanup_path)
    return report
""",
        "issues": [
            {
                "label": "file_not_closed",
                "keywords": ["close", "resource"],
                "aliases": ["open file leak", "file handle leak"],
                "weight": 1.5,
            },
            {
                "label": "inefficient_loop",
                "keywords": ["range", "loop"],
                "aliases": ["index-based iteration", "manual indexing"],
                "weight": 1.2,
            },
            {
                "label": "command_injection",
                "keywords": ["system", "injection"],
                "aliases": ["shell injection", "unsafe shell command"],
                "weight": 2.5,
            },
        ],
    },
    {
        "id": 4,
        "title": "Revenue Analytics Batch Pipeline",
        "domain": "data",
        "difficulty": "extreme",
        "objective": "Review a pandas-based revenue pipeline that computes flags and writes analyst-facing exports.",
        "tags": ["analytics", "pandas", "batch"],
        "code": """
import pandas as pd


def enrich_defaults(df):
    df["value"] = df["value"].fillna(0)
    df["region"] = df["region"].fillna("unknown")
    return df


def process_data(df):
    df = enrich_defaults(df)

    for index in range(len(df)):
        if df.iloc[index]["value"] > 100:
            df.iloc[index]["flag"] = True
        else:
            df.iloc[index]["flag"] = False

        if df.iloc[index]["region"] == "enterprise":
            df.iloc[index]["priority"] = "high"
        else:
            df.iloc[index]["priority"] = "normal"

    return df


def save(df):
    df.to_csv("output.csv")


def load(path):
    return pd.read_csv(path)


def run_pipeline(path):
    data = load(path)
    processed = process_data(data)
    save(processed)
    return {
        "rows": len(processed),
        "high_priority": int((processed["priority"] == "high").sum()),
    }
""",
        "issues": [
            {
                "label": "inefficient_dataframe_loop",
                "keywords": ["loop", "iloc"],
                "aliases": ["row-by-row dataframe mutation", "non-vectorized pandas logic"],
                "weight": 2.0,
            },
            {
                "label": "no_error_handling",
                "keywords": ["exception", "try"],
                "aliases": ["missing error handling", "pipeline failures unhandled"],
                "weight": 1.5,
            },
            {
                "label": "hardcoded_path",
                "keywords": ["path", "output"],
                "aliases": ["hardcoded output file", "fixed export path"],
                "weight": 1.2,
            },
        ],
    },
    {
        "id": 10,
        "title": "Welcome Email Formatter",
        "domain": "messaging",
        "difficulty": "easy",
        "objective": "Review a helper that prepares welcome emails before they are sent by a separate worker.",
        "tags": ["messaging", "templates", "privacy"],
        "code": """
def build_email(user):
    subject = "Welcome " + user["name"]
    body = "Hi " + user["name"] + ", thanks for joining."
    return {"subject": subject, "body": body}


def send_preview(user):
    email = build_email(user)
    print("preview email for", user["email"])
    return email
""",
        "issues": [
            {
                "label": "pii_logging",
                "keywords": ["email", "logging", "pii"],
                "aliases": ["personal data in logs", "sensitive user data logging"],
                "weight": 1.2,
            },
        ],
    },
], key=lambda task: task["title"].lower())
