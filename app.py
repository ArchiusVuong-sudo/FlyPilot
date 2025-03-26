import streamlit as st
import os
import docx2txt
import pandas as pd
import io
import tempfile
import openai
import json
import base64
import pdfplumber
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from datetime import datetime
import xml.etree.ElementTree
import hashlib
import pickle
import uuid
import re  # Add this for regex operations
import shutil

# Load environment variables
load_dotenv()

# Set up OpenAI API
openai.api_key = st.session_state.get("openai_api_key", os.getenv("OPENAI_API_KEY"))

# User and team data storage
USER_DB_FILE = "user_credentials.pkl"
TEAM_DB_FILE = "team_data.pkl"
USER_DATA_DIR = "user_data"
TEAM_DATA_DIR = "team_data"
NOTIFICATION_FILE = "notifications.pkl"

# Ensure data directories exist
for directory in [USER_DATA_DIR, TEAM_DATA_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_team_activities(team_id, limit=20):
    """Get the most recent team activities"""
    team_dir = os.path.join(TEAM_DATA_DIR, team_id)
    activity_log_file = os.path.join(team_dir, "activity_log.pkl")
    
    if os.path.exists(activity_log_file):
        with open(activity_log_file, "rb") as f:
            activities = pickle.load(f)
        
        # Return the most recent activities (limited)
        return activities[-limit:] if limit > 0 else activities
    
    return []

# Activity log functions
def add_team_activity(team_id, username, action, item=None):
    """Add an activity to the team's activity log"""
    team_dir = os.path.join(TEAM_DATA_DIR, team_id)
    activity_log_file = os.path.join(team_dir, "activity_log.pkl")
    
    # Load existing activities
    activities = []
    if os.path.exists(activity_log_file):
        with open(activity_log_file, "rb") as f:
            activities = pickle.load(f)
    
    # Add new activity
    activities.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "username": username,
        "action": action,
        "item": item
    })
    
    # Save activities
    with open(activity_log_file, "wb") as f:
        pickle.dump(activities, f)


def generate_document_summary(doc_name, document_text):
    """Generate a summary of the document using OpenAI"""
    try:
        # Limit text to avoid API limits
        text_for_summary = document_text[:5000]  # Take first 5000 chars for summary
        
        # Call OpenAI API for summarization with improved prompt
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert document analyst that creates comprehensive summaries. Your summaries should identify key information, main topics, and potential relationships to other documents. The summary will be used as context for a chatbot to answer user questions, so include important background information and context."},
                {"role": "user", "content": f"Please provide a comprehensive summary of the following document titled '{doc_name}':\n\n{text_for_summary}\n\nInclude key points, main topics, and any important contextual information that would help understand this document's relationship to other potential documents in the collection."}
            ],
            temperature=0.5,
            max_tokens=500  # Increased token limit for more comprehensive summaries
        )
        
        summary = response.choices[0].message.content
        return summary
    
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def get_chat_download_link(content, filename, link_text):
    """Generate a download link for chat content"""
    # Encode the content as base64
    b64 = base64.b64encode(content.encode()).decode()
    
    # Create the download link
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

# Load or create user database
def load_user_db():
    if os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def extract_text_from_file(uploaded_file):
    """Extract text content from various file types"""
    try:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        
        if file_extension == "pdf":
            # Extract text from PDF
            with pdfplumber.open(uploaded_file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                return text
        
        elif file_extension in ["docx", "doc"]:
            # Extract text from Word document
            text = docx2txt.process(uploaded_file)
            return text
        
        elif file_extension in ["xlsx", "xls"]:
            # Extract text from Excel file
            df = pd.read_excel(uploaded_file)
            return df.to_string()
        
        elif file_extension == "xml":
            # Parse XML file
            tree = xml.etree.ElementTree.parse(uploaded_file)
            root = tree.getroot()
            
            # Convert XML to string representation
            text = ""
            for element in root.iter():
                if element.text and element.text.strip():
                    text += f"{element.tag}: {element.text.strip()}\n"
            return text
        
        else:
            return f"Error: Unsupported file format: {file_extension}"
    
    except Exception as e:
        return f"Error: {str(e)}"


def search_in_documents(search_query):
    """Search for text in all documents"""
    results = {}
    
    for doc_name, doc_info in st.session_state.document_contents.items():
        content = doc_info["content"]
        search_query_lower = search_query.lower()
        
        if search_query_lower in content.lower():
            # Find all occurrences
            matches = []
            start_pos = 0
            content_lower = content.lower()
            
            while True:
                pos = content_lower.find(search_query_lower, start_pos)
                if pos == -1:
                    break
                
                # Get context around the match
                context_start = max(0, pos - 100)
                context_end = min(len(content), pos + len(search_query) + 100)
                context = content[context_start:context_end]
                
                # Highlight the match in context (could improve this with better formatting)
                match_in_context = content[pos:pos+len(search_query)]
                
                matches.append({
                    "match": match_in_context,
                    "context": context,
                    "position": pos
                })
                
                start_pos = pos + 1
            
            if matches:
                results[doc_name] = matches
    
    return results


def save_user_db(user_db):
    with open(USER_DB_FILE, "wb") as f:
        pickle.dump(user_db, f)

# Load or create team database
def load_team_db():
    if os.path.exists(TEAM_DB_FILE):
        with open(TEAM_DB_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_team_db(team_db):
    with open(TEAM_DB_FILE, "wb") as f:
        pickle.dump(team_db, f)

# User authentication
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate(username, password):
    user_db = load_user_db()
    if username in user_db and user_db[username]["password"] == hash_password(password):
        return True
    return False

# Team Management Functions
def create_team(team_name, owner_username):
    """Create a new team with the given owner."""
    team_db = load_team_db()
    user_db = load_user_db()
    
    # Generate a unique team ID
    team_id = str(uuid.uuid4())
    
    # Create team entry
    team_db[team_id] = {
        "name": team_name,
        "owner": owner_username,
        "members": [owner_username],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "documents": {},
        "folders": {"root": []}
    }
    
    # Update user record to include team
    if "teams" not in user_db[owner_username]:
        user_db[owner_username]["teams"] = []
    
    user_db[owner_username]["teams"].append(team_id)
    
    # Create team directory
    team_dir = os.path.join(TEAM_DATA_DIR, team_id)
    if not os.path.exists(team_dir):
        os.makedirs(team_dir)
    
    save_team_db(team_db)
    save_user_db(user_db)
    return team_id

def add_member_to_team(team_id, username):
    """Add a user to a team."""
    team_db = load_team_db()
    user_db = load_user_db()
    
    if team_id not in team_db or username not in user_db:
        return False
    
    if username in team_db[team_id]["members"]:
        return True  # Already a member
    
    # Add user to team
    team_db[team_id]["members"].append(username)
    
    # Update user record
    if "teams" not in user_db[username]:
        user_db[username]["teams"] = []
    
    if team_id not in user_db[username]["teams"]:
        user_db[username]["teams"].append(team_id)
    
    # Send notification to the added user
    team_name = team_db[team_id]["name"]
    inviter = st.session_state.username
    notification_message = f"{inviter} added you to the team '{team_name}'"
    add_notification(username, notification_message, "team", team_id)
    
    save_team_db(team_db)
    save_user_db(user_db)
    return True

def remove_member_from_team(team_id, username):
    """Remove a user from a team."""
    team_db = load_team_db()
    user_db = load_user_db()
    
    if team_id not in team_db or username not in user_db:
        return False
    
    # Cannot remove the owner
    if team_db[team_id]["owner"] == username:
        return False
    
    # Remove user from team
    if username in team_db[team_id]["members"]:
        team_db[team_id]["members"].remove(username)
    
    # Update user record
    if "teams" in user_db[username] and team_id in user_db[username]["teams"]:
        user_db[username]["teams"].remove(team_id)
    
    # Send notification to the removed user
    team_name = team_db[team_id]["name"]
    remover = st.session_state.username
    notification_message = f"{remover} removed you from the team '{team_name}'"
    add_notification(username, notification_message, "system", None)
    
    save_team_db(team_db)
    save_user_db(user_db)
    return True

def leave_team(team_id, username):
    """Allow a user to leave a team."""
    team_db = load_team_db()
    user_db = load_user_db()
    
    if team_id not in team_db or username not in user_db:
        return False, "Team or user not found"
    
    # Prevent owner from leaving
    if team_db[team_id]["owner"] == username:
        return False, "Team owners cannot leave their teams. You must either delete the team or transfer ownership first."
    
    # Remove user from team
    if username in team_db[team_id]["members"]:
        team_db[team_id]["members"].remove(username)
    
    # Update user record
    if "teams" in user_db[username] and team_id in user_db[username]["teams"]:
        user_db[username]["teams"].remove(team_id)
    
    # Notify the team owner
    team_name = team_db[team_id]["name"]
    owner = team_db[team_id]["owner"]
    notification_message = f"{username} has left your team '{team_name}'"
    add_notification(owner, notification_message, "team", team_id)
    
    # Log activity
    add_team_activity(team_id, username, "left team")
    
    save_team_db(team_db)
    save_user_db(user_db)
    return True, f"You have left the team '{team_name}'."

def delete_team(team_id, username):
    """Delete a team. Only the team owner can delete a team."""
    team_db = load_team_db()
    user_db = load_user_db()
    
    # Verify team exists
    if team_id not in team_db:
        return False, "Team not found"
    
    # Verify user is the team owner
    if team_db[team_id]["owner"] != username:
        return False, "Only the team owner can delete a team"
    
    team_name = team_db[team_id]["name"]
    team_members = team_db[team_id]["members"].copy()  # Create a copy to avoid modification during iteration
    
    # Remove team from all members' records
    for member in team_members:
        if member in user_db and "teams" in user_db[member] and team_id in user_db[member]["teams"]:
            user_db[member]["teams"].remove(team_id)
            
            # Send notification to team members
            if member != username:  # Don't notify the owner who's deleting
                notification_message = f"Team '{team_name}' has been deleted by {username}"
                add_notification(member, notification_message, "system", None)
    
    # Delete team from database
    del team_db[team_id]
    
    # Delete team data directory
    team_dir = os.path.join(TEAM_DATA_DIR, team_id)
    if os.path.exists(team_dir):
        try:
            shutil.rmtree(team_dir)
        except Exception as e:
            # Continue even if file deletion fails
            print(f"Warning: Could not delete team directory: {e}")
    
    save_team_db(team_db)
    save_user_db(user_db)
    return True, f"Team '{team_name}' has been successfully deleted"

def get_user_teams(username):
    """Get all teams that a user belongs to."""
    user_db = load_user_db()
    team_db = load_team_db()
    
    if username not in user_db or "teams" not in user_db[username]:
        return []
    
    teams = []
    for team_id in user_db[username]["teams"]:
        if team_id in team_db:
            teams.append({
                "id": team_id,
                "name": team_db[team_id]["name"],
                "owner": team_db[team_id]["owner"],
                "member_count": len(team_db[team_id]["members"]),
                "is_owner": team_db[team_id]["owner"] == username
            })
    
    return teams

# Load team data
def load_team_data(team_id):
    """Load team document data."""
    team_data_file = os.path.join(TEAM_DATA_DIR, team_id, "team_data.pkl")
    if os.path.exists(team_data_file):
        with open(team_data_file, "rb") as f:
            return pickle.load(f)
    return {
        "document_contents": {},
        "document_summaries": {},
        "folders": {"root": []},
        "current_folder": "root"
    }

def save_team_data(team_id, team_data):
    """Save team document data."""
    team_dir = os.path.join(TEAM_DATA_DIR, team_id)
    if not os.path.exists(team_dir):
        os.makedirs(team_dir)
    
    team_data_file = os.path.join(team_dir, "team_data.pkl")
    with open(team_data_file, "wb") as f:
        pickle.dump(team_data, f)

# Load user session data
def load_user_session(user_id):
    user_data_file = os.path.join(USER_DATA_DIR, user_id, "session_data.pkl")
    if os.path.exists(user_data_file):
        with open(user_data_file, "rb") as f:
            return pickle.load(f)
    return {
        "messages": [],
        "document_contents": {},
        "chat_history": [],
        "current_chat_id": None,
        "document_summaries": {},
        "custom_prompt": "",
        "search_results": {},
        "favorites": [],
        "dark_mode": False,
        "message_filter": "all",
        "folders": {"root": []},
        "current_folder": "root",
        "current_team_id": None
    }

# Save user session data
def save_user_session(user_id):
    # Skip if user_id is None
    if user_id is None:
        return
        
    user_dir = os.path.join(USER_DATA_DIR, user_id)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    
    user_data_file = os.path.join(user_dir, "session_data.pkl")
    
    session_data = {
        "messages": st.session_state.messages,
        "document_contents": st.session_state.document_contents,
        "chat_history": st.session_state.chat_history,
        "current_chat_id": st.session_state.current_chat_id,
        "document_summaries": st.session_state.document_summaries,
        "custom_prompt": st.session_state.custom_prompt,
        "search_results": st.session_state.search_results,
        "favorites": st.session_state.favorites,
        "dark_mode": st.session_state.dark_mode,
        "message_filter": st.session_state.message_filter,
        "folders": st.session_state.folders,
        "current_folder": st.session_state.current_folder,
        "current_team_id": st.session_state.get("current_team_id", None)
    }
    
    with open(user_data_file, "wb") as f:
        pickle.dump(session_data, f)

# Initialize notification system
def load_notifications():
    if os.path.exists(NOTIFICATION_FILE):
        with open(NOTIFICATION_FILE, 'rb') as f:
            return pickle.load(f)
    return {}  # Format: {username: [notification1, notification2, ...]}

def save_notifications(notifications):
    with open(NOTIFICATION_FILE, 'wb') as f:
        pickle.dump(notifications, f)

def add_notification(username, message, source_type, source_id):
    notifications = load_notifications()
    if username not in notifications:
        notifications[username] = []
    
    # Add notification with timestamp
    notifications[username].append({
        "message": message,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "read": False,
        "source_type": source_type,  # "team" or "system"
        "source_id": source_id       # team_id if related to team
    })
    
    save_notifications(notifications)

def mark_notification_read(username, index):
    notifications = load_notifications()
    if username in notifications and 0 <= index < len(notifications[username]):
        notifications[username][index]["read"] = True
        save_notifications(notifications)
        return True
    return False

def get_user_notifications(username):
    notifications = load_notifications()
    return notifications.get(username, [])

def clear_user_notifications(username):
    notifications = load_notifications()
    if username in notifications:
        notifications[username] = []
        save_notifications(notifications)
        return True
    return False

# Clear user notifications
def clear_user_notifications(username):
    notifications = load_notifications()
    if username in notifications:
        notifications[username] = []
        save_notifications(notifications)
        return True
    return False

# System message creation
def create_system_message():
    """Create a system message for the AI based on loaded documents and custom prompt."""
    
    # Start with base instructions
    base_message = "あなたは文書分析を支援するAIアシスタントです。回答には常に関連する文書情報を参照し、コンテキストと背景を提供してください。"
    
    # Add info about loaded documents
    if st.session_state.document_contents:
        doc_names = list(st.session_state.document_contents.keys())
        base_message += f"{len(doc_names)}つの文書にアクセスできます: {', '.join(doc_names)}。"
        
        # Add document summaries if available
        if st.session_state.document_summaries:
            base_message += "以下は文書の包括的な要約です。これらを使用して文書間の関係を理解し、回答に豊かなコンテキストを提供してください:\n\n"
            for doc_name, summary in st.session_state.document_summaries.items():
                base_message += f"--- 文書: {doc_name} ---\n{summary}\n\n"
    else:
        base_message += "まだ文書がアップロードされていません。"
    
    # Strong instruction to always reference document information
    base_message += "\n重要: 回答には常に文書要約から関連情報を組み込んでください。質問に答える場合は、特定の文書を引用し、適切な場合は文書間の関係を説明してください。文書要約に関連するコンテンツが存在する場合は、「わかりません」や「情報がありません」などとは決して応答しないでください。"
    
    # Add custom instructions if set
    if st.session_state.custom_prompt:
        base_message += f"\n追加指示: {st.session_state.custom_prompt}"
    
    return base_message

# Save current state to appropriate location based on view mode
def save_current_state():
    if st.session_state.view_mode == "personal":
        save_user_session(st.session_state.user_id)
    elif st.session_state.view_mode == "team" and st.session_state.current_team_id:
        team_data = {
            "document_contents": st.session_state.document_contents,
            "document_summaries": st.session_state.document_summaries,
            "folders": st.session_state.folders,
            "current_folder": st.session_state.current_folder
        }
        save_team_data(st.session_state.current_team_id, team_data)

# App title and configuration
st.set_page_config(page_title="FlyPilot - Welcome", layout="wide")

# Display welcome message at the top if user is authenticated
if "username" in st.session_state and st.session_state.username:
    title_text = f"FlyPilot - Welcome, {st.session_state.username}!"
    if "view_mode" in st.session_state and st.session_state.view_mode == "team" and "current_team_id" in st.session_state and st.session_state.current_team_id:
        team_db = load_team_db()
        team_name = team_db.get(st.session_state.current_team_id, {}).get("name", "Unknown Team")
        title_text += f" (Team: {team_name})"
    st.title(title_text)
else:
    # Display welcome message for unauthenticated users
    st.title("Welcome to FlyPilot")
    st.write("Please login or register using the sidebar to get started.")
    
    # Add a brief description or features overview
    st.markdown("""
    ## Features:
    - Chat with your documents
    - Organize and manage document libraries
    - Team collaboration
    - Document analytics and insights
    - AI-powered document analysis
    """)
    
    # Add a call to action
    st.write("Sign in to start analyzing your documents!")

# Initialize session state for authentication
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None
if "current_team_id" not in st.session_state:
    st.session_state.current_team_id = None
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "personal"  # Can be "personal" or "team"
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "team_description" not in st.session_state:
    st.session_state.team_description = ""
    
# Initialize document and user data related variables
if "document_contents" not in st.session_state:
    st.session_state.document_contents = {}
if "document_summaries" not in st.session_state:
    st.session_state.document_summaries = {}
if "folders" not in st.session_state:
    st.session_state.folders = {"root": []}
if "current_folder" not in st.session_state:
    st.session_state.current_folder = "root"
if "custom_prompt" not in st.session_state:
    st.session_state.custom_prompt = ""
if "search_results" not in st.session_state:
    st.session_state.search_results = {}
if "favorites" not in st.session_state:
    st.session_state.favorites = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "message_filter" not in st.session_state:
    st.session_state.message_filter = "all"
if "show_folder_form" not in st.session_state:
    st.session_state.show_folder_form = False
if "show_team_form" not in st.session_state:
    st.session_state.show_team_form = False
if "show_invite_form" not in st.session_state:
    st.session_state.show_invite_form = False

# Login / Registration Page
def login_page():
    st.sidebar.title("FlyPilot - Welcome")
    
    # Use selectbox instead of tabs for better sidebar compatibility
    auth_mode = st.sidebar.selectbox(
        "Choose option:",
        options=["Login", "Register", "Recover Password"],
        key="auth_mode"
    )
    
    if auth_mode == "Login":
        st.sidebar.subheader("Login")
        username = st.sidebar.text_input("Username", key="login_username")
        password = st.sidebar.text_input("Password", type="password", key="login_password")
        
        col1, col2 = st.sidebar.columns([1, 3])
        with col1:
            if st.sidebar.button("Login", key="login_button"):
                if authenticate(username, password):
                    user_db = load_user_db()
                    st.session_state.authenticated = True
                    st.session_state.user_id = user_db[username]["user_id"]
                    st.session_state.username = username
                    
                    # Load user's session data
                    user_session = load_user_session(st.session_state.user_id)
                    for key, value in user_session.items():
                        st.session_state[key] = value
                    
                    st.sidebar.success("Login successful!")
                    st.rerun()
                else:
                    st.sidebar.error("Invalid username or password")
    
    elif auth_mode == "Register":
        st.sidebar.subheader("Register")
        new_username = st.sidebar.text_input("Choose Username", key="reg_username")
        new_password = st.sidebar.text_input("Choose Password", type="password", key="reg_password")
        confirm_password = st.sidebar.text_input("Confirm Password", type="password", key="confirm_password")
        security_question = st.sidebar.selectbox(
            "Security Question", 
            options=[
                "What was your first pet's name?",
                "What city were you born in?",
                "What is your mother's maiden name?",
                "What was the name of your first school?",
                "What is your favorite book?"
            ]
        )
        security_answer = st.sidebar.text_input("Security Answer")
        
        if st.sidebar.button("Register", key="register_button"):
            if new_password != confirm_password:
                st.sidebar.error("Passwords do not match")
            elif len(new_username) < 3:
                st.sidebar.error("Username must be at least 3 characters")
            elif len(new_password) < 6:
                st.sidebar.error("Password must be at least 6 characters")
            elif not security_answer:
                st.sidebar.error("Security answer is required for password recovery")
            else:
                user_db = load_user_db()
                if new_username in user_db:
                    st.sidebar.error("Username already exists")
                else:
                    # Create user entry in database with security question/answer
                    user_db[new_username] = {
                        "password": hash_password(new_password),
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "user_id": str(uuid.uuid4()),
                        "security_question": security_question,
                        "security_answer": hashlib.sha256(security_answer.lower().encode()).hexdigest()
                    }
                    
                    # Create user-specific data directory
                    user_dir = os.path.join(USER_DATA_DIR, user_db[new_username]["user_id"])
                    if not os.path.exists(user_dir):
                        os.makedirs(user_dir)
                    
                    save_user_db(user_db)
                    st.sidebar.success("Registration successful! Please login.")
    
    elif auth_mode == "Recover Password":
        st.sidebar.subheader("Recover Password")
        recovery_username = st.sidebar.text_input("Username", key="recovery_username")
        
        # Check if username exists and show security question
        if recovery_username:
            user_db = load_user_db()
            if recovery_username in user_db and "security_question" in user_db[recovery_username]:
                st.sidebar.write(f"Security Question: {user_db[recovery_username]['security_question']}")
                security_answer = st.sidebar.text_input("Answer", type="password", key="recovery_answer")
                new_password = st.sidebar.text_input("New Password", type="password", key="new_password")
                confirm_new_password = st.sidebar.text_input("Confirm New Password", type="password", key="confirm_new_password")
                
                if st.sidebar.button("Reset Password", key="reset_password_button"):
                    # Verify security answer
                    hashed_answer = hashlib.sha256(security_answer.lower().encode()).hexdigest()
                    if hashed_answer == user_db[recovery_username]["security_answer"]:
                        if new_password != confirm_new_password:
                            st.sidebar.error("Passwords do not match")
                        elif len(new_password) < 6:
                            st.sidebar.error("Password must be at least 6 characters")
                        else:
                            # Update password
                            user_db[recovery_username]["password"] = hash_password(new_password)
                            save_user_db(user_db)
                            st.sidebar.success("Password reset successful! You can now login with your new password.")
                    else:
                        st.sidebar.error("Incorrect security answer")
            else:
                st.sidebar.error("Username not found or recovery information not available")


# Main app function
def main_app():
    # Initialize session state variables
    if "messages" not in st.session_state:
        session_data = load_user_session(st.session_state.user_id)
        
        # Document data
        st.session_state.document_contents = session_data["document_contents"]
        st.session_state.document_summaries = session_data["document_summaries"]
        st.session_state.folders = session_data["folders"]
        st.session_state.current_folder = session_data["current_folder"]
        
        # Chat data
        st.session_state.messages = session_data.get("messages", [])
        st.session_state.chat_history = session_data.get("chat_history", [])
        st.session_state.current_chat_id = session_data.get("current_chat_id")
        
        # Other settings
        st.session_state.custom_prompt = session_data.get("custom_prompt", "")
        st.session_state.search_results = session_data.get("search_results", {})
        st.session_state.favorites = session_data.get("favorites", [])
        st.session_state.dark_mode = session_data.get("dark_mode", False)
        st.session_state.message_filter = session_data.get("message_filter", "all")
        
        # Team related data
        st.session_state.view_mode = session_data.get("view_mode", "personal")
        st.session_state.current_team_id = session_data.get("current_team_id")
        
        # UI state
        st.session_state.show_folder_form = False
        st.session_state.show_team_form = False
        st.session_state.show_invite_form = False
        
        # If view mode is team, load team data
        if st.session_state.view_mode == "team" and st.session_state.current_team_id:
            team_data = load_team_data(st.session_state.current_team_id)
            
            # Update session state with team data
            st.session_state.document_contents = team_data["document_contents"]
            st.session_state.document_summaries = team_data["document_summaries"]
            st.session_state.folders = team_data["folders"]
            st.session_state.current_folder = team_data["current_folder"]

    # Initialize session state variables if they don't exist
    if "show_folder_form" not in st.session_state:
        st.session_state.show_folder_form = False
    if "show_team_form" not in st.session_state:
        st.session_state.show_team_form = False
    if "show_invite_form" not in st.session_state:
        st.session_state.show_invite_form = False

    # Welcome title has been moved to the top of the page

    # Switch between personal and team view modes
    def switch_to_personal_mode():
        st.session_state.view_mode = "personal"
        st.session_state.current_team_id = None
        save_user_session(st.session_state.user_id)
        st.rerun()
    
    def switch_to_team_mode(team_id):
        # Save current personal state if we're switching from personal mode
        if st.session_state.view_mode == "personal":
            save_user_session(st.session_state.user_id)
        
        # Switch to team mode
        st.session_state.view_mode = "team"
        st.session_state.current_team_id = team_id
        
        # Load team data
        team_data = load_team_data(team_id)
        
        # Update session state with team data
        st.session_state.document_contents = team_data["document_contents"]
        st.session_state.document_summaries = team_data["document_summaries"]
        st.session_state.folders = team_data["folders"]
        st.session_state.current_folder = team_data["current_folder"]
        
        save_user_session(st.session_state.user_id)
        st.rerun()

    # Sidebar
    with st.sidebar:
        # User information and logout
        st.write(f"👤 {st.session_state.username}")
        
        # User information and mode selection
        user_teams = get_user_teams(st.session_state.username)
        
        # View mode selector
        st.subheader("Mode Selection")
        
        mode_col1, mode_col2 = st.columns(2)
        with mode_col1:
            if st.button("Personal", 
                        type="primary" if st.session_state.view_mode == "personal" else "secondary",
                        use_container_width=True):
                switch_to_personal_mode()
        
        with mode_col2:
            if user_teams and st.button("Team", 
                                     type="primary" if st.session_state.view_mode == "team" else "secondary",
                                     use_container_width=True):
                if st.session_state.current_team_id is None and user_teams:
                    # Default to first team if none selected
                    switch_to_team_mode(user_teams[0]["id"])
                elif st.session_state.current_team_id:
                    switch_to_team_mode(st.session_state.current_team_id)
        
        if st.session_state.view_mode == "team" and st.session_state.current_team_id:
            team_db = load_team_db()
            current_team = team_db.get(st.session_state.current_team_id, {})
            if current_team:
                st.write(f"**Current Team:** {current_team.get('name', 'Unknown')}")
             
        # Team management section
        if user_teams or True:  # Always show team section
            st.header("Teams")
            
            # Team creation
            if st.button("Create New Team", key="sidebar_create_team_button"):
                st.session_state.show_team_form = True
            
            # Create team form
            if st.session_state.show_team_form:
                with st.form(key="create_team_form", clear_on_submit=True):
                    st.subheader("Create New Team")
                    new_team_name = st.text_input("Team Name", key="new_team_name")
                    team_description = st.text_area("Team Description (optional)", key="team_description_input")
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        submit_button = st.form_submit_button("Create")
                    with col2:
                        cancel_button = st.form_submit_button("Cancel")
                    
                    if submit_button and new_team_name:
                        # Create the team
                        team_id = create_team(new_team_name, st.session_state.username)
                        
                        # Add description to team data if provided
                        if team_description:
                            team_db = load_team_db()
                            if team_id in team_db:
                                team_db[team_id]["description"] = team_description
                                save_team_db(team_db)
                        
                        st.session_state.show_team_form = False
                        st.success(f"Team '{new_team_name}' created!")
                        # Switch to the new team
                        switch_to_team_mode(team_id)
                    
                    elif cancel_button or (submit_button and not new_team_name):
                        st.session_state.show_team_form = False
                        st.rerun()
            
            # Team selection if in team mode
            if st.session_state.view_mode == "team" and user_teams:
                # Display team selector
                team_options = {team["id"]: team["name"] for team in user_teams}
                selected_team = st.selectbox(
                    "Select Team",
                    options=list(team_options.keys()),
                    format_func=lambda x: team_options[x],
                    key="team_selector",
                    index=list(team_options.keys()).index(st.session_state.current_team_id) if st.session_state.current_team_id in team_options else 0
                )
                
                if selected_team != st.session_state.current_team_id:
                    switch_to_team_mode(selected_team)
                
                # Team member management (only for team owners)
                current_team_info = next((t for t in user_teams if t["id"] == st.session_state.current_team_id), None)
                if current_team_info and current_team_info["is_owner"]:
                    st.subheader("Team Management")
                    
                    # Show team members
                    team_db = load_team_db()
                    current_team = team_db.get(st.session_state.current_team_id, {})
                    
                    if current_team:
                        members = current_team.get("members", [])
                        st.write(f"**Members ({len(members)}):**")
                        
                        for member in members:
                            mem_col1, mem_col2 = st.columns([3, 1])
                            with mem_col1:
                                owner_badge = " 👑" if member == current_team["owner"] else ""
                                st.write(f"{member}{owner_badge}")
                            
                            with mem_col2:
                                if member != current_team["owner"] and member != st.session_state.username:
                                    if st.button("Remove", key=f"remove_{member}"):
                                        if remove_member_from_team(st.session_state.current_team_id, member):
                                            st.success(f"Removed {member} from team")
                                            save_current_state()
                                            st.rerun()
                    
                    # Invite members
                    if st.button("Invite Member", key="sidebar_invite_member_button"):
                        st.session_state.show_invite_form = True
                    
                    if st.session_state.show_invite_form:
                        with st.form(key="invite_member_form", clear_on_submit=True):
                            st.subheader("Invite Team Member")
                            username_to_invite = st.text_input("Username", key="invite_username")
                            col1, col2 = st.columns([1, 1])
                            with col1:
                                invite_submit = st.form_submit_button("Invite")
                            with col2:
                                invite_cancel = st.form_submit_button("Cancel")
                            
                            if invite_submit and username_to_invite:
                                user_db = load_user_db()
                                if username_to_invite not in user_db:
                                    st.error(f"User '{username_to_invite}' does not exist")
                                elif add_member_to_team(st.session_state.current_team_id, username_to_invite):
                                    st.success(f"Added {username_to_invite} to team")
                                    st.session_state.show_invite_form = False
                                    save_current_state()
                                    st.rerun()
                            
                            elif invite_cancel:
                                st.session_state.show_invite_form = False
                                st.rerun()
        
        # Document and folder section
        st.header("Document Quick Access")
        
        # Show document and folder count
        doc_count = len(st.session_state.document_contents)
        folder_count = len(st.session_state.folders) - 1  # Exclude root
        st.write(f"📚 {doc_count} document{'s' if doc_count != 1 else ''} in {folder_count + 1} folder{'s' if folder_count != 0 else ''}")
        
        # Context label for team/personal mode
        if st.session_state.view_mode == "team":
            team_db = load_team_db()
            team_name = team_db.get(st.session_state.current_team_id, {}).get("name", "Unknown Team")
            st.info(f"Viewing Team Documents: {team_name}")
        else:
            st.info("Viewing Personal Documents")
            
        # Show folders in sidebar
        if len(st.session_state.folders) > 1:
            st.subheader("Folders")
            for folder_name in st.session_state.folders.keys():
                folder_docs = len(st.session_state.folders[folder_name])
                folder_display = "Root Folder" if folder_name == "root" else folder_name
                if st.button(f"📁 {folder_display} ({folder_docs})", key=f"folder_sidebar_{folder_name}"):
                    st.session_state.current_folder = folder_name
                    save_current_state()
                    st.rerun()
        
        # Chat history section
        st.header("Chat History")
        
        # New chat button
        if st.button("New Chat", key="new_chat_button"):
            chat_id = datetime.now().strftime("%Y%m%d%H%M%S")
            st.session_state.chat_history.append({
                "id": chat_id,
                "name": f"Chat {len(st.session_state.chat_history) + 1}",
                "messages": []
            })
            st.session_state.current_chat_id = chat_id
            st.session_state.messages = []
            save_current_state()
            st.rerun()
        
        # Display chat history
        if st.session_state.chat_history:
            st.text_input("Search chats...", key="chat_search")
            
        for idx, chat in enumerate(st.session_state.chat_history):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                # Highlight current chat
                if st.session_state.current_chat_id == chat["id"]:
                    button_label = f"🔵 {chat['name']}"
                else:
                    button_label = chat["name"]
                    
                if st.button(button_label, key=f"select_chat_{chat['id']}"):
                    st.session_state.current_chat_id = chat["id"]
                    st.session_state.messages = chat["messages"]
                    save_current_state()
                    st.rerun()
            with col2:
                # Rename option
                if st.button("✏️", key=f"rename_{chat['id']}"):
                    chat["name"] = f"Chat {idx+1} (Edited)"
                    save_current_state()
                    st.rerun()
            with col3:
                # Delete option
                if st.button("❌", key=f"delete_chat_{chat['id']}"):
                    st.session_state.chat_history.pop(idx)
                    if st.session_state.current_chat_id == chat["id"]:
                        st.session_state.current_chat_id = None
                        st.session_state.messages = []
                    save_current_state()
                    st.rerun()
        
        # Favorites quick access
        if st.session_state.favorites:
            st.header("Favorites")
            for i, fav in enumerate(st.session_state.favorites[:3]):  # Show only first 3
                truncated = fav["content"][:50] + "..." if len(fav["content"]) > 50 else fav["content"]
                if st.button(f"⭐ {truncated}", key=f"fav_sidebar_{i}_{id(fav)}"):
                    # Create a new chat with this favorite or add to current chat
                    if st.session_state.current_chat_id is None:
                        chat_id = datetime.now().strftime("%Y%m%d%H%M%S")
                        st.session_state.chat_history.append({
                            "id": chat_id,
                            "name": "Chat from favorite",
                            "messages": []
                        })
                        st.session_state.current_chat_id = chat_id
                        st.session_state.messages = []
                    
                    # Add as a message from the appropriate role
                    st.session_state.messages.append(fav)
                    
                    # Update chat history
                    for chat in st.session_state.chat_history:
                        if chat["id"] == st.session_state.current_chat_id:
                            chat["messages"] = st.session_state.messages
                    
                    save_current_state()
                    st.rerun()

# Main layout with tabs - Only show if authenticated
if st.session_state.authenticated:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Chat", "Document Management", "Analytics", "Teams", "Settings"])

    with tab1:
        # Chat interface
        if st.session_state.current_chat_id is not None:
            # Chat header with export and filter options
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.subheader("Chat")
            with col2:
                message_filter = st.selectbox(
                    "Filter messages:",
                    options=["all", "user", "assistant"],
                    format_func=lambda x: {"all": "All Messages", "user": "Your Messages", "assistant": "Assistant Responses"}[x],
                    key="message_filter_select"
                )
                st.session_state.message_filter = message_filter
            with col3:
                if st.session_state.messages:
                    # Create exportable chat content
                    chat_export = ""
                    for msg in st.session_state.messages:
                        role = "You" if msg["role"] == "user" else "Assistant"
                        chat_export += f"{role}: {msg['content']}\n\n"
                    
                    # Create download link
                    chat_filename = f"chat_export_{st.session_state.current_chat_id}.txt"
                    download_link = get_chat_download_link(chat_export, chat_filename, "Export Chat")
                    st.markdown(download_link, unsafe_allow_html=True)
            
            # Create a container for chat messages with fixed height and scrolling
            chat_container = st.container(height=400, border=True)
            
            # Prepare filtered messages
            filtered_messages = st.session_state.messages
            if st.session_state.message_filter != "all":
                filtered_messages = [msg for msg in st.session_state.messages if msg["role"] == st.session_state.message_filter]
            
            # Display chat messages - reversed to show newest at the bottom
            with chat_container:
                # Add dummy element at the top to push content down when full
                st.empty()
                
                # Display messages in reverse order
                for idx, message in enumerate(filtered_messages):
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                        
                        # Add message actions
                        col1, col2 = st.columns([1, 10])
                        with col1:
                            if st.button("⭐", key=f"fav_{message['role']}_{idx}_{id(message)}"):
                                if message not in st.session_state.favorites:
                                    st.session_state.favorites.append(message)
                                    st.success("Added to favorites")
                                    # Save user session after adding to favorites
                                    save_user_session(st.session_state.user_id)
                                    st.rerun()
            
            # Input for new message at the bottom
            prompt = st.chat_input("Ask a question about your documents")
            if prompt:
                # Add user message to chat
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Update chat history immediately
                for chat in st.session_state.chat_history:
                    if chat["id"] == st.session_state.current_chat_id:
                        chat["messages"] = st.session_state.messages
                
                # Store the prompt in session state for processing after rerun
                st.session_state.last_prompt = prompt
                
                # Force a rerun to show the user message in the container
                st.rerun()
                
            # Process the AI response after rerun if there's a last_prompt
            if "last_prompt" in st.session_state:
                prompt = st.session_state.last_prompt
                
                # Clear the last_prompt to avoid reprocessing
                del st.session_state.last_prompt
                
                # Add a message placeholder inside the chat container
                with chat_container:
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        
                        # Check if it's a search query
                        if prompt.lower().startswith("search:"):
                            search_query = prompt[7:].strip()  # Remove 'search:' prefix
                            search_results = search_in_documents(search_query)
                            st.session_state.search_results = search_results
                            
                            # Format search results as response
                            if search_results:
                                search_response = f"**'{search_query}'の検索結果:**\n\n"
                                for doc, matches in search_results.items():
                                    search_response += f"**{doc}内:**\n"
                                    for i, match in enumerate(matches):
                                        search_response += f"- 一致 {i+1}: ...{match['context']}...\n\n"
                            else:
                                search_response = f"アップロードされた文書には'{search_query}'の検索結果は見つかりませんでした。"
                            
                            # Display and update the message
                            message_placeholder.markdown(search_response)
                            
                            # Add response to messages
                            st.session_state.messages.append({"role": "assistant", "content": search_response})
                            
                        else:
                            # Prepare messages for API call
                            system_message = create_system_message()
                            api_messages = [{"role": "system", "content": system_message}]
                            
                            # Add conversation history
                            for message in st.session_state.messages:
                                api_messages.append({"role": message["role"], "content": message["content"]})
                            
                            try:
                                # Call OpenAI API
                                full_response = ""
                                
                                # Stream the response
                                for response in openai.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=api_messages,
                                    stream=True,
                                    temperature=0.5,  # Lower temperature for more focused responses
                                    presence_penalty=0.6,  # Encourage the model to include diverse information
                                    frequency_penalty=0.1,  # Slight penalty for repetition
                                    max_tokens=800  # Allow longer responses to include adequate document context
                                ):
                                    # Extract content from the response
                                    if response.choices[0].delta.content is not None:
                                        content_chunk = response.choices[0].delta.content
                                        full_response += content_chunk
                                        
                                        # Update the message in real-time with a cursor
                                        message_placeholder.markdown(full_response + "▌")
                                
                                # Final update without cursor
                                message_placeholder.markdown(full_response)
                                
                                # Add assistant response to messages
                                st.session_state.messages.append({"role": "assistant", "content": full_response})
                                
                            except Exception as e:
                                # Display error in the message
                                error_message = f"OpenAI APIとの通信エラー: {str(e)}"
                                message_placeholder.error(error_message)
                                
                                # Add error message to chat
                                st.session_state.messages.append({"role": "assistant", "content": error_message})
                
                # Update chat history after response
                for chat in st.session_state.chat_history:
                    if chat["id"] == st.session_state.current_chat_id:
                        chat["messages"] = st.session_state.messages
                
                # Save the updated session
                save_user_session(st.session_state.user_id)
        else:
            # No chat selected, show welcome message
            st.info("👈 Create a new chat or select an existing chat from the sidebar to get started")
            
            if not st.session_state.chat_history and st.button("Start New Chat", key="start_new_chat_button"):
                chat_id = datetime.now().strftime("%Y%m%d%H%M%S")
                st.session_state.chat_history.append({
                    "id": chat_id,
                    "name": "Chat 1",
                    "messages": []
                })
                st.session_state.current_chat_id = chat_id
                # Save user session after creating new chat
                save_user_session(st.session_state.user_id)
                st.rerun()

    with tab2:
        # Document Management Tab Organization
        # Main container for document management
        doc_container = st.container()
        with doc_container:
            # Two column layout for main sections
            doc_col1, doc_col2 = st.columns([3, 2])
            
            with doc_col1:
                # Document Library Section
                st.subheader("📚 Document Library")
                
                # Show current folder path and document count
                current_folder_display = "Root Folder" if st.session_state.current_folder == "root" else st.session_state.current_folder
                folder_docs = st.session_state.folders.get(st.session_state.current_folder, [])
                st.write(f"**Current Location:** {current_folder_display} ({len(folder_docs)} documents)")
                
                # Document list with actions
                if st.session_state.document_contents:
                    # Filter documents by current folder
                    folder_documents = [doc for doc in st.session_state.document_contents.keys() 
                                      if st.session_state.document_contents[doc].get("folder", "root") == st.session_state.current_folder]
                    
                    if folder_documents:
                        # Display documents in the current folder
                        st.write("**Documents in this folder:**")
                        for doc_name in folder_documents:
                            doc_info = st.session_state.document_contents[doc_name]
                            
                            with st.container(border=True):
                                doc_action_col1, doc_action_col2 = st.columns([4, 1])
                                with doc_action_col1:
                                    st.write(f"**{doc_name}**")
                                    st.caption(f"Uploaded: {doc_info.get('upload_time', 'Unknown')}")
                                    if "user_context" in doc_info and doc_info["user_context"]:
                                        st.info(doc_info["user_context"])
                                
                                with doc_action_col2:
                                    # Removed View button, keeping only Delete
                                    st.button("Delete", key=f"delete_{doc_name}")
                    else:
                        st.info(f"No documents in {current_folder_display}. Upload a document or select a different folder.")
                else:
                    st.info("No documents uploaded yet. Use the upload section to add documents.")
            
                # Search Results Display - Now displays instantly without expander
                if st.session_state.search_results:
                    with st.container(border=True):
                        st.subheader("🔍 Search Results")
                        
                        # Display search results count
                        result_count = sum(len(matches) for matches in st.session_state.search_results.values())
                        st.write(f"Found {result_count} matches in {len(st.session_state.search_results)} documents")
                        
                        # Show results for each document
                        for doc_name, matches in st.session_state.search_results.items():
                            st.write(f"**In {doc_name}:**")
                            
                            for i, match in enumerate(matches[:3]):  # Show only first 3 matches per document
                                with st.container():
                                    st.caption(f"Match {i+1}")
                                    st.markdown(f"...{match['context']}...")
                            
                            if len(matches) > 3:
                                st.caption(f"...and {len(matches) - 3} more matches in this document")
                            
                            st.write("---")  # Add separator between documents
            
            with doc_col2:
                # Folder Navigation
                with st.container(border=True):
                    st.subheader("📁 Folder Navigation")
                    
                    # Folder actions row
                    folder_action_col1, folder_action_col2 = st.columns([3, 1])
                    with folder_action_col1:
                        # Folder dropdown
                        folders = list(st.session_state.folders.keys())
                        selected_folder = st.selectbox(
                            "Select Folder:", 
                            options=folders,
                            index=folders.index(st.session_state.current_folder),
                            format_func=lambda x: "Root Folder" if x == "root" else x
                        )
                        if selected_folder != st.session_state.current_folder:
                            st.session_state.current_folder = selected_folder
                            save_current_state()
                            st.rerun()
                    
                    with folder_action_col2:
                        # New folder button
                        if st.button("New Folder", key="new_folder_button_top", use_container_width=True):
                            st.session_state.show_folder_form = True
                
                # New folder form
                if st.session_state.show_folder_form:
                    with st.container(border=True):
                        with st.form(key="new_folder_form", clear_on_submit=True):
                            st.subheader("Create New Folder")
                            new_folder_name = st.text_input("Folder Name", key="new_folder_name_input")
                            col1, col2 = st.columns([1, 1])
                            with col1:
                                submit_button = st.form_submit_button("Create")
                            with col2:
                                cancel_button = st.form_submit_button("Cancel")
                            
                            if submit_button and new_folder_name:
                                if new_folder_name not in st.session_state.folders:
                                    # Create the new folder
                                    st.session_state.folders[new_folder_name] = []
                                    st.session_state.current_folder = new_folder_name
                                    st.session_state.show_folder_form = False
                                    save_current_state()
                                    st.success(f"Folder '{new_folder_name}' created!")
                                    st.rerun()
                                else:
                                    st.error(f"Folder '{new_folder_name}' already exists!")
                            
                            elif cancel_button:
                                st.session_state.show_folder_form = False
                                st.rerun()
                
                # Document Upload Section
                with st.container(border=True):
                    st.subheader("📤 Upload Document")
                    
                    # File uploader
                    uploaded_file = st.file_uploader(
                        "Select File", 
                        type=["docx", "xlsx", "xls", "pdf", "xml"], 
                        help="Upload .docx, .pdf, Excel or XML files for analysis"
                    )
                    
                    # Document context input
                    doc_context = st.text_area(
                        "Document Context (optional)", 
                        placeholder="Add any additional context or description for this document...",
                        help="This context will help the AI better understand the document's purpose."
                    )
                    
                    # Target folder selection
                    target_folder = st.selectbox(
                        "Upload to Folder:",
                        options=list(st.session_state.folders.keys()),
                        index=folders.index(st.session_state.current_folder),
                        format_func=lambda x: "Root Folder" if x == "root" else x
                    )
                    
                    # Process button
                    if uploaded_file is not None:
                        if st.button("Process Document", key="process_doc_button", use_container_width=True):
                            # Extract text from the uploaded file
                            document_text = extract_text_from_file(uploaded_file)
                            
                            # Check if extraction returned an error message
                            if document_text.startswith("Error:"):
                                st.error(document_text)
                            else:
                                with st.spinner("Processing document..."):
                                    # Store document content in session state
                                    file_name = uploaded_file.name
                                    
                                    # Generate document summary
                                    summary = generate_document_summary(file_name, document_text)
                                    
                                    # Save document information
                                    st.session_state.document_contents[file_name] = {
                                        "content": document_text,
                                        "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        "user_context": doc_context,
                                        "folder": target_folder
                                    }
                                    
                                    # Save summary
                                    st.session_state.document_summaries[file_name] = summary
                                    
                                    # Add to folder's document list
                                    if file_name not in st.session_state.folders[target_folder]:
                                        st.session_state.folders[target_folder].append(file_name)
                                    
                                    # Log activity if in team mode
                                    if st.session_state.view_mode == "team" and st.session_state.current_team_id:
                                        add_team_activity(
                                            st.session_state.current_team_id,
                                            st.session_state.username,
                                            "uploaded document",
                                            file_name
                                        )
                                    
                                    # Save state
                                    save_current_state()
                                    st.success(f"Document '{file_name}' processed successfully!")
                                    st.rerun()
                
                # Document Tools
                with st.container(border=True):
                    st.subheader("🔍 Document Tools")
                    
                    # Document search
                    search_query = st.text_input("Search Documents:", placeholder="Enter search term...")
                    search_button = st.button("Search", key="search_docs_button", use_container_width=True)
                    
                    # Only perform search when the button is clicked or on first entry
                    if search_query and (search_button or "last_search_query" not in st.session_state or search_query != st.session_state.last_search_query):
                        with st.spinner("Searching documents..."):
                            # Store current query to prevent repeated searches for same term
                            st.session_state.last_search_query = search_query
                            
                            # Perform search
                            search_results = search_in_documents(search_query)
                            st.session_state.search_results = search_results
                            
                            # Display search results count
                            result_count = sum(len(matches) for matches in search_results.values())
                            if result_count > 0:
                                st.success(f"Found {result_count} matches in {len(search_results)} documents")
                            else:
                                st.info("No matches found")


    with tab3:
        # Analytics tab 
        st.subheader("Document Analytics")
        
        if not st.session_state.document_contents:
            st.info("No documents to analyze. Please upload documents first.")
        else:
            # Document statistics
            st.subheader("Document Statistics")
            
            # Prepare data for chart
            doc_names = list(st.session_state.document_contents.keys())
            doc_lengths = [len(doc_info["content"]) for doc_info in st.session_state.document_contents.values()]
            
            # Create a bar chart of document lengths
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(doc_names, doc_lengths)
            ax.set_ylabel('Character Count')
            ax.set_title('Document Size Comparison')
            
            # Add labels on top of each bar
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Chat analysis
            if st.session_state.chat_history:
                st.subheader("Chat Analysis")
                
                # Calculate statistics
                total_chats = len(st.session_state.chat_history)
                total_messages = sum(len(chat["messages"]) for chat in st.session_state.chat_history)
                user_messages = sum(sum(1 for msg in chat["messages"] if msg["role"] == "user") for chat in st.session_state.chat_history)
                assistant_messages = sum(sum(1 for msg in chat["messages"] if msg["role"] == "assistant") for chat in st.session_state.chat_history)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Chats", total_chats)
                with col2:
                    st.metric("Total Messages", total_messages)
                with col3:
                    st.metric("Your Messages", user_messages)
                with col4:
                    st.metric("AI Responses", assistant_messages)
                
                # Message distribution pie chart
                if user_messages > 0 or assistant_messages > 0:  # Only create pie chart if there are messages
                    fig2, ax2 = plt.subplots()
                    ax2.pie([user_messages or 0.1, assistant_messages or 0.1], labels=['User', 'Assistant'], 
                           autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
                    ax2.set_title('Message Distribution')
                    st.pyplot(fig2)
                else:
                    st.info("No messages to display in the chart yet. Start a conversation to see analytics.")
            
            # Favorites section
            if st.session_state.favorites:
                st.subheader("Favorite Messages")
                for i, fav in enumerate(st.session_state.favorites):
                    with st.expander(f"Favorite {i+1} ({fav['role']})"):
                        st.write(fav["content"])
                        if st.button("Remove from favorites", key=f"remove_fav_{i}_{id(fav)}"):
                            st.session_state.favorites.pop(i)
                            st.rerun()
            else:
                st.info("No favorite messages yet. Star messages in chat to add them to favorites.")

    with tab4:
        # Teams tab for team management
        st.subheader("Team Management")
        
        # Get user teams
        user_teams = get_user_teams(st.session_state.username)
        
        # Team creation section
        team_creation_col1, team_creation_col2 = st.columns([3, 1])
        with team_creation_col1:
            st.write("Create a new team to share documents with other users")
        with team_creation_col2:
            if st.button("Create Team", key="create_team_tab"):
                st.session_state.show_team_form = True
        
        # Create team form
        if st.session_state.show_team_form:
            with st.form(key="create_team_form", clear_on_submit=True):
                st.subheader("Create New Team")
                new_team_name = st.text_input("Team Name", key="new_team_name")
                team_description = st.text_area("Team Description (optional)", key="team_description_input")
                col1, col2 = st.columns([1, 1])
                with col1:
                    submit_button = st.form_submit_button("Create")
                with col2:
                    cancel_button = st.form_submit_button("Cancel")
                
                if submit_button and new_team_name:
                    # Create the team
                    team_id = create_team(new_team_name, st.session_state.username)
                    
                    # Add description to team data if provided
                    if team_description:
                        team_db = load_team_db()
                        if team_id in team_db:
                            team_db[team_id]["description"] = team_description
                            save_team_db(team_db)
                    
                    st.session_state.show_team_form = False
                    st.success(f"Team '{new_team_name}' created!")
                    
                    # Switch to the new team
                    st.session_state.view_mode = "team"
                    st.session_state.current_team_id = team_id
                    save_user_session(st.session_state.user_id)
                    st.rerun()
                
                elif cancel_button:
                    st.session_state.show_team_form = False
                    st.rerun()
        
        # Team listing and management
        if user_teams:
            st.subheader("Your Teams")
            
            team_tabs = st.tabs([team["name"] for team in user_teams])
            
            for i, team in enumerate(user_teams):
                with team_tabs[i]:
                    team_id = team["id"]
                    team_db = load_team_db()
                    current_team = team_db.get(team_id, {})
                    
                    # Create subtabs for team management
                    team_subtabs = st.tabs(["Overview", "Members", "Activity Log"])
                    
                    with team_subtabs[0]:  # Overview tab
                        # Team details
                        st.write(f"**Team Name:** {team['name']}")
                        st.write(f"**Owner:** {team['owner']}")
                        st.write(f"**Members:** {team['member_count']}")
                        
                        if "description" in current_team:
                            st.write(f"**Description:** {current_team['description']}")
                        
                        # View mode button
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            if st.button("View Documents", key=f"view_team_docs_{team_id}"):
                                st.session_state.view_mode = "team"
                                st.session_state.current_team_id = team_id
                                
                                # Load team data
                                team_data = load_team_data(team_id)
                                
                                # Update session state with team data
                                st.session_state.document_contents = team_data["document_contents"]
                                st.session_state.document_summaries = team_data["document_summaries"]
                                st.session_state.folders = team_data["folders"]
                                st.session_state.current_folder = team_data["current_folder"]
                                
                                # Log activity
                                add_team_activity(team_id, st.session_state.username, "viewed team documents")
                                
                                st.rerun()
                        
                        # Team document statistics
                        team_data = load_team_data(team_id)
                        doc_count = len(team_data["document_contents"])
                        
                        # Display document count
                        st.write(f"**Team Documents:** {doc_count}")
                        
                        # Add Delete Team button (only for team owners)
                        if team["is_owner"]:
                            st.write("---")
                            st.subheader("Danger Zone")
                            st.error("Deleting a team will permanently remove all team documents and data.")
                            
                            delete_col1, delete_col2 = st.columns([1, 3])
                            with delete_col1:
                                if st.button("Delete Team", key=f"delete_team_{team_id}"):
                                    # Add confirmation dialog
                                    st.session_state[f"show_delete_confirmation_{team_id}"] = True
                            
                            # Show confirmation dialog if button was clicked
                            if st.session_state.get(f"show_delete_confirmation_{team_id}", False):
                                with st.form(key=f"delete_team_form_{team_id}"):
                                    st.write(f"**Are you sure you want to delete team '{team['name']}'?**")
                                    st.write("This action cannot be undone. All team documents and data will be permanently deleted.")
                                    
                                    # Require typing the team name for confirmation
                                    confirmation_input = st.text_input(
                                        f"Type '{team['name']}' to confirm deletion",
                                        key=f"delete_confirm_input_{team_id}"
                                    )
                                    
                                    col1, col2 = st.columns([1, 1])
                                    with col1:
                                        confirm_button = st.form_submit_button("Delete Team Permanently")
                                    with col2:
                                        cancel_button = st.form_submit_button("Cancel")
                                    
                                    if confirm_button:
                                        if confirmation_input == team['name']:
                                            # Delete the team
                                            success, message = delete_team(team_id, st.session_state.username)
                                            
                                            if success:
                                                st.success(message)
                                                # If we're currently viewing this team, switch to personal mode
                                                if st.session_state.view_mode == "team" and st.session_state.current_team_id == team_id:
                                                    st.session_state.view_mode = "personal"
                                                    st.session_state.current_team_id = None
                                                    # Reset document display
                                                    user_session = load_user_session(st.session_state.user_id)
                                                    st.session_state.document_contents = user_session["document_contents"]
                                                    st.session_state.document_summaries = user_session["document_summaries"]
                                                    st.session_state.folders = user_session["folders"]
                                                    st.session_state.current_folder = user_session["current_folder"]
                                                
                                                # Clear confirmation state and rerun to update UI
                                                st.session_state.pop(f"show_delete_confirmation_{team_id}", None)
                                                st.rerun()
                                            else:
                                                st.error(message)
                                        else:
                                            st.error("Team name doesn't match. Deletion canceled.")
                                    
                                    if cancel_button:
                                        # Clear confirmation state and rerun to update UI
                                        st.session_state.pop(f"show_delete_confirmation_{team_id}", None)
                                        st.rerun()
                    
                    with team_subtabs[1]:  # Members tab
                        # Team members section
                        st.subheader("Team Members")
                        
                        members = current_team.get("members", [])
                        
                        # Create a table for members
                        member_data = []
                        for member in members:
                            role = "Owner" if member == current_team["owner"] else "Member"
                            member_data.append({"Username": member, "Role": role})
                        
                        st.table(member_data)
                        
                        # Member management (only for team owners)
                        if team["is_owner"]:
                            st.subheader("Manage Members")
                            
                            # Invite members
                            with st.expander("Invite New Members"):
                                invite_username = st.text_input("Username to invite", key=f"invite_input_{team_id}")
                                if st.button("Send Invitation", key=f"invite_button_{team_id}"):
                                    user_db = load_user_db()
                                    if invite_username not in user_db:
                                        st.error(f"User '{invite_username}' does not exist")
                                    elif invite_username in members:
                                        st.warning(f"{invite_username} is already a member")
                                    elif add_member_to_team(team_id, invite_username):
                                        # Log activity
                                        add_team_activity(team_id, st.session_state.username, "added member", invite_username)
                                        st.success(f"Added {invite_username} to team")
                                        st.rerun()
                            
                            # Remove members
                            with st.expander("Remove Members"):
                                # Only show non-owner members
                                removable_members = [m for m in members if m != current_team["owner"]]
                                if removable_members:
                                    member_to_remove = st.selectbox(
                                        "Select member to remove",
                                        options=removable_members,
                                        key=f"remove_select_{team_id}"
                                    )
                                    
                                    if st.button("Remove Member", key=f"remove_button_{team_id}"):
                                        if remove_member_from_team(team_id, member_to_remove):
                                            # Log activity
                                            add_team_activity(team_id, st.session_state.username, "removed member", member_to_remove)
                                            st.success(f"Removed {member_to_remove} from team")
                                            st.rerun()
                                else:
                                    st.info("No members to remove")
                    
                    with team_subtabs[2]:  # Activity Log tab
                        st.subheader("Recent Team Activity")
                        
                        # Get recent activities
                        activities = get_team_activities(team_id)
                        
                        if activities:
                            # Create a table of activities
                            activity_data = []
                            for activity in reversed(activities):  # Display newest first
                                activity_data.append({
                                    "When": activity["timestamp"],
                                    "User": activity["username"],
                                    "Action": activity["action"],
                                    "Details": activity["item"] if activity["item"] else ""
                                })
                            
                            # Use a dataframe for better display
                            activity_df = pd.DataFrame(activity_data)
                            st.dataframe(activity_df, use_container_width=True)
                        else:
                            st.info("No team activity recorded yet.")
        else:
            st.info("You don't have any teams yet. Create a team to share documents with other users.")

    with tab5:
        # Settings
        st.subheader("Chat Settings")
        
        # Custom system prompt
        st.write("Customize the system prompt for the AI assistant:")
        custom_prompt = st.text_area(
            "Custom Instructions", 
            value=st.session_state.custom_prompt if st.session_state.custom_prompt else "When answering questions, please reference specific sections of the documents and provide detailed explanations.",
            height=100
        )
        
        if st.button("Save Custom Instructions", key="save_instructions_button"):
            st.session_state.custom_prompt = custom_prompt
            save_user_session(st.session_state.user_id)
            st.success("Custom instructions saved!")
        
        # Account settings
        st.subheader("Account Settings")
        
        # Password change
        with st.expander("Change Password"):
            current_password = st.text_input("Current Password", type="password", key="current_password")
            new_password = st.text_input("New Password", type="password", key="settings_new_password")
            confirm_new_password = st.text_input("Confirm New Password", type="password", key="settings_confirm_password")
            
            if st.button("Update Password", key="update_password_button"):
                # Verify current password
                user_db = load_user_db()
                if hash_password(current_password) != user_db[st.session_state.username]["password"]:
                    st.error("Current password is incorrect")
                elif new_password != confirm_new_password:
                    st.error("New passwords do not match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    # Update password
                    user_db[st.session_state.username]["password"] = hash_password(new_password)
                    save_user_db(user_db)
                    st.success("Password updated successfully!")
        
        # Model parameters
        st.subheader("Model Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1,
                                   help="Higher values make output more random, lower values more deterministic")
        
        with col2:
            max_tokens = st.slider("Max Response Length", min_value=100, max_value=1000, value=500, step=50,
                                  help="Maximum number of tokens in the response")
        
        # UI Settings
        st.subheader("UI Settings")
        
        # OpenAI API Key Settings
        st.subheader("OpenAI API Key")
        with st.expander("Manage API Key"):
            st.write("Enter your OpenAI API key to use the AI features. Your key will be stored securely and only used for this application.")
            
            # Get current API key from session state or environment
            current_api_key = st.session_state.get("openai_api_key", os.getenv("OPENAI_API_KEY", ""))
            
            # Show masked API key if exists
            if current_api_key:
                masked_key = current_api_key[:4] + "..." + current_api_key[-4:]
                st.info(f"Current API Key: {masked_key}")
            
            # API key input
            new_api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                value=current_api_key if current_api_key else "",
                help="Enter your OpenAI API key. You can get one from https://platform.openai.com/api-keys"
            )
            
            # Save button
            if st.button("Save API Key"):
                if new_api_key:
                    # Update session state
                    st.session_state.openai_api_key = new_api_key
                    
                    # Update OpenAI client
                    openai.api_key = new_api_key
                    
                    # Test the API key with a simple request
                    try:
                        # Make a minimal API call to test the key
                        openai.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": "test"}],
                            max_tokens=1
                        )
                        st.success("API key saved and validated successfully!")
                    except Exception as e:
                        st.error(f"Error validating API key: {str(e)}")
                else:
                    st.error("Please enter an API key")
        
        # Import/Export Settings
        st.subheader("Import/Export")
        
        # Export all data
        if st.button("Export All Data", key="export_data_button"):
            export_data = {
                "document_contents": {k: {"content": v["content"], 
                                        "upload_time": v["upload_time"],
                                        "user_context": v.get("user_context", ""),
                                        "folder": v.get("folder", "root")} 
                                    for k, v in st.session_state.document_contents.items()},
                "document_summaries": st.session_state.document_summaries,
                "chat_history": st.session_state.chat_history,
                "favorites": st.session_state.favorites,
                "custom_prompt": st.session_state.custom_prompt,
                "folders": st.session_state.folders
            }
            
            export_json = json.dumps(export_data)
            b64 = base64.b64encode(export_json.encode()).decode()
            export_filename = f"Flypilot_export_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
            st.markdown(
                f'<a href="data:file/json;base64,{b64}" download="{export_filename}">Download All Data</a>',
                unsafe_allow_html=True
            )
        
        # Import data
        st.write("Import data from a previous export:")
        import_file = st.file_uploader("Upload export file", type=["json"])
        if import_file is not None:
            if st.button("Import Data", key="import_data_button"):
                try:
                    import_data = json.loads(import_file.getvalue())
                    
                    # Update session state with imported data
                    st.session_state.document_contents.update(import_data.get("document_contents", {}))
                    st.session_state.document_summaries.update(import_data.get("document_summaries", {}))
                    st.session_state.chat_history.extend(import_data.get("chat_history", []))
                    st.session_state.favorites.extend(import_data.get("favorites", []))
                    st.session_state.custom_prompt = import_data.get("custom_prompt", st.session_state.custom_prompt)
                    st.session_state.folders.update(import_data.get("folders", {}))
                    
                    st.success("Data imported successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error importing data: {str(e)}")

        # Chat History Management
        st.subheader("Chat History Management")
        
        if st.session_state.chat_history:
            if st.button("Clear All Chat History", key="clear_chat_history_button"):
                confirm = st.checkbox("Are you sure? This cannot be undone.")
                if confirm:
                    st.session_state.chat_history = []
                    st.session_state.current_chat_id = None
                    st.session_state.messages = []
                    st.success("All chat history cleared!")
                    st.rerun()
        else:
            st.info("No chat history to clear.")

# Main application flow
if st.session_state.authenticated:
    # Save user session on every page load
    if st.session_state.user_id:
        main_app()
        # Save session at the end of interaction
        if st.session_state.username:  # Only save if we have a valid user
            save_user_session(st.session_state.user_id)
            
        # Add user info to sidebar
        sidebar_col1, sidebar_col2 = st.sidebar.columns([4, 1])
        with sidebar_col1:            
            # Check for notifications
            user_notifications = get_user_notifications(st.session_state.username)
            unread_count = sum(1 for n in user_notifications if not n["read"])
            
            # Display notification indicator
            if unread_count > 0:
                st.sidebar.markdown(f"### 🔔 Notifications ({unread_count})")
                
                # Show notification panel
                with st.sidebar.expander("View Notifications", expanded=True):
                    for i, notification in enumerate(user_notifications):
                        # Format timestamp
                        timestamp = notification["timestamp"]
                        
                        # Create a distinctive background for unread notifications
                        if not notification["read"]:
                            st.markdown(
                                f"""
                                <div style="background-color: rgba(100,149,237,0.2); padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                                    <p><b>{notification["message"]}</b></p>
                                    <p style="font-size: 0.8em; color: gray;">{timestamp}</p>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                            
                            # Mark as read button
                            if st.button(f"Mark as read", key=f"mark_read_{i}"):
                                mark_notification_read(st.session_state.username, i)
                                st.rerun()
                        else:
                            st.markdown(
                                f"""
                                <div style="background-color: rgba(220,220,220,0.2); padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                                    <p>{notification["message"]}</p>
                                    <p style="font-size: 0.8em; color: gray;">{timestamp}</p>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                    
                    # Clear all button
                    if user_notifications:
                        if st.button("Clear All Notifications", key="clear_notifications_button"):
                            clear_user_notifications(st.session_state.username)
                            st.rerun()

        with sidebar_col2:
            # Logout button
            if st.sidebar.button("Logout", key="sidebar_logout_button2"):
                # Only save if authenticated and user_id exists
                if st.session_state.get("authenticated", False) and st.session_state.get("user_id") is not None:
                    save_user_session(st.session_state.user_id)
                    
                # Clear all session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
    else:
        login_page()
else:
    # For unauthenticated users, only show login page
    login_page()
