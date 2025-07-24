import sqlite3

DB_NAME = 'rag.db'  # Name of the SQLite database file

# Establish a connection to the SQLite database and set row_factory for dict-like access
def db_connect():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

# Create the application_logs table if it doesn't exist
# This table stores chat history for each session
def create_application_logs():
    conn = db_connect()
    conn.execute('''CREATE TABLE IF NOT EXISTS application_logs
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     session_id TEXT,
                     user_query TEXT,
                     model_response TEXT,
                     model TEXT,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.close()

# Create the document_store table if it doesn't exist
# This table stores uploaded document metadata
def create_document_store():
    conn = db_connect()
    conn.execute('''CREATE TABLE IF NOT EXISTS document_store
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     filename TEXT,
                     upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.close()

# Insert a new chat log entry into application_logs
# session_id: conversation session identifier
# user_query: user's question
# model_response: model's answer
# model: model name used
def insert_application_logs(session_id, user_query, model_response, model):
    conn = db_connect()
    conn.execute('INSERT INTO application_logs (session_id, user_query, model_response, model) VALUES (?, ?, ?, ?)',
                 (session_id, user_query, model_response, model))
    conn.commit()
    conn.close()

# Retrieve chat history for a given session_id, ordered by creation time
# Returns a list of message dicts for use in chat history
def get_rag_history(session_id):
    conn = db_connect()
    cursor = conn.cursor()
    cursor.execute('SELECT user_query, model_response FROM application_logs WHERE session_id = ? ORDER BY created_at', (session_id,))
    messages = []
    for row in cursor.fetchall():
        messages.extend([
            {"role": "human", "content": row['user_query']},
            {"role": "ai", "content": row['model_response']}
        ])
    conn.close()
    return messages

# Insert a new document record into document_store
# Returns the file_id (primary key) of the inserted document
def insert_document_record(filename):
    conn = db_connect()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO document_store (filename) VALUES (?)', (filename,))
    file_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return file_id

# Delete a document record from document_store by file_id
# Returns True if successful
def delete_document_record(file_id):
    conn = db_connect()
    conn.execute('DELETE FROM document_store WHERE id = ?', (file_id,))
    conn.commit()
    conn.close()
    return True

# Retrieve all document records, ordered by upload time (most recent first)
# Returns a list of dicts with id, filename, and upload_timestamp
def get_all_documents():
    conn = db_connect()
    cursor = conn.cursor()
    cursor.execute('SELECT id, filename, upload_timestamp FROM document_store ORDER BY upload_timestamp DESC')
    documents = cursor.fetchall()
    conn.close()
    return [dict(doc) for doc in documents]

# Initialize the database tables on import
create_application_logs()
create_document_store()