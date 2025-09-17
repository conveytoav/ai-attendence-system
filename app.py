from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import sqlite3
from datetime import datetime, timedelta
import os
import json
import csv
import io
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

def upload_db_to_drive():
        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()  # Opens browser for authentication
        drive = GoogleDrive(gauth)

        # Path to your database
        db_file = "attendance.db"  # Make sure it's in the root directory of your app

        if os.path.exists(db_file):
            file_drive = drive.CreateFile({'title': db_file})
            file_drive.SetContentFile(db_file)
            file_drive.Upload()
            print(f"{db_file} uploaded to Google Drive.")
        else:
            print(f"{db_file} not found!")

@app.route('/upload_drive')
def upload_drive():
    upload_db_to_drive()
    return 'Database uploaded to Google Drive successfully!'


class AttendanceManager:
    def __init__(self):
        self.init_db()
    
    def init_db(self):
        """Initialize the database with enhanced schema"""
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        
        # Create attendance table with additional fields
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                status TEXT DEFAULT 'present',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Check if status column exists, if not add it
        cursor.execute("PRAGMA table_info(attendance)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'status' not in columns:
            print("Adding status column to existing table...")
            cursor.execute("ALTER TABLE attendance ADD COLUMN status TEXT DEFAULT 'present'")
        
        # Check if created_at column exists, if not add it
        if 'created_at' not in columns:
            print("Adding created_at column to existing table...")
            cursor.execute("ALTER TABLE attendance ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        
        conn.commit()
        conn.close()
        print("Database initialized successfully")
    
    def get_attendance_data(self, date):
        """Get attendance data for a specific date"""
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        
        # Check if status column exists
        cursor.execute("PRAGMA table_info(attendance)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'status' in columns:
            cursor.execute("""
                SELECT name, time, status FROM attendance 
                WHERE date = ? ORDER BY time ASC
            """, (date,))
        else:
            cursor.execute("""
                SELECT name, time, 'present' as status FROM attendance 
                WHERE date = ? ORDER BY time ASC
            """, (date,))
        
        attendance_data = cursor.fetchall()
        conn.close()
        
        print(f"DEBUG: Looking for date: {date}")
        print(f"DEBUG: Found {len(attendance_data)} records")
        for record in attendance_data:
            print(f"DEBUG: Record: {record}")
        
        return attendance_data
    
    def add_attendance(self, name, date, time, status='present'):
        """Add new attendance record"""
        try:
            conn = sqlite3.connect('attendance.db')
            cursor = conn.cursor()
            
            print(f"DEBUG: Adding attendance - Name: {name}, Date: {date}, Time: {time}, Status: {status}")
            
            cursor.execute("""
                INSERT INTO attendance (name, date, time, status) 
                VALUES (?, ?, ?, ?)
            """, (name, date, time, status))
            
            conn.commit()
            
            # Verify the record was added
            cursor.execute("SELECT * FROM attendance WHERE name = ? AND date = ? AND time = ?", (name, date, time))
            result = cursor.fetchone()
            print(f"DEBUG: Added record verification: {result}")
            
            conn.close()
            return True
        except Exception as e:
            print(f"ERROR: Failed to add attendance - {str(e)}")
            return False
    
    def reset_database(self):
        """Reset the database - WARNING: This will delete all data"""
        try:
            conn = sqlite3.connect('attendance.db')
            cursor = conn.cursor()
            cursor.execute("DROP TABLE IF EXISTS attendance")
            conn.commit()
            conn.close()
            print("Database reset successfully")
            self.init_db()
        except Exception as e:
            print(f"Error resetting database: {e}")

    def get_all_attendance(self):
        """Get all attendance records for debugging"""
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM attendance ORDER BY date DESC, time DESC")
        all_records = cursor.fetchall()
        conn.close()
        
        print(f"DEBUG: Total records in database: {len(all_records)}")
        for record in all_records:
            print(f"DEBUG: All records: {record}")
        
        return all_records

# Initialize attendance manager
attendance_manager = AttendanceManager()


@app.route('/')
def index():
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"DEBUG: Index route called, today: {today}")
    
    # Debug: Show all records
    attendance_manager.get_all_attendance()
    
    return render_template('index.html', 
                         selected_date=today, 
                         no_data=False,
                         attendance_data=None)

@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    if request.method == 'GET':
        # Handle GET request with date parameter
        selected_date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    else:
        # Handle POST request from form
        selected_date = request.form.get('selected_date')
    
    print(f"DEBUG: Attendance route called with date: {selected_date}")
    
    if not selected_date:
        flash('Please select a date!', 'error')
        today = datetime.now().strftime('%Y-%m-%d')
        return render_template('index.html', 
                             selected_date=today, 
                             no_data=False,
                             attendance_data=None)
    
    attendance_data = attendance_manager.get_attendance_data(selected_date)
    
    if not attendance_data:
        print("DEBUG: No attendance data found")
        return render_template('index.html', 
                             selected_date=selected_date, 
                             no_data=True,
                             attendance_data=None)
    
    print(f"DEBUG: Found {len(attendance_data)} attendance records")
    return render_template('index.html', 
                         selected_date=selected_date, 
                         attendance_data=attendance_data,
                         no_data=False)

@app.route('/add_attendance', methods=['POST'])
def add_attendance():
    name = request.form.get('name', '').strip()
    date = request.form.get('date')
    time = request.form.get('time')
    status = request.form.get('status', 'present')
    
    print(f"DEBUG: Add attendance called with: name={name}, date={date}, time={time}, status={status}")
    
    # Validate input
    if not name or not date or not time:
        flash('Please fill all required fields!', 'error')
        print(f"DEBUG: Missing required fields: name={name}, date={date}, time={time}")
        return redirect(url_for('index'))
    
    # Check for duplicate entry
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM attendance WHERE name = ? AND date = ? AND time = ?", (name, date, time))
    existing = cursor.fetchone()
    conn.close()
    
    if existing:
        flash('Attendance record already exists for this person at this time!', 'error')
        return redirect(url_for('index'))
    
    success = attendance_manager.add_attendance(name, date, time, status)
    if success:
        flash('Attendance added successfully!', 'success')
        print(f"DEBUG: Attendance added successfully, redirecting to view date: {date}")
        
        # Get updated attendance data
        attendance_data = attendance_manager.get_attendance_data(date)
        print(f"DEBUG: After adding, found {len(attendance_data)} records for date {date}")
        
        return render_template('index.html', 
                             selected_date=date, 
                             attendance_data=attendance_data,
                             no_data=len(attendance_data) == 0)
    else:
        flash('Failed to add attendance!', 'error')
    
    return redirect(url_for('index'))

@app.route('/debug')
def debug():
    """Debug route to check database contents"""
    all_records = attendance_manager.get_all_attendance()
    return jsonify({
        'total_records': len(all_records),
        'records': all_records
    })

@app.route('/api/attendance/<date>')
def api_attendance(date):
    """API endpoint for attendance data"""
    attendance_data = attendance_manager.get_attendance_data(date)
    return jsonify({
        'date': date,
        'count': len(attendance_data),
        'attendees': [{'name': row[0], 'time': row[1], 'status': row[2]} for row in attendance_data]
    })

@app.route('/delete_attendance/<int:record_id>', methods=['POST'])
def delete_attendance(record_id):
    """Delete an attendance record"""
    try:
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        cursor.execute("DELETE FROM attendance WHERE id = ?", (record_id,))
        conn.commit()
        conn.close()
        flash('Attendance record deleted successfully!', 'success')
    except Exception as e:
        flash(f'Error deleting record: {str(e)}', 'error')
    
    return redirect(url_for('index'))

@app.route('/reset_db', methods=['POST'])
def reset_db():
    """Reset database route - use with caution"""
    attendance_manager.reset_database()
    flash('Database reset successfully! All data has been cleared.', 'success')
    return redirect(url_for('index'))


def export_and_upload():
    # Step 1: Connect to the database
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    # Step 2: Query attendance table
    cursor.execute("SELECT * FROM attendance")
    rows = cursor.fetchall()

    # Step 3: Get column names
    column_names = [description[0] for description in cursor.description]

    # Step 4: Export to CSV
    csv_filename = "attendance_export.csv"
    with open(csv_filename, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(column_names)  # write header
        writer.writerows(rows)         # write data

    conn.close()

    # Step 5: Authenticate with Google Drive
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()  # One-time login
    drive = GoogleDrive(gauth)

    # Step 6: Upload CSV
    upload_file = drive.CreateFile({'title': csv_filename})
    upload_file.SetContentFile(csv_filename)
    upload_file.Upload()

    return f"✅ Uploaded {csv_filename} to Google Drive successfully."

@app.route('/export_drive')
def export_drive():
    try:
        msg = export_and_upload()
        return msg
    except Exception as e:
        return f"❌ Error: {str(e)}"

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True)