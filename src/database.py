"""
Simple database module for malnutrition detection pipeline.
Stores uploaded training data metadata and training metrics for retraining purposes.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

class MalnutritionDB:
    """Simple SQLite database for storing training data and metrics."""
    
    def __init__(self, db_path: str = "../data/malnutrition.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()
        logger.info(f"Database initialized at: {db_path}")
    
    def init_database(self):
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Table for uploaded training data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS uploaded_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    original_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    class_label TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    used_for_training BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # Table for training sessions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    end_time DATETIME,
                    status TEXT DEFAULT 'started',
                    total_epochs INTEGER,
                    final_accuracy REAL,
                    final_loss REAL,
                    model_path TEXT
                )
            ''')
            
            # Table for training metrics (epoch-by-epoch)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    epoch INTEGER NOT NULL,
                    accuracy REAL NOT NULL,
                    loss REAL NOT NULL,
                    val_accuracy REAL NOT NULL,
                    val_loss REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
                )
            ''')
            
            # Table for model performance history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT NOT NULL,
                    accuracy REAL NOT NULL,
                    precision_score REAL NOT NULL,
                    recall_score REAL NOT NULL,
                    f1_score REAL NOT NULL,
                    test_samples INTEGER NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            logger.info("Database tables created/verified successfully")
    
    def save_uploaded_data(self, filename: str, original_name: str, file_path: str, 
                          class_label: str, file_size: int) -> int:
        """
        Save uploaded training data metadata to database.
        
        Args:
            filename: Unique filename
            original_name: Original uploaded filename
            file_path: Path where file is stored
            class_label: Classification label (malnourished/overnourished)
            file_size: File size in bytes
            
        Returns:
            Database record ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO uploaded_data 
                (filename, original_name, file_path, class_label, file_size)
                VALUES (?, ?, ?, ?, ?)
            ''', (filename, original_name, file_path, class_label, file_size))
            
            record_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Saved uploaded data: {original_name} -> {class_label}")
            return record_id
    
    def get_uploaded_data_for_training(self) -> List[Dict]:
        """
        Get all uploaded data that hasn't been used for training yet.
        
        Returns:
            List of uploaded data records
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM uploaded_data 
                WHERE used_for_training = FALSE
                ORDER BY upload_timestamp DESC
            ''')
            
            records = [dict(row) for row in cursor.fetchall()]
            logger.info(f"Retrieved {len(records)} unused training files from database")
            return records
    
    def mark_data_as_used(self, record_ids: List[int]):
        """
        Mark uploaded data as used for training.
        
        Args:
            record_ids: List of database record IDs to mark as used
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            placeholders = ','.join(['?' for _ in record_ids])
            cursor.execute(f'''
                UPDATE uploaded_data 
                SET used_for_training = TRUE 
                WHERE id IN ({placeholders})
            ''', record_ids)
            
            conn.commit()
            logger.info(f"Marked {len(record_ids)} files as used for training")
    
    def start_training_session(self, session_id: str, total_epochs: int) -> None:
        """
        Start a new training session.
        
        Args:
            session_id: Unique session identifier
            total_epochs: Total number of training epochs
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO training_sessions 
                (session_id, total_epochs, status)
                VALUES (?, ?, 'training')
            ''', (session_id, total_epochs))
            
            conn.commit()
            logger.info(f"Started training session: {session_id}")
    
    def save_training_metrics(self, session_id: str, epoch: int, 
                             accuracy: float, loss: float, 
                             val_accuracy: float, val_loss: float) -> None:
        """
        Save training metrics for an epoch.
        
        Args:
            session_id: Training session ID
            epoch: Epoch number
            accuracy: Training accuracy
            loss: Training loss
            val_accuracy: Validation accuracy
            val_loss: Validation loss
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO training_metrics 
                (session_id, epoch, accuracy, loss, val_accuracy, val_loss)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (session_id, epoch, accuracy, loss, val_accuracy, val_loss))
            
            conn.commit()
            logger.debug(f"Saved metrics for epoch {epoch} in session {session_id}")
    
    def complete_training_session(self, session_id: str, final_accuracy: float, 
                                 final_loss: float, model_path: str) -> None:
        """
        Mark training session as completed.
        
        Args:
            session_id: Training session ID
            final_accuracy: Final model accuracy
            final_loss: Final model loss
            model_path: Path to saved model
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE training_sessions 
                SET end_time = CURRENT_TIMESTAMP, status = 'completed',
                    final_accuracy = ?, final_loss = ?, model_path = ?
                WHERE session_id = ?
            ''', (final_accuracy, final_loss, model_path, session_id))
            
            conn.commit()
            logger.info(f"Completed training session: {session_id}")
    
    def get_training_history(self, limit: int = 50) -> List[Dict]:
        """
        Get recent training history.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of training metric records
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT tm.*, ts.start_time as session_start
                FROM training_metrics tm
                JOIN training_sessions ts ON tm.session_id = ts.session_id
                ORDER BY tm.timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            records = [dict(row) for row in cursor.fetchall()]
            return records
    
    def save_model_performance(self, model_version: str, accuracy: float, 
                              precision: float, recall: float, f1: float, 
                              test_samples: int) -> None:
        """
        Save model performance metrics.
        
        Args:
            model_version: Model version identifier
            accuracy: Model accuracy
            precision: Precision score
            recall: Recall score
            f1: F1 score
            test_samples: Number of test samples
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO model_performance 
                (model_version, accuracy, precision_score, recall_score, f1_score, test_samples)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (model_version, accuracy, precision, recall, f1, test_samples))
            
            conn.commit()
            logger.info(f"Saved performance metrics for model: {model_version}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Count uploaded files
            cursor.execute("SELECT COUNT(*) FROM uploaded_data")
            total_files = cursor.fetchone()[0]
            
            # Count unused files
            cursor.execute("SELECT COUNT(*) FROM uploaded_data WHERE used_for_training = FALSE")
            unused_files = cursor.fetchone()[0]
            
            # Count training sessions
            cursor.execute("SELECT COUNT(*) FROM training_sessions")
            training_sessions = cursor.fetchone()[0]
            
            # Count training metrics
            cursor.execute("SELECT COUNT(*) FROM training_metrics")
            training_metrics = cursor.fetchone()[0]
            
            return {
                "total_uploaded_files": total_files,
                "unused_files": unused_files,
                "training_sessions": training_sessions,
                "training_metrics_records": training_metrics,
                "database_path": self.db_path
            }

# Global database instance
db = MalnutritionDB()

def get_database() -> MalnutritionDB:
    """Get global database instance."""
    return db

if __name__ == "__main__":
    # Test database functionality
    logging.basicConfig(level=logging.INFO)
    
    # Initialize database
    test_db = MalnutritionDB("../data/test_malnutrition.db")
    
    # Test saving uploaded data
    file_id = test_db.save_uploaded_data(
        filename="test_001.jpg",
        original_name="child_photo.jpg", 
        file_path="/data/uploaded/test_001.jpg",
        class_label="malnourished",
        file_size=12345
    )
    print(f"✅ Saved file with ID: {file_id}")
    
    # Test training session
    session_id = "test_session_001"
    test_db.start_training_session(session_id, 20)
    
    # Test saving metrics
    test_db.save_training_metrics(session_id, 1, 0.85, 0.45, 0.82, 0.48)
    test_db.save_training_metrics(session_id, 2, 0.88, 0.42, 0.85, 0.45)
    
    # Complete session
    test_db.complete_training_session(session_id, 0.95, 0.25, "/models/test_model.h5")
    
    # Get stats
    stats = test_db.get_database_stats()
    print(f"📊 Database stats: {stats}")
    
    print("✅ Database functionality test completed!") 