#!/usr/bin/env python3
"""
Database management CLI for malnutrition detection pipeline.
Provides commands to view, clear, and manage the database.
"""

import argparse
import sys
import os
from datetime import datetime
from database import MalnutritionDB
import logging

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def view_stats(db: MalnutritionDB):
    """Display database statistics."""
    print("\n📊 DATABASE STATISTICS")
    print("=" * 50)
    
    stats = db.get_database_stats()
    for key, value in stats.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print()

def view_uploaded_data(db: MalnutritionDB):
    """Display uploaded training data."""
    print("\n📁 UPLOADED TRAINING DATA")
    print("=" * 50)
    
    import sqlite3
    with sqlite3.connect(db.db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, filename, original_name, class_label, file_size, 
                   upload_timestamp, used_for_training
            FROM uploaded_data 
            ORDER BY upload_timestamp DESC
            LIMIT 20
        ''')
        
        records = cursor.fetchall()
        
        if not records:
            print("  No uploaded data found.")
            return
        
        print(f"{'ID':<4} {'Original Name':<20} {'Label':<15} {'Size':<10} {'Used':<6} {'Upload Time'}")
        print("-" * 80)
        
        for record in records:
            used_status = "✅" if record['used_for_training'] else "❌"
            size_mb = f"{record['file_size'] / 1024 / 1024:.1f}MB"
            upload_time = record['upload_timestamp'][:16]  # Show date and time only
            
            print(f"{record['id']:<4} {record['original_name'][:19]:<20} "
                  f"{record['class_label']:<15} {size_mb:<10} {used_status:<6} {upload_time}")

def view_training_sessions(db: MalnutritionDB):
    """Display training sessions."""
    print("\n🚀 TRAINING SESSIONS")
    print("=" * 50)
    
    import sqlite3
    with sqlite3.connect(db.db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT session_id, start_time, end_time, status, 
                   total_epochs, final_accuracy, final_loss
            FROM training_sessions 
            ORDER BY start_time DESC
            LIMIT 10
        ''')
        
        records = cursor.fetchall()
        
        if not records:
            print("  No training sessions found.")
            return
        
        print(f"{'Session ID':<20} {'Status':<10} {'Epochs':<8} {'Accuracy':<10} {'Start Time'}")
        print("-" * 80)
        
        for record in records:
            accuracy = f"{record['final_accuracy']*100:.1f}%" if record['final_accuracy'] else "N/A"
            start_time = record['start_time'][:16] if record['start_time'] else "N/A"
            
            print(f"{record['session_id'][:19]:<20} {record['status']:<10} "
                  f"{record['total_epochs'] or 'N/A':<8} {accuracy:<10} {start_time}")

def view_training_metrics(db: MalnutritionDB, session_id: str = None):
    """Display training metrics."""
    print(f"\n📈 TRAINING METRICS{f' - {session_id}' if session_id else ''}")
    print("=" * 50)
    
    import sqlite3
    with sqlite3.connect(db.db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if session_id:
            cursor.execute('''
                SELECT * FROM training_metrics 
                WHERE session_id = ?
                ORDER BY epoch
            ''', (session_id,))
        else:
            cursor.execute('''
                SELECT * FROM training_metrics 
                ORDER BY timestamp DESC
                LIMIT 20
            ''')
        
        records = cursor.fetchall()
        
        if not records:
            print("  No training metrics found.")
            return
        
        print(f"{'Session':<20} {'Epoch':<6} {'Accuracy':<10} {'Loss':<8} {'Val Acc':<10} {'Val Loss':<8}")
        print("-" * 80)
        
        for record in records:
            print(f"{record['session_id'][:19]:<20} {record['epoch']:<6} "
                  f"{record['accuracy']:.3f}({record['accuracy']*100:.1f}%)<10 "
                  f"{record['loss']:.4f}<8 {record['val_accuracy']:.3f}({record['val_accuracy']*100:.1f}%)<10 "
                  f"{record['val_loss']:.4f}<8")

def clear_database(db: MalnutritionDB, confirm: bool = False):
    """Clear all database data."""
    if not confirm:
        print("\n⚠️  WARNING: This will delete ALL database data!")
        print("   - All uploaded data records")
        print("   - All training sessions")
        print("   - All training metrics")
        print("   - All model performance records")
        
        response = input("\nAre you sure you want to continue? (type 'yes' to confirm): ")
        if response.lower() != 'yes':
            print("❌ Operation cancelled.")
            return
    
    print("\n🗑️  Clearing database...")
    
    import sqlite3
    with sqlite3.connect(db.db_path) as conn:
        cursor = conn.cursor()
        
        # Clear all tables
        tables = ['training_metrics', 'training_sessions', 'uploaded_data', 'model_performance']
        
        for table in tables:
            cursor.execute(f'DELETE FROM {table}')
            affected = cursor.rowcount
            print(f"   Cleared {affected} records from {table}")
        
        conn.commit()
    
    print("✅ Database cleared successfully!")

def clear_unused_data(db: MalnutritionDB):
    """Clear only unused uploaded data."""
    print("\n🗑️  Clearing unused uploaded data...")
    
    import sqlite3
    with sqlite3.connect(db.db_path) as conn:
        cursor = conn.cursor()
        
        # Count unused data first
        cursor.execute('SELECT COUNT(*) FROM uploaded_data WHERE used_for_training = FALSE')
        count = cursor.fetchone()[0]
        
        if count == 0:
            print("   No unused data found.")
            return
        
        # Delete unused data
        cursor.execute('DELETE FROM uploaded_data WHERE used_for_training = FALSE')
        conn.commit()
        
        print(f"✅ Cleared {count} unused data records!")

def backup_database(db: MalnutritionDB, backup_path: str = None):
    """Create a backup of the database."""
    if not backup_path:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f"../data/malnutrition_backup_{timestamp}.db"
    
    import shutil
    try:
        shutil.copy2(db.db_path, backup_path)
        print(f"✅ Database backed up to: {backup_path}")
    except Exception as e:
        print(f"❌ Backup failed: {e}")

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Malnutrition Detection Database Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python db_manager.py stats                    # View database statistics
  python db_manager.py view uploaded           # View uploaded data
  python db_manager.py view sessions           # View training sessions
  python db_manager.py view metrics            # View training metrics
  python db_manager.py clear all               # Clear entire database
  python db_manager.py clear unused            # Clear only unused data
  python db_manager.py backup                  # Backup database
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Stats command
    subparsers.add_parser('stats', help='Show database statistics')
    
    # View command
    view_parser = subparsers.add_parser('view', help='View database content')
    view_parser.add_argument('type', choices=['uploaded', 'sessions', 'metrics'], 
                           help='Type of data to view')
    view_parser.add_argument('--session', help='Session ID for metrics (optional)')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear database data')
    clear_parser.add_argument('type', choices=['all', 'unused'], 
                            help='Type of data to clear')
    clear_parser.add_argument('--yes', action='store_true', 
                            help='Skip confirmation prompt')
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Backup database')
    backup_parser.add_argument('--path', help='Backup file path (optional)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    setup_logging()
    
    # Initialize database
    try:
        db = MalnutritionDB()
        print(f"🗃️  Connected to database: {db.db_path}")
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        sys.exit(1)
    
    # Execute commands
    try:
        if args.command == 'stats':
            view_stats(db)
        
        elif args.command == 'view':
            if args.type == 'uploaded':
                view_uploaded_data(db)
            elif args.type == 'sessions':
                view_training_sessions(db)
            elif args.type == 'metrics':
                view_training_metrics(db, args.session)
        
        elif args.command == 'clear':
            if args.type == 'all':
                clear_database(db, args.yes)
            elif args.type == 'unused':
                clear_unused_data(db)
        
        elif args.command == 'backup':
            backup_database(db, args.path)
            
    except Exception as e:
        print(f"❌ Error executing command: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 