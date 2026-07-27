"""
eDNA Classification System - Advanced GUI with RBAC & Blockchain
Made by Team AGNI
Version: 2.0 Final
"""

import sys
import sqlite3
import pickle
import hashlib
import joblib
from datetime import datetime
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QTabWidget,
    QTableWidget, QTableWidgetItem, QProgressBar, QGroupBox,
    QFileDialog, QMessageBox, QHeaderView, QDialog, QComboBox,
    QFormLayout, QDialogButtonBox, QCheckBox, QSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QPalette, QColor, QPainter
from PyQt6.QtCharts import QChart, QChartView, QPieSeries

import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

# Optional: Blockchain integration
try:
    from web3 import Web3
    BLOCKCHAIN_AVAILABLE = True
except ImportError:
    BLOCKCHAIN_AVAILABLE = False
    print("Warning: web3 not installed. Blockchain features disabled.")


class BlockchainManager:
    """Manages blockchain interactions for prediction logging"""
    
    def __init__(self):
        self.enabled = False
        self.w3 = None
        self.contract = None
        self.wallet = None
        self.private_key = None
        
        if BLOCKCHAIN_AVAILABLE:
            self.rpc_url = "https://eth-sepolia.g.alchemy.com/v2/eBJicIGT5fCVstSsNDYNn"
            self.contract_address = "0x4daDCa8f7e34651F3670A702A3D41EB962bE0fd2"
            self.wallet = "0x2615F9daC89A9A21bf4B1496213e407977D81A86"
            self.private_key = "YOUR_PRIVATE_KEY"
            
            self.abi = [
                {
                    "inputs": [
                        {"name": "dnaHash", "type": "string"},
                        {"name": "predictedClass", "type": "string"},
                        {"name": "confidence", "type": "uint256"}
                    ],
                    "name": "logPrediction",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                }
            ]
    
    def connect(self):
        """Initialize blockchain connection"""
        if not BLOCKCHAIN_AVAILABLE:
            return False
            
        try:
            self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
            if self.w3.is_connected():
                self.contract = self.w3.eth.contract(
                    address=self.contract_address,
                    abi=self.abi
                )
                self.enabled = True
                return True
            return False
        except Exception as e:
            print(f"Blockchain connection failed: {e}")
            return False
    
    def log_prediction(self, sequence, label, confidence):
        """Log prediction to blockchain"""
        if not self.enabled or not self.w3:
            return None
        
        try:
            # Create DNA hash
            dna_hash = hashlib.sha256(sequence.encode()).hexdigest()
            
            # Get nonce
            nonce = self.w3.eth.get_transaction_count(self.wallet)
            
            # Build transaction
            tx = self.contract.functions.logPrediction(
                dna_hash,
                label,
                int(confidence * 100)
            ).build_transaction({
                'from': self.wallet,
                'nonce': nonce,
                'gas': 500000,
                'gasPrice': self.w3.to_wei('10', 'gwei')
            })
            
            # Sign and send
            signed = self.w3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
            
            return tx_hash.hex()
            
        except Exception as e:
            print(f"Blockchain logging failed: {e}")
            return None


class DatabaseManager:
    """SQLite database manager for users, predictions, and audit logs"""
    
    def __init__(self, db_path="edna_system.db"):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        return sqlite3.connect(self.db_path)
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                full_name TEXT,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                active INTEGER DEFAULT 1
            )
        ''')
        
        # Predictions table with blockchain hash
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                sequence TEXT NOT NULL,
                result TEXT NOT NULL,
                confidence REAL NOT NULL,
                blockchain_tx TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Audit logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action TEXT NOT NULL,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Models table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                model_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_path TEXT NOT NULL,
                trained_by INTEGER,
                accuracy REAL,
                num_classes INTEGER,
                num_samples INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                active INTEGER DEFAULT 0,
                FOREIGN KEY (trained_by) REFERENCES users (user_id)
            )
        ''')
        
        conn.commit()
        
        # Create default admin user if no users exist
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            self.create_user("admin", "admin123", "Administrator", "Admin User", "admin@agni.team")
            self.create_user("scientist", "scientist123", "Scientist", "Research Scientist", "scientist@agni.team")
            self.create_user("viewer", "viewer123", "Viewer", "Data Viewer", "viewer@agni.team")
        
        conn.close()
    
    def hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, username, password, role, full_name="", email=""):
        """Create a new user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            password_hash = self.hash_password(password)
            cursor.execute('''
                INSERT INTO users (username, password_hash, role, full_name, email)
                VALUES (?, ?, ?, ?, ?)
            ''', (username, password_hash, role, full_name, email))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()
    
    def authenticate_user(self, username, password):
        """Authenticate user and return user data"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        password_hash = self.hash_password(password)
        cursor.execute('''
            SELECT user_id, username, role, full_name, email, active
            FROM users
            WHERE username = ? AND password_hash = ? AND active = 1
        ''', (username, password_hash))
        
        user = cursor.fetchone()
        
        if user:
            # Update last login
            cursor.execute('''
                UPDATE users SET last_login = CURRENT_TIMESTAMP
                WHERE user_id = ?
            ''', (user[0],))
            conn.commit()
        
        conn.close()
        return user
    
    def log_action(self, user_id, action, details=""):
        """Log user action"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO audit_logs (user_id, action, details)
            VALUES (?, ?, ?)
        ''', (user_id, action, details))
        
        conn.commit()
        conn.close()
    
    def save_prediction(self, user_id, sequence, result, confidence, blockchain_tx=None):
        """Save prediction to database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (user_id, sequence, result, confidence, blockchain_tx)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, sequence[:100], result, confidence, blockchain_tx))
        
        conn.commit()
        conn.close()
    
    def get_predictions(self, user_id=None, limit=100):
        """Get predictions history"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if user_id:
            cursor.execute('''
                SELECT p.timestamp, p.sequence, p.result, p.confidence, u.username, p.blockchain_tx
                FROM predictions p
                JOIN users u ON p.user_id = u.user_id
                WHERE p.user_id = ?
                ORDER BY p.timestamp DESC
                LIMIT ?
            ''', (user_id, limit))
        else:
            cursor.execute('''
                SELECT p.timestamp, p.sequence, p.result, p.confidence, u.username, p.blockchain_tx
                FROM predictions p
                JOIN users u ON p.user_id = u.user_id
                ORDER BY p.timestamp DESC
                LIMIT ?
            ''', (limit,))
        
        predictions = cursor.fetchall()
        conn.close()
        return predictions
    
    def save_model_info(self, model_name, model_path, trained_by, accuracy, num_classes, num_samples):
        """Save model information"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Deactivate all previous models
        cursor.execute("UPDATE models SET active = 0")
        
        cursor.execute('''
            INSERT INTO models (model_name, model_path, trained_by, accuracy, num_classes, num_samples, active)
            VALUES (?, ?, ?, ?, ?, ?, 1)
        ''', (model_name, model_path, trained_by, accuracy, num_classes, num_samples))
        
        conn.commit()
        conn.close()
    
    def get_all_users(self):
        """Get all users (admin only)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_id, username, role, full_name, email, created_at, last_login, active
            FROM users
            ORDER BY user_id
        ''')
        
        users = cursor.fetchall()
        conn.close()
        return users
    
    def get_audit_logs(self, limit=100):
        """Get audit logs (admin only)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT a.timestamp, u.username, a.action, a.details
            FROM audit_logs a
            JOIN users u ON a.user_id = u.user_id
            ORDER BY a.timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        logs = cursor.fetchall()
        conn.close()
        return logs


class LoginDialog(QDialog):
    """Login dialog for user authentication"""
    
    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.user_data = None
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("eDNA System Login - Team AGNI")
        self.setFixedSize(450, 380)
        self.setModal(True)
        
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)
        
        # Logo/Title
        title = QLabel("eDNA Classification System")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #ffffff; margin-bottom: 10px;")
        
        subtitle = QLabel("Made by Team AGNI")
        subtitle.setFont(QFont("Segoe UI", 11))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #cccccc; margin-bottom: 20px;")
        
        # Form
        form_layout = QFormLayout()
        form_layout.setSpacing(15)
        
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Enter username")
        self.username_input.setMinimumHeight(35)
        
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Enter password")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setMinimumHeight(35)
        self.password_input.returnPressed.connect(self.attempt_login)
        
        form_layout.addRow("Username:", self.username_input)
        form_layout.addRow("Password:", self.password_input)
        
        # Login button
        login_btn = QPushButton("Login")
        login_btn.setMinimumHeight(40)
        login_btn.clicked.connect(self.attempt_login)
        login_btn.setDefault(True)
        
        # Default credentials info
        blockchain_status = "🔗 Blockchain: Enabled" if BLOCKCHAIN_AVAILABLE else "⚠️ Blockchain: Disabled"
        info_label = QLabel(
            f"{blockchain_status}\n\n"
            "Default Users:\n"
            "• admin / admin123 (Administrator)\n"
            "• scientist / scientist123 (Scientist)\n"
            "• viewer / viewer123 (Viewer)"
        )
        info_label.setFont(QFont("Segoe UI", 8))
        info_label.setStyleSheet("color: #999999; margin-top: 15px; padding: 10px; background-color: #2a2a2a; border-radius: 4px;")
        
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addLayout(form_layout)
        layout.addWidget(login_btn)
        layout.addWidget(info_label)
        layout.addStretch()
        
        self.setLayout(layout)
        
        # Apply dark theme
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1a1a;
            }
            QLineEdit {
                background-color: #2a2a2a;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 8px;
                color: #ffffff;
                font-size: 11pt;
            }
            QLineEdit:focus {
                border: 2px solid #000000;
                background-color: #333333;
            }
            QPushButton {
                background-color: #000000;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: 600;
                font-size: 11pt;
            }
            QPushButton:hover {
                background-color: #333333;
            }
            QPushButton:pressed {
                background-color: #444444;
            }
            QLabel {
                color: #ffffff;
            }
        """)
    
    def attempt_login(self):
        username = self.username_input.text().strip()
        password = self.password_input.text()
        
        if not username or not password:
            QMessageBox.warning(self, "Error", "Please enter both username and password")
            return
        
        user = self.db_manager.authenticate_user(username, password)
        
        if user:
            self.user_data = {
                'user_id': user[0],
                'username': user[1],
                'role': user[2],
                'full_name': user[3],
                'email': user[4],
                'active': user[5]
            }
            self.db_manager.log_action(user[0], "LOGIN", f"User {username} logged in")
            self.accept()
        else:
            QMessageBox.warning(self, "Login Failed", "Invalid username or password")
            self.password_input.clear()


class ModelTrainer(QThread):
    """Background thread for model training with normalized k-mers"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(object, object, object, object, float)
    error = pyqtSignal(str)

    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path

    def run(self):
        try:
            self.progress.emit("Loading dataset...")
            
            # Clean DNA
            def clean_sequence(seq):
                if pd.isna(seq):
                    return ""
                seq = seq.upper()
                return re.sub(r"[^ACGTU]", "", seq)

            # Load CSV
            df = pd.read_csv(self.csv_path, header=None)
            df.columns = ["sequence", "class", "genus"]
            
            df["sequence"] = df["sequence"].apply(clean_sequence)
            df = df[df["sequence"].str.len() > 10]
            df["class"] = df["class"].fillna("Unknown")
            
            self.progress.emit(f"Original dataset: {df.shape[0]} sequences")

            # Remove rare classes
            counts = df["class"].value_counts()
            valid_classes = counts[counts >= 3].index
            df = df[df["class"].isin(valid_classes)]
            
            self.progress.emit(f"After cleaning: {df.shape[0]} sequences, {df['class'].nunique()} classes")

            if df["class"].nunique() < 2:
                raise ValueError("Not enough classes to train.")

            # K-mer features (NORMALIZED like in blockchain version)
            K = 4
            
            def kmer_counts(seq):
                return Counter(seq[i:i+K] for i in range(len(seq)-K+1))

            self.progress.emit("Building k-mer space...")
            
            all_kmers = sorted({
                k for seq in df["sequence"]
                for k in kmer_counts(seq)
            })
            
            def vectorize(seq):
                c = kmer_counts(seq)
                total = sum(c.values()) + 1e-8  # Normalization
                return [c.get(k, 0) / total for k in all_kmers]

            self.progress.emit(f"Vectorizing {len(df)} sequences...")
            X = np.array([vectorize(seq) for seq in df["sequence"]])

            # Label encoding
            encoder = LabelEncoder()
            y = encoder.fit_transform(df["class"])

            # Train test split
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=42
                )
                self.progress.emit("Using stratified split")
            except:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                self.progress.emit("Using random split")

            # XGBoost training
            self.progress.emit("Training XGBoost model...")
            model = XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                eval_metric="mlogloss"
            )
            
            model.fit(X_train, y_train)
            
            accuracy = model.score(X_test, y_test)
            self.progress.emit(f"Training complete! Test accuracy: {accuracy:.2%}")
            
            self.finished.emit(model, encoder, all_kmers, df, accuracy)
            
        except Exception as e:
            self.error.emit(str(e))


class UserManagementDialog(QDialog):
    """User management dialog (Admin only)"""
    
    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self.db_manager = db_manager
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("User Management")
        self.setMinimumSize(800, 500)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("User Management")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        
        # Users table
        self.users_table = QTableWidget()
        self.users_table.setColumnCount(7)
        self.users_table.setHorizontalHeaderLabels([
            "ID", "Username", "Role", "Full Name", "Email", "Last Login", "Active"
        ])
        self.users_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        add_btn = QPushButton("Add User")
        add_btn.clicked.connect(self.add_user)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.load_users)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(refresh_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        
        layout.addWidget(title)
        layout.addWidget(self.users_table)
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
        self.load_users()
    
    def load_users(self):
        """Load all users into table"""
        users = self.db_manager.get_all_users()
        
        self.users_table.setRowCount(len(users))
        for row, user in enumerate(users):
            for col, value in enumerate(user):
                if col == 6:  # Active column
                    value = "Yes" if value == 1 else "No"
                self.users_table.setItem(row, col, QTableWidgetItem(str(value) if value else ""))
    
    def add_user(self):
        """Add new user dialog"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add New User")
        dialog.setMinimumWidth(400)
        
        layout = QFormLayout()
        
        username_input = QLineEdit()
        password_input = QLineEdit()
        password_input.setEchoMode(QLineEdit.EchoMode.Password)
        role_combo = QComboBox()
        role_combo.addItems(["Administrator", "Scientist", "Viewer"])
        fullname_input = QLineEdit()
        email_input = QLineEdit()
        
        layout.addRow("Username:", username_input)
        layout.addRow("Password:", password_input)
        layout.addRow("Role:", role_combo)
        layout.addRow("Full Name:", fullname_input)
        layout.addRow("Email:", email_input)
        
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        
        layout.addRow(buttons)
        dialog.setLayout(layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            if self.db_manager.create_user(
                username_input.text(),
                password_input.text(),
                role_combo.currentText(),
                fullname_input.text(),
                email_input.text()
            ):
                QMessageBox.information(self, "Success", "User created successfully")
                self.load_users()
            else:
                QMessageBox.warning(self, "Error", "Username already exists")


class eDNAMainWindow(QMainWindow):
    def __init__(self, user_data, db_manager, blockchain_manager):
        super().__init__()
        self.user_data = user_data
        self.db_manager = db_manager
        self.blockchain_manager = blockchain_manager
        self.model = None
        self.encoder = None
        self.all_kmers = None
        self.df = None
        self.prediction_history = []
        self.confidence_threshold = 0.7
        
        self.init_ui()
        
    def init_ui(self):
        blockchain_status = " + Blockchain" if self.blockchain_manager.enabled else ""
        self.setWindowTitle(f"eDNA System{blockchain_status} - {self.user_data['full_name']} ({self.user_data['role']})")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set dark theme
        self.set_theme()
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Header
        header = self.create_header()
        main_layout.addWidget(header)
        
        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setFont(QFont("Segoe UI", 10))
        
        # Tab 1: Model Training (Admin & Scientist only)
        if self.user_data['role'] in ['Administrator', 'Scientist']:
            self.tab_training = self.create_training_tab()
            self.tabs.addTab(self.tab_training, "Model Training")
        
        # Tab 2: Prediction
        self.tab_prediction = self.create_prediction_tab()
        self.tabs.addTab(self.tab_prediction, "DNA Classification")
        
        # Tab 3: Analytics
        self.tab_analytics = self.create_analytics_tab()
        self.tabs.addTab(self.tab_analytics, "Analytics & Reports")
        
        # Tab 4: Admin Panel (Admin only)
        if self.user_data['role'] == 'Administrator':
            self.tab_admin = self.create_admin_tab()
            self.tabs.addTab(self.tab_admin, "Administration")
        
        main_layout.addWidget(self.tabs)
        
        # Status bar
        blockchain_text = " | Blockchain: Connected" if self.blockchain_manager.enabled else ""
        self.statusBar().showMessage(f"Logged in as: {self.user_data['username']} ({self.user_data['role']}){blockchain_text}")
        self.statusBar().setFont(QFont("Segoe UI", 9))
        
    def set_theme(self):
        """Set dark theme with black accents"""
        palette = QPalette()
        
        # Dark colors
        palette.setColor(QPalette.ColorRole.Window, QColor(26, 26, 26))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Base, QColor(42, 42, 42))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(52, 52, 52))
        palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Button, QColor(42, 42, 42))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 0, 0))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        
        self.setPalette(palette)
        
        # Stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
            }
            QTabWidget::pane {
                border: 1px solid #444444;
                background-color: #1a1a1a;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #2a2a2a;
                color: #cccccc;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-weight: 500;
            }
            QTabBar::tab:selected {
                background-color: #1a1a1a;
                border-bottom: 2px solid #000000;
                color: #ffffff;
            }
            QTabBar::tab:hover {
                background-color: #3a3a3a;
            }
            QPushButton {
                background-color: #000000;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: 500;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #2a2a2a;
            }
            QPushButton:pressed {
                background-color: #3a3a3a;
            }
            QPushButton:disabled {
                background-color: #444444;
                color: #888888;
            }
            QLineEdit, QTextEdit {
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 8px;
                background-color: #2a2a2a;
                color: #ffffff;
                font-size: 10pt;
            }
            QLineEdit:focus, QTextEdit:focus {
                border: 2px solid #000000;
                background-color: #333333;
            }
            QGroupBox {
                border: 1px solid #444444;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: 600;
                background-color: #1a1a1a;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                color: #ffffff;
            }
            QTableWidget {
                border: 1px solid #444444;
                border-radius: 4px;
                background-color: #2a2a2a;
                gridline-color: #444444;
                color: #ffffff;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #000000;
                color: #ffffff;
                padding: 8px;
                border: none;
                border-bottom: 2px solid #444444;
                font-weight: 600;
            }
            QProgressBar {
                border: 1px solid #444444;
                border-radius: 4px;
                text-align: center;
                background-color: #2a2a2a;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background-color: #000000;
                border-radius: 3px;
            }
            QLabel {
                color: #ffffff;
            }
            QComboBox {
                background-color: #2a2a2a;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 5px;
            }
            QComboBox:focus {
                border: 2px solid #000000;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #2a2a2a;
                color: #ffffff;
                selection-background-color: #000000;
            }
        """)
        
    def create_header(self):
        """Create header section"""
        header_widget = QWidget()
        header_layout = QVBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 10)
        
        # Title row
        title_row = QHBoxLayout()
        
        title_col = QVBoxLayout()
        title = QLabel("eDNA Classification System")
        title.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        title.setStyleSheet("color: #ffffff; margin-bottom: 5px;")
        
        blockchain_text = " with Blockchain Integration" if self.blockchain_manager.enabled else ""
        subtitle = QLabel(f"Advanced Machine Learning for Environmental DNA Analysis{blockchain_text}")
        subtitle.setFont(QFont("Segoe UI", 11))
        subtitle.setStyleSheet("color: #999999; margin-bottom: 10px;")
        
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        
        title_row.addLayout(title_col)
        title_row.addStretch()
        
        # User info
        user_info = QLabel(f"User: {self.user_data['full_name']}\nRole: {self.user_data['role']}")
        user_info.setFont(QFont("Segoe UI", 9))
        user_info.setStyleSheet("color: #cccccc; padding: 10px; background-color: #2a2a2a; border-radius: 4px;")
        user_info.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        title_row.addWidget(user_info)
        
        # Team credit
        team = QLabel("Made by Team AGNI")
        team.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        team.setStyleSheet("color: #ffffff; padding: 8px 16px; background-color: #000000; border-radius: 4px;")
        team.setAlignment(Qt.AlignmentFlag.AlignCenter)
        team.setMaximumWidth(200)
        
        header_layout.addLayout(title_row)
        header_layout.addWidget(team)
        
        return header_widget
    
    def create_training_tab(self):
        """Create model training tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Dataset section
        dataset_group = QGroupBox("Dataset Configuration")
        dataset_layout = QVBoxLayout()
        
        file_layout = QHBoxLayout()
        self.csv_path_edit = QLineEdit()
        self.csv_path_edit.setPlaceholderText("Select CSV file containing eDNA sequences...")
        self.csv_path_edit.setText("output_10000.csv")
        
        browse_btn = QPushButton("Browse")
        browse_btn.setMaximumWidth(100)
        browse_btn.clicked.connect(self.browse_csv)
        
        file_layout.addWidget(self.csv_path_edit)
        file_layout.addWidget(browse_btn)
        dataset_layout.addLayout(file_layout)
        
        dataset_group.setLayout(dataset_layout)
        layout.addWidget(dataset_group)
        
        # Model name
        model_name_group = QGroupBox("Model Configuration")
        model_name_layout = QFormLayout()
        
        self.model_name_input = QLineEdit()
        self.model_name_input.setText(f"edna_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        model_name_layout.addRow("Model Name:", self.model_name_input)
        model_name_group.setLayout(model_name_layout)
        layout.addWidget(model_name_group)
        
        # Training controls
        controls_group = QGroupBox("Training Controls")
        controls_layout = QVBoxLayout()
        
        self.train_btn = QPushButton("Start Training")
        self.train_btn.setMinimumHeight(40)
        self.train_btn.clicked.connect(self.start_training)
        
        self.training_progress = QProgressBar()
        self.training_progress.setMinimum(0)
        self.training_progress.setMaximum(0)
        self.training_progress.hide()
        
        controls_layout.addWidget(self.train_btn)
        controls_layout.addWidget(self.training_progress)
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Training log
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout()
        
        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setFont(QFont("Consolas", 9))
        self.training_log.setMaximumHeight(300)
        
        log_layout.addWidget(self.training_log)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        layout.addStretch()
        
        return tab
    
    def create_prediction_tab(self):
        """Create prediction tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Input section
        input_group = QGroupBox("DNA Sequence Input")
        input_layout = QVBoxLayout()
        
        self.sequence_input = QTextEdit()
        self.sequence_input.setPlaceholderText("Enter DNA sequence (ACGT format)...\nExample: ATCGATCGATCG")
        self.sequence_input.setMaximumHeight(120)
        self.sequence_input.setFont(QFont("Consolas", 11))
        
        btn_layout = QHBoxLayout()
        self.predict_btn = QPushButton("Classify Sequence")
        self.predict_btn.setMinimumHeight(40)
        self.predict_btn.clicked.connect(self.predict_sequence)
        self.predict_btn.setEnabled(False)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setMaximumWidth(100)
        self.clear_btn.setStyleSheet("background-color: #444444;")
        self.clear_btn.clicked.connect(self.sequence_input.clear)
        
        btn_layout.addWidget(self.predict_btn)
        btn_layout.addWidget(self.clear_btn)
        
        # Confidence threshold
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("Confidence Threshold:")
        threshold_label.setFont(QFont("Segoe UI", 10))
        
        self.threshold_input = QLineEdit("0.70")
        self.threshold_input.setMaximumWidth(80)
        self.threshold_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Blockchain toggle
        self.blockchain_checkbox = QCheckBox("Log to Blockchain")
        self.blockchain_checkbox.setChecked(self.blockchain_manager.enabled)
        self.blockchain_checkbox.setEnabled(BLOCKCHAIN_AVAILABLE)
        
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.threshold_input)
        threshold_layout.addStretch()
        threshold_layout.addWidget(self.blockchain_checkbox)
        
        input_layout.addWidget(self.sequence_input)
        input_layout.addLayout(btn_layout)
        input_layout.addLayout(threshold_layout)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Result section
        result_group = QGroupBox("Classification Result")
        result_layout = QVBoxLayout()
        
        self.result_label = QLabel("No prediction yet")
        self.result_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setMinimumHeight(60)
        self.result_label.setStyleSheet("background-color: #2a2a2a; border-radius: 4px; padding: 15px; color: #ffffff;")
        
        self.blockchain_status_label = QLabel("")
        self.blockchain_status_label.setFont(QFont("Segoe UI", 9))
        self.blockchain_status_label.setStyleSheet("color: #999999; padding: 5px;")
        self.blockchain_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        result_layout.addWidget(self.result_label)
        result_layout.addWidget(self.blockchain_status_label)
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        
        # History section
        history_group = QGroupBox("Prediction History")
        history_layout = QVBoxLayout()
        
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels(["Timestamp", "Sequence", "Result", "Confidence", "Blockchain TX"])
        self.history_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.history_table.setMaximumHeight(250)
        
        history_layout.addWidget(self.history_table)
        history_group.setLayout(history_layout)
        layout.addWidget(history_group)
        
        layout.addStretch()
        
        return tab
    
    def create_analytics_tab(self):
        """Create analytics tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Charts section
        charts_group = QGroupBox("Dataset Analytics")
        charts_layout = QHBoxLayout()
        
        # Class distribution chart
        self.class_chart_view = QChartView()
        self.class_chart_view.setMinimumHeight(300)
        self.class_chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        charts_layout.addWidget(self.class_chart_view)
        charts_group.setLayout(charts_layout)
        layout.addWidget(charts_group)
        
        # Statistics section
        stats_group = QGroupBox("Dataset Statistics")
        stats_layout = QVBoxLayout()
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(150)
        self.stats_text.setFont(QFont("Segoe UI", 10))
        
        stats_layout.addWidget(self.stats_text)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Report generation
        report_group = QGroupBox("Model & Report Management")
        report_layout = QHBoxLayout()
        
        self.generate_report_btn = QPushButton("Generate PDF Report")
        self.generate_report_btn.setMinimumHeight(40)
        self.generate_report_btn.clicked.connect(self.generate_report)
        self.generate_report_btn.setEnabled(False)
        
        self.load_model_btn = QPushButton("Load Model (PKL)")
        self.load_model_btn.setMinimumHeight(40)
        self.load_model_btn.clicked.connect(self.load_model)
        
        report_layout.addWidget(self.generate_report_btn)
        report_layout.addWidget(self.load_model_btn)
        report_group.setLayout(report_layout)
        layout.addWidget(report_group)
        
        layout.addStretch()
        
        return tab
    
    def create_admin_tab(self):
        """Create admin tab (Administrator only)"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # User management
        user_group = QGroupBox("User Management")
        user_layout = QVBoxLayout()
        
        manage_users_btn = QPushButton("Manage Users")
        manage_users_btn.setMinimumHeight(40)
        manage_users_btn.clicked.connect(self.open_user_management)
        
        user_layout.addWidget(manage_users_btn)
        user_group.setLayout(user_layout)
        layout.addWidget(user_group)
        
        # Audit logs
        audit_group = QGroupBox("Audit Logs")
        audit_layout = QVBoxLayout()
        
        self.audit_table = QTableWidget()
        self.audit_table.setColumnCount(4)
        self.audit_table.setHorizontalHeaderLabels(["Timestamp", "User", "Action", "Details"])
        self.audit_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        refresh_logs_btn = QPushButton("Refresh Logs")
        refresh_logs_btn.clicked.connect(self.load_audit_logs)
        
        audit_layout.addWidget(self.audit_table)
        audit_layout.addWidget(refresh_logs_btn)
        audit_group.setLayout(audit_layout)
        layout.addWidget(audit_group)
        
        # Load initial logs
        self.load_audit_logs()
        
        layout.addStretch()
        
        return tab
    
    def browse_csv(self):
        """Browse for CSV file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "", "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            self.csv_path_edit.setText(file_path)
    
    def start_training(self):
        """Start model training"""
        csv_path = self.csv_path_edit.text()
        
        if not csv_path or not Path(csv_path).exists():
            QMessageBox.warning(self, "Error", "Please select a valid CSV file")
            return
        
        self.train_btn.setEnabled(False)
        self.training_progress.show()
        self.training_log.clear()
        self.training_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting training pipeline...")
        
        # Log action
        self.db_manager.log_action(
            self.user_data['user_id'],
            "TRAIN_MODEL",
            f"Started training with {csv_path}"
        )
        
        # Start training thread
        self.trainer = ModelTrainer(csv_path)
        self.trainer.progress.connect(self.update_training_log)
        self.trainer.finished.connect(self.training_complete)
        self.trainer.error.connect(self.training_error)
        self.trainer.start()
    
    def update_training_log(self, message):
        """Update training log"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.training_log.append(f"[{timestamp}] {message}")
        self.statusBar().showMessage(message)
    
    def training_complete(self, model, encoder, all_kmers, df, accuracy):
        """Handle training completion"""
        self.model = model
        self.encoder = encoder
        self.all_kmers = all_kmers
        self.df = df
        
        self.training_progress.hide()
        self.train_btn.setEnabled(True)
        self.predict_btn.setEnabled(True)
        self.generate_report_btn.setEnabled(True)
        
        self.training_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Model ready for predictions!")
        self.statusBar().showMessage("Model training complete")
        
        # Save model as PKL (using joblib like in blockchain version)
        model_name = self.model_name_input.text()
        model_path = f"models/{model_name}.pkl"
        Path("models").mkdir(exist_ok=True)
        
        # Save using joblib (compatible with blockchain version)
        try:
            joblib.dump((model, all_kmers, encoder), model_path)
            self.training_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Model saved to {model_path} (joblib format)")
        except Exception as e:
            self.training_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Warning: joblib save failed, using pickle")
            # Fallback to pickle
            model_data = {
                'model': model,
                'encoder': encoder,
                'all_kmers': all_kmers,
                'df': df,
                'accuracy': accuracy,
                'trained_at': datetime.now().isoformat(),
                'trained_by': self.user_data['username']
            }
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            self.training_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Model saved to {model_path} (pickle format)")
        
        # Save to database
        self.db_manager.save_model_info(
            model_name,
            model_path,
            self.user_data['user_id'],
            accuracy,
            df['class'].nunique(),
            len(df)
        )
        
        # Log action
        self.db_manager.log_action(
            self.user_data['user_id'],
            "MODEL_TRAINED",
            f"Model {model_name} trained with {accuracy:.2%} accuracy"
        )
        
        # Update analytics
        self.update_analytics()
        
        QMessageBox.information(self, "Success", f"Model training completed!\nAccuracy: {accuracy:.2%}\nSaved as: {model_path}")
    
    def training_error(self, error_msg):
        """Handle training error"""
        self.training_progress.hide()
        self.train_btn.setEnabled(True)
        self.training_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {error_msg}")
        
        # Log error
        self.db_manager.log_action(
            self.user_data['user_id'],
            "TRAINING_ERROR",
            error_msg
        )
        
        QMessageBox.critical(self, "Training Error", f"Training failed:\n{error_msg}")
    
    def predict_sequence(self):
        """Predict DNA sequence classification with optional blockchain logging"""
        if self.model is None:
            QMessageBox.warning(self, "Error", "Please train or load a model first")
            return
        
        sequence = self.sequence_input.toPlainText().strip()
        
        if not sequence:
            QMessageBox.warning(self, "Error", "Please enter a DNA sequence")
            return
        
        # Clean sequence
        sequence = sequence.upper()
        sequence = re.sub(r"[^ACGTU]", "", sequence)
        
        if len(sequence) < 4:
            QMessageBox.warning(self, "Error", "Sequence too short (minimum 4 bases)")
            return
        
        try:
            threshold = float(self.threshold_input.text())
        except:
            threshold = 0.7
            self.threshold_input.setText("0.70")
        
        # Vectorize (with normalization)
        K = 4
        def kmer_counts(seq):
            return Counter(seq[i:i+K] for i in range(len(seq)-K+1))
        
        def vectorize(seq):
            c = kmer_counts(seq)
            total = sum(c.values()) + 1e-8
            return [c.get(k, 0) / total for k in self.all_kmers]
        
        vec = np.array([vectorize(sequence)])
        
        # Predict
        probs = self.model.predict_proba(vec)[0]
        confidence = max(probs)
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        seq_display = sequence[:20] + "..." if len(sequence) > 20 else sequence
        
        blockchain_tx = None
        
        if confidence < threshold:
            result = "UNKNOWN"
            self.result_label.setText(f"Classification: UNKNOWN\nConfidence: {confidence:.2%}")
            self.result_label.setStyleSheet("background-color: #3a3a00; color: #ffff00; border-radius: 4px; padding: 15px; font-size: 14pt; font-weight: bold;")
            self.blockchain_status_label.setText("")
        else:
            pred = self.model.predict(vec)
            label = self.encoder.inverse_transform(pred)[0]
            result = label
            self.result_label.setText(f"Classification: {label}\nConfidence: {confidence:.2%}")
            self.result_label.setStyleSheet("background-color: #003a00; color: #00ff00; border-radius: 4px; padding: 15px; font-size: 14pt; font-weight: bold;")
            
            # Log to blockchain if enabled
            if self.blockchain_checkbox.isChecked() and self.blockchain_manager.enabled:
                self.blockchain_status_label.setText("🔗 Logging to blockchain...")
                QApplication.processEvents()  # Update UI
                
                blockchain_tx = self.blockchain_manager.log_prediction(sequence, label, confidence)
                
                if blockchain_tx:
                    self.blockchain_status_label.setText(f"🔗 Blockchain TX: {blockchain_tx[:16]}...")
                    self.blockchain_status_label.setStyleSheet("color: #00ff00; padding: 5px;")
                else:
                    self.blockchain_status_label.setText("⚠️ Blockchain logging failed")
                    self.blockchain_status_label.setStyleSheet("color: #ffaa00; padding: 5px;")
            else:
                self.blockchain_status_label.setText("")
        
        # Save to database
        self.db_manager.save_prediction(
            self.user_data['user_id'],
            sequence,
            result,
            confidence,
            blockchain_tx
        )
        
        # Log action
        blockchain_note = f" (Blockchain: {blockchain_tx[:16]}...)" if blockchain_tx else ""
        self.db_manager.log_action(
            self.user_data['user_id'],
            "PREDICTION",
            f"Classified sequence as {result} with {confidence:.2%} confidence{blockchain_note}"
        )
        
        # Add to history
        row = self.history_table.rowCount()
        self.history_table.insertRow(row)
        self.history_table.setItem(row, 0, QTableWidgetItem(timestamp))
        self.history_table.setItem(row, 1, QTableWidgetItem(seq_display))
        self.history_table.setItem(row, 2, QTableWidgetItem(result))
        self.history_table.setItem(row, 3, QTableWidgetItem(f"{confidence:.2%}"))
        self.history_table.setItem(row, 4, QTableWidgetItem(blockchain_tx[:16] + "..." if blockchain_tx else "N/A"))
        
        self.prediction_history.append({
            'timestamp': timestamp,
            'sequence': sequence,
            'result': result,
            'confidence': confidence,
            'blockchain_tx': blockchain_tx
        })
    
    def load_model(self):
        """Load model from PKL file (supports both joblib and pickle formats)"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Model", "models", "Pickle Files (*.pkl);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # Try joblib format first (blockchain version)
            try:
                model, all_kmers, encoder = joblib.load(file_path)
                self.model = model
                self.encoder = encoder
                self.all_kmers = all_kmers
                self.df = None
                accuracy = 0
                trained_by = "Unknown"
                trained_at = "Unknown"
                load_method = "joblib"
            except:
                # Fallback to pickle format (GUI version)
                with open(file_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.model = model_data['model']
                self.encoder = model_data['encoder']
                self.all_kmers = model_data['all_kmers']
                self.df = model_data.get('df')
                accuracy = model_data.get('accuracy', 0)
                trained_by = model_data.get('trained_by', 'Unknown')
                trained_at = model_data.get('trained_at', 'Unknown')
                load_method = "pickle"
            
            self.predict_btn.setEnabled(True)
            self.generate_report_btn.setEnabled(True)
            
            # Update analytics if df is available
            if self.df is not None:
                self.update_analytics()
            
            # Log action
            self.db_manager.log_action(
                self.user_data['user_id'],
                "LOAD_MODEL",
                f"Loaded model from {file_path} ({load_method} format)"
            )
            
            QMessageBox.information(
                self,
                "Model Loaded",
                f"Model loaded successfully ({load_method} format)!\n\n"
                f"Accuracy: {accuracy:.2%}\n"
                f"Trained by: {trained_by}\n"
                f"Trained at: {trained_at}\n"
                f"K-mers: {len(self.all_kmers)}"
            )
            
            self.statusBar().showMessage(f"Model loaded: {Path(file_path).name}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{str(e)}")
    
    def update_analytics(self):
        """Update analytics visualizations"""
        if self.df is None:
            return
        
        # Class distribution pie chart
        class_counts = self.df['class'].value_counts()
        
        series = QPieSeries()
        for cls, count in class_counts.head(10).items():
            series.append(f"{cls} ({count})", count)
        
        for slice in series.slices():
            slice.setLabelVisible(True)
            slice.setLabelPosition(slice.LabelPosition.LabelOutside)
        
        chart = QChart()
        chart.addSeries(series)
        chart.setTitle("Top 10 Class Distribution")
        chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)
        chart.legend().setAlignment(Qt.AlignmentFlag.AlignRight)
        chart.setBackgroundBrush(QColor(26, 26, 26))
        chart.setTitleBrush(QColor(255, 255, 255))
        chart.legend().setLabelColor(QColor(255, 255, 255))
        
        self.class_chart_view.setChart(chart)
        
        # Statistics
        stats_text = f"""
Dataset Overview:
• Total Sequences: {len(self.df):,}
• Number of Classes: {self.df['class'].nunique():,}
• Average Sequence Length: {self.df['sequence'].str.len().mean():.1f} bases
• Min Sequence Length: {self.df['sequence'].str.len().min()} bases
• Max Sequence Length: {self.df['sequence'].str.len().max()} bases

Top 5 Classes:
"""
        for idx, (cls, count) in enumerate(class_counts.head(5).items(), 1):
            stats_text += f"{idx}. {cls}: {count:,} sequences ({count/len(self.df)*100:.1f}%)\n"
        
        self.stats_text.setText(stats_text.strip())
    
    def generate_report(self):
        """Generate PDF report with proper error handling"""
        if self.model is None:
            QMessageBox.warning(self, "Error", "Please train or load a model first")
            return
        
        # Create reports directory if it doesn't exist
        Path("reports").mkdir(exist_ok=True)
        
        default_filename = f"reports/eDNA_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Report", default_filename,
            "PDF Files (*.pdf)"
        )
        
        if not file_path:
            return
        
        try:
            self.create_pdf_report(file_path)
            
            # Log action
            self.db_manager.log_action(
                self.user_data['user_id'],
                "GENERATE_REPORT",
                f"Generated PDF report: {file_path}"
            )
            
            QMessageBox.information(self, "Success", f"Report generated successfully!\n\n{file_path}")
            self.statusBar().showMessage(f"Report saved: {file_path}")
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            QMessageBox.critical(self, "Error", f"Failed to generate report:\n{str(e)}\n\nDetails:\n{error_details}")
    
    def create_pdf_report(self, filename):
        """Create PDF report using ReportLab with proper error handling"""
        try:
            doc = SimpleDocTemplate(
                filename,
                pagesize=letter,
                leftMargin=0.75*inch,
                rightMargin=0.75*inch,
                topMargin=0.75*inch,
                bottomMargin=0.75*inch
            )
            
            story = []
            styles = getSampleStyleSheet()
            
            # Custom title style
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1a1a1a'),
                spaceAfter=12,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            )
            
            # Add title
            story.append(Paragraph("eDNA Classification System Report", title_style))
            story.append(Paragraph("Made by Team AGNI", styles['Normal']))
            story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            story.append(Paragraph(f"Generated by: {self.user_data['full_name']} ({self.user_data['role']})", styles['Normal']))
            story.append(Spacer(1, 0.3*inch))
            
            # Dataset Overview (if available)
            if self.df is not None:
                story.append(Paragraph("Dataset Overview", styles['Heading2']))
                
                overview_data = [
                    ['Metric', 'Value'],
                    ['Total Sequences', f"{len(self.df):,}"],
                    ['Number of Classes', f"{self.df['class'].nunique():,}"],
                    ['Average Sequence Length', f"{self.df['sequence'].str.len().mean():.1f} bases"],
                    ['Min Sequence Length', f"{self.df['sequence'].str.len().min()} bases"],
                    ['Max Sequence Length', f"{self.df['sequence'].str.len().max()} bases"],
                ]
                
                overview_table = Table(overview_data, colWidths=[3*inch, 3*inch])
                overview_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.black),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ]))
                
                story.append(overview_table)
                story.append(Spacer(1, 0.3*inch))
                
                # Class Distribution
                story.append(Paragraph("Top 10 Class Distribution", styles['Heading2']))
                
                class_counts = self.df['class'].value_counts().head(10)
                class_data = [['Rank', 'Class', 'Count', 'Percentage']]
                
                for idx, (cls, count) in enumerate(class_counts.items(), 1):
                    percentage = f"{count/len(self.df)*100:.2f}%"
                    class_data.append([str(idx), str(cls), f"{count:,}", percentage])
                
                class_table = Table(class_data, colWidths=[0.75*inch, 2.5*inch, 1.5*inch, 1.5*inch])
                class_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.black),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 11),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ]))
                
                story.append(class_table)
                story.append(Spacer(1, 0.3*inch))
            
            # Prediction History from database
            predictions = self.db_manager.get_predictions(self.user_data['user_id'], limit=20)
            
            if predictions:
                story.append(PageBreak())
                story.append(Paragraph("Recent Predictions", styles['Heading2']))
                
                pred_data = [['Timestamp', 'Result', 'Confidence', 'Blockchain']]
                
                for pred in predictions:
                    blockchain_tx = pred[5] if len(pred) > 5 and pred[5] else "N/A"
                    if blockchain_tx != "N/A" and len(blockchain_tx) > 16:
                        blockchain_tx = blockchain_tx[:16] + "..."
                    
                    pred_data.append([
                        str(pred[0])[:19],  # Timestamp
                        str(pred[2]),       # Result
                        f"{pred[3]:.2%}",   # Confidence
                        blockchain_tx       # Blockchain TX
                    ])
                
                pred_table = Table(pred_data, colWidths=[1.5*inch, 2*inch, 1*inch, 1.75*inch])
                pred_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.black),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                ]))
                
                story.append(pred_table)
            
            # Footer
            story.append(Spacer(1, 0.5*inch))
            story.append(Paragraph("—" * 50, styles['Normal']))
            story.append(Paragraph("eDNA Classification System - Powered by XGBoost ML", styles['Normal']))
            blockchain_footer = " with Blockchain Integration" if BLOCKCHAIN_AVAILABLE else ""
            story.append(Paragraph(f"Made by Team AGNI{blockchain_footer}", styles['Normal']))
            
            # Build PDF
            doc.build(story)
            
        except Exception as e:
            raise Exception(f"PDF generation error: {str(e)}")
    
    def open_user_management(self):
        """Open user management dialog"""
        dialog = UserManagementDialog(self.db_manager, self)
        dialog.exec()
        
        # Log action
        self.db_manager.log_action(
            self.user_data['user_id'],
            "ADMIN_ACCESS",
            "Accessed user management"
        )
    
    def load_audit_logs(self):
        """Load audit logs"""
        logs = self.db_manager.get_audit_logs(limit=100)
        
        self.audit_table.setRowCount(len(logs))
        for row, log in enumerate(logs):
            for col, value in enumerate(log):
                self.audit_table.setItem(row, col, QTableWidgetItem(str(value)))


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    # Initialize database
    db_manager = DatabaseManager()
    
    # Initialize blockchain (optional)
    blockchain_manager = BlockchainManager()
    if BLOCKCHAIN_AVAILABLE:
        print("Connecting to blockchain...")
        if blockchain_manager.connect():
            print("✓ Blockchain connected")
        else:
            print("✗ Blockchain connection failed (will run without blockchain)")
    
    # Show login dialog
    login_dialog = LoginDialog(db_manager)
    
    if login_dialog.exec() == QDialog.DialogCode.Accepted:
        user_data = login_dialog.user_data
        
        # Show main window
        window = eDNAMainWindow(user_data, db_manager, blockchain_manager)
        window.show()
        
        sys.exit(app.exec())
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
