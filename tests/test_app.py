import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import sqlite3

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import app as flask_app
from app import app, _SQLiteCursorShim

# Use a test database path
TEST_DB_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_app.db")
flask_app.DB_PATH = TEST_DB_PATH

class TestFlaskApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create the test DB
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)
        flask_app.init_db()

    @classmethod
    def tearDownClass(cls):
        # Remove the test DB
        if os.path.exists(TEST_DB_PATH):
            try:
                os.remove(TEST_DB_PATH)
            except Exception:
                pass

    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        app.config['SECRET_KEY'] = 'test-secret-key'
        self.client = app.test_client()

    def test_sqlite_cursor_shim(self):
        """Test SQLiteCursorShim translation of mysql syntax"""
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        shim = _SQLiteCursorShim(cursor)
        
        # Test query parameter translation (%s to ?)
        shim.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
            ("testuser_shim", "test_shim@test.com", "hash123")
        )
        self.assertEqual(shim.lastrowid, 1)

        # Test SELECT LAST_INSERT_ID() emulation
        shim.execute("SELECT LAST_INSERT_ID()")
        res = shim.fetchone()
        self.assertEqual(res, (1,))
        conn.close()

    def test_home_page_route(self):
        """Test homepage routing and template rendering"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_about_route(self):
        """Test about page route"""
        response = self.client.get('/about')
        self.assertEqual(response.status_code, 200)

    def test_contact_route(self):
        """Test contact page route"""
        response = self.client.get('/contact')
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.location, 'https://ca-t.psu.ac.th/')


if __name__ == '__main__':
    unittest.main()
