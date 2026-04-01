import Database from 'better-sqlite3';
import bcrypt from 'bcryptjs';

const db = new Database('fitness.db');

db.exec(`
CREATE TABLE IF NOT EXISTS coaches (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  role TEXT NOT NULL DEFAULT 'coach',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS students (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  age INTEGER NOT NULL,
  gender TEXT NOT NULL,
  phone TEXT,
  notes TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS assessments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  student_id INTEGER NOT NULL,
  head_position INTEGER NOT NULL,
  shoulders_symmetry INTEGER NOT NULL,
  thoracic_curvature INTEGER NOT NULL,
  hip_alignment INTEGER NOT NULL,
  knee_alignment INTEGER NOT NULL,
  feet_positioning INTEGER NOT NULL,
  ankle_dorsiflexion INTEGER NOT NULL,
  hip_rotation INTEGER NOT NULL,
  shoulder_flexion INTEGER NOT NULL,
  thoracic_rotation INTEGER NOT NULL,
  plank_time INTEGER NOT NULL,
  single_leg_balance INTEGER NOT NULL,
  squat_pattern INTEGER NOT NULL,
  push_pull_quality INTEGER NOT NULL,
  pain_scale INTEGER NOT NULL,
  injury_notes TEXT,
  trainer_observations TEXT,
  movement_quality_score REAL NOT NULL,
  risk_flag INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
);
`);

const defaultUser = process.env.COACH_USERNAME || 'coach';
const defaultPass = process.env.COACH_PASSWORD || 'coach123';
const existingCoach = db.prepare('SELECT id FROM coaches WHERE username = ?').get(defaultUser);

if (!existingCoach) {
  db.prepare('INSERT INTO coaches (username, password_hash, role) VALUES (?, ?, ?)').run(
    defaultUser,
    bcrypt.hashSync(defaultPass, 10),
    'coach'
  );
}

export default db;
