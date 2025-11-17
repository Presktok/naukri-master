from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash, send_file
import sqlite3
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re
import string
import pdfplumber
import PyPDF2
from docx import Document

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

CORS(app)

# --- Jinja filters ---
@app.template_filter('format_date')
def format_date(value, fmt='%B %d, %Y'):
    if not value:
        return ''
    # Accept datetime, sqlite ISO string, or other strings
    try:
        if isinstance(value, str):
            # Try multiple parse strategies
            try:
                dt = datetime.fromisoformat(value)
            except ValueError:
                try:
                    # Fallback: common formats
                    dt = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
                except Exception:
                    return value
        else:
            dt = value
        return dt.strftime(fmt)
    except Exception:
        return value

# --- Raw SQLite helpers ---
DB_PATH = os.path.join(os.path.dirname(__file__), 'instance', 'job_portal.db')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()
    # Users
    cur.execute(
        '''CREATE TABLE IF NOT EXISTS user (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            skills TEXT NOT NULL,
            experience TEXT NOT NULL,
            education TEXT NOT NULL,
            location TEXT NOT NULL,
            phone TEXT NOT NULL,
            resume_summary TEXT,
            resume_path TEXT,
            resume_uploaded_at TEXT,
            created_at TEXT
        )'''
    )
    # Ensure new columns exist when upgrading older databases
    try:
        cols = [row[1] for row in cur.execute('PRAGMA table_info(user)').fetchall()]
        if 'resume_path' not in cols:
            cur.execute('ALTER TABLE user ADD COLUMN resume_path TEXT')
        if 'resume_uploaded_at' not in cols:
            cur.execute('ALTER TABLE user ADD COLUMN resume_uploaded_at TEXT')
        if 'resume_skills' not in cols:
            cur.execute('ALTER TABLE user ADD COLUMN resume_skills TEXT')
        conn.commit()
    except Exception:
        # If PRAGMA fails for any reason, proceed without interrupting init
        pass
    # Jobs
    cur.execute(
        '''CREATE TABLE IF NOT EXISTS job (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            company TEXT NOT NULL,
            description TEXT NOT NULL,
            required_skills TEXT NOT NULL,
            experience_required TEXT NOT NULL,
            location TEXT NOT NULL,
            salary TEXT,
            job_type TEXT NOT NULL,
            posted_by TEXT NOT NULL,
            contact_email TEXT NOT NULL,
            created_at TEXT
        )'''
    )
    # Applications
    cur.execute(
        '''CREATE TABLE IF NOT EXISTS application (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            job_id INTEGER NOT NULL,
            applied_at TEXT,
            status TEXT DEFAULT 'Applied',
            FOREIGN KEY(user_id) REFERENCES user(id) ON DELETE CASCADE,
            FOREIGN KEY(job_id) REFERENCES job(id) ON DELETE CASCADE
        )'''
    )
    conn.commit()
    conn.close()

# AI Recommendation System
class JobRecommendationEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.job_vectors = None
        self.jobs_data = None
        
    def preprocess_text(self, text):
        """Clean and preprocess text for better matching"""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text
    
    def update_job_vectors(self):
        """Update job vectors when new jobs are added"""
        conn = get_db()
        rows = conn.execute('SELECT * FROM job').fetchall()
        conn.close()
        if not rows:
            self.jobs_data = []
            self.job_vectors = None
            return
        # Convert to list of dicts
        jobs = [dict(row) for row in rows]
        job_texts = []
        for job in jobs:
            combined_text = f"{job['title']} {job['description']} {job['required_skills']}"
            job_texts.append(self.preprocess_text(combined_text))
        self.jobs_data = jobs
        self.job_vectors = self.vectorizer.fit_transform(job_texts)
    
    def get_recommendations(self, user_skills, user_experience, top_n=5):
        """Get job recommendations for a user based on their skills.
        Adds skill_match, exp_match, and matching_skills for UI breakdown.
        """
        # Always update job vectors to ensure latest jobs are included
        self.update_job_vectors()
            
        if self.job_vectors is None or len(self.jobs_data) == 0:
            return []
        
        # Preprocess user skills for vector similarity
        user_profile = self.preprocess_text(user_skills)
        
        # Check if vectorizer has been fitted
        if not hasattr(self.vectorizer, 'vocabulary_') or len(self.vectorizer.vocabulary_) == 0:
            self.update_job_vectors()
        
        try:
            user_vector = self.vectorizer.transform([user_profile])
        except ValueError:
            # If transform fails (vocabulary mismatch), refit
            self.update_job_vectors()
            user_vector = self.vectorizer.transform([user_profile])
        
        # Calculate similarity scores
        similarity_scores = cosine_similarity(user_vector, self.job_vectors).flatten()
        
        # Normalize candidate skills list (from profile or resume)
        candidate_skills = [s.strip().lower() for s in (user_skills or '').split(',') if s.strip()]
        
        # Check ALL jobs, not just top N, to ensure we don't miss any matches
        recommendations = []
        # Helper for experience buckets
        exp_levels = {
            '0-1 years': 0,
            '1-3 years': 1,
            '3-5 years': 2,
            '5-10 years': 3,
            '10+ years': 4
        }

        # Process ALL jobs to find matches
        for idx in range(len(self.jobs_data)):
            job_dict = self.jobs_data[idx]
            # Compute skill match first to check if there's any match
            required = [s.strip().lower() for s in (job_dict.get('required_skills') or '').split(',') if s.strip()]
            has_skill_match = bool(set(required) & set(candidate_skills)) if required and candidate_skills else False
            
            # Also check for partial matches (e.g., "javascript" matches "js" or vice versa)
            has_partial_match = False
            if required and candidate_skills:
                for req_skill in required:
                    for cand_skill in candidate_skills:
                        # Check if one skill contains the other (for partial matches)
                        if req_skill in cand_skill or cand_skill in req_skill:
                            has_partial_match = True
                            break
                    if has_partial_match:
                        break
            
            # Include job if:
            # 1. Similarity > 0.001 (very low threshold)
            # 2. There's at least one exact matching skill
            # 3. There's a partial skill match
            # 4. OR if there are 20 or fewer jobs total, show all jobs to ensure visibility
            # This ensures newly posted jobs always appear
            show_all_jobs = len(self.jobs_data) <= 20
            if similarity_scores[idx] > 0.001 or has_skill_match or has_partial_match or show_all_jobs:
                # Compute skill match percentage based on required skills vs candidate skills
                if required:
                    matched = sorted(list(set(required) & set(candidate_skills)))
                    skill_match = round((len(matched) / len(required)) * 100, 1)
                else:
                    matched = []
                    skill_match = 0.0

                # Compute experience match (bucket distance)
                req_idx = exp_levels.get(job_dict.get('experience_required', '').strip(), None)
                user_idx = exp_levels.get((user_experience or '').strip(), None)
                if req_idx is not None and user_idx is not None:
                    diff = abs(req_idx - user_idx)
                    exp_match = max(0, 100 - diff * 25)  # 0,25,50,75,100
                else:
                    exp_match = 0

                # Calculate combined priority score
                # Higher priority for jobs where BOTH skills and experience match well
                # Formula: (skill_match * 0.6 + exp_match * 0.4) * bonus_multiplier
                # Bonus multiplier: 1.5x if both are >= 70%, 1.3x if both are >= 50%, 1.1x if both are >= 30%
                base_score = (skill_match * 0.6 + exp_match * 0.4)
                
                if skill_match >= 70 and exp_match >= 70:
                    priority_multiplier = 1.5  # Excellent match in both
                elif skill_match >= 50 and exp_match >= 50:
                    priority_multiplier = 1.3  # Good match in both
                elif skill_match >= 30 and exp_match >= 30:
                    priority_multiplier = 1.1  # Decent match in both
                else:
                    priority_multiplier = 1.0  # No bonus
                
                priority_score = base_score * priority_multiplier
                
                # Also consider the original similarity score (weighted 30%)
                combined_score = (priority_score * 0.7) + (float(similarity_scores[idx]) * 100 * 0.3)

                recommendations.append({
                    'job': job_dict,
                    'similarity_score': float(similarity_scores[idx]),
                    'match_percentage': round(float(similarity_scores[idx]) * 100, 1),
                    'skill_match': skill_match,
                    'exp_match': exp_match,
                    'matching_skills': matched,
                    'priority_score': round(combined_score, 2),
                    'both_matched': skill_match >= 50 and exp_match >= 50  # Flag for jobs with good match in both
                })
        
        # Sort by priority score (highest first) - jobs with both skills and experience match will rank higher
        recommendations.sort(key=lambda x: x['priority_score'], reverse=True)
        
        # Return top N recommendations after sorting (but include all if less than top_n)
        return recommendations[:top_n] if len(recommendations) > top_n else recommendations

# Initialize recommendation engine
recommendation_engine = JobRecommendationEngine()

# Resume Parser Class
class ResumeParser:
    def __init__(self):
        # Common skills keywords
        self.skill_keywords = [
            'python', 'java', 'javascript', 'react', 'angular', 'vue', 'node.js', 'django', 'flask',
            'sql', 'mysql', 'postgresql', 'mongodb', 'aws', 'azure', 'docker', 'kubernetes',
            'machine learning', 'deep learning', 'ai', 'data science', 'tensorflow', 'pytorch',
            'html', 'css', 'bootstrap', 'git', 'github', 'agile', 'scrum', 'rest api',
            'c++', 'c#', '.net', 'php', 'ruby', 'go', 'rust', 'swift', 'kotlin',
            'excel', 'power bi', 'tableau', 'salesforce', 'project management'
        ]
        
        # Education keywords
        self.education_keywords = [
            'bachelor', 'master', 'phd', 'degree', 'university', 'college', 'diploma',
            'bsc', 'msc', 'mba', 'engineering', 'computer science', 'information technology'
        ]
        
        # Experience patterns
        self.experience_patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
            r'experience\s*(?:of\s*)?(\d+)\+?\s*(?:years?|yrs?)',
            r'(\d+)\s*(?:years?|yrs?)\s*(?:in|with)'
        ]
    
    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF file"""
        text = ""
        try:
            # Try pdfplumber first (better for structured text)
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
        except:
            try:
                # Fallback to PyPDF2
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
            except Exception as e:
                print(f"Error extracting PDF: {e}")
        return text
    
    def extract_text_from_docx(self, file_path):
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            print(f"Error extracting DOCX: {e}")
            return ""
    
    def extract_text(self, file_path, filename):
        """Extract text from resume file based on extension"""
        file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        
        if file_ext == 'pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_ext in ['docx', 'doc']:
            return self.extract_text_from_docx(file_path)
        else:
            return ""
    
    def extract_skills(self, text):
        """Extract skills from resume text"""
        text_lower = text.lower()
        found_skills = []
        
        for skill in self.skill_keywords:
            if skill.lower() in text_lower:
                found_skills.append(skill.title())
        
        # Also look for common patterns like "Skills:", "Technical Skills:", etc.
        skills_section_pattern = r'(?:skills?|technical\s+skills?|core\s+skills?|competencies?)[:\-]?\s*(.+?)(?:\n\n|\n[A-Z]|$)'
        match = re.search(skills_section_pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            skills_text = match.group(1)
            # Extract comma or newline separated skills
            skills_list = re.split(r'[,;\n•\-\*]', skills_text)
            for skill in skills_list:
                skill = skill.strip()
                if len(skill) > 2 and len(skill) < 50:
                    found_skills.append(skill)
        
        # Remove duplicates and return
        return list(set(found_skills))
    
    def extract_experience(self, text):
        """Extract years of experience from resume text"""
        text_lower = text.lower()
        
        for pattern in self.experience_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                years = int(match.group(1))
                if years <= 1:
                    return "0-1 years"
                elif years <= 3:
                    return "1-3 years"
                elif years <= 5:
                    return "3-5 years"
                elif years <= 10:
                    return "5-10 years"
                else:
                    return "10+ years"
        
        # Default if not found
        return "0-1 years"
    
    def extract_detailed_experience(self, text):
        """Extract detailed work experience including job titles, companies, dates, and descriptions"""
        experience_entries = []
        
        # Common section headers for experience
        experience_section_patterns = [
            r'(?:work\s+)?experience[:\-]?\s*\n',
            r'employment\s+history[:\-]?\s*\n',
            r'professional\s+experience[:\-]?\s*\n',
            r'career\s+history[:\-]?\s*\n',
            r'employment[:\-]?\s*\n'
        ]
        
        # Find experience section
        experience_section = None
        for pattern in experience_section_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                # Get text after experience section
                start_pos = match.end()
                # Try to find next major section (Education, Skills, etc.)
                next_section_pattern = r'(?:education|skills?|projects?|certifications?|achievements?|awards?)[:\-]?\s*\n'
                next_match = re.search(next_section_pattern, text[start_pos:], re.IGNORECASE | re.MULTILINE)
                if next_match:
                    experience_section = text[start_pos:start_pos + next_match.start()]
                else:
                    experience_section = text[start_pos:start_pos + 2000]  # Take next 2000 chars
                break
        
        # If no explicit section found, look for date patterns that indicate work experience
        if not experience_section:
            # Look for date patterns (MM/YYYY, MM-YYYY, Month YYYY, etc.)
            date_pattern = r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}|(?:0?[1-9]|1[0-2])[/\-]\d{4}|\d{4}\s*[-–]\s*\d{4}'
            if re.search(date_pattern, text, re.IGNORECASE):
                # Take a reasonable chunk of text that likely contains experience
                experience_section = text[:3000]
        
        if not experience_section:
            return []
        
        # Split into potential job entries (look for date patterns or bullet points)
        # Pattern: Date range or single date, followed by job title and company
        entry_patterns = [
            # Pattern 1: Date range, Job Title, Company
            r'((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}|(?:0?[1-9]|1[0-2])[/\-]\d{4}|\d{4})\s*[-–]\s*((?:present|current|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}|(?:0?[1-9]|1[0-2])[/\-]\d{4}|\d{4}|present|current).*?\n(.+?)\n(.+?)(?:\n|$)',
            # Pattern 2: Job Title at Company, Date range
            r'(.+?)\s+(?:at|@)\s+(.+?)\s+((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}|(?:0?[1-9]|1[0-2])[/\-]\d{4}|\d{4})\s*[-–]\s*((?:present|current|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}|(?:0?[1-9]|1[0-2])[/\-]\d{4}|\d{4}|present|current)',
        ]
        
        # Try to extract entries using line-by-line analysis
        lines = experience_section.split('\n')
        current_entry = {}
        entries = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check if line contains a date pattern (likely start of new entry)
            date_match = re.search(r'((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}|(?:0?[1-9]|1[0-2])[/\-]\d{4}|\d{4})\s*[-–]\s*((?:present|current|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}|(?:0?[1-9]|1[0-2])[/\-]\d{4}|\d{4}|present|current)', line, re.IGNORECASE)
            
            if date_match:
                # Save previous entry if exists
                if current_entry:
                    entries.append(current_entry)
                
                # Start new entry
                current_entry = {
                    'date_range': date_match.group(0).strip(),
                    'job_title': '',
                    'company': '',
                    'description': []
                }
                
                # Try to extract job title and company from same or next lines
                remaining_text = line.replace(date_match.group(0), '').strip()
                if remaining_text:
                    # Check if it contains "at" or "@" (company indicator)
                    if ' at ' in remaining_text.lower() or ' @ ' in remaining_text.lower():
                        parts = re.split(r'\s+(?:at|@)\s+', remaining_text, flags=re.IGNORECASE)
                        if len(parts) >= 2:
                            current_entry['job_title'] = parts[0].strip()
                            current_entry['company'] = parts[1].strip()
                    else:
                        current_entry['job_title'] = remaining_text
                
                # Check next few lines for company or description
                for j in range(1, min(5, len(lines) - i)):
                    next_line = lines[i + j].strip()
                    if not next_line:
                        continue
                    
                    # If company not found yet and line doesn't look like description
                    if not current_entry['company'] and len(next_line) < 100:
                        # Check for company indicators
                        if not re.search(r'^[•\-\*]', next_line):  # Not a bullet point
                            current_entry['company'] = next_line
                    elif len(next_line) > 20:  # Likely description
                        if next_line.startswith(('•', '-', '*')) or not current_entry['description']:
                            desc = re.sub(r'^[•\-\*\d+\.]\s*', '', next_line)
                            if desc:
                                current_entry['description'].append(desc)
            
            # If we have a current entry, collect description lines
            elif current_entry:
                if line.startswith(('•', '-', '*')) or (len(line) > 30 and not re.search(r'^\d+', line)):
                    desc = re.sub(r'^[•\-\*\d+\.]\s*', '', line)
                    if desc and len(desc) > 10:
                        current_entry['description'].append(desc)
        
        # Add last entry
        if current_entry:
            entries.append(current_entry)
        
        # If no structured entries found, try simpler pattern matching
        if not entries:
            # Look for job titles (common patterns)
            job_title_patterns = [
                r'(?:senior|junior|lead|principal|staff)?\s*(?:software|web|full.?stack|front.?end|back.?end|devops|data|machine learning|ai|ml|engineer|developer|analyst|manager|specialist|consultant|architect)',
            ]
            
            for pattern in job_title_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Get context around match
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 200)
                    context = text[start:end]
                    
                    # Try to extract company and date from context
                    entry = {
                        'job_title': match.group(0).strip(),
                        'company': '',
                        'date_range': '',
                        'description': []
                    }
                    
                    # Look for company name (common words after "at" or "@")
                    company_match = re.search(r'(?:at|@)\s+([A-Z][a-zA-Z\s&]+)', context)
                    if company_match:
                        entry['company'] = company_match.group(1).strip()
                    
                    # Look for date
                    date_match = re.search(r'((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}|(?:0?[1-9]|1[0-2])[/\-]\d{4}|\d{4})\s*[-–]\s*((?:present|current|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}|(?:0?[1-9]|1[0-2])[/\-]\d{4}|\d{4}|present|current)', context, re.IGNORECASE)
                    if date_match:
                        entry['date_range'] = date_match.group(0).strip()
                    
                    if entry['job_title'] or entry['company']:
                        entries.append(entry)
                        if len(entries) >= 5:  # Limit to 5 most recent
                            break
                
                if entries:
                    break
        
        # Clean and format entries
        formatted_entries = []
        for entry in entries[:5]:  # Limit to 5 most recent experiences
            formatted_entry = {
                'job_title': entry.get('job_title', 'Not specified').strip(),
                'company': entry.get('company', 'Not specified').strip(),
                'date_range': entry.get('date_range', 'Not specified').strip(),
                'description': ' | '.join(entry.get('description', [])[:3]) if entry.get('description') else 'No description available'
            }
            if formatted_entry['job_title'] != 'Not specified' or formatted_entry['company'] != 'Not specified':
                formatted_entries.append(formatted_entry)
        
        return formatted_entries
    
    def extract_education(self, text):
        """Extract education information from resume text"""
        text_lower = text.lower()
        education_info = []
        
        for edu_keyword in self.education_keywords:
            if edu_keyword in text_lower:
                # Try to extract the full education line
                pattern = rf'.*{re.escape(edu_keyword)}.*'
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    education_info.extend(matches[:2])  # Take first 2 matches
        
        return " | ".join(education_info[:3]) if education_info else "Not specified"
    
    def parse_resume(self, file_path, filename):
        """Main method to parse resume and extract information"""
        text = self.extract_text(file_path, filename)
        
        if not text:
            return None
        
        # Extract detailed experience
        detailed_experience = self.extract_detailed_experience(text)
        
        parsed_data = {
            'skills': ', '.join(self.extract_skills(text)),
            'experience': self.extract_experience(text),  # Years of experience
            'detailed_experience': detailed_experience,  # Detailed work history
            'education': self.extract_education(text),
            'resume_text': text[:1000]  # Store first 1000 chars for reference
        }
        
        return parsed_data

# Initialize resume parser
resume_parser = ResumeParser()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.form
        conn = get_db()
        cur = conn.cursor()
        # Check if user exists
        if cur.execute('SELECT 1 FROM user WHERE username = ?', (data['username'],)).fetchone():
            conn.close()
            flash('Username already exists!')
            return render_template('register.html')
        if cur.execute('SELECT 1 FROM user WHERE email = ?', (data['email'],)).fetchone():
            conn.close()
            flash('Email already registered!')
            return render_template('register.html')
        # Handle optional resume upload
        resume_path = None
        resume_uploaded_at = None
        if 'resume' in request.files:
            file = request.files['resume']
            if file and file.filename:
                allowed_extensions = {'pdf', 'docx', 'doc'}
                file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
                if file_ext in allowed_extensions:
                    filename = secure_filename(file.filename)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
                    filename = timestamp + filename
                    # Store relative path under uploads
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    resume_path = os.path.join('uploads', filename)
                    resume_uploaded_at = datetime.utcnow().isoformat()
        cur.execute(
            'INSERT INTO user (username, email, password_hash, full_name, skills, experience, education, location, phone, resume_summary, resume_path, resume_uploaded_at, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
            (
                data['username'],
                data['email'],
                generate_password_hash(data['password']),
                data['full_name'],
                data['skills'],
                data['experience'],
                data['education'],
                data['location'],
                data['phone'],
                data.get('resume_summary', ''),
                resume_path,
                resume_uploaded_at,
                datetime.utcnow().isoformat()
            )
        )
        conn.commit()
        conn.close()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.form
        conn = get_db()
        row = conn.execute('SELECT * FROM user WHERE username = ?', (data['username'],)).fetchone()
        # If credentials valid
        if row and check_password_hash(row['password_hash'], data['password']):
            conn.close()
            session['user_id'] = row['id']
            session['username'] = row['username']
            session['user_type'] = 'job_seeker'
            flash('Login successful!')
            return redirect(url_for('dashboard'))
        else:
            conn.close()
            flash('Invalid username or password!')
    
    return render_template('login.html')

@app.route('/recruiter_login', methods=['GET', 'POST'])
def recruiter_login():
    if request.method == 'POST':
        # Simple recruiter authentication (in production, use proper auth)
        recruiter_name = request.form['recruiter_name']
        session['recruiter_name'] = recruiter_name
        session['user_type'] = 'recruiter'
        flash('Recruiter login successful!')
        return redirect(url_for('recruiter_dashboard'))
    
    return render_template('recruiter_login.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    conn = get_db()
    user_row = conn.execute('SELECT * FROM user WHERE id = ?', (session['user_id'],)).fetchone()
    conn.close()
    if not user_row:
        flash('User not found')
        return redirect(url_for('login'))
    user = dict(user_row)
    
    # Combine profile skills and resume-extracted skills for better matching
    profile_skills = user.get('skills', '') or ''
    resume_skills = user.get('resume_skills', '') or ''
    
    # Merge skills (remove duplicates, case-insensitive)
    all_user_skills = set()
    if profile_skills:
        all_user_skills.update([s.strip().lower() for s in profile_skills.split(',') if s.strip()])
    if resume_skills:
        all_user_skills.update([s.strip().lower() for s in resume_skills.split(',') if s.strip()])
    combined_skills = ', '.join(sorted([s.title() for s in all_user_skills]))
    
    # Use combined skills for recommendations (get more to allow better prioritization)
    recommendations = recommendation_engine.get_recommendations(
        combined_skills, user['experience'], top_n=10
    )
    
    # Collect all unique matched skills across all recommendations
    all_matched_skills = set()
    for rec in recommendations:
        if rec.get('matching_skills'):
            for skill in rec['matching_skills']:
                all_matched_skills.add(skill.lower())
    all_matched_skills = sorted([s.title() for s in all_matched_skills])
    
    # Separate resume skills for display
    resume_skills_list = []
    if resume_skills:
        resume_skills_list = [s.strip().title() for s in resume_skills.split(',') if s.strip()]
    
    return render_template('dashboard.html', user=user, recommendations=recommendations, 
                         all_matched_skills=all_matched_skills, resume_skills_list=resume_skills_list)

def calculate_applicant_match(job_dict, user_dict):
    """Calculate skill match and experience match for an applicant against a job"""
    # Experience levels mapping
    exp_levels = {
        '0-1 years': 0,
        '1-3 years': 1,
        '3-5 years': 2,
        '5-10 years': 3,
        '10+ years': 4
    }
    
    # Get required skills from job
    required_skills = [s.strip().lower() for s in (job_dict.get('required_skills') or '').split(',') if s.strip()]
    
    # Get candidate skills (combine profile and resume skills)
    profile_skills = [s.strip().lower() for s in (user_dict.get('skills') or '').split(',') if s.strip()]
    resume_skills = [s.strip().lower() for s in (user_dict.get('resume_skills') or '').split(',') if s.strip()]
    candidate_skills = list(set(profile_skills + resume_skills))
    
    # Calculate skill match
    if required_skills:
        matched_skills = sorted(list(set(required_skills) & set(candidate_skills)))
        skill_match = round((len(matched_skills) / len(required_skills)) * 100, 1)
    else:
        matched_skills = []
        skill_match = 0.0
    
    # Calculate experience match
    req_exp = job_dict.get('experience_required', '').strip()
    user_exp = user_dict.get('experience', '').strip()
    
    req_idx = exp_levels.get(req_exp, None)
    user_idx = exp_levels.get(user_exp, None)
    
    if req_idx is not None and user_idx is not None:
        diff = abs(req_idx - user_idx)
        exp_match = max(0, 100 - diff * 25)  # 0, 25, 50, 75, 100
    else:
        exp_match = 0
    
    return {
        'skill_match': skill_match,
        'exp_match': exp_match,
        'matched_skills': matched_skills,
        'overall_match': round((skill_match * 0.7 + exp_match * 0.3), 1)  # Weighted average
    }

@app.route('/recruiter_dashboard')
def recruiter_dashboard():
    if 'recruiter_name' not in session:
        return redirect(url_for('recruiter_login'))
    conn = get_db()
    job_rows = conn.execute('SELECT * FROM job WHERE posted_by = ? ORDER BY datetime(created_at) DESC', (session['recruiter_name'],)).fetchall()
    jobs_with_applications = []
    for job_row in job_rows:
        job_dict = dict(job_row)
        app_rows = conn.execute('SELECT a.*, u.full_name, u.email, u.skills, u.resume_skills, u.resume_path, u.experience FROM application a JOIN user u ON a.user_id = u.id WHERE a.job_id = ? ORDER BY datetime(a.applied_at) DESC', (job_dict['id'],)).fetchall()
        applications = []
        for a in app_rows:
            # Convert Row to dict for easier handling
            app_dict = dict(a)
            user_dict = {
                'full_name': app_dict['full_name'],
                'email': app_dict['email'],
                'skills': app_dict['skills'],
                'resume_skills': app_dict.get('resume_skills'),
                'resume_path': app_dict.get('resume_path'),
                'experience': app_dict.get('experience'),
            }
            
            # Calculate match scores
            match_scores = calculate_applicant_match(job_dict, user_dict)
            
            applications.append({
                'id': app_dict['id'],
                'user_id': app_dict['user_id'],
                'job_id': app_dict['job_id'],
                'applied_at': app_dict['applied_at'],
                'status': app_dict['status'],
                'user': user_dict,
                'match_scores': match_scores
            })
        
        # Sort applications by overall match (best matches first)
        applications.sort(key=lambda x: x['match_scores']['overall_match'], reverse=True)
        
        jobs_with_applications.append({
            'job': job_dict,
            'applications': applications,
            'application_count': len(applications)
        })
    conn.close()
    return render_template('recruiter_dashboard.html', jobs_with_applications=jobs_with_applications, recruiter_name=session['recruiter_name'])

@app.route('/post_job', methods=['GET', 'POST'])
def post_job():
    if 'recruiter_name' not in session:
        return redirect(url_for('recruiter_login'))
    
    if request.method == 'POST':
        data = request.form
        conn = get_db()
        conn.execute(
            'INSERT INTO job (title, company, description, required_skills, experience_required, location, salary, job_type, posted_by, contact_email, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
            (
                data['title'],
                data['company'],
                data['description'],
                data['required_skills'],
                data['experience_required'],
                data['location'],
                data.get('salary', ''),
                data['job_type'],
                session['recruiter_name'],
                data['contact_email'],
                datetime.utcnow().isoformat()
            )
        )
        conn.commit()
        conn.close()
        
        # Update recommendation engine with new job
        recommendation_engine.update_job_vectors()
        
        flash('Job posted successfully!')
        return redirect(url_for('recruiter_dashboard'))
    
    return render_template('post_job.html')

@app.route('/jobs')
def jobs():
    conn = get_db()
    rows = conn.execute('SELECT * FROM job ORDER BY datetime(created_at) DESC').fetchall()
    conn.close()
    jobs_list = [dict(r) for r in rows]
    return render_template('jobs.html', jobs=jobs_list)

@app.route('/view_job/<int:job_id>')
def view_job(job_id):
    conn = get_db()
    row = conn.execute('SELECT * FROM job WHERE id = ?', (job_id,)).fetchone()
    conn.close()
    if not row:
        if 'recruiter_name' in session:
            flash('Job not found!')
            return redirect(url_for('recruiter_dashboard'))
        return redirect(url_for('jobs'))
    
    job = dict(row)
    # Check if user is the job owner (recruiter)
    is_owner = 'recruiter_name' in session and job.get('posted_by') == session.get('recruiter_name')
    
    return render_template('view_job.html', job=job, is_owner=is_owner)

@app.route('/edit_job/<int:job_id>', methods=['GET', 'POST'])
def edit_job(job_id):
    if 'recruiter_name' not in session:
        return redirect(url_for('recruiter_login'))
    conn = get_db()
    job_row = conn.execute('SELECT * FROM job WHERE id = ?', (job_id,)).fetchone()
    if not job_row:
        conn.close()
        flash('Job not found!')
        return redirect(url_for('recruiter_dashboard'))
    job = dict(job_row)
    # Check ownership
    if job['posted_by'] != session['recruiter_name']:
        conn.close()
        flash('You can only edit your own jobs!')
        return redirect(url_for('recruiter_dashboard'))
    
    if request.method == 'POST':
        data = request.form
        conn.execute(
            'UPDATE job SET title = ?, company = ?, description = ?, required_skills = ?, experience_required = ?, location = ?, salary = ?, job_type = ?, contact_email = ? WHERE id = ?',
            (
                data['title'], data['company'], data['description'], data['required_skills'], data['experience_required'],
                data['location'], data.get('salary', ''), data['job_type'], data['contact_email'], job_id
            )
        )
        conn.commit()
        conn.close()
        
        # Update recommendation engine with modified job
        recommendation_engine.update_job_vectors()
        
        flash('Job updated successfully!')
        return redirect(url_for('recruiter_dashboard'))
    
    conn.close()
    return render_template('edit_job.html', job=job)

@app.route('/delete_job/<int:job_id>', methods=['POST'])
def delete_job(job_id):
    if 'recruiter_name' not in session:
        return redirect(url_for('recruiter_login'))
    conn = get_db()
    row = conn.execute('SELECT posted_by FROM job WHERE id = ?', (job_id,)).fetchone()
    if not row:
        conn.close()
        flash('Job not found!')
        return redirect(url_for('recruiter_dashboard'))
    if row['posted_by'] != session['recruiter_name']:
        conn.close()
        flash('You can only delete your own jobs!')
        return redirect(url_for('recruiter_dashboard'))
    # Delete dependent applications then job
    conn.execute('DELETE FROM application WHERE job_id = ?', (job_id,))
    conn.execute('DELETE FROM job WHERE id = ?', (job_id,))
    conn.commit()
    conn.close()
    
    # Update recommendation engine
    recommendation_engine.update_job_vectors()
    
    flash('Job deleted successfully!')
    return redirect(url_for('recruiter_dashboard'))

@app.route('/apply_job/<int:job_id>', methods=['POST'])
def apply_job(job_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please login to apply for jobs'})
    
    user_id = session['user_id']
    conn = get_db()
    if conn.execute('SELECT 1 FROM application WHERE user_id = ? AND job_id = ?', (user_id, job_id)).fetchone():
        conn.close()
        return jsonify({'success': False, 'message': 'You have already applied for this job'})
    conn.execute(
        'INSERT INTO application (user_id, job_id, applied_at, status) VALUES (?,?,?,?)',
        (user_id, job_id, datetime.utcnow().isoformat(), 'Applied')
    )
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'message': 'Application submitted successfully!'})

@app.route('/upload_user_resume', methods=['POST'])
def upload_user_resume():
    """Route for job seekers to upload/update their resume from dashboard"""
    if 'user_id' not in session:
        flash('Please login to upload resume.')
        return redirect(url_for('login'))
    
    if 'resume' not in request.files:
        flash('No file selected!')
        return redirect(url_for('dashboard'))
    
    file = request.files['resume']
    if file.filename == '':
        flash('No file selected!')
        return redirect(url_for('dashboard'))
    
    # Check file extension
    allowed_extensions = {'pdf', 'docx', 'doc'}
    file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    
    if file_ext not in allowed_extensions:
        flash('Invalid file type! Please upload PDF, DOCX, or DOC files only.')
        return redirect(url_for('dashboard'))
    
    # Get current user to check for existing resume
    conn = get_db()
    user_row = conn.execute('SELECT * FROM user WHERE id = ?', (session['user_id'],)).fetchone()
    
    # Delete old resume if exists
    if user_row and user_row['resume_path']:
        old_file_path = os.path.join(os.path.dirname(__file__), user_row['resume_path'])
        try:
            if os.path.exists(old_file_path):
                os.remove(old_file_path)
        except:
            pass
    
    # Save new file
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
    filename = timestamp + filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    resume_path = os.path.join('uploads', filename)
    resume_uploaded_at = datetime.utcnow().isoformat()
    
    # Parse resume to extract skills
    parsed_data = resume_parser.parse_resume(file_path, file.filename)
    resume_skills = None
    if parsed_data and parsed_data.get('skills'):
        resume_skills = parsed_data['skills']
    
    # Update database with resume path and extracted skills
    # First check if resume_skills column exists, if not we'll add it
    try:
        cols = [row[1] for row in conn.execute('PRAGMA table_info(user)').fetchall()]
        if 'resume_skills' not in cols:
            conn.execute('ALTER TABLE user ADD COLUMN resume_skills TEXT')
            conn.commit()
    except:
        pass
    
    conn.execute(
        'UPDATE user SET resume_path = ?, resume_uploaded_at = ?, resume_skills = ? WHERE id = ?',
        (resume_path, resume_uploaded_at, resume_skills, session['user_id'])
    )
    conn.commit()
    conn.close()
    
    if resume_skills:
        flash(f'Resume uploaded successfully! Extracted {len(resume_skills.split(","))} skills from your resume.')
    else:
        flash('Resume uploaded successfully!')
    return redirect(url_for('dashboard'))

@app.route('/remove_resume', methods=['POST'])
def remove_resume():
    """Route to remove user's resume"""
    if 'user_id' not in session:
        flash('Please login to remove resume.')
        return redirect(url_for('login'))
    
    conn = get_db()
    user_row = conn.execute('SELECT * FROM user WHERE id = ?', (session['user_id'],)).fetchone()
    
    if not user_row:
        conn.close()
        flash('User not found')
        return redirect(url_for('dashboard'))
    
    # Delete resume file if exists
    if user_row['resume_path']:
        resume_path = os.path.join(os.path.dirname(__file__), user_row['resume_path'])
        try:
            if os.path.exists(resume_path):
                os.remove(resume_path)
        except:
            pass
    
    # Update database to remove resume
    conn.execute(
        'UPDATE user SET resume_path = NULL, resume_uploaded_at = NULL, resume_skills = NULL WHERE id = ?',
        (session['user_id'],)
    )
    conn.commit()
    conn.close()
    
    flash('Resume removed successfully! Your profile skills are still being used for job matching.')
    return redirect(url_for('dashboard'))

@app.route('/download_resume')
def download_resume():
    """Route to download user's resume (for job seekers) or applicant's resume (for recruiters)"""
    user_id_param = request.args.get('user_id')
    
    # If user_id is provided, it's a recruiter downloading an applicant's resume
    if user_id_param and 'recruiter_name' in session:
        try:
            applicant_id = int(user_id_param)
            conn = get_db()
            user_row = conn.execute('SELECT * FROM user WHERE id = ?', (applicant_id,)).fetchone()
            conn.close()
            
            if not user_row or not user_row['resume_path']:
                flash('No resume found for this applicant!')
                return redirect(url_for('recruiter_dashboard'))
            
            resume_path = os.path.join(os.path.dirname(__file__), user_row['resume_path'])
            if not os.path.exists(resume_path):
                flash('Resume file not found!')
                return redirect(url_for('recruiter_dashboard'))
            
            return send_file(resume_path, as_attachment=True, download_name=f"{user_row['full_name']}_resume.pdf")
        except:
            flash('Invalid request!')
            return redirect(url_for('recruiter_dashboard'))
    
    # Otherwise, it's a job seeker downloading their own resume
    if 'user_id' not in session:
        flash('Please login to download resume.')
        return redirect(url_for('login'))
    
    conn = get_db()
    user_row = conn.execute('SELECT * FROM user WHERE id = ?', (session['user_id'],)).fetchone()
    conn.close()
    
    if not user_row or not user_row['resume_path']:
        flash('No resume found!')
        return redirect(url_for('dashboard'))
    
    resume_path = os.path.join(os.path.dirname(__file__), user_row['resume_path'])
    if not os.path.exists(resume_path):
        flash('Resume file not found!')
        return redirect(url_for('dashboard'))
    
    return send_file(resume_path, as_attachment=True)

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.')
    return redirect(url_for('index'))

# API endpoints
@app.route('/api/recommendations/<int:user_id>')
def api_recommendations(user_id):
    conn = get_db()
    user = conn.execute('SELECT * FROM user WHERE id = ?', (user_id,)).fetchone()
    if not user:
        conn.close()
        return jsonify({'error': 'User not found'}), 404
    
    recommendations = recommendation_engine.get_recommendations(
        user['skills'], user['experience']
    )
    
    result = []
    for rec in recommendations:
        job = rec['job']
        result.append({
            'id': job['id'],
            'title': job['title'],
            'company': job['company'],
            'description': job['description'][:200] + '...',
            'location': job['location'],
            'match_percentage': rec['match_percentage']
        })
    conn.close()
    return jsonify(result)

@app.route('/api/job/<int:job_id>')
def api_job_details(job_id):
    """API endpoint to get job details"""
    conn = get_db()
    row = conn.execute('SELECT * FROM job WHERE id = ?', (job_id,)).fetchone()
    conn.close()
    if not row:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(dict(row))

@app.route('/api/check_application/<int:job_id>')
def check_application(job_id):
    if 'user_id' not in session:
        return jsonify({'applied': False})
    
    user_id = session['user_id']
    conn = get_db()
    exists = conn.execute('SELECT 1 FROM application WHERE user_id = ? AND job_id = ?', (user_id, job_id)).fetchone()
    conn.close()
    return jsonify({'applied': exists is not None})

if __name__ == '__main__':
    os.makedirs(os.path.join(os.path.dirname(__file__), 'instance'), exist_ok=True)
    init_db()
    # Update job vectors after ensuring tables exist
    recommendation_engine.update_job_vectors()
    app.run(debug=True)
