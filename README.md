# AI Job Portal with Resume Parser

A modern, AI-powered job portal application built with Flask that helps employers find the right candidates and job seekers find their dream jobs. Features intelligent resume parsing and job matching using machine learning.

## 🚀 Features

### For Employers
- **Resume Parser**: Upload candidate resumes (PDF/DOCX) and get AI-powered job suggestions
- **Detailed Experience Extraction**: Automatically extracts job titles, companies, dates, and descriptions from resumes
- **Job Posting**: Create and manage job listings with detailed requirements
- **AI Job Matching**: Intelligent matching between candidate profiles and job requirements
- **Dashboard**: Comprehensive dashboard to manage posted jobs and applications

### For Job Seekers
- **AI Recommendations**: Get personalized job recommendations based on your skills and experience
- **Job Search**: Browse and search through available job listings
- **Easy Application**: One-click job application process
- **Profile Management**: Manage your profile, skills, and experience

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: Bootstrap 5, HTML5, CSS3, JavaScript
- **AI/ML**: scikit-learn (TF-IDF, Cosine Similarity)
- **Database**: SQLite
- **Resume Parsing**: pdfplumber, PyPDF2, python-docx
- **NLP**: NLTK

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-job-portal.git
   cd ai-job-portal
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize the database**
   The database will be automatically created when you run the application for the first time.

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   Open your browser and navigate to `http://localhost:5000`

## 📁 Project Structure

```
ai-job-portal/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .gitignore           # Git ignore file
├── instance/             # Database files (auto-generated)
├── uploads/              # Temporary resume uploads (auto-generated)
├── static/
│   └── css/
│       └── style.css     # Custom styles
└── templates/
    ├── base.html         # Base template
    ├── index.html        # Home page
    ├── login.html        # Job seeker login
    ├── register.html     # Job seeker registration
    ├── dashboard.html    # Job seeker dashboard
    ├── employer_login.html    # Employer login
    ├── employer_dashboard.html # Employer dashboard
    ├── post_job.html     # Post new job
    ├── jobs.html         # Browse jobs
    ├── upload_resume.html # Resume upload page
    └── resume_suggestions.html # Job suggestions based on resume
```

## 🎯 Key Features Explained

### Resume Parsing
The application uses advanced text extraction and pattern matching to parse resumes:
- **Skills Extraction**: Identifies technical skills, programming languages, and tools
- **Experience Extraction**: Extracts years of experience and detailed work history
- **Education Extraction**: Parses educational background and qualifications
- **Work History**: Extracts job titles, companies, dates, and job descriptions

### AI Job Matching
Uses TF-IDF vectorization and cosine similarity to match:
- Candidate skills with job requirements
- Experience levels
- Job descriptions with candidate profiles

## 🔐 Usage

### For Employers

1. **Login/Register**: Use the employer login page (any company name works in demo mode)
2. **Post Jobs**: Create job listings with detailed requirements
3. **Upload Resumes**: Upload candidate resumes to get AI-powered job suggestions
4. **View Matches**: See matching jobs with match percentages and skill highlights

### For Job Seekers

1. **Register**: Create an account with your profile information
2. **Login**: Access your personalized dashboard
3. **Get Recommendations**: View AI-recommended jobs based on your profile
4. **Apply**: Apply to jobs with one click

## 🎨 UI/UX Features

- Modern, professional design with gradient themes
- Responsive layout for all devices
- Smooth animations and transitions
- Drag-and-drop file upload
- Interactive job cards with hover effects
- Visual match scores and skill highlighting

## 📝 Configuration

The application uses default settings. To customize:

- **Secret Key**: Update `app.config['SECRET_KEY']` in `app.py`
- **Upload Folder**: Modify `app.config['UPLOAD_FOLDER']` in `app.py`
- **Database**: SQLite database is created automatically in `instance/` folder

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Your Name - [Your GitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- Bootstrap for the UI framework
- Font Awesome for icons
- scikit-learn for machine learning capabilities
- All the open-source libraries that made this project possible

---

**Note**: This is a demo application. For production use, implement proper authentication, security measures, and database optimization.

