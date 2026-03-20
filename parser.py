import pdfplumber
import spacy
from typing import List, Dict

nlp = spacy.load("en_core_web_sm")

SENTIMENT_WEIGHTS = {
    "architected": 0.95,
    "designed":    0.90,
    "built":       0.85,
    "developed":   0.85,
    "implemented": 0.80,
    "led":         0.80,
    "created":     0.75,
    "used":        0.60,
    "worked":      0.55,
    "assisted":    0.35,
    "familiar":    0.25,
    "learning":    0.15,
}

TECH_SKILLS = [
    "python", "javascript", "java", "react", "angular",
    "machine learning", "deep learning", "nlp",
    "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy",
    "sql", "postgresql", "mongodb", "docker", "kubernetes",
    "aws", "gcp", "azure", "git", "data analysis",
    "data science", "statistics", "flask", "django", "fastapi",
]

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.lower()

def get_sentiment_weight(text: str, skill: str) -> float:
    weight = 0.5
    skill_pos = text.find(skill)
    if skill_pos == -1:
        return 0.0
    context = text[max(0, skill_pos-100): skill_pos+100]
    for word, w in SENTIMENT_WEIGHTS.items():
        if word in context:
            weight = max(weight, w)
    return round(weight, 2)

def extract_skills(text: str) -> List[Dict]:
    found = []
    for skill in TECH_SKILLS:
        if skill in text:
            w = get_sentiment_weight(text, skill)
            found.append({
                "skill": skill,
                "mastery": w,
                "level": "expert" if w >= 0.8 else
                         "intermediate" if w >= 0.5 else
                         "beginner"
            })
    return found

def parse_resume(pdf_path: str) -> Dict:
    text = extract_text_from_pdf(pdf_path)
    skills = extract_skills(text)
    return {"skills": skills, "total": len(skills)}

def parse_jd(jd_text: str) -> List[Dict]:
    text = jd_text.lower()
    return [{"skill": s, "required_mastery": 0.8}
            for s in TECH_SKILLS if s in text]

def compute_gap(resume_skills: List[Dict], jd_skills: List[Dict]) -> Dict:
    resume_map = {s["skill"]: s["mastery"] for s in resume_skills}
    gaps = []
    matched = []
    for jd in jd_skills:
        skill = jd["skill"]
        current = resume_map.get(skill, 0.0)
        gap = 0.8 - current
        if gap > 0.1:
            gaps.append({
                "skill": skill,
                "current": current,
                "required": 0.8,
                "gap": round(gap, 2),
                "priority": "HIGH" if gap > 0.5 else "MEDIUM"
            })
        else:
            matched.append(skill)
    score = len(matched) / len(jd_skills) * 100 if jd_skills else 0
    return {
        "gaps": sorted(gaps, key=lambda x: x["gap"], reverse=True),
        "matched": matched,
        "match_score": round(score, 1),
        "total_gaps": len(gaps)
    }

if __name__ == "__main__":
    sample_jd = """
    Data Scientist role requiring:
    Python, Machine Learning, TensorFlow,
    SQL, Docker, Statistics, Data Analysis
    """
    jd_skills = parse_jd(sample_jd)
    print(f"JD requires: {[s['skill'] for s in jd_skills]}")

    sample_resume_text = """
    I architected machine learning pipelines using python and tensorflow.
    Developed data analysis tools with pandas and sql.
    Familiar with docker.
    """
    resume_skills = extract_skills(sample_resume_text.lower())
    result = compute_gap(resume_skills, jd_skills)
    print(f"\nMatch Score: {result['match_score']}%")
    print(f"Gaps found: {result['total_gaps']}")
    for g in result['gaps']:
        print(f"  {g['skill']:20} current={g['current']} gap={g['gap']} [{g['priority']}]")