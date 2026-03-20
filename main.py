from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile
import os

from parser import parse_resume, parse_jd, compute_gap
from algorithm import grkt_generate_pathway

app = FastAPI(title="G-RKT API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "G-RKT API is running!"}

@app.post("/analyze")
async def analyze(
    resume: UploadFile = File(...),
    jd_text: str = Form(...)
):
    # Save uploaded resume temporarily
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf"
    ) as tmp:
        content = await resume.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Step 1: Parse resume
        resume_data = parse_resume(tmp_path)
        resume_skills = resume_data["skills"]

        # Step 2: Parse JD
        jd_skills = parse_jd(jd_text)

        # Step 3: Compute gap
        gap_result = compute_gap(resume_skills, jd_skills)

        # Step 4: G-RKT pathway
        pathway_result = grkt_generate_pathway(
            resume_skills,
            gap_result["gaps"]
        )

        return JSONResponse({
            "status": "success",
            "resume_skills": resume_skills,
            "jd_skills": jd_skills,
            "match_score": gap_result["match_score"],
            "gaps": gap_result["gaps"],
            "pathway": pathway_result["pathway"],
            "total_hours": pathway_result["total_hours"],
            "hours_saved": pathway_result["hours_saved_vs_generic"],
            "final_match_score": pathway_result["final_match_score"]
        })

    finally:
        os.unlink(tmp_path)

@app.post("/analyze-text")
async def analyze_text(
    resume_text: str = Form(...),
    jd_text: str = Form(...)
):
    from parser import extract_skills
    resume_skills = extract_skills(resume_text.lower())
    jd_skills = parse_jd(jd_text)
    gap_result = compute_gap(resume_skills, jd_skills)
    pathway_result = grkt_generate_pathway(
        resume_skills,
        gap_result["gaps"]
    )
    return JSONResponse({
        "status": "success",
        "resume_skills": resume_skills,
        "match_score": gap_result["match_score"],
        "gaps": gap_result["gaps"],
        "pathway": pathway_result["pathway"],
        "total_hours": pathway_result["total_hours"],
        "hours_saved": pathway_result["hours_saved_vs_generic"],
        "final_match_score": pathway_result["final_match_score"]
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)