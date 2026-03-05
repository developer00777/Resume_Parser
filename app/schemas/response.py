from typing import Optional
from pydantic import BaseModel, Field


class Experience(BaseModel):
    company: str | None = None
    title: str | None = None
    duration: str | None = None
    description: str | None = None


class Education(BaseModel):
    institution: str | None = None
    degree: str | None = None
    field_of_study: str | None = None
    year: str | None = None
    grade: str | None = None


class Project(BaseModel):
    name: str | None = None
    duration: str | None = None
    description: str | None = None


class Certification(BaseModel):
    name: str | None = None
    issuer: str | None = None


class Award(BaseModel):
    name: str | None = None
    year: str | None = None


class ResumeScore(BaseModel):
    # ── 7-category weighted score matrix (each raw score 0–10, weighted to 100) ──
    overall: int = Field(0, ge=0, le=100, description="Overall weighted score out of 100")

    # Individual category raw scores (0–10)
    contact_information: int = Field(0, ge=0, le=10, description="Contact info completeness (weight 5%)")
    professional_summary: int = Field(0, ge=0, le=10, description="Summary clarity and strength (weight 15%)")
    work_experience: int = Field(0, ge=0, le=10, description="Experience structure and impact (weight 25%)")
    skills: int = Field(0, ge=0, le=10, description="Skills relevance and categorization (weight 20%)")
    education_certifications: int = Field(0, ge=0, le=10, description="Education and certifications (weight 10%)")
    achievements_projects: int = Field(0, ge=0, le=10, description="Projects, awards, measurable results (weight 15%)")
    format_design: int = Field(0, ge=0, le=10, description="Layout, readability, formatting (weight 10%)")

    # Interpretation band
    grade: str = Field("Poor", description="Excellent / Good / Average / Poor")
    remarks: str | None = None


# ── Standard web-app response (existing) ─────────────────────────────────────

class ResumeData(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    number: str | None = Field(None, description="Alternate/secondary contact number")
    current_location: str | None = None
    skills: list[str] = []
    experience: list[Experience] = []
    education: list[Education] = []
    projects: list[Project] = []
    certifications: list[Certification] = []
    awards: list[Award] = []
    summary: str | None = None
    resume_score: ResumeScore = ResumeScore()


class ParseResponse(BaseModel):
    success: bool
    data: ResumeData
    processing_time_ms: float


# ── Salesforce SCSCHAMPS-mapped response ──────────────────────────────────────
# Field names match SCSCHAMPS__<FieldName>__c exactly (prefix/suffix stripped).

class SalesforceResumeData(BaseModel):
    # Contact / identity
    Title: str | None = None                  # SCSCHAMPS__Title__c
    AadharNumber: str | None = None           # SCSCHAMPS__AadharNumber__c
    AlternateEmail: str | None = None         # SCSCHAMPS__AlternateEmail__c
    AlternatePhoneNumber: str | None = None   # SCSCHAMPS__AlternatePhoneNumber__c
    Phone: str | None = None                  # SCSCHAMPS__Phone__c
    PhoneNumber: str | None = None            # SCSCHAMPS__PhoneNumber__c
    LinkedIn_URL: str | None = None           # SCSCHAMPS__LinkedIn_URL__c
    Web_address: str | None = None            # SCSCHAMPS__Web_address__c
    DateOfBirth: str | None = None            # SCSCHAMPS__DateOfBirth__c
    Gender: str | None = None                 # SCSCHAMPS__Gender__c

    # Location
    City: str | None = None                   # SCSCHAMPS__City__c
    State: str | None = None                  # SCSCHAMPS__State__c
    Current_Location: str | None = None       # SCSCHAMPS__Current_Location__c
    Preferred_Location: str | None = None     # SCSCHAMPS__Preferred_Location__c

    # Professional
    CurrentDesignation: str | None = None     # SCSCHAMPS__CurrentDesignation__c
    Designation: str | None = None            # SCSCHAMPS__Designation__c
    Department: str | None = None             # SCSCHAMPS__Department__c
    Company: str | None = None                # SCSCHAMPS__Company__c
    CurrentCompany: str | None = None         # SCSCHAMPS__CurrentCompany__c
    CurrentDuration: str | None = None        # SCSCHAMPS__CurrentDuration__c
    Years_of_Experience: str | None = None    # SCSCHAMPS__Years_of_Experience__c

    # Skills
    Primary_Skills: str | None = None         # SCSCHAMPS__Primary_Skills__c  (newline-sep list)
    Technical_Skills: str | None = None       # SCSCHAMPS__Technical_Skills__c
    General_Skills: str | None = None         # SCSCHAMPS__General_Skills__c
    SkillList: str | None = None              # SCSCHAMPS__SkillList__c  (comma-sep)
    AutoPopulate_Skillset: bool = False       # SCSCHAMPS__AutoPopulate_Skillset__c
    Key_Skillsets_del: str | None = None      # SCSCHAMPS__Key_Skillsets_del__c

    # Compensation / availability
    Current_CTC: str | None = None            # SCSCHAMPS__Current_CTC__c
    Expected_CTC: str | None = None           # SCSCHAMPS__Expected_CTC__c
    Notice_Period: str | None = None          # SCSCHAMPS__Notice_Period__c
    Available_To_Start: str | None = None     # SCSCHAMPS__Available_To_Start_c

    # Resume content
    ResumeRich: str | None = None             # SCSCHAMPS__ResumeRich__c  (HTML/rich text summary)
    Resume_URL: str | None = None             # SCSCHAMPS__Resume_URL__c
    Resume_Attachment_Id: str | None = None   # SCSCHAMPS__Resume_Attachment_Id__c
    Resume: str | None = None                 # SCSCHAMPS__Resume__c

    # Candidate meta
    Candidate_Status: str | None = None       # SCSCHAMPS__Candidate_Status__c
    Background_Check: str | None = None       # SCSCHAMPS__Background_Check__c
    Source: str | None = None                 # SCSCHAMPS__Source__c
    Talent_Id: str | None = None              # SCSCHAMPS__Talent_Id__c
    Job_Id: str | None = None                 # SCSCHAMPS__Job_Id__c
    job: str | None = None                    # SCSCHAMPS__job__c
    Lead: str | None = None                   # SCSCHAMPS__Lead__c
    Recruiter: str | None = None              # SCSCHAMPS__Recruiter__c
    converted_from_lead: bool = False         # SCSCHAMPS__converted_from_lead__c

    # Scoring (extra, not in SCSCHAMPS — returned for convenience)
    resume_score: ResumeScore = ResumeScore()


class SalesforceParseResponse(BaseModel):
    success: bool
    data: SalesforceResumeData
    processing_time_ms: float


# ── Shared utility models ─────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    success: bool = False
    detail: str


class HealthResponse(BaseModel):
    status: str
    ollama_connected: bool
    model: str


class ModelInfo(BaseModel):
    name: str
    size: int | None = None
    modified_at: str | None = None


class ModelsResponse(BaseModel):
    success: bool = True
    models: list[ModelInfo]


# ── Helpers ───────────────────────────────────────────────────────────────────

def map_to_salesforce(parsed: dict) -> SalesforceResumeData:
    """Map the internal parsed dict to SalesforceResumeData field names."""
    skills: list[str] = parsed.get("skills", [])
    experience: list[dict] = parsed.get("experience", [])
    score = parsed.get("resume_score", {})

    # Current company / designation from first experience entry
    current_exp = experience[0] if experience else {}

    # Build comma-separated skill list
    skill_list = ", ".join(skills) if skills else None

    # Build rich HTML summary from experience entries
    exp_html_parts = []
    for exp in experience:
        co = exp.get("company", "")
        ti = exp.get("title", "")
        du = exp.get("duration", "")
        de = exp.get("description", "")
        exp_html_parts.append(f"<b>{ti}</b> at {co} ({du})<br/>{de}")
    resume_rich = "<br/><br/>".join(exp_html_parts) if exp_html_parts else parsed.get("summary")

    return SalesforceResumeData(
        # Contact
        Phone=parsed.get("phone"),
        PhoneNumber=parsed.get("phone"),
        AlternatePhoneNumber=parsed.get("number"),
        Current_Location=parsed.get("current_location"),
        City=_extract_city(parsed.get("current_location")),
        State=_extract_state(parsed.get("current_location")),

        # Current role
        CurrentCompany=current_exp.get("company"),
        CurrentDesignation=current_exp.get("title"),
        CurrentDuration=current_exp.get("duration"),
        Designation=current_exp.get("title"),
        Company=current_exp.get("company"),

        # Skills
        Primary_Skills="\n".join(skills[:10]) if skills else None,
        Technical_Skills="\n".join(skills) if skills else None,
        SkillList=skill_list,
        AutoPopulate_Skillset=bool(skills),

        # Resume content
        ResumeRich=resume_rich,

        # Score
        resume_score=ResumeScore(**score) if isinstance(score, dict) else ResumeScore(),
    )


def _extract_city(location: str | None) -> str | None:
    """Best-effort: 'City, State' → 'City'."""
    if not location:
        return None
    parts = [p.strip() for p in location.split(",")]
    return parts[0] if parts else None


def _extract_state(location: str | None) -> str | None:
    """Best-effort: 'City, State' → 'State'."""
    if not location:
        return None
    parts = [p.strip() for p in location.split(",")]
    return parts[1] if len(parts) > 1 else None
