from typing import Optional, List
from pydantic import BaseModel, Field


class Experience(BaseModel):
    company: str | None = None
    title: str | None = None
    duration: str | None = None
    description: str | None = None
    department: str | None = None


class Education(BaseModel):
    institution: str | None = None
    degree: str | None = None
    field_of_study: str | None = None
    start_year: int | None = None
    end_year: int | None = None
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
    overall: int = Field(0, ge=0, le=100)
    contact_information: int = Field(0, ge=0, le=10)
    professional_summary: int = Field(0, ge=0, le=10)
    work_experience: int = Field(0, ge=0, le=10)
    skills: int = Field(0, ge=0, le=10)
    education_certifications: int = Field(0, ge=0, le=10)
    achievements_projects: int = Field(0, ge=0, le=10)
    format_design: int = Field(0, ge=0, le=10)
    grade: str = Field("Poor")
    remarks: str | None = None


# ── Client-1 response schema ──────────────────────────────────────────────────
# Field names match the client's Salesforce org API names exactly.

class ClientResumeData(BaseModel):
    # Identity
    Name: str | None = None                          # full name (Formula field)
    FullName__c: str | None = None                   # complete official name
    Nationality__c: str | None = None
    Date_of_Birth__c: str | None = None              # YYYY-MM-DD
    Years_of_Experience__c: float | None = None      # numeric years
    Current_Location__c: str | None = None
    CurrentDesignation__c: str | None = None
    Email: str | None = None
    PhoneNumber__c: str | None = None                # primary mobile
    SCSCHAMPS__PhoneNumber__c: str | None = None     # secondary/alternate phone
    CurrentCompany__c: str | None = None
    Consultant__c: None = None                       # always null — assigned in SF
    Type_1__c: str | None = None                     # Graduate | Non Graduate
    Spoken_Language__c: List[str] = Field(default_factory=list)  # multi-select array


class ClientParseResponse(BaseModel):
    success: bool
    processing_time_ms: float
    data: ClientResumeData


# ── Standard web-app response ─────────────────────────────────────────────────

class ResumeData(BaseModel):
    # Contact
    first_name: str | None = None
    last_name: str | None = None
    name: str | None = None
    email: str | None = None
    alternate_email: str | None = None
    phone: str | None = None
    number: str | None = Field(None, description="Alternate/secondary contact number")
    current_location: str | None = None
    linkedin_url: str | None = None
    web_address: str | None = None

    # Personal
    date_of_birth: str | None = None
    gender: str | None = None
    nationality: str | None = None
    father_name: str | None = None
    mother_name: str | None = None
    aadhar_number: str | None = None
    pan_number: str | None = None
    passport_number: str | None = None
    blood_group: str | None = None
    languages_known: str | None = None
    marital_status: str | None = None

    # Skills (categorized)
    skills: str | None = None
    primary_skills: str | None = None
    technical_skills: str | None = None
    general_skills: str | None = None

    # Experience
    experience: str | None = None
    total_years_of_experience: float | None = None
    number_of_companies: int | None = None
    current_company: str | None = None
    current_designation: str | None = None
    current_ctc: str | None = None
    expected_ctc: str | None = None
    notice_period: str | None = None
    current_employment_status: str | None = None
    industry: str | None = None
    preferred_location: str | None = None

    # Education
    education: str | None = None
    highest_degree: str | None = None
    qualification_1: str | None = None
    qualification_1_type: str | None = None
    institute_1: str | None = None
    qualification_2: str | None = None
    qualification_2_type: str | None = None
    institute_2: str | None = None
    education_detail: str | None = None

    # Other
    projects: str | None = None
    certifications: str | None = None
    awards: str | None = None
    summary: str | None = None
    overall_score: int | None = None
    grade: str | None = None
    resume_text: str | None = None


class ParseResponse(BaseModel):
    success: bool
    data: ResumeData
    processing_time_ms: float


# ── Salesforce SCSCHAMPS-mapped response ──────────────────────────────────────

class SalesforceResumeData(BaseModel):
    # Contact / Identity
    FirstName: str | None = None
    LastName: str | None = None
    Name: str | None = None
    Email: str | None = None
    Title: str | None = None
    AadharNumber: str | None = None
    AlternateEmail: str | None = None
    AlternatePhoneNumber: str | None = None
    Phone: str | None = None
    PhoneNumber: str | None = None
    MobilePhone: str | None = None
    LinkedIn_URL: str | None = None
    Web_address: str | None = None
    DateOfBirth: str | None = None
    Birthdate: str | None = None
    Gender: str | None = None
    Blood_Group: str | None = None
    Father_s_Name: str | None = None
    MotherName: str | None = None
    Nationnality: str | None = None
    PAN_Number: str | None = None
    Passport_Number: str | None = None
    LanguagesKnown: str | None = None

    # Location
    City: str | None = None
    State: str | None = None
    Current_Location: str | None = None
    Preferred_Location: str | None = None

    # Professional
    CurrentDesignation: str | None = None
    Designation: str | None = None
    Department: str | None = None
    Company: str | None = None
    CurrentCompany: str | None = None
    CurrentDuration: float | None = None
    Years_of_Experience: float | None = None
    No_of_companies_worked_in: int | None = None
    Current_Employment: str | None = None
    Industry: str | None = None

    # Skills
    Primary_Skills: str | None = None
    Technical_Skills: str | None = None
    General_Skills: str | None = None
    SkillList: str | None = None
    Skill_List: str | None = None
    AutoPopulate_Skillset: str | None = None
    Key_Skillsets_del: str | None = None

    # Education
    Education: str | None = None
    Highest_Degree: str | None = None
    education_start_year: int | None = None
    Education_End_Year: int | None = None
    Education_year: bool = False
    educationDetail: str | None = None
    Qualification_1: str | None = None
    Qualification_1_Type: str | None = None
    Qualification_2: str | None = None
    Qualification_2_Type: str | None = None
    Institute_1: str | None = None
    Institute_2: str | None = None
    Certification: str | None = None
    Awards: str | None = None

    # Compensation / Availability
    Current_CTC: str | None = None
    Expected_CTC: str | None = None
    Notice_Period: str | None = None
    Available_To_Start: str | None = None

    # Resume Content
    ResumeRich: str | None = None
    Resume: str | None = None
    TextResume: str | None = None
    Resume_URL: str | None = None
    Resume_Attachment_Id: str | None = None
    Date_Parsed_Text: str | None = None

    # Scoring
    Candidate_Score: int | None = None
    Resume_Score: float | None = None
    resume_score: ResumeScore = ResumeScore()

    # Candidate Meta
    Candidate_Status: str | None = None
    Status: str | None = None
    Background_Check: str | None = None
    Source: str | None = None
    Talent_Id: str | None = None
    Job_Id: str | None = None
    job: str | None = None
    Lead: str | None = None
    Recruiter: str | None = None
    converted_from_lead: bool = False
    Ampliz_Contact: bool = False
    Ampliz_Talent_Name: str | None = None


class SalesforceParseResponse(BaseModel):
    success: bool
    data: SalesforceResumeData
    processing_time_ms: float


# ── Shared utility models ─────────────────────────────────────────────────────

class BulkParseItem(BaseModel):
    filename: str
    success: bool
    data: Optional["ResumeData"] = None
    error: Optional[str] = None
    processing_time_ms: float = 0.0


class BulkParseResponse(BaseModel):
    success: bool
    total: int
    parsed: int
    failed: int
    results: list[BulkParseItem]
    total_processing_time_ms: float


class BulkSalesforceParseItem(BaseModel):
    filename: str
    success: bool
    data: Optional["SalesforceResumeData"] = None
    error: Optional[str] = None
    processing_time_ms: float = 0.0


class BulkSalesforceParseResponse(BaseModel):
    success: bool
    total: int
    parsed: int
    failed: int
    results: list[BulkSalesforceParseItem]
    total_processing_time_ms: float


class BulkClientParseItem(BaseModel):
    filename: str
    success: bool
    data: Optional["ClientResumeData"] = None
    error: Optional[str] = None
    processing_time_ms: float = 0.0


class BulkClientParseResponse(BaseModel):
    success: bool
    total: int
    parsed: int
    failed: int
    results: list[BulkClientParseItem]
    total_processing_time_ms: float


class BulkJobStatus(BaseModel):
    job_id: str
    status: str  # "processing" | "completed" | "failed"
    total: int
    result: Optional["BulkParseResponse"] = None
    error: Optional[str] = None


class ErrorResponse(BaseModel):
    success: bool = False
    detail: str


class HealthResponse(BaseModel):
    status: str
    openrouter_connected: bool
    model: str


class ModelInfo(BaseModel):
    name: str
    size: int | None = None
    modified_at: str | None = None


class ModelsResponse(BaseModel):
    success: bool = True
    models: list[ModelInfo]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_current_job(exp: dict) -> bool:
    duration = (exp.get("duration") or "").lower()
    return any(kw in duration for kw in ("present", "current", "now", "till date", "ongoing"))


def _extract_city(location: str | None) -> str | None:
    if not location:
        return None
    parts = [p.strip() for p in location.split(",")]
    return parts[0] if parts else None


def _extract_state(location: str | None) -> str | None:
    if not location:
        return None
    parts = [p.strip() for p in location.split(",")]
    return parts[1] if len(parts) > 1 else None


def _parse_ctc_to_number(ctc_str: str | None) -> float | None:
    if not ctc_str:
        return None
    import re
    cleaned = re.sub(r'[₹$,\s]', '', ctc_str.upper())
    match = re.search(r'(\d+(?:\.\d+)?)', cleaned)
    if not match:
        return None
    value = float(match.group(1))
    if 'LPA' in cleaned or 'LAKH' in cleaned or 'LAC' in cleaned:
        value = value * 100000
    elif 'CR' in cleaned:
        value = value * 10000000
    return value


def _parse_duration_years(duration_str: str | None) -> float | None:
    if not duration_str:
        return None
    import re
    match = re.search(r'(\d+(?:\.\d+)?)', str(duration_str))
    return float(match.group(1)) if match else None


def _infer_candidate_type(parsed: dict) -> str | None:
    """
    Determine if the candidate is a Graduate or Non Graduate based on their
    education. Any completed college/university degree (Bachelor's or above)
    counts as Graduate; diploma-only, high-school-only, or no education
    returns Non Graduate.
    """
    education: list[dict] = parsed.get("education", [])
    highest_degree: str = (parsed.get("highest_degree") or "").lower()

    # Keywords that indicate a recognised degree-level qualification
    graduate_kw = (
        "bachelor", "b.sc", "b.sc.", "bsc", "b.e", "b.e.", "be ",
        "b.tech", "btech", "b.com", "bcom", "b.a", "b.a.", "ba ",
        "b.s", "b.s.", "bs ", "llb", "mbbs", "bba", "bca",
        "master", "m.sc", "m.sc.", "msc", "m.e", "m.e.", "me ",
        "m.tech", "mtech", "mba", "m.com", "mcom", "m.a", "m.a.", "ma ",
        "m.s", "m.s.", "ms ", "llm", "mca",
        "phd", "ph.d", "ph.d.", "doctorate", "doctor of",
        "degree", "graduate", "graduation",
        "engineering", "technology", "science", "arts", "commerce",
    )

    if highest_degree and any(kw in highest_degree for kw in graduate_kw):
        return "Graduate"

    for edu in education:
        degree = (edu.get("degree") or "").lower()
        field = (edu.get("field_of_study") or "").lower()
        combined = f"{degree} {field}"
        if any(kw in combined for kw in graduate_kw):
            return "Graduate"

    # Has some education listed but no degree-level qualification found
    if education or highest_degree:
        return "Non Graduate"

    return None


def _languages_to_list(languages_known: str | None) -> list[str]:
    """Convert comma/semicolon-separated language string to a clean list."""
    if not languages_known:
        return []
    import re
    # Split on comma, semicolon, slash, or ' and '
    parts = re.split(r'[,;/]|\band\b', languages_known, flags=re.IGNORECASE)
    result = []
    for part in parts:
        cleaned = part.strip().strip(".")
        if cleaned:
            result.append(cleaned)
    return result


def map_to_client(parsed: dict) -> ClientResumeData:
    """Map the internal parsed dict to the client-1 field schema."""
    return ClientResumeData(
        Name=parsed.get("name"),
        FullName__c=parsed.get("name"),
        Nationality__c=parsed.get("nationality"),
        Date_of_Birth__c=parsed.get("date_of_birth"),
        Years_of_Experience__c=parsed.get("total_years_of_experience"),
        Current_Location__c=parsed.get("current_location"),
        CurrentDesignation__c=parsed.get("current_designation"),
        Email=parsed.get("email"),
        PhoneNumber__c=parsed.get("phone"),
        SCSCHAMPS__PhoneNumber__c=parsed.get("number"),
        CurrentCompany__c=parsed.get("current_company"),
        Consultant__c=None,
        Type_1__c=_infer_candidate_type(parsed),
        Spoken_Language__c=_languages_to_list(parsed.get("languages_known")),
    )


def map_to_salesforce(parsed: dict, raw_text: str | None = None) -> SalesforceResumeData:
    """Map the internal parsed dict to SalesforceResumeData field names."""
    from datetime import date

    skills: list[str] = parsed.get("skills", [])
    primary_skills: list[str] = parsed.get("primary_skills", [])
    technical_skills: list[str] = parsed.get("technical_skills", [])
    general_skills: list[str] = parsed.get("general_skills", [])
    experience: list[dict] = parsed.get("experience", [])
    education: list[dict] = parsed.get("education", [])
    certifications: list[dict] = parsed.get("certifications", [])
    awards: list[dict] = parsed.get("awards", [])
    score = parsed.get("resume_score", {})

    current_exp = next(
        (e for e in experience if _is_current_job(e)),
        experience[0] if experience else {},
    )

    skill_list = ", ".join(skills) if skills else None
    primary_skill_list = ", ".join(primary_skills) if primary_skills else None
    tech_skill_text = "\n".join(technical_skills) if technical_skills else None
    general_skill_text = "\n".join(general_skills) if general_skills else None

    exp_html_parts = []
    for exp in experience:
        co = exp.get("company", "")
        ti = exp.get("title", "")
        du = exp.get("duration", "")
        de = exp.get("description", "")
        exp_html_parts.append(f"<b>{ti}</b> at {co} ({du})<br/>{de}")
    resume_rich = "<br/><br/>".join(exp_html_parts) if exp_html_parts else parsed.get("summary")

    exp_text_parts = []
    for exp in experience:
        co = exp.get("company", "")
        ti = exp.get("title", "")
        du = exp.get("duration", "")
        de = exp.get("description", "")
        exp_text_parts.append(f"{ti} at {co} ({du})\n{de}")
    resume_text = "\n\n".join(exp_text_parts) if exp_text_parts else None

    cert_text = ", ".join(c.get("name", "") for c in certifications if c.get("name")) or None
    awards_text = ", ".join(
        f"{a.get('name', '')}" + (f" ({a['year']})" if a.get('year') else "")
        for a in awards if a.get("name")
    ) or None

    edu_first = education[0] if education else {}
    edu_start_year = edu_first.get("start_year")
    edu_end_year = edu_first.get("end_year")
    has_edu_year = bool(edu_start_year or edu_end_year)
    edu_str = parsed.get("highest_degree") or edu_first.get("degree")

    score_obj = ResumeScore(**score) if isinstance(score, dict) else ResumeScore()

    return SalesforceResumeData(
        FirstName=parsed.get("first_name"),
        LastName=parsed.get("last_name"),
        Name=parsed.get("name"),
        Email=parsed.get("email"),
        Title=current_exp.get("title"),
        AadharNumber=parsed.get("aadhar_number"),
        AlternateEmail=parsed.get("alternate_email"),
        AlternatePhoneNumber=parsed.get("number"),
        Phone=parsed.get("phone"),
        PhoneNumber=parsed.get("phone"),
        MobilePhone=parsed.get("phone"),
        LinkedIn_URL=parsed.get("linkedin_url"),
        Web_address=parsed.get("web_address"),
        DateOfBirth=parsed.get("date_of_birth"),
        Birthdate=parsed.get("date_of_birth"),
        Gender=parsed.get("gender"),
        Blood_Group=parsed.get("blood_group"),
        Father_s_Name=parsed.get("father_name"),
        MotherName=parsed.get("mother_name"),
        Nationnality=parsed.get("nationality"),
        PAN_Number=parsed.get("pan_number"),
        Passport_Number=parsed.get("passport_number"),
        LanguagesKnown=parsed.get("languages_known"),
        Current_Location=parsed.get("current_location"),
        City=_extract_city(parsed.get("current_location")),
        State=_extract_state(parsed.get("current_location")),
        Preferred_Location=parsed.get("preferred_location"),
        CurrentCompany=parsed.get("current_company") or current_exp.get("company"),
        CurrentDesignation=parsed.get("current_designation") or current_exp.get("title"),
        CurrentDuration=_parse_duration_years(current_exp.get("duration")),
        Designation=current_exp.get("title"),
        Company=parsed.get("current_company") or current_exp.get("company"),
        Department=current_exp.get("department"),
        Years_of_Experience=parsed.get("total_years_of_experience"),
        No_of_companies_worked_in=parsed.get("number_of_companies"),
        Current_Employment=parsed.get("current_employment_status"),
        Industry=parsed.get("industry"),
        Primary_Skills=primary_skill_list,
        Technical_Skills=tech_skill_text,
        General_Skills=general_skill_text,
        SkillList=skill_list,
        Skill_List=primary_skill_list,
        AutoPopulate_Skillset=skill_list,
        Key_Skillsets_del=tech_skill_text,
        Education=edu_str,
        Highest_Degree=parsed.get("highest_degree"),
        education_start_year=edu_start_year,
        Education_End_Year=edu_end_year,
        Education_year=has_edu_year,
        educationDetail=parsed.get("education_detail"),
        Qualification_1=parsed.get("qualification_1"),
        Qualification_1_Type=parsed.get("qualification_1_type"),
        Qualification_2=parsed.get("qualification_2"),
        Qualification_2_Type=parsed.get("qualification_2_type"),
        Institute_1=parsed.get("institute_1"),
        Institute_2=parsed.get("institute_2"),
        Certification=cert_text,
        Awards=awards_text,
        Current_CTC=parsed.get("current_ctc"),
        Expected_CTC=parsed.get("expected_ctc"),
        Notice_Period=parsed.get("notice_period"),
        ResumeRich=resume_rich,
        Resume=resume_text,
        TextResume=raw_text,
        Date_Parsed_Text=date.today().isoformat(),
        Candidate_Score=score_obj.overall,
        Resume_Score=float(score_obj.overall),
        resume_score=score_obj,
    )
