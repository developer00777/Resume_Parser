from typing import Optional
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
    skills: str | None = None           # comma-separated string (all skills)
    primary_skills: str | None = None   # comma-separated primary skills
    technical_skills: str | None = None # comma-separated technical skills
    general_skills: str | None = None   # comma-separated general/soft skills

    # Experience
    experience: str | None = None       # newline-separated: "Company | Title | Duration | Description"
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
    education: str | None = None        # newline-separated: "Institution | Degree | Field | Year | Grade"
    highest_degree: str | None = None
    qualification_1: str | None = None
    qualification_1_type: str | None = None
    institute_1: str | None = None
    qualification_2: str | None = None
    qualification_2_type: str | None = None
    institute_2: str | None = None
    education_detail: str | None = None

    # Other
    projects: str | None = None         # newline-separated: "Name | Duration | Description"
    certifications: str | None = None   # newline-separated: "Name | Issuer"
    awards: str | None = None           # newline-separated: "Name | Year"
    summary: str | None = None
    overall_score: int | None = None
    grade: str | None = None

    # Full raw text extracted from the uploaded resume (PDF/DOCX)
    resume_text: str | None = None


class ParseResponse(BaseModel):
    success: bool
    data: ResumeData
    processing_time_ms: float


# ── Salesforce SCSCHAMPS-mapped response ──────────────────────────────────────
# Field names match Salesforce API names. Both SCSCHAMPS__*__c and custom *__c fields.

class SalesforceResumeData(BaseModel):
    # ── Contact / Identity ──────────────────────────────────────────────────
    FirstName: str | None = None                  # FirstName
    LastName: str | None = None                   # LastName (required in SF)
    Name: str | None = None                       # Full Name
    Email: str | None = None                      # Email
    Title: str | None = None                      # SCSCHAMPS__Title__c
    AadharNumber: str | None = None               # AadharNumber__c / SCSCHAMPS__AadharNumber__c
    AlternateEmail: str | None = None             # AlternateEmail__c / SCSCHAMPS__AlternateEmail__c
    AlternatePhoneNumber: str | None = None       # AlternatePhoneNumber__c / SCSCHAMPS__AlternatePhoneNumber__c
    Phone: str | None = None                      # SCSCHAMPS__Phone__c
    PhoneNumber: str | None = None                # PhoneNumber__c / SCSCHAMPS__PhoneNumber__c
    MobilePhone: str | None = None                # MobilePhone
    LinkedIn_URL: str | None = None               # LinkedinURL__c / SCSCHAMPS__LinkedIn_URL__c
    Web_address: str | None = None                # SCSCHAMPS__Web_address__c
    DateOfBirth: str | None = None                # DateOfBirth__c / SCSCHAMPS__DateOfBirth__c
    Birthdate: str | None = None                  # Birthdate
    Gender: str | None = None                     # Gender__c / SCSCHAMPS__Gender__c
    Blood_Group: str | None = None                # Blood_Group__c
    Father_s_Name: str | None = None              # Father_s_Name__c
    MotherName: str | None = None                 # MotherName__c
    Nationnality: str | None = None               # Nationnality__c (SF spelling)
    PAN_Number: str | None = None                 # PAN_Number__c
    Passport_Number: str | None = None            # Passport_Number__c
    LanguagesKnown: str | None = None             # LanguagesKnown__c

    # ── Location ────────────────────────────────────────────────────────────
    City: str | None = None                       # SCSCHAMPS__City__c
    State: str | None = None                      # SCSCHAMPS__State__c
    Current_Location: str | None = None           # Current_Location__c / SCSCHAMPS__Current_Location__c
    Preferred_Location: str | None = None         # Preferred_Location__c / SCSCHAMPS__Preferred_Location__c

    # ── Professional ────────────────────────────────────────────────────────
    CurrentDesignation: str | None = None         # CurrentDesignation__c / SCSCHAMPS__CurrentDesignation__c
    Designation: str | None = None                # SCSCHAMPS__Designation__c
    Department: str | None = None                 # Department / SCSCHAMPS__Department__c
    Company: str | None = None                    # SCSCHAMPS__Company__c
    CurrentCompany: str | None = None             # CurrentCompany__c / SCSCHAMPS__CurrentCompany__c
    CurrentDuration: float | None = None          # SCSCHAMPS__CurrentDuration__c (double)
    Years_of_Experience: float | None = None      # Years_of_Experience__c / SCSCHAMPS__Years_of_Experience__c
    No_of_companies_worked_in: int | None = None  # No_of_companies_worked_in__c
    Current_Employment: str | None = None         # Current_Employment__c
    Industry: str | None = None                   # Industry__c

    # ── Skills ──────────────────────────────────────────────────────────────
    Primary_Skills: str | None = None             # SCSCHAMPS__Primary_Skills__c
    Technical_Skills: str | None = None           # SCSCHAMPS__Technical_Skills__c
    General_Skills: str | None = None             # SCSCHAMPS__General_Skills__c
    SkillList: str | None = None                  # SCSCHAMPS__SkillList__c / SkillList__c
    Skill_List: str | None = None                 # Skill_List__c (AI)
    AutoPopulate_Skillset: str | None = None      # SCSCHAMPS__AutoPopulate_Skillset__c (textarea)
    Key_Skillsets_del: str | None = None          # SCSCHAMPS__Key_Skillsets_del__c

    # ── Education ───────────────────────────────────────────────────────────
    Education: str | None = None                  # Education__c
    Highest_Degree: str | None = None             # Highest_Degree__c
    education_start_year: int | None = None       # education_start_year__c
    Education_End_Year: int | None = None         # Education_End_Year__c
    # Nullable, not False-by-default. A boolean that is always present gets
    # assigned on every parse, so a False default silently clears whatever
    # Salesforce already held whenever the resume says nothing about it.
    Education_year: bool | None = None            # Education_year__c
    educationDetail: str | None = None            # educationDetail__c
    Qualification_1: str | None = None            # Qualification_1__c
    Qualification_1_Type: str | None = None       # Qualification_1_Type__c
    Qualification_2: str | None = None            # Qualification_2__c
    Qualification_2_Type: str | None = None       # Qualification_2_Type__c
    Institute_1: str | None = None                # Institute_1__c
    Institute_2: str | None = None                # Institute_2__c
    Certification: str | None = None              # Certification__c
    Awards: str | None = None                     # Awards__c

    # ── Compensation / Availability ─────────────────────────────────────────
    Current_CTC: str | None = None                # Current_CTC__c / SCSCHAMPS__Current_CTC__c
    Expected_CTC: str | None = None               # Expected_CTC__c / SCSCHAMPS__Expected_CTC__c
    Notice_Period: str | None = None              # Notice_Period__c / SCSCHAMPS__Notice_Period__c
    Available_To_Start: str | None = None         # SCSCHAMPS__Available_To_Start__c

    # ── Resume Content ──────────────────────────────────────────────────────
    ResumeRich: str | None = None                 # SCSCHAMPS__ResumeRich__c (HTML)
    Resume: str | None = None                     # SCSCHAMPS__Resume__c (work details)
    TextResume: str | None = None                 # TextResume__c (raw text)
    Resume_URL: str | None = None                 # SCSCHAMPS__Resume_URL__c
    Resume_Attachment_Id: str | None = None       # SCSCHAMPS__Resume_Attachment_Id__c
    Date_Parsed_Text: str | None = None           # Date_Parsed_Text__c

    # ── Scoring ─────────────────────────────────────────────────────────────
    Candidate_Score: int | None = None            # Candidate_Score__c
    Resume_Score: float | None = None             # Resume_Score__c
    # Named score_breakdown, NOT resume_score: Apex identifiers are
    # case-insensitive, so a wrapper class cannot declare both `Resume_Score`
    # and `resume_score`. With the old name, JSON.deserialize into a typed Apex
    # class does not compile and callers are forced onto deserializeUntyped.
    score_breakdown: ResumeScore = ResumeScore()

    # ── Candidate Meta (populated by Salesforce, not resume) ────────────────
    Candidate_Status: str | None = None           # SCSCHAMPS__Candidate_Status__c
    Status: str | None = None                     # Status__c
    Background_Check: str | None = None           # SCSCHAMPS__Background_Check__c
    Source: str | None = None                     # Source__c / SCSCHAMPS__Source__c
    Talent_Id: str | None = None                  # SCSCHAMPS__Talent_Id__c
    Job_Id: str | None = None                     # SCSCHAMPS__Job_Id__c
    job: str | None = None                        # SCSCHAMPS__job__c
    Lead: str | None = None                       # SCSCHAMPS__Lead__c
    Recruiter: str | None = None                  # SCSCHAMPS__Recruiter__c
    # Nullable for the same reason as Education_year — a resume never carries
    # these, so they must never be written from a parse result.
    converted_from_lead: bool | None = None       # SCSCHAMPS__converted_from_lead__c
    Ampliz_Contact: bool | None = None            # SCSCHAMPS__Ampliz_Contact__c
    Ampliz_Talent_Name: str | None = None         # SCSCHAMPS__Ampliz_Talent_Name__c


class ExtractionStatus(BaseModel):
    """
    Whether the parse actually saw the whole resume and got an answer from every
    LLM call.

    Without this, an empty field is ambiguous: the resume may have had no
    experience section, or the chunk that extracts experience may have been rate
    limited. The first is data; the second is a failure that must not be written
    to a system of record.
    """
    complete: bool = True
    chunks_total: int = 0
    chunks_ok: int = 0
    failed_chunks: list[str] = Field(default_factory=list)
    failed_sections: list[str] = Field(
        default_factory=list,
        description="Sections empty because their LLM call failed, not because the resume was silent",
    )
    errors: dict[str, str] = Field(default_factory=dict)
    text_truncated: bool = False
    text_length: int = 0
    text_sent: int = 0


def extraction_from_parsed(parsed: dict) -> ExtractionStatus:
    raw = parsed.get("extraction")
    return ExtractionStatus(**raw) if isinstance(raw, dict) else ExtractionStatus()


class SalesforceParseResponse(BaseModel):
    success: bool
    data: SalesforceResumeData
    processing_time_ms: float
    extraction: ExtractionStatus = Field(default_factory=ExtractionStatus)


class SalesforceUpdateResponse(BaseModel):
    """Result of parsing a resume and writing the fields back onto the record."""
    success: bool
    record_id: str
    dry_run: bool = Field(
        False, description="True when the payload was computed and validated but not written",
    )
    fields_written: dict[str, object] = Field(
        default_factory=dict,
        description="Salesforce API name -> value actually sent (or that would be sent)",
    )
    fields_skipped: dict[str, str] = Field(
        default_factory=dict, description="Logical field name -> why it was withheld",
    )
    extraction: ExtractionStatus = Field(default_factory=ExtractionStatus)
    processing_time_ms: float = 0.0


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
    """Return True if the duration field suggests an ongoing role."""
    duration = (exp.get("duration") or "").lower()
    return any(kw in duration for kw in ("present", "current", "now", "till date", "ongoing"))


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


def _as_int(value: object) -> int | None:
    """Coerce a count to int, or None. Guards the same crash as to_number()."""
    from app.services.salesforce_coerce import to_number
    number = to_number(value)
    return int(number) if number is not None else None


def _parse_duration_years(duration_str: str | None) -> float | None:
    """Extract numeric duration from strings like '2.5 years', '3 yrs'."""
    if not duration_str:
        return None
    import re
    match = re.search(r'(\d+(?:\.\d+)?)', str(duration_str))
    return float(match.group(1)) if match else None


def map_to_salesforce(parsed: dict, raw_text: str | None = None) -> SalesforceResumeData:
    """
    Map the internal parsed dict to SalesforceResumeData field names.

    Values are coerced into shapes Salesforce accepts on the way out — dates to
    ISO, compensation to a number where the field is numeric, picklists snapped
    to their value set, long text clipped to its declared length. Anything that
    cannot be coerced becomes None: Salesforce rejects a whole record over one
    bad value, so a null field is always cheaper than a rejected DML.
    """
    from datetime import date

    from app.services.salesforce_coerce import (
        email_or_none,
        iso_date,
        money_or_text,
        pick,
        to_number,
        truncate,
    )

    skills: list[str] = parsed.get("skills", [])
    primary_skills: list[str] = parsed.get("primary_skills", [])
    technical_skills: list[str] = parsed.get("technical_skills", [])
    general_skills: list[str] = parsed.get("general_skills", [])
    experience: list[dict] = parsed.get("experience", [])
    education: list[dict] = parsed.get("education", [])
    certifications: list[dict] = parsed.get("certifications", [])
    awards: list[dict] = parsed.get("awards", [])
    score = parsed.get("resume_score", {})

    # Current company: prefer any entry marked as ongoing/present;
    # otherwise fall back to the first entry (LLM is instructed to sort newest-first).
    current_exp = next(
        (e for e in experience if _is_current_job(e)),
        experience[0] if experience else {},
    )

    # Build comma-separated skill list
    skill_list = ", ".join(skills) if skills else None
    primary_skill_list = ", ".join(primary_skills) if primary_skills else None
    tech_skill_text = "\n".join(technical_skills) if technical_skills else None
    general_skill_text = "\n".join(general_skills) if general_skills else None

    # Build rich HTML summary from experience entries
    exp_html_parts = []
    for exp in experience:
        co = exp.get("company", "")
        ti = exp.get("title", "")
        du = exp.get("duration", "")
        de = exp.get("description", "")
        exp_html_parts.append(f"<b>{ti}</b> at {co} ({du})<br/>{de}")
    resume_rich = "<br/><br/>".join(exp_html_parts) if exp_html_parts else parsed.get("summary")

    # Build plain-text resume (work details)
    exp_text_parts = []
    for exp in experience:
        co = exp.get("company", "")
        ti = exp.get("title", "")
        du = exp.get("duration", "")
        de = exp.get("description", "")
        exp_text_parts.append(f"{ti} at {co} ({du})\n{de}")
    resume_text = "\n\n".join(exp_text_parts) if exp_text_parts else None

    # Certification text
    cert_text = ", ".join(c.get("name", "") for c in certifications if c.get("name")) or None

    # Awards text
    awards_text = ", ".join(
        f"{a.get('name', '')}" + (f" ({a['year']})" if a.get('year') else "")
        for a in awards if a.get("name")
    ) or None

    # Education fields
    edu_first = education[0] if education else {}
    edu_second = education[1] if len(education) > 1 else {}
    edu_start_year = edu_first.get("start_year")
    edu_end_year = edu_first.get("end_year")
    # None, not False, when the resume carried no education section at all —
    # otherwise every parse writes False over whatever Salesforce held.
    has_edu_year: bool | None = (
        bool(edu_start_year or edu_end_year) if education else None
    )

    # Education summary string
    edu_str = parsed.get("highest_degree") or edu_first.get("degree")

    # Score values
    score_obj = ResumeScore(**score) if isinstance(score, dict) else ResumeScore()

    return SalesforceResumeData(
        # Contact / Identity
        FirstName=parsed.get("first_name"),
        LastName=parsed.get("last_name"),
        Name=parsed.get("name"),
        # Salesforce Email fields validate format server-side; a malformed
        # address fails the whole record.
        Email=email_or_none(parsed.get("email")),
        Title=current_exp.get("title"),
        AadharNumber=parsed.get("aadhar_number"),
        AlternateEmail=email_or_none(parsed.get("alternate_email")),
        AlternatePhoneNumber=parsed.get("number"),
        Phone=parsed.get("phone"),
        PhoneNumber=parsed.get("phone"),
        MobilePhone=parsed.get("phone"),
        LinkedIn_URL=parsed.get("linkedin_url"),
        Web_address=parsed.get("web_address"),
        # Both are Salesforce Date fields — they take yyyy-MM-dd and nothing
        # else. The LLM emits whatever the resume said ("12 May 1990").
        DateOfBirth=iso_date(parsed.get("date_of_birth")),
        Birthdate=iso_date(parsed.get("date_of_birth")),
        Gender=pick(parsed.get("gender"), "Gender"),
        Blood_Group=parsed.get("blood_group"),
        Father_s_Name=parsed.get("father_name"),
        MotherName=parsed.get("mother_name"),
        Nationnality=parsed.get("nationality"),
        PAN_Number=parsed.get("pan_number"),
        Passport_Number=parsed.get("passport_number"),
        LanguagesKnown=parsed.get("languages_known"),

        # Location
        Current_Location=parsed.get("current_location"),
        City=_extract_city(parsed.get("current_location")),
        State=_extract_state(parsed.get("current_location")),
        Preferred_Location=parsed.get("preferred_location"),

        # Professional
        CurrentCompany=parsed.get("current_company") or current_exp.get("company"),
        CurrentDesignation=parsed.get("current_designation") or current_exp.get("title"),
        CurrentDuration=_parse_duration_years(current_exp.get("duration")),
        Designation=current_exp.get("title"),
        Company=parsed.get("current_company") or current_exp.get("company"),
        Department=current_exp.get("department"),
        # The LLM emits whatever the resume said — "5 years", "5+". Both are
        # strings, and a string reaching a float field raises a pydantic
        # ValidationError that fails the entire parse with a 500.
        Years_of_Experience=to_number(parsed.get("total_years_of_experience")),
        No_of_companies_worked_in=_as_int(parsed.get("number_of_companies")),
        # Restricted picklists in most orgs — free-text LLM output would fail
        # the record with INVALID_OR_NULL_FOR_RESTRICTED_PICKLIST.
        Current_Employment=pick(parsed.get("current_employment_status"), "Current_Employment"),
        Industry=pick(parsed.get("industry"), "Industry"),

        # Skills
        Primary_Skills=primary_skill_list,
        Technical_Skills=tech_skill_text,
        General_Skills=general_skill_text,
        SkillList=skill_list,
        Skill_List=primary_skill_list,
        AutoPopulate_Skillset=skill_list,
        Key_Skillsets_del=tech_skill_text,

        # Education
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

        # Compensation / Availability
        # Currency in some orgs, Text in others.
        Current_CTC=money_or_text(parsed.get("current_ctc"), "Current_CTC"),
        Expected_CTC=money_or_text(parsed.get("expected_ctc"), "Expected_CTC"),
        Notice_Period=pick(parsed.get("notice_period"), "Notice_Period"),

        # Resume content — clipped to each field's declared length. A Text Area
        # caps at 255 and a Long Text Area at 131,072; one character over
        # either fails the record with STRING_TOO_LONG.
        ResumeRich=truncate(resume_rich, "ResumeRich"),
        Resume=truncate(resume_text, "Resume"),
        TextResume=truncate(raw_text, "TextResume"),
        Date_Parsed_Text=date.today().isoformat(),

        # Scoring
        Candidate_Score=score_obj.overall,
        Resume_Score=float(score_obj.overall),
        score_breakdown=score_obj,
    )
