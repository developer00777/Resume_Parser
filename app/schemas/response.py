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
    Country__c: str | None = None
    CurrentDesignation__c: str | None = None
    Email: str | None = None
    PhoneNumber__c: str | None = None                # primary mobile
    SCSCHAMPS__PhoneNumber__c: str | None = None     # secondary/alternate phone
    CurrentCompany__c: str | None = None
    Consultant__c: None = None                       # always null — assigned in SF
    Type_1__c: str | None = None                     # Graduate | Non Graduate
    Spoken_Language__c: List[str] = Field(default_factory=list)  # multi-select array
    MaritalStatus__c: str | None = None               # picklist
    Gender__c: str | None = None                      # picklist

    # Education Background
    Graduation_Year2__c: str | None = None           # Text(15) — graduation year
    Institution_College__c: str | None = None        # Text(225) — institution/college name

    # Professional Degree
    Name__c: str | None = None                       # Text(255) — professional/highest degree name
    Year__c: str | None = None                       # Date (YYYY-MM-DD) — degree completion year


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
    country: str | None = Field(None, description="Country name derived from the phone number's dialing code")
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
    Country: str | None = None

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


# Phone country-calling-code -> country name. Ordered longest-prefix-first within
# each length group so lookup can match greedily (e.g. "1242" before "1").
_PHONE_COUNTRY_CODES: dict[str, str] = {
    "1242": "Bahamas", "1246": "Barbados", "1264": "Anguilla", "1268": "Antigua and Barbuda",
    "1284": "British Virgin Islands", "1340": "U.S. Virgin Islands", "1345": "Cayman Islands",
    "1441": "Bermuda", "1473": "Grenada", "1649": "Turks and Caicos Islands", "1658": "Jamaica",
    "1664": "Montserrat", "1670": "Northern Mariana Islands", "1671": "Guam", "1684": "American Samoa",
    "1721": "Sint Maarten", "1758": "Saint Lucia", "1767": "Dominica", "1784": "Saint Vincent and the Grenadines",
    "1787": "Puerto Rico", "1809": "Dominican Republic", "1829": "Dominican Republic", "1849": "Dominican Republic",
    "1868": "Trinidad and Tobago", "1869": "Saint Kitts and Nevis", "1876": "Jamaica", "1939": "Puerto Rico",
    "1": "United States",
    "212": "Morocco", "213": "Algeria", "216": "Tunisia", "218": "Libya",
    "220": "Gambia", "221": "Senegal", "222": "Mauritania", "223": "Mali", "224": "Guinea",
    "225": "Ivory Coast", "226": "Burkina Faso", "227": "Niger", "228": "Togo", "229": "Benin",
    "230": "Mauritius", "231": "Liberia", "232": "Sierra Leone", "233": "Ghana", "234": "Nigeria",
    "235": "Chad", "236": "Central African Republic", "237": "Cameroon", "238": "Cape Verde",
    "239": "Sao Tome and Principe", "240": "Equatorial Guinea", "241": "Gabon", "242": "Republic of the Congo",
    "243": "Democratic Republic of the Congo", "244": "Angola", "245": "Guinea-Bissau",
    "246": "British Indian Ocean Territory", "248": "Seychelles", "249": "Sudan", "250": "Rwanda",
    "251": "Ethiopia", "252": "Somalia", "253": "Djibouti", "254": "Kenya", "255": "Tanzania",
    "256": "Uganda", "257": "Burundi", "258": "Mozambique", "260": "Zambia", "261": "Madagascar",
    "262": "Reunion", "263": "Zimbabwe", "264": "Namibia", "265": "Malawi", "266": "Lesotho",
    "267": "Botswana", "268": "Eswatini", "269": "Comoros", "290": "Saint Helena", "291": "Eritrea",
    "297": "Aruba", "298": "Faroe Islands", "299": "Greenland",
    "27": "South Africa", "20": "Egypt",
    "30": "Greece", "31": "Netherlands", "32": "Belgium", "33": "France", "34": "Spain",
    "36": "Hungary", "39": "Italy",
    "40": "Romania", "41": "Switzerland", "43": "Austria", "44": "United Kingdom", "45": "Denmark",
    "46": "Sweden", "47": "Norway", "48": "Poland", "49": "Germany",
    "351": "Portugal", "352": "Luxembourg", "353": "Ireland", "354": "Iceland", "355": "Albania",
    "356": "Malta", "357": "Cyprus", "358": "Finland", "359": "Bulgaria",
    "370": "Lithuania", "371": "Latvia", "372": "Estonia", "373": "Moldova", "374": "Armenia",
    "375": "Belarus", "376": "Andorra", "377": "Monaco", "378": "San Marino", "379": "Vatican City",
    "380": "Ukraine", "381": "Serbia", "382": "Montenegro", "383": "Kosovo", "385": "Croatia",
    "386": "Slovenia", "387": "Bosnia and Herzegovina", "389": "North Macedonia",
    "420": "Czech Republic", "421": "Slovakia", "423": "Liechtenstein",
    "60": "Malaysia", "61": "Australia", "62": "Indonesia", "63": "Philippines", "64": "New Zealand",
    "65": "Singapore", "66": "Thailand",
    "7": "Russia",
    "81": "Japan", "82": "South Korea", "84": "Vietnam", "86": "China",
    "90": "Turkey", "91": "India", "92": "Pakistan", "93": "Afghanistan", "94": "Sri Lanka",
    "95": "Myanmar", "98": "Iran",
    "670": "East Timor", "672": "Norfolk Island", "673": "Brunei", "674": "Nauru", "675": "Papua New Guinea",
    "676": "Tonga", "677": "Solomon Islands", "678": "Vanuatu", "679": "Fiji", "680": "Palau",
    "681": "Wallis and Futuna", "682": "Cook Islands", "683": "Niue", "685": "Samoa", "686": "Kiribati",
    "687": "New Caledonia", "688": "Tuvalu", "689": "French Polynesia", "690": "Tokelau",
    "691": "Micronesia", "692": "Marshall Islands",
    "850": "North Korea", "852": "Hong Kong", "853": "Macau", "855": "Cambodia", "856": "Laos",
    "880": "Bangladesh", "886": "Taiwan",
    "960": "Maldives", "961": "Lebanon", "962": "Jordan", "963": "Syria", "964": "Iraq",
    "965": "Kuwait", "966": "Saudi Arabia", "967": "Yemen", "968": "Oman", "970": "Palestine",
    "971": "United Arab Emirates", "972": "Israel", "973": "Bahrain", "974": "Qatar",
    "975": "Bhutan", "976": "Mongolia", "977": "Nepal", "992": "Tajikistan", "993": "Turkmenistan",
    "994": "Azerbaijan", "995": "Georgia", "996": "Kyrgyzstan", "998": "Uzbekistan",
    "54": "Argentina", "55": "Brazil", "56": "Chile", "57": "Colombia", "58": "Venezuela",
    "51": "Peru", "52": "Mexico", "53": "Cuba",
    "500": "Falkland Islands", "501": "Belize", "502": "Guatemala", "503": "El Salvador",
    "504": "Honduras", "505": "Nicaragua", "506": "Costa Rica", "507": "Panama", "508": "Saint Pierre and Miquelon",
    "509": "Haiti", "590": "Guadeloupe", "591": "Bolivia", "592": "Guyana", "593": "Ecuador",
    "594": "French Guiana", "595": "Paraguay", "596": "Martinique", "597": "Suriname",
    "598": "Uruguay", "599": "Curacao",
}


def _country_from_phone(phone: str | None) -> str | None:
    """Derive a country name from a phone number's leading dialing code (e.g. '+91...' -> 'India')."""
    if not phone:
        return None
    import re
    match = re.search(r'\+(\d{1,4})', phone)
    if not match:
        return None
    digits = match.group(1)
    # Greedy longest-prefix match: try 4, 3, 2, then 1 digit(s).
    for length in (4, 3, 2, 1):
        code = digits[:length]
        if code in _PHONE_COUNTRY_CODES:
            return _PHONE_COUNTRY_CODES[code]
    return None


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


def _education_graduation_year(parsed: dict) -> str | None:
    """Return the graduation/end year from the first education entry as a string."""
    education: list = parsed.get("education", [])
    if not education:
        return None
    edu_first = education[0] if isinstance(education, list) else {}
    year = edu_first.get("end_year")
    return str(year) if year else None


def _education_institution(parsed: dict) -> str | None:
    """Return the institution name from the first education entry."""
    education: list = parsed.get("education", [])
    if not education:
        return None
    edu_first = education[0] if isinstance(education, list) else {}
    return edu_first.get("institution")


def _professional_degree_year(parsed: dict) -> str | None:
    """Return the degree completion year as YYYY-MM-DD (Date type) from the first education entry."""
    education: list = parsed.get("education", [])
    if not education:
        return None
    edu_first = education[0] if isinstance(education, list) else {}
    year = edu_first.get("end_year")
    if year:
        return f"{year}-01-01"
    return None


def map_to_client(parsed: dict) -> ClientResumeData:
    """Map the internal parsed dict to the client-1 field schema."""
    return ClientResumeData(
        Name=parsed.get("name"),
        FullName__c=parsed.get("name"),
        Nationality__c=parsed.get("nationality"),
        Date_of_Birth__c=parsed.get("date_of_birth"),
        Years_of_Experience__c=parsed.get("total_years_of_experience"),
        Current_Location__c=parsed.get("current_location"),
        Country__c=_country_from_phone(parsed.get("phone") or parsed.get("number")),
        CurrentDesignation__c=parsed.get("current_designation"),
        Email=parsed.get("email"),
        PhoneNumber__c=parsed.get("phone"),
        SCSCHAMPS__PhoneNumber__c=parsed.get("number"),
        CurrentCompany__c=parsed.get("current_company"),
        Consultant__c=None,
        Type_1__c=_infer_candidate_type(parsed),
        Spoken_Language__c=_languages_to_list(parsed.get("languages_known")),
        MaritalStatus__c=parsed.get("marital_status"),
        Gender__c=parsed.get("gender"),
        # Education Background
        Graduation_Year2__c=_education_graduation_year(parsed),
        Institution_College__c=_education_institution(parsed),
        # Professional Degree
        Name__c=parsed.get("highest_degree"),
        Year__c=_professional_degree_year(parsed),
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
        Country=_country_from_phone(parsed.get("phone") or parsed.get("number")),
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
