"""
Tests for Salesforce value coercion and the Apex-safety changes to the payload.

The point of every case: Salesforce rejects an entire record over one bad value,
so anything that cannot be coerced must become None rather than reach the org.
"""
import pytest

from app.services import salesforce_coerce as sc
from app.services.salesforce_coerce import (
    FieldSpec, iso_date, money_or_text, pick, to_number, truncate,
)


@pytest.fixture
def specs():
    original = dict(sc.FIELD_SPECS)
    sc.FIELD_SPECS.clear()
    yield sc.FIELD_SPECS
    sc.FIELD_SPECS.clear()
    sc.FIELD_SPECS.update(original)


class TestIsoDate:
    @pytest.mark.parametrize("raw,expected", [
        ("1990-05-12", "1990-05-12"), ("1990-5-12", "1990-05-12"),
        ("12/05/1990", "1990-05-12"), ("12-05-1990", "1990-05-12"),
        ("12.05.1990", "1990-05-12"),
        ("05/22/1990", "1990-05-22"),   # month > 12 -> unambiguously month-first
        ("12 May 1990", "1990-05-12"), ("12th May 1990", "1990-05-12"),
        ("May 12, 1990", "1990-05-12"), ("  1990-05-12  ", "1990-05-12"),
    ])
    def test_parses_the_forms_resumes_use(self, raw, expected):
        assert iso_date(raw) == expected

    @pytest.mark.parametrize("raw", [
        None, "", "   ", "not a date", "Present",
        "May 1990",    # no day — filling in the 1st would be a fabrication
        "1990-02-31",  # not a real calendar date
        "1850-01-01",  # implausible: extraction artefact
        "2099-01-01",  # implausible: far future
    ])
    def test_refuses_to_guess(self, raw):
        assert iso_date(raw) is None

    def test_two_digit_year_resolves_to_the_past(self):
        assert iso_date("12/05/90") == "1990-05-12"


class TestToNumber:
    @pytest.mark.parametrize("raw,expected", [
        ("12 LPA", 1_200_000.0), ("12LPA", 1_200_000.0), ("15 lakhs", 1_500_000.0),
        ("₹15,00,000", 1_500_000.0), ("2.5 Cr", 25_000_000.0),
        ("50K", 50_000.0), ("$90,000", 90_000.0), (12.5, 12.5), (12, 12.0),
    ])
    def test_extracts_amounts(self, raw, expected):
        assert to_number(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "Negotiable", "As per company norms"])
    def test_no_number_means_none(self, raw):
        assert to_number(raw) is None


class TestPick:
    def test_passes_through_when_unverified(self, specs):
        assert pick("Anything At All", "Gender") == "Anything At All"

    def test_snaps_onto_the_value_set(self, specs):
        specs["Gender"] = FieldSpec("Gender__c", kind="picklist", values=("Male", "Female"))
        assert pick("male", "Gender") == "Male"
        assert pick("  FEMALE ", "Gender") == "Female"

    def test_ignores_spacing_differences(self, specs):
        specs["Notice_Period"] = FieldSpec("Notice_Period__c", kind="picklist",
                                           values=("30 Days", "60 Days"))
        assert pick("30days", "Notice_Period") == "30 Days"

    def test_unmatched_value_becomes_none(self, specs):
        specs["Gender"] = FieldSpec("Gender__c", kind="picklist", values=("Male", "Female"))
        assert pick("Prefer not to say", "Gender") is None

    def test_unrestricted_picklist_passes_through(self, specs):
        specs["Industry"] = FieldSpec("Industry__c", kind="picklist", values=())
        assert pick("Agritech", "Industry") == "Agritech"


class TestTruncate:
    def test_passes_through_when_length_unknown(self, specs):
        assert truncate("x" * 5000, "TextResume") == "x" * 5000

    def test_clips_to_declared_length(self, specs):
        specs["Resume"] = FieldSpec("SCSCHAMPS__Resume__c", kind="text", length=255)
        assert len(truncate("x" * 1000, "Resume")) == 255

    def test_leaves_short_values_alone(self, specs):
        specs["Resume"] = FieldSpec("SCSCHAMPS__Resume__c", kind="text", length=255)
        assert truncate("short", "Resume") == "short"

    def test_empty_becomes_none(self, specs):
        assert truncate("", "Resume") is None
        assert truncate(None, "Resume") is None


class TestMoneyOrText:
    def test_string_preserved_when_field_is_text(self, specs):
        assert money_or_text("12 LPA", "Current_CTC") == "12 LPA"

    def test_parsed_when_field_is_currency(self, specs):
        specs["Current_CTC"] = FieldSpec("Current_CTC__c", kind="number")
        assert money_or_text("12 LPA", "Current_CTC") == 1_200_000.0

    def test_none_stays_none(self, specs):
        assert money_or_text(None, "Current_CTC") is None


class TestCoerceReport:
    def test_unsafe_without_a_describe(self, specs):
        report = sc.coerce_report(["Gender", "Current_CTC"])
        assert report.describe_loaded is False and report.safe_to_write is False
        assert report.unverified_fields == ["Current_CTC", "Gender"]

    def test_safe_once_loaded(self, specs):
        specs["Gender"] = FieldSpec("Gender__c", kind="picklist", values=("Male",))
        report = sc.coerce_report(["Gender"])
        assert report.describe_loaded is True and report.safe_to_write is True


class TestApexSafePayload:
    def _parsed(self, **overrides):
        base = {
            "name": "Asha Rao", "email": "asha@example.com", "phone": "+911234567890",
            "skills": [], "experience": [], "education": [], "projects": [],
            "certifications": [], "awards": [], "summary": None,
            "resume_score": {"overall": 72, "grade": "Good"},
        }
        base.update(overrides)
        return base

    def test_score_breakdown_replaces_the_colliding_name(self):
        from app.schemas.response import map_to_salesforce
        sf = map_to_salesforce(self._parsed())
        assert sf.score_breakdown.overall == 72
        assert not hasattr(sf, "resume_score")
        emitted = sf.model_dump()
        assert "score_breakdown" in emitted and "resume_score" not in emitted
        assert emitted["Resume_Score"] == 72.0

    def test_free_text_dob_is_coerced(self):
        from app.schemas.response import map_to_salesforce
        sf = map_to_salesforce(self._parsed(date_of_birth="12th May 1990"))
        assert sf.DateOfBirth == "1990-05-12" and sf.Birthdate == "1990-05-12"

    def test_unparseable_dob_becomes_none(self):
        from app.schemas.response import map_to_salesforce
        sf = map_to_salesforce(self._parsed(date_of_birth="sometime in the nineties"))
        assert sf.DateOfBirth is None

    def test_booleans_are_null_not_false(self):
        from app.schemas.response import map_to_salesforce
        sf = map_to_salesforce(self._parsed())
        assert sf.Education_year is None
        assert sf.converted_from_lead is None
        assert sf.Ampliz_Contact is None

    def test_education_year_set_when_education_exists(self):
        from app.schemas.response import map_to_salesforce
        sf = map_to_salesforce(self._parsed(
            education=[{"institution": "IIT", "degree": "B.Tech", "end_year": 2012}]))
        assert sf.Education_year is True
