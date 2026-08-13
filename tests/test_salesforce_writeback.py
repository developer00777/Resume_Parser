"""
Tests for the Describe-driven field mapping, its lifecycle, and the write-back
gating rules.

The gating rules are the point of the endpoint: a field must be *withheld*
rather than written whenever the parser cannot stand behind the value. Writing a
blank over real Salesforce data is worse than writing nothing.
"""
from unittest.mock import AsyncMock, patch

import pytest

from app.routes.salesforce import _writable_payload
from app.schemas.response import ExtractionStatus, SalesforceResumeData
from app.services import salesforce_coerce as sc
from app.services import salesforce_schema as schema


@pytest.fixture(autouse=True)
def clean_state():
    schema.reset_for_tests()
    yield
    schema.reset_for_tests()


def _field(name, sf_type="string", length=255, updateable=True, **extra):
    return {"name": name, "type": sf_type, "length": length,
            "updateable": updateable, **extra}


class TestDescribeLoading:
    def test_prefers_the_namespaced_field(self):
        describe = {"name": "SCSCHAMPS__Candidate__c", "fields": [
            _field("SCSCHAMPS__LinkedIn_URL__c"), _field("LinkedIn_URL__c")]}
        report = sc.load_specs_from_describe(describe, ["LinkedIn_URL"])
        assert report.resolved["LinkedIn_URL"] == "SCSCHAMPS__LinkedIn_URL__c"

    def test_falls_back_to_the_org_local_field(self):
        describe = {"name": "X", "fields": [_field("PAN_Number__c")]}
        report = sc.load_specs_from_describe(describe, ["PAN_Number"])
        assert report.resolved["PAN_Number"] == "PAN_Number__c"

    def test_resolves_standard_fields(self):
        describe = {"name": "Contact", "fields": [_field("Email", length=80)]}
        report = sc.load_specs_from_describe(describe, ["Email"])
        assert report.resolved["Email"] == "Email"
        assert sc.spec_for("Email").length == 80

    def test_reports_missing_fields(self):
        describe = {"name": "X", "fields": [_field("Email")]}
        report = sc.load_specs_from_describe(describe, ["Email", "Nonexistent_Field"])
        assert report.unresolved == ["Nonexistent_Field"]
        assert sc.spec_for("Nonexistent_Field") is None

    def test_excludes_read_only_fields(self):
        describe = {"name": "X", "fields": [
            _field("Computed_Score__c", sf_type="double", updateable=False)]}
        report = sc.load_specs_from_describe(describe, ["Computed_Score"])
        assert report.not_updateable == ["Computed_Score -> Computed_Score__c"]
        assert sc.spec_for("Computed_Score") is None

    def test_captures_restricted_picklist_values(self):
        describe = {"name": "X", "fields": [
            _field("Gender__c", sf_type="picklist", restrictedPicklist=True,
                   picklistValues=[{"value": "Male", "active": True},
                                   {"value": "Female", "active": True},
                                   {"value": "Legacy", "active": False}])]}
        sc.load_specs_from_describe(describe, ["Gender"])
        assert sc.spec_for("Gender").values == ("Male", "Female")
        assert sc.pick("male", "Gender") == "Male"
        assert sc.pick("Unspecified", "Gender") is None

    def test_unrestricted_picklist_keeps_empty_value_set(self):
        describe = {"name": "X", "fields": [
            _field("Industry__c", sf_type="picklist", restrictedPicklist=False,
                   picklistValues=[{"value": "IT", "active": True}])]}
        sc.load_specs_from_describe(describe, ["Industry"])
        assert sc.spec_for("Industry").values == ()
        assert sc.pick("Agritech", "Industry") == "Agritech"

    def test_currency_becomes_numeric(self):
        describe = {"name": "X", "fields": [
            _field("Current_CTC__c", sf_type="currency", length=None)]}
        sc.load_specs_from_describe(describe, ["Current_CTC"])
        assert sc.spec_for("Current_CTC").kind == "number"
        assert sc.money_or_text("12 LPA", "Current_CTC") == 1_200_000.0

    def test_reload_leaves_no_stale_entries(self):
        sc.load_specs_from_describe(
            {"name": "X", "fields": [_field("Email"), _field("Gender__c")]},
            ["Email", "Gender"])
        assert sc.spec_for("Gender") is not None
        sc.load_specs_from_describe({"name": "X", "fields": [_field("Email")]},
                                    ["Email", "Gender"])
        assert sc.spec_for("Gender") is None
        assert sc.spec_for("Email") is not None


class TestSchemaLifecycle:
    """The gap that made write-back fail silently after every deploy."""

    DESCRIBE = {"name": "SCSCHAMPS__Candidate__c",
                "fields": [_field("SCSCHAMPS__Phone__c")]}

    @pytest.mark.asyncio
    async def test_startup_load_populates_the_mapping(self, monkeypatch):
        monkeypatch.setattr(schema.settings, "sf_client_id", "x")
        monkeypatch.setattr(schema.settings, "sf_client_secret", "y")
        with patch("app.services.salesforce.fetch_candidate_describe",
                   new_callable=AsyncMock) as fetch:
            fetch.return_value = self.DESCRIBE
            await schema.load_at_startup()

        assert schema.state().loaded is True
        assert schema.state().field_count >= 1

    @pytest.mark.asyncio
    async def test_startup_failure_is_not_fatal(self, monkeypatch):
        # Salesforce being down at boot must not poison the process for its
        # whole lifetime, nor stop the upload endpoints working.
        monkeypatch.setattr(schema.settings, "sf_client_id", "x")
        monkeypatch.setattr(schema.settings, "sf_client_secret", "y")
        with patch("app.services.salesforce.fetch_candidate_describe",
                   new_callable=AsyncMock) as fetch:
            fetch.side_effect = RuntimeError("connection refused")
            await schema.load_at_startup()

        assert schema.state().loaded is False
        assert "connection refused" in schema.state().last_error

    @pytest.mark.asyncio
    async def test_first_use_retries_after_a_failed_startup(self, monkeypatch):
        monkeypatch.setattr(schema.settings, "sf_client_id", "x")
        monkeypatch.setattr(schema.settings, "sf_client_secret", "y")
        with patch("app.services.salesforce.fetch_candidate_describe",
                   new_callable=AsyncMock) as fetch:
            fetch.side_effect = RuntimeError("down")
            await schema.load_at_startup()
            assert schema.state().loaded is False

            fetch.side_effect = None
            fetch.return_value = self.DESCRIBE
            state = await schema.ensure_loaded()

        assert state.loaded is True, "must recover without a restart"

    @pytest.mark.asyncio
    async def test_a_loaded_mapping_is_not_refetched(self, monkeypatch):
        monkeypatch.setattr(schema.settings, "sf_client_id", "x")
        monkeypatch.setattr(schema.settings, "sf_client_secret", "y")
        monkeypatch.setattr(schema.settings, "sf_describe_ttl_minutes", 60)
        with patch("app.services.salesforce.fetch_candidate_describe",
                   new_callable=AsyncMock) as fetch:
            fetch.return_value = self.DESCRIBE
            await schema.ensure_loaded()
            await schema.ensure_loaded()
            await schema.ensure_loaded()

        assert fetch.await_count == 1, "a fresh mapping must not re-describe"

    @pytest.mark.asyncio
    async def test_force_reloads_even_when_fresh(self, monkeypatch):
        monkeypatch.setattr(schema.settings, "sf_client_id", "x")
        monkeypatch.setattr(schema.settings, "sf_client_secret", "y")
        with patch("app.services.salesforce.fetch_candidate_describe",
                   new_callable=AsyncMock) as fetch:
            fetch.return_value = self.DESCRIBE
            await schema.ensure_loaded()
            await schema.ensure_loaded(force=True)

        assert fetch.await_count == 2

    @pytest.mark.asyncio
    async def test_a_failed_refresh_keeps_the_old_mapping(self, monkeypatch):
        # A stale mapping that worked beats no mapping at all.
        monkeypatch.setattr(schema.settings, "sf_client_id", "x")
        monkeypatch.setattr(schema.settings, "sf_client_secret", "y")
        with patch("app.services.salesforce.fetch_candidate_describe",
                   new_callable=AsyncMock) as fetch:
            fetch.return_value = self.DESCRIBE
            await schema.ensure_loaded()
            assert sc.spec_for("Phone") is not None

            fetch.side_effect = RuntimeError("transient")
            await schema.ensure_loaded(force=True)

        assert sc.spec_for("Phone") is not None, "must not discard a working mapping"

    @pytest.mark.asyncio
    async def test_no_credentials_means_no_describe_attempt(self, monkeypatch):
        monkeypatch.setattr(schema.settings, "sf_client_id", "")
        monkeypatch.setattr(schema.settings, "sf_client_secret", "")
        with patch("app.services.salesforce.fetch_candidate_describe",
                   new_callable=AsyncMock) as fetch:
            await schema.load_at_startup()
        assert fetch.await_count == 0
        assert schema.state().loaded is False


class TestWritablePayload:
    def _load(self, *names):
        sc.load_specs_from_describe(
            {"name": "SCSCHAMPS__Candidate__c",
             "fields": [_field(f"SCSCHAMPS__{n}__c") for n in names]}, list(names))

    def test_maps_onto_org_api_names(self):
        self._load("Phone", "Email")
        written, _ = _writable_payload(
            SalesforceResumeData(Phone="+911234567890", Email="a@b.com"), set())
        assert written == {"SCSCHAMPS__Phone__c": "+911234567890",
                           "SCSCHAMPS__Email__c": "a@b.com"}

    def test_none_values_never_written(self):
        self._load("Phone")
        written, skipped = _writable_payload(SalesforceResumeData(Phone=None), set())
        assert written == {}
        assert skipped["Phone"] == "no value extracted"

    def test_org_owned_fields_never_written(self):
        self._load("Candidate_Status", "Talent_Id", "Phone")
        written, skipped = _writable_payload(SalesforceResumeData(
            Candidate_Status="Hired", Talent_Id="T-1", Phone="+91123"), set())
        assert written == {"SCSCHAMPS__Phone__c": "+91123"}
        assert skipped["Candidate_Status"] == "Salesforce owns this field"

    def test_failed_section_fields_are_withheld(self):
        # Chunk C died, so experience is empty because the call failed — not
        # because the candidate has no job history. Writing these erases a real
        # employment record.
        self._load("CurrentCompany", "Phone")
        written, skipped = _writable_payload(
            SalesforceResumeData(CurrentCompany="Acme", Phone="+91123"), {"experience"})
        assert written == {"SCSCHAMPS__Phone__c": "+91123"}
        assert "refusing to overwrite" in skipped["CurrentCompany"]

    def test_written_when_the_section_succeeded(self):
        self._load("CurrentCompany")
        written, _ = _writable_payload(SalesforceResumeData(CurrentCompany="Acme"), set())
        assert written == {"SCSCHAMPS__CurrentCompany__c": "Acme"}

    def test_unmapped_fields_withheld_not_guessed(self):
        self._load("Phone")
        written, skipped = _writable_payload(
            SalesforceResumeData(Phone="+91123", Email="a@b.com"), set())
        assert written == {"SCSCHAMPS__Phone__c": "+91123"}
        assert skipped["Email"] == "no updateable field of this name in the org"

    def test_nested_breakdown_is_not_a_field(self):
        self._load("Phone")
        written, skipped = _writable_payload(SalesforceResumeData(Phone="+91123"), set())
        assert "score_breakdown" not in written and "score_breakdown" not in skipped

    def test_nothing_writable_before_a_describe(self):
        written, skipped = _writable_payload(
            SalesforceResumeData(Phone="+91123", Email="a@b.com"), set())
        assert written == {}
        assert skipped["Phone"] == "no updateable field of this name in the org"


class TestExtractionStatus:
    def test_defaults(self):
        status = ExtractionStatus()
        assert status.complete is True and status.failed_sections == []

    def test_carries_failed_sections(self):
        status = ExtractionStatus(complete=False, chunks_total=3, chunks_ok=2,
                                  failed_chunks=["chunk_c"],
                                  failed_sections=["education", "experience"])
        assert status.complete is False and "experience" in status.failed_sections
