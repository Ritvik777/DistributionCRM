import unittest

from agents.intent import (
    extract_part_reference,
    infer_leads_limit,
    is_crm_request,
    is_leads_by_part_enquiry,
    is_leads_time_window_fetch,
    is_outreach_request,
    is_recent_leads_list_fetch,
    is_simple_leads_fetch,
    is_singular_lead_fetch,
    parse_leads_time_window_minutes,
    wants_salesforce_leads_for_outreach,
)
from services.salesforce_client import _part_like_patterns


class IntentTests(unittest.TestCase):
    def test_part_enquiry_not_simple_leads_fetch(self):
        query = "show all leads who enquired about Part No. LED-RED-5MM"
        self.assertTrue(is_leads_by_part_enquiry(query))
        self.assertFalse(is_simple_leads_fetch(query))
        self.assertEqual(extract_part_reference(query), "LED-RED-5MM")

    def test_simple_latest_leads(self):
        query = "show last 10 leads"
        self.assertTrue(is_simple_leads_fetch(query))
        self.assertFalse(is_leads_by_part_enquiry(query))
        self.assertEqual(infer_leads_limit(query), 10)

    def test_singular_lead(self):
        self.assertEqual(infer_leads_limit("show the latest lead"), 1)
        self.assertTrue(is_singular_lead_fetch("show last lead in crm"))
        self.assertTrue(is_singular_lead_fetch("show last crm lead"))
        self.assertTrue(is_singular_lead_fetch("show latest crm lead"))
        self.assertFalse(is_singular_lead_fetch("show last 10 leads"))

    def test_recent_leads_list(self):
        query = "show all crm leads from last 10 minutes"
        self.assertTrue(is_leads_time_window_fetch(query))
        self.assertTrue(is_recent_leads_list_fetch("show latest crm leads"))
        self.assertFalse(is_recent_leads_list_fetch(query))
        self.assertEqual(parse_leads_time_window_minutes(query), 10)
        self.assertEqual(infer_leads_limit("show all crm leads"), 50)

    def test_outreach_over_crm_for_email(self):
        query = "email rgaur@company.com about LED-RED-5MM availability"
        self.assertTrue(is_outreach_request(query))
        self.assertFalse(is_crm_request(query))

    def test_crm_part_filter_query(self):
        query = "show all leads who enquired about Part No. LED-RED-5MM"
        self.assertTrue(is_crm_request(query))

    def test_outreach_salesforce_leads(self):
        query = "email the leads from salesforce about our new catalog"
        self.assertTrue(wants_salesforce_leads_for_outreach(query))


class SalesforcePatternTests(unittest.TestCase):
    def test_part_like_patterns_flexible(self):
        patterns = _part_like_patterns("LED-RED-5MM")
        self.assertTrue(any("LED" in pattern and "RED" in pattern for pattern in patterns))


if __name__ == "__main__":
    unittest.main()
