import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from pathlib import Path
import docx

# Import the FastAPI app and GeminiClient
from app.main import app


class TestQuotaResilientFallback(unittest.TestCase):

    @patch("app.llm.GeminiClient.generate_text")
    @patch("app.llm.GeminiClient.generate_structured")
    def test_fallback_under_quota_exhaustion(
        self, mock_gen_struct, mock_gen_text
    ):
        """Verifies that the agent generates a high-quality report and hides all API errors under 429."""
        # 1. Force GeminiClient methods to raise 429 Quota Exceeded exceptions
        mock_gen_text.side_effect = Exception(
            "429 RESOURCE_EXHAUSTED: Quota exceeded for metric free tier"
        )
        mock_gen_struct.side_effect = Exception(
            "429 RESOURCE_EXHAUSTED: Quota exceeded for metric free tier"
        )

        # 2. Query the endpoint using TestClient
        client = TestClient(app)
        payload = {"request": "Write a business plan for a tech startup"}

        response = client.post("/agent", json=payload)

        # 3. Verify response is 200 OK and status is success
        self.assertEqual(response.status_code, 200)
        resp_json = response.json()
        self.assertEqual(resp_json["status"], "success")

        # 4. Verify document was created and exists
        doc_metadata = resp_json.get("document_metadata")
        self.assertIsNotNone(
            doc_metadata, "Document metadata should not be None"
        )

        file_path = doc_metadata.get("file_path")
        self.assertTrue(
            Path(file_path).exists(),
            f"Document file at {file_path} should exist",
        )

        # 5. Open and inspect document content
        doc = docx.Document(file_path)
        full_text = []
        for p in doc.paragraphs:
            full_text.append(p.text)
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    full_text.append(cell.text)

        combined_text = "\n".join(full_text).lower()

        # Verify that there are no raw Gemini exception words or error strings
        forbidden_words = [
            "429",
            "exception",
            "error",
            "quota",
            "resource_exhausted",
        ]
        for word in forbidden_words:
            self.assertNotIn(
                word,
                combined_text,
                f"Generated document contains forbidden API error keyword: '{word}'",
            )

        # Verify that the report has actual professional structure and text
        self.assertIn("business plan", combined_text)
        self.assertGreater(
            len(combined_text), 200, "Document should contain substantial text"
        )


if __name__ == "__main__":
    unittest.main()
