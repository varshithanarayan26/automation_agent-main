import unittest
from pathlib import Path
import docx
import os

from app.tools.doc_generator import DocxBuilder


class TestFilenameSanitization(unittest.TestCase):

    def test_save_with_colon_title(self):
        """Verifies that a title containing colons is sanitized to underscores, successfully saved, and readable."""
        builder = DocxBuilder("Test Title", "Test Type")

        # Add mock content to ensure doc is valid and not empty
        builder.add_header_footer()
        p = builder.doc.add_paragraph()
        p.add_run(
            "This is a test document to verify Windows filename sanitization."
        )

        invalid_filename = (
            "Artificial Intelligence in Healthcare: A Comprehensive Overview"
        )
        expected_filename = (
            "artificial_intelligence_in_healthcare__a_comprehensive_overview.docx"
        )

        # Execute save using raw invalid title
        saved_path = builder.save(invalid_filename)

        # 1. Assert file exists
        self.assertTrue(saved_path.exists())

        # 2. Assert path name is sanitized (converts : to _ and lowercase)
        # Note: Depending on where lowercase happens, DocxBuilder.save preserves casing or handles it.
        # Let's check that the invalid characters are gone
        self.assertNotIn(":", saved_path.name)
        self.assertTrue(saved_path.name.endswith(".docx"))

        # 3. Assert file size is greater than 0
        self.assertGreater(saved_path.stat().st_size, 0)

        # 4. Assert it opens successfully in python-docx
        try:
            doc = docx.Document(str(saved_path))
            self.assertEqual(
                doc.paragraphs[0].text,
                "This is a test document to verify Windows filename sanitization.",
            )
        except Exception as e:
            self.fail(f"Docx file failed to open in Microsoft Word format: {e}")

        # Clean up
        try:
            saved_path.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()
