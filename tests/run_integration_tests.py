import httpx
import os
import json
from pathlib import Path
from loguru import logger

SERVER_URL = "http://127.0.0.1:8000"

def run_test_case(name: str, request_text: str):
    logger.info(f"=== Running Test Case: {name} ===")
    logger.info(f"Request: '{request_text}'")

    payload = {"request": request_text}
    
    try:
        # Increase timeout since agent does multi-step research and self-check
        response = httpx.post(f"{SERVER_URL}/agent", json=payload, timeout=90.0)
        
        if response.status_code != 200:
            logger.error(f"Test case '{name}' failed with status code {response.status_code}")
            logger.error(response.text)
            return False

        data = response.json()
        logger.info(f"Agent Status: {data.get('status')}")
        logger.info(f"Summary: {data.get('summary')}")
        
        ref = data.get("reflection_result")
        if ref:
            logger.info(f"Audit Score: {ref.get('quality_score')}/100")
            logger.info(f"Audit Approved: {ref.get('is_approved')}")
            logger.info(f"Corrections Required: {len(ref.get('required_corrections', []))}")
        
        doc_meta = data.get("document_metadata")
        if doc_meta:
            logger.info(f"Document Generated: {doc_meta.get('file_name')}")
            logger.info(f"File Path: {doc_meta.get('file_path')}")
            logger.info(f"Word Count: {doc_meta.get('word_count')}")
            
            # Verify download endpoint
            dl_path = data.get("download_path")
            if dl_path:
                logger.info(f"Testing download endpoint: {SERVER_URL}{dl_path}")
                dl_response = httpx.get(f"{SERVER_URL}{dl_path}")
                if dl_response.status_code == 200:
                    logger.info("Download endpoint verified! File bytes received.")
                else:
                    logger.error(f"Download failed with status code {dl_response.status_code}")
                    return False
        else:
            logger.error("No document metadata returned in response!")
            return False

        logger.info(f"=== Completed Test Case: {name} successfully ===\n")
        return True

    except Exception as e:
        logger.exception(f"Exception encountered during test '{name}': {e}")
        return False

def main():
    # 1. Healthcheck check
    logger.info("Checking API Health...")
    r = httpx.get(SERVER_URL)
    logger.info(f"Health Response: {r.json()}")

    # 2. Simple request test case
    simple_req = "Create a meeting summary for a discussion about a new employee onboarding process."
    simple_ok = run_test_case("Simple Business Request", simple_req)

    # 3. Complex request test case
    complex_req = (
        "Produce a comprehensive project launch document for Project Apex. "
        "It needs to contain cybersecurity compliance details, a detailed timeline, "
        "but we are not sure who the target audience is (maybe developers or executive board? satisfy both), "
        "and some standard procedures. Make appropriate assumptions and research cybersecurity standard frameworks."
    )
    complex_ok = run_test_case("Complex Ambiguous Request", complex_req)

    if simple_ok and complex_ok:
        logger.info("ALL INTEGRATION TESTS PASSED SUCCESSFULLY!")
    else:
        logger.error("SOME INTEGRATION TESTS FAILED.")

if __name__ == "__main__":
    main()
