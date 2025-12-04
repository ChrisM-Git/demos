#!/usr/bin/env python3
"""
Simple Palo Alto Networks AI Security Demo
Test prompt inspection and model response scanning
"""

import aisecurity
from aisecurity.scan.inline.scanner import Scanner
from aisecurity.generated_openapi_client.models.ai_profile import AiProfile
from aisecurity.scan.models.content import Content

# =============================================================================
# CONFIGURATION - CUSTOMIZE THESE VALUES
# =============================================================================

# Your Palo Alto Networks API credentials
API_KEY = ""  # Add your Palo Alto API key here
SECURITY_PROFILE = ""  # Add your security profile name here (e.g., "MyCompany")

# Model configuration
MODEL_NAME = ""  # Put your model here (e.g., "gpt-4", "claude-3", "llama3.2")


# =============================================================================
# SETUP
# =============================================================================

def initialize_security ():
    """Initialize Palo Alto AI Security SDK"""
    if not API_KEY:
        raise ValueError ("Please set your API_KEY in the configuration section")

    aisecurity.init (api_key=API_KEY)

    profile = AiProfile (profile_name=SECURITY_PROFILE)
    scanner = Scanner ()

    print (f"✅ Security initialized with profile: {SECURITY_PROFILE}")
    return scanner, profile


def scan_prompt (scanner, profile, user_prompt):
    """Scan user prompt for security threats"""
    print (f"\n🔍 Scanning prompt: {user_prompt[:50]}...")

    result = scanner.sync_scan (
        ai_profile=profile,
        content=Content (prompt=user_prompt),
        metadata={"scan_type": "prompt_inspection"}
    )

    print (f"✅ Prompt scan complete - Verdict: {result.verdict}")
    return result


def scan_response (scanner, profile, user_prompt, model_response):
    """Scan model response for security issues"""
    print (f"\n🔍 Scanning model response: {model_response[:50]}...")

    result = scanner.sync_scan (
        ai_profile=profile,
        content=Content (
            prompt=user_prompt,
            response=model_response
        ),
        metadata={"scan_type": "response_inspection"}
    )

    print (f"✅ Response scan complete - Verdict: {result.verdict}")
    return result


# =============================================================================
# DEMO
# =============================================================================

def main ():
    """Run a simple demo"""

    # Initialize security
    scanner, profile = initialize_security ()

    # Example user prompt
    test_prompt = "How do I reset my password?"

    # Scan the prompt
    prompt_result = scan_prompt (scanner, profile, test_prompt)

    # Simulate model response (replace with actual model call)
    # model_response = your_model_call(test_prompt)  # Put your model API call here
    test_response = "To reset your password, go to Settings > Security > Reset Password"

    # Scan the response
    response_result = scan_response (scanner, profile, test_prompt, test_response)

    print ("\n" + "=" * 60)
    print ("Demo complete!")
    print (f"Prompt verdict: {prompt_result.verdict}")
    print (f"Response verdict: {response_result.verdict}")
    print ("=" * 60)


if __name__ == "__main__":
    main ()