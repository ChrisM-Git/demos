#!/usr/bin/env python3
"""
Security Audit Script - Check what files are publicly accessible
Run this to verify which sensitive files are exposed on your GCP instance.
"""

import requests
import sys
from urllib.parse import urljoin

# ANSI color codes
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def check_file_accessibility(base_url, file_path):
    """Check if a file is publicly accessible."""
    url = urljoin(base_url, file_path)
    try:
        response = requests.get(url, timeout=5, allow_redirects=False)
        return response.status_code, response.text[:200] if response.status_code == 200 else None
    except requests.RequestException as e:
        return None, str(e)

def main():
    if len(sys.argv) < 2:
        print(f"{RED}Usage: python3 security_audit.py <your-gcp-instance-ip-or-domain>{RESET}")
        print(f"Example: python3 security_audit.py https://35.123.45.67")
        sys.exit(1)

    base_url = sys.argv[1]
    if not base_url.startswith(('http://', 'https://')):
        base_url = 'https://' + base_url

    # List of sensitive files to check
    sensitive_files = [
        # Environment and config files
        '.env',
        '.env.local',
        '.env.production',
        '.env.development',
        'config.py',
        'settings.py',

        # Python files
        'demo.py',
        'upload_server.py',
        'mcp_server.py',
        'mcp_demo.py',

        # Package management
        'requirements.txt',
        'Pipfile',
        'Pipfile.lock',
        'package.json',
        'package-lock.json',
        'yarn.lock',
        'composer.json',

        # Version control
        '.git/config',
        '.git/HEAD',
        '.gitignore',

        # Other sensitive files
        '.htaccess',
        'README.md',
        'docker-compose.yml',
        'Dockerfile',
        '.dockerignore',

        # Subdirectory requirements
        'mcp/requirements.txt',
        'ms-rt/requirements.txt',
        'healthcare/requirements.txt',
    ]

    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}Security Audit for: {base_url}{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")

    exposed_files = []
    blocked_files = []
    error_files = []

    for file_path in sensitive_files:
        status_code, content = check_file_accessibility(base_url, file_path)

        if status_code == 200:
            exposed_files.append((file_path, content))
            print(f"{RED}[EXPOSED]{RESET} {file_path} - Status: {status_code}")
            if content:
                print(f"  {YELLOW}Preview: {content[:100]}...{RESET}")
        elif status_code in [403, 404]:
            blocked_files.append(file_path)
            print(f"{GREEN}[BLOCKED]{RESET} {file_path} - Status: {status_code}")
        elif status_code is None:
            error_files.append(file_path)
            print(f"{YELLOW}[ERROR]{RESET} {file_path} - {content}")
        else:
            print(f"{YELLOW}[OTHER]{RESET} {file_path} - Status: {status_code}")

    # Summary
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}SUMMARY{RESET}")
    print(f"{BLUE}{'='*70}{RESET}")
    print(f"{RED}Exposed files: {len(exposed_files)}{RESET}")
    print(f"{GREEN}Blocked files: {len(blocked_files)}{RESET}")
    print(f"{YELLOW}Errors: {len(error_files)}{RESET}")

    if exposed_files:
        print(f"\n{RED}{'='*70}{RESET}")
        print(f"{RED}CRITICAL: The following files are publicly accessible:{RESET}")
        print(f"{RED}{'='*70}{RESET}")
        for file_path, _ in exposed_files:
            print(f"  - {file_path}")
        print(f"\n{RED}ACTION REQUIRED:{RESET}")
        print(f"1. Update your nginx configuration to block these files")
        print(f"2. If .env or config files were exposed, rotate all API keys immediately")
        print(f"3. Review access logs to see if these files were accessed")
        return 1
    else:
        print(f"\n{GREEN}Good! No critical files appear to be exposed.{RESET}")
        return 0

if __name__ == "__main__":
    sys.exit(main())
