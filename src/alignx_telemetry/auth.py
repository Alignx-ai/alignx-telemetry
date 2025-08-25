import os
import httpx


class AuthManager:
    def __init__(self, api_base_url: str = None):
        self.api_base_url = (
            api_base_url or os.getenv("ALIGNX_API_BASE_URL") or "https://app.alignx.ai"
        )

    def validate_key(self, license_key: str) -> dict:
        """Validates license key against the auth service"""
        try:
            with httpx.Client(
                timeout=httpx.Timeout(
                    connect=15.0,
                    read=20.0,
                    write=10.0,
                    pool=2.0,
                ),
            ) as client:
                response = client.get(
                    f"{self.api_base_url}/api/v1.0/licenses/{license_key}/validate",
                    headers={"X-License-Key": license_key},
                )

                if response.status_code == 200:
                    data = response.json()
                    return data
                else:
                    return None
        except Exception:
            return None
