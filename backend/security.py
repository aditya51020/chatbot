import os
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    Basic Role-Based Access Control (RBAC) middleware.
    Validates a simple bearer token for admin operations like uploading and scanning.
    """
    expected_token = os.getenv("ADMIN_TOKEN", "supersecret-gov-token")
    if credentials.credentials != expected_token:
        raise HTTPException(
            status_code=403,
            detail="Forbidden: Invalid authorization token for admin operation",
        )
    return True
