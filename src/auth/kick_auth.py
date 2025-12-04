"""
Kick OAuth Authentication Handler

Handles OAuth token generation and automatic refresh for Kick API.
Uses Authorization Code Flow with PKCE.
"""
import asyncio
import json
import logging
import os
import time
import webbrowser
import secrets
import hashlib
import base64
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlencode, parse_qs, urlparse

import aiohttp

logger = logging.getLogger(__name__)

# Kick OAuth endpoints
KICK_AUTH_URL = "https://id.kick.com/oauth/authorize"
KICK_TOKEN_URL = "https://id.kick.com/oauth/token"
KICK_REVOKE_URL = "https://id.kick.com/oauth/revoke"
KICK_API_BASE = "https://api.kick.com/public/v1"

# Default scopes
DEFAULT_SCOPES = [
    "user:read",
    "channel:read",
    "chat:write",
    "events:subscribe",
    "channel:read",
    "kicks:read"
]


@dataclass
class KickTokens:
    """Stores Kick OAuth tokens"""
    access_token: str
    refresh_token: str
    expires_at: float  # Unix timestamp
    token_type: str = "bearer"
    scopes: list = None # type: ignore
    
    def __post_init__(self):
        if self.scopes is None:
            self.scopes = []
    
    @property
    def is_expired(self) -> bool:
        """Check if the access token is expired (with 5 min buffer)"""
        return time.time() >= (self.expires_at - 300)
    
    def to_dict(self) -> dict:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "token_type": self.token_type,
            "scopes": self.scopes,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "KickTokens":
        return cls(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=data["expires_at"],
            token_type=data.get("token_type", "bearer"),
            scopes=data.get("scopes", []),
        )


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for OAuth callback"""
    
    def do_GET(self):
        """Handle the OAuth callback"""
        parsed = urlparse(self.path)
        
        if parsed.path == "/callback":
            query = parse_qs(parsed.query)
            
            # Debug logging
            print(f"Callback received: {self.path}")
            
            if "code" in query:
                self.server.auth_code = query["code"][0] # type: ignore
                self.server.auth_state = query.get("state", [None])[0] # type: ignore
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(b"""
                    <html><body style="font-family: Arial; text-align: center; padding: 50px;">
                    <h1>&#10004; Authorization Successful!</h1>
                    <p>You can close this window and return to the terminal.</p>
                    </body></html>
                """)
            elif "error" in query:
                error_desc = query.get("error_description", query["error"])[0]
                self.server.auth_error = error_desc # type: ignore
                print(f"Auth Error from server: {error_desc}")
                
                self.send_response(400)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(f"""
                    <html><body style="font-family: Arial; text-align: center; padding: 50px;">
                    <h1>&#10008; Authorization Failed</h1>
                    <p>{error_desc}</p>
                    </body></html>
                """.encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress HTTP server logs"""
        pass


class KickAuth:
    """
    Handles Kick OAuth authentication with automatic token refresh.
    Uses PKCE flow.
    """
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        token_file: Optional[Path] = None,
        redirect_port: int = 3000,
        redirect_host: str = "localhost",
        scopes: Optional[list] = None
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_port = redirect_port
        self.redirect_host = redirect_host
        self.redirect_uri = f"http://{redirect_host}:{redirect_port}/callback"
        self.scopes = scopes or DEFAULT_SCOPES
        
        # Token storage
        self.token_file = token_file or Path.home() / ".kick_tokens.json"
        self._tokens: Optional[KickTokens] = None
        
        # PKCE State
        self._code_verifier: Optional[str] = None
        
        # Load existing tokens
        self._load_tokens()
    
    def _load_tokens(self):
        """Load tokens from file if exists"""
        if self.token_file.exists():
            try:
                with open(self.token_file, "r") as f:
                    data = json.load(f)
                self._tokens = KickTokens.from_dict(data)
                logger.info("Loaded existing Kick tokens")
            except Exception as e:
                logger.warning(f"Failed to load tokens: {e}")
                self._tokens = None
    
    def _save_tokens(self):
        """Save tokens to file"""
        if self._tokens:
            try:
                with open(self.token_file, "w") as f:
                    json.dump(self._tokens.to_dict(), f, indent=2)
                logger.info(f"Saved tokens to {self.token_file}")
            except Exception as e:
                logger.error(f"Failed to save tokens: {e}")
    
    def _generate_pkce(self) -> Tuple[str, str]:
        """Generate PKCE verifier and challenge"""
        verifier = secrets.token_urlsafe(32)
        digest = hashlib.sha256(verifier.encode()).digest()
        challenge = base64.urlsafe_b64encode(digest).rstrip(b'=').decode()
        return verifier, challenge

    def get_auth_url(self) -> str:
        """Generate the authorization URL with PKCE and workaround"""
        self._code_verifier, challenge = self._generate_pkce()
        # Use hex to ensure safe characters, though urlsafe is usually fine
        self._auth_state = secrets.token_hex(16)
        
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(self.scopes),
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": self._auth_state
        }
        
        # Workaround for 127.0.0.1: Add 'redirect=127.0.0.1' BEFORE redirect_uri
        # We construct the query string manually to ensure order if needed, 
        # though urlencode usually preserves order in recent Python versions.
        # But to be safe and follow the docs exactly:
        
        query_parts = []
        
        # Add workaround param if needed
        # NOTE: Some users report needing this even for localhost
        # If we are using localhost, we might still need to tell Kick to treat it as 127.0.0.1
        # But we must be careful not to break the redirect_uri matching.
        
        if "127.0.0.1" in self.redirect_host:
            query_parts.append("redirect=127.0.0.1")
            
        query_parts.append(urlencode(params))
        
        full_query = "&".join(query_parts)
        return f"{KICK_AUTH_URL}?{full_query}"
    
    async def authenticate(self, open_browser: bool = True) -> KickTokens:
        """
        Perform OAuth authentication flow.
        """
        # Start local callback server
        server = HTTPServer((self.redirect_host, self.redirect_port), OAuthCallbackHandler)
        server.auth_code = None # type: ignore
        server.auth_error = None # type: ignore
        server.auth_state = None # type: ignore
        
        # Run server in background thread
        server_thread = threading.Thread(target=server.handle_request, daemon=True)
        server_thread.start()
        
        # Generate and open auth URL
        auth_url = self.get_auth_url()
        print(f"\n{'='*60}")
        print("Kick Authorization Required")
        print(f"{'='*60}")
        print(f"\nPlease authorize in your browser.")
        print(f"\nIf the browser doesn't open, visit:\n{auth_url}\n")
        
        if open_browser:
            webbrowser.open(auth_url)
        
        # Wait for callback
        server_thread.join(timeout=120)
        server.server_close()
        
        if server.auth_error: # type: ignore
            raise Exception(f"Authorization failed: {server.auth_error}") # type: ignore
        
        if not server.auth_code: # type: ignore
            raise Exception("Authorization timed out - no code received")
            
        # Validate state
        if server.auth_state != self._auth_state: # type: ignore
            # If state mismatch, it might be due to browser/server encoding differences
            # or a legitimate security issue. We'll log it but maybe allow it if user forces?
            # For now, strict check.
            print(f"DEBUG: Sent state: {self._auth_state}")
            print(f"DEBUG: Recv state: {server.auth_state}") # type: ignore
            raise Exception(f"State mismatch! Expected {self._auth_state}, got {server.auth_state}") # type: ignore
        
        # Exchange code for tokens
        self._tokens = await self._exchange_code(server.auth_code) # type: ignore
        self._save_tokens()
        
        print("✓ Authorization successful!")
        print(f"{'='*60}\n")
        
        return self._tokens
    
    async def _exchange_code(self, code: str) -> KickTokens:
        """Exchange authorization code for tokens"""
        if not self._code_verifier:
            raise Exception("PKCE code verifier missing")
            
        async with aiohttp.ClientSession() as session:
            data = {
                "grant_type": "authorization_code",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "code": code,
                "redirect_uri": self.redirect_uri,
                "code_verifier": self._code_verifier
            }
            
            async with session.post(KICK_TOKEN_URL, data=data) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise Exception(f"Token exchange failed: {error}")
                
                result = await resp.json()
                
                # Kick returns expires_in (seconds)
                expires_in = result.get("expires_in", 3600)
                
                return KickTokens(
                    access_token=result["access_token"],
                    refresh_token=result["refresh_token"],
                    expires_at=time.time() + expires_in,
                    token_type=result.get("token_type", "bearer"),
                    scopes=result.get("scope", "").split(" ") if isinstance(result.get("scope"), str) else result.get("scope", []),
                )
    
    async def refresh_tokens(self) -> KickTokens:
        """Refresh the access token using the refresh token"""
        if not self._tokens or not self._tokens.refresh_token:
            raise Exception("No refresh token available - need to authenticate first")
        
        logger.info("Refreshing Kick access token...")
        
        async with aiohttp.ClientSession() as session:
            data = {
                "grant_type": "refresh_token",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": self._tokens.refresh_token,
            }
            
            async with session.post(KICK_TOKEN_URL, data=data) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    logger.error(f"Token refresh failed: {error}")
                    self._tokens = None
                    raise Exception(f"Token refresh failed: {error}")
                
                result = await resp.json()
                
                expires_in = result.get("expires_in", 3600)
                
                self._tokens = KickTokens(
                    access_token=result["access_token"],
                    refresh_token=result["refresh_token"],
                    expires_at=time.time() + expires_in,
                    token_type=result.get("token_type", "bearer"),
                    scopes=result.get("scope", "").split(" ") if isinstance(result.get("scope"), str) else result.get("scope", []),
                )
                
                self._save_tokens()
                logger.info("Token refreshed successfully")
                
                return self._tokens
    
    async def validate_token(self) -> bool:
        """Validate the current access token"""
        if not self._tokens:
            return False
            
        # Check local expiration first
        if self._tokens.is_expired:
            return False
            
        # Try to call a simple endpoint to verify
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self._tokens.access_token}"}
            
            # Use users/me endpoint if available, or just trust local expiry + refresh
            # Kick API v1 doesn't have a standard "validate" endpoint like Twitch
            # We'll try fetching self user info
            try:
                async with session.get(f"{KICK_API_BASE}/users", headers=headers) as resp:
                    if resp.status == 200:
                        return True
                    elif resp.status == 401:
                        return False
                    else:
                        # Other error, assume valid if not 401?
                        return True
            except Exception:
                return True # Assume valid if network error?
    
    async def get_valid_token(self) -> Optional[str]:
        """Get a valid access token, refreshing if necessary."""
        if not self._tokens:
            return None
        
        if self._tokens.is_expired:
            try:
                await self.refresh_tokens()
            except Exception:
                return None
        
        return self._tokens.access_token
    
    async def revoke_tokens(self):
        """Revoke the current tokens"""
        if not self._tokens:
            return
        
        async with aiohttp.ClientSession() as session:
            data = {
                "token": self._tokens.access_token,
                "token_hint_type": "access_token"
            }
            # Kick docs say POST /oauth/revoke with token param
            # Headers: Content-Type: application/x-www-form-urlencoded
            
            await session.post(KICK_REVOKE_URL, data=data)
        
        self._tokens = None
        if self.token_file.exists():
            self.token_file.unlink()
        
        logger.info("Tokens revoked")

    @property
    def has_tokens(self) -> bool:
        return self._tokens is not None
    
    @property
    def tokens(self) -> Optional[KickTokens]:
        return self._tokens


async def main():
    """CLI tool for Kick authentication"""
    import argparse
    from dotenv import load_dotenv
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check for credentials in environment
    env_client_id = os.getenv("KICK_CLIENT_ID")
    env_client_secret = os.getenv("KICK_CLIENT_SECRET")
    
    parser = argparse.ArgumentParser(description="Kick OAuth Token Manager")
    parser.add_argument("--client-id", default=env_client_id, help="Kick Client ID (or set KICK_CLIENT_ID in .env)")
    parser.add_argument("--client-secret", default=env_client_secret, help="Kick Client Secret (or set KICK_CLIENT_SECRET in .env)")
    parser.add_argument("--refresh", action="store_true", help="Force refresh the token")
    parser.add_argument("--revoke", action="store_true", help="Revoke the current token")
    parser.add_argument("--show", action="store_true", help="Show the current token")
    parser.add_argument("--host", default="localhost", help="Redirect host (use 127.0.0.1 if you get state/redirect errors)")
    
    args = parser.parse_args()
    
    if not args.client_id or not args.client_secret:
        print("\nError: Missing Credentials")
        print("Please provide --client-id and --client-secret arguments")
        print("OR set KICK_CLIENT_ID and KICK_CLIENT_SECRET in your .env file.\n")
        return
    
    auth = KickAuth(
        args.client_id, 
        args.client_secret, 
        redirect_host=args.host
    )
    
    if args.revoke:
        await auth.revoke_tokens()
        print("Tokens revoked")
        return
    
    if args.show:
        if auth.has_tokens and auth.tokens:
            print(f"Access Token: {auth.tokens.access_token[:20]}...")
            print(f"Expires: {time.ctime(auth.tokens.expires_at)}")
        else:
            print("No tokens stored")
        return
    
    if args.refresh:
        if auth.has_tokens:
            await auth.refresh_tokens()
            print("✓ Token refreshed")
        else:
            print("No tokens to refresh")
        return
    
    # Default: authenticate
    if auth.has_tokens and auth.tokens and not auth.tokens.is_expired:
        print("Existing valid tokens found.")
        response = input("Re-authenticate anyway? [y/N]: ")
        if response.lower() != 'y':
            return
    
    await auth.authenticate()
    if auth.tokens:
        print(f"\nAdd this to your .env file:")
        print(f"KICK_OAUTH_TOKEN={auth.tokens.access_token}")
        print(f"KICK_REFRESH_TOKEN={auth.tokens.refresh_token}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
