"""
Twitch OAuth Authentication Handler

Handles OAuth token generation and automatic refresh for Twitch API.
Uses the Device Code Flow for easy CLI authentication.
"""
import asyncio
import json
import logging
import os
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlencode, parse_qs, urlparse
import threading

import aiohttp

logger = logging.getLogger(__name__)

# Twitch OAuth endpoints
TWITCH_AUTH_URL = "https://id.twitch.tv/oauth2/authorize"
TWITCH_TOKEN_URL = "https://id.twitch.tv/oauth2/token"
TWITCH_VALIDATE_URL = "https://id.twitch.tv/oauth2/validate"
TWITCH_REVOKE_URL = "https://id.twitch.tv/oauth2/revoke"

# Default scopes for chat reading
DEFAULT_SCOPES = [
    "chat:read",
    "user:read:email",
    "user:read:chat",
]


@dataclass
class TwitchTokens:
    """Stores Twitch OAuth tokens"""
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
    
    @property
    def oauth_token(self) -> str:
        """Get token in oauth:xxx format for IRC"""
        return f"oauth:{self.access_token}"
    
    def to_dict(self) -> dict:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "token_type": self.token_type,
            "scopes": self.scopes,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TwitchTokens":
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
            
            if "code" in query:
                self.server.auth_code = query["code"][0]
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
                self.server.auth_error = query.get("error_description", query["error"])[0]
                self.send_response(400)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(f"""
                    <html><body style="font-family: Arial; text-align: center; padding: 50px;">
                    <h1>&#10008; Authorization Failed</h1>
                    <p>{self.server.auth_error}</p>
                    </body></html>
                """.encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress HTTP server logs"""
        pass


class TwitchAuth:
    """
    Handles Twitch OAuth authentication with automatic token refresh.
    
    Usage:
        auth = TwitchAuth(client_id, client_secret)
        
        # First time - get new tokens
        tokens = await auth.authenticate()
        
        # Later - get valid token (auto-refreshes if needed)
        token = await auth.get_valid_token()
    """
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        token_file: Optional[Path] = None,
        redirect_port: int = 3000,
        scopes: Optional[list] = None
    ):
        """
        Initialize Twitch auth handler.
        
        Args:
            client_id: Twitch application client ID
            client_secret: Twitch application client secret
            token_file: Path to store tokens (default: ~/.twitch_tokens.json)
            redirect_port: Local port for OAuth callback
            scopes: OAuth scopes to request
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_port = redirect_port
        self.redirect_uri = f"http://localhost:{redirect_port}/callback"
        self.scopes = scopes or DEFAULT_SCOPES
        
        # Token storage
        self.token_file = token_file or Path.home() / ".twitch_tokens.json"
        self._tokens: Optional[TwitchTokens] = None
        
        # Load existing tokens
        self._load_tokens()
    
    def _load_tokens(self):
        """Load tokens from file if exists"""
        if self.token_file.exists():
            try:
                with open(self.token_file, "r") as f:
                    data = json.load(f)
                self._tokens = TwitchTokens.from_dict(data)
                logger.info("Loaded existing Twitch tokens")
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
    
    def get_auth_url(self) -> str:
        """Generate the authorization URL"""
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": " ".join(self.scopes),
        }
        return f"{TWITCH_AUTH_URL}?{urlencode(params)}"
    
    async def authenticate(self, open_browser: bool = True) -> TwitchTokens:
        """
        Perform OAuth authentication flow.
        
        Opens browser for user to authorize, then exchanges code for tokens.
        
        Args:
            open_browser: Whether to automatically open the browser
            
        Returns:
            TwitchTokens with access and refresh tokens
        """
        # Start local callback server
        server = HTTPServer(("localhost", self.redirect_port), OAuthCallbackHandler)
        server.auth_code = None
        server.auth_error = None
        
        # Run server in background thread
        server_thread = threading.Thread(target=server.handle_request, daemon=True)
        server_thread.start()
        
        # Generate and open auth URL
        auth_url = self.get_auth_url()
        print(f"\n{'='*60}")
        print("Twitch Authorization Required")
        print(f"{'='*60}")
        print(f"\nPlease authorize in your browser.")
        print(f"\nIf the browser doesn't open, visit:\n{auth_url}\n")
        
        if open_browser:
            webbrowser.open(auth_url)
        
        # Wait for callback
        server_thread.join(timeout=120)
        server.server_close()
        
        if server.auth_error:
            raise Exception(f"Authorization failed: {server.auth_error}")
        
        if not server.auth_code:
            raise Exception("Authorization timed out - no code received")
        
        # Exchange code for tokens
        self._tokens = await self._exchange_code(server.auth_code)
        self._save_tokens()
        
        print("✓ Authorization successful!")
        print(f"{'='*60}\n")
        
        return self._tokens
    
    async def _exchange_code(self, code: str) -> TwitchTokens:
        """Exchange authorization code for tokens"""
        async with aiohttp.ClientSession() as session:
            data = {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": self.redirect_uri,
            }
            
            async with session.post(TWITCH_TOKEN_URL, data=data) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise Exception(f"Token exchange failed: {error}")
                
                result = await resp.json()
                
                return TwitchTokens(
                    access_token=result["access_token"],
                    refresh_token=result["refresh_token"],
                    expires_at=time.time() + result["expires_in"],
                    token_type=result.get("token_type", "bearer"),
                    scopes=result.get("scope", []),
                )
    
    async def refresh_tokens(self) -> TwitchTokens:
        """Refresh the access token using the refresh token"""
        if not self._tokens or not self._tokens.refresh_token:
            raise Exception("No refresh token available - need to authenticate first")
        
        logger.info("Refreshing Twitch access token...")
        
        async with aiohttp.ClientSession() as session:
            data = {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "refresh_token",
                "refresh_token": self._tokens.refresh_token,
            }
            
            async with session.post(TWITCH_TOKEN_URL, data=data) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    logger.error(f"Token refresh failed: {error}")
                    # Clear invalid tokens
                    self._tokens = None
                    raise Exception(f"Token refresh failed: {error}")
                
                result = await resp.json()
                
                self._tokens = TwitchTokens(
                    access_token=result["access_token"],
                    refresh_token=result["refresh_token"],
                    expires_at=time.time() + result["expires_in"],
                    token_type=result.get("token_type", "bearer"),
                    scopes=result.get("scope", []),
                )
                
                self._save_tokens()
                logger.info("Token refreshed successfully")
                
                return self._tokens
    
    async def validate_token(self) -> bool:
        """Validate the current access token"""
        if not self._tokens:
            return False
        
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"OAuth {self._tokens.access_token}"}
            
            async with session.get(TWITCH_VALIDATE_URL, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.debug(f"Token valid, expires in {data.get('expires_in', 0)}s")
                    return True
                return False
    
    async def get_valid_token(self) -> Optional[str]:
        """
        Get a valid access token, refreshing if necessary.
        
        Returns:
            Valid access token or None if unable to get one
        """
        if not self._tokens:
            logger.warning("No tokens available - need to authenticate")
            return None
        
        # Check if expired
        if self._tokens.is_expired:
            try:
                await self.refresh_tokens()
            except Exception as e:
                logger.error(f"Failed to refresh token: {e}")
                return None
        
        return self._tokens.access_token
    
    async def get_oauth_token(self) -> Optional[str]:
        """Get token in oauth:xxx format for Twitch IRC"""
        token = await self.get_valid_token()
        if token:
            return f"oauth:{token}"
        return None
    
    async def revoke_tokens(self):
        """Revoke the current tokens"""
        if not self._tokens:
            return
        
        async with aiohttp.ClientSession() as session:
            data = {
                "client_id": self.client_id,
                "token": self._tokens.access_token,
            }
            
            await session.post(TWITCH_REVOKE_URL, data=data)
        
        # Clear tokens
        self._tokens = None
        if self.token_file.exists():
            self.token_file.unlink()
        
        logger.info("Tokens revoked")
    
    @property
    def has_tokens(self) -> bool:
        """Check if tokens are available"""
        return self._tokens is not None
    
    @property
    def tokens(self) -> Optional[TwitchTokens]:
        """Get current tokens"""
        return self._tokens


async def main():
    """CLI tool for Twitch authentication"""
    import argparse
    from dotenv import load_dotenv
    
    # Load .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Twitch OAuth Token Manager"
    )
    parser.add_argument(
        "--client-id",
        default=os.getenv("TWITCH_CLIENT_ID"),
        help="Twitch Client ID (or set TWITCH_CLIENT_ID env var)"
    )
    parser.add_argument(
        "--client-secret",
        default=os.getenv("TWITCH_CLIENT_SECRET"),
        help="Twitch Client Secret (or set TWITCH_CLIENT_SECRET env var)"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force refresh the token"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the current token"
    )
    parser.add_argument(
        "--revoke",
        action="store_true",
        help="Revoke the current token"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the current token"
    )
    
    args = parser.parse_args()
    
    if not args.client_id or not args.client_secret:
        print("Error: TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET are required")
        print("\nGet these from https://dev.twitch.tv/console/apps")
        print("\nSet them as environment variables or pass via --client-id and --client-secret")
        return
    
    auth = TwitchAuth(args.client_id, args.client_secret)
    
    if args.revoke:
        await auth.revoke_tokens()
        print("Tokens revoked")
        return
    
    if args.validate:
        if await auth.validate_token():
            print("✓ Token is valid")
        else:
            print("✗ Token is invalid or expired")
        return
    
    if args.show:
        if auth.has_tokens:
            print(f"Access Token: {auth.tokens.access_token[:20]}...")
            print(f"Expires: {time.ctime(auth.tokens.expires_at)}")
            print(f"Expired: {auth.tokens.is_expired}")
            print(f"OAuth Token: {auth.tokens.oauth_token[:25]}...")
        else:
            print("No tokens stored")
        return
    
    if args.refresh:
        if auth.has_tokens:
            await auth.refresh_tokens()
            print("✓ Token refreshed")
            print(f"New expiry: {time.ctime(auth.tokens.expires_at)}")
        else:
            print("No tokens to refresh - run without --refresh first")
        return
    
    # Default: authenticate
    if auth.has_tokens and not auth.tokens.is_expired:
        print("Existing valid tokens found.")
        print(f"Expires: {time.ctime(auth.tokens.expires_at)}")
        
        response = input("Re-authenticate anyway? [y/N]: ")
        if response.lower() != 'y':
            print(f"\nYour OAuth token: {auth.tokens.oauth_token}")
            return
    
    await auth.authenticate()
    print(f"\nYour OAuth token: {auth.tokens.oauth_token}")
    print(f"\nAdd this to your .env file:")
    print(f"TWITCH_OAUTH_TOKEN={auth.tokens.oauth_token}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
