"""
Kick API Client

Handles authenticated API calls to Kick using OAuth tokens.
This bypasses Cloudflare blocking by using proper authentication.

IMPORTANT: Kick has TWO types of tokens:
1. User Access Token - For user-specific actions (user:read, events:subscribe)
   - Obtained via Authorization Code + PKCE flow
   - Works with: /users endpoint
   
2. App Access Token - For public data access (channel:read)
   - Obtained via Client Credentials flow
   - Works with: /channels endpoint
   
The /channels endpoint requires an APP ACCESS TOKEN, not a user token!
"""
import os
import json
import time
import logging
from typing import Optional, Dict, Any
from pathlib import Path

import aiohttp

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Token file paths
USER_TOKEN_FILE = Path.home() / ".kick_tokens.json"
APP_TOKEN_FILE = Path.home() / ".kick_app_token.json"
CHATROOM_CACHE_FILE = Path.home() / ".kick_chatroom_cache.json"


class KickAPI:
    """
    Authenticated Kick API client.
    
    Uses both User tokens and App tokens depending on the endpoint:
    - /users -> User Access Token
    - /channels -> App Access Token (Client Credentials)
    """
    
    BASE_URL = "https://api.kick.com/public/v1"
    TOKEN_URL = "https://id.kick.com/oauth/token"
    
    def __init__(self, token: Optional[str] = None, client_id: Optional[str] = None, client_secret: Optional[str] = None):
        """
        Initialize Kick API client.
        
        Args:
            token: OAuth user access token (or uses token file / KICK_OAUTH_TOKEN from env)
            client_id: Client ID (or uses KICK_CLIENT_ID from env)
            client_secret: Client Secret (or uses KICK_CLIENT_SECRET from env) - needed for app token
        """
        self.client_id = client_id or os.getenv("KICK_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("KICK_CLIENT_SECRET")
        
        # User token - for /users endpoint
        self.user_token = token
        if not self.user_token:
            self._load_user_token()
        if not self.user_token:
            self.user_token = os.getenv("KICK_OAUTH_TOKEN")
            if self.user_token:
                logger.info("Using KICK_OAUTH_TOKEN from environment")
        
        # App token - for /channels endpoint (will be fetched on demand)
        self.app_token: Optional[str] = None
        self.app_token_expires: float = 0
        self._load_app_token()
        
        # Chatroom ID cache - maps slug -> chatroom_id
        self.chatroom_cache: Dict[str, int] = {}
        self._load_chatroom_cache()
    
    def _load_chatroom_cache(self):
        """Load chatroom ID cache from file"""
        if CHATROOM_CACHE_FILE.exists():
            try:
                with open(CHATROOM_CACHE_FILE, "r") as f:
                    self.chatroom_cache = json.load(f)
                logger.info(f"Loaded {len(self.chatroom_cache)} cached chatroom IDs")
            except Exception as e:
                logger.warning(f"Failed to load chatroom cache: {e}")
    
    def _save_chatroom_cache(self):
        """Save chatroom ID cache to file"""
        try:
            with open(CHATROOM_CACHE_FILE, "w") as f:
                json.dump(self.chatroom_cache, indent=2, fp=f)
            logger.info(f"Saved chatroom cache to {CHATROOM_CACHE_FILE}")
        except Exception as e:
            logger.warning(f"Failed to save chatroom cache: {e}")
    
    def set_chatroom_id(self, slug: str, chatroom_id: int):
        """Manually set a chatroom ID for a channel (bypasses API lookup)"""
        self.chatroom_cache[slug.lower()] = chatroom_id
        self._save_chatroom_cache()
        logger.info(f"Cached chatroom ID for {slug}: {chatroom_id}")
    
    def _load_user_token(self):
        """Load user token from the kick_auth token file"""
        if USER_TOKEN_FILE.exists():
            try:
                with open(USER_TOKEN_FILE, "r") as f:
                    data = json.load(f)
                
                # Check if token is expired
                expires_at = data.get("expires_at", 0)
                if time.time() >= expires_at:
                    logger.warning("User token from file is expired")
                    return
                
                self.user_token = data.get("access_token")
                if self.user_token:
                    logger.info("Loaded Kick user token from file")
            except Exception as e:
                logger.warning(f"Failed to load user token from file: {e}")
    
    def _load_app_token(self):
        """Load app token from file if available and not expired"""
        if APP_TOKEN_FILE.exists():
            try:
                with open(APP_TOKEN_FILE, "r") as f:
                    data = json.load(f)
                
                expires_at = data.get("expires_at", 0)
                if time.time() < expires_at:
                    self.app_token = data.get("access_token")
                    self.app_token_expires = expires_at
                    if self.app_token:
                        logger.info("Loaded Kick app token from file")
                else:
                    logger.info("App token from file is expired, will refresh")
            except Exception as e:
                logger.warning(f"Failed to load app token from file: {e}")
    
    def _save_app_token(self, token: str, expires_in: int):
        """Save app token to file"""
        try:
            data = {
                "access_token": token,
                "expires_at": time.time() + expires_in - 60,  # 60s buffer
                "token_type": "Bearer"
            }
            with open(APP_TOKEN_FILE, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved app token to {APP_TOKEN_FILE}")
        except Exception as e:
            logger.warning(f"Failed to save app token: {e}")
    
    async def _get_app_token(self) -> Optional[str]:
        """
        Get an App Access Token using Client Credentials flow.
        
        This is required for endpoints like /channels that need app-level access.
        """
        # Return cached token if valid
        if self.app_token and time.time() < self.app_token_expires:
            return self.app_token
        
        if not self.client_id or not self.client_secret:
            logger.error("Client ID and Secret required for App Access Token")
            logger.error("Set KICK_CLIENT_ID and KICK_CLIENT_SECRET in .env file")
            return None
        
        logger.info("Fetching new App Access Token via Client Credentials...")
        
        try:
            async with aiohttp.ClientSession() as session:
                data = {
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                }
                
                async with session.post(self.TOKEN_URL, data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.app_token = result.get("access_token")
                        expires_in = result.get("expires_in", 3600)
                        self.app_token_expires = time.time() + expires_in - 60
                        
                        # Save to file for reuse
                        if self.app_token:
                            self._save_app_token(self.app_token, expires_in)
                        
                        logger.info("Successfully obtained App Access Token")
                        return self.app_token
                    else:
                        error = await response.text()
                        logger.error(f"Failed to get App Token: {response.status} - {error}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error getting App Token: {e}")
            return None
    
    def _get_headers(self, token: Optional[str] = None) -> Dict[str, str]:
        """Get request headers with authentication"""
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers
    
    async def get_channel_data(self, slug: str) -> Optional[Dict[str, Any]]:
        """
        Fetch channel metadata using the official API.
        
        This uses an APP ACCESS TOKEN (client credentials) because
        the /channels endpoint requires channel:read scope which is
        only available to app tokens, not user tokens.
        
        NOTE: The Public API does NOT return chatroom info!
        Use get_chatroom_id() which has a fallback to scrape the HTML.
        
        Args:
            slug: Channel username/slug
            
        Returns:
            Channel data dict or None on error
        """
        # Get app token (will fetch if needed)
        app_token = await self._get_app_token()
        if not app_token:
            logger.error("Could not obtain App Access Token for /channels endpoint")
            return None
        
        # Public API uses query parameters: /channels?slug=xxx
        url = f"{self.BASE_URL}/channels"
        params = {"slug": slug}
        
        logger.info(f"[API] Fetching data for: {slug}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self._get_headers(app_token), params=params) as response:
                    if response.status == 401:
                        logger.error("[API] 401 Unauthorized. Token is invalid or expired.")
                        # Try to refresh token once
                        self.app_token = None
                        self.app_token_expires = 0
                        app_token = await self._get_app_token()
                        if app_token:
                            async with session.get(url, headers=self._get_headers(app_token), params=params) as retry_response:
                                if retry_response.status == 200:
                                    result = await retry_response.json()
                                    data = result.get("data", [])
                                    if isinstance(data, list) and len(data) > 0:
                                        return data[0]
                                    return data
                        return None
                    
                    if response.status == 403:
                        logger.error("[API] 403 Forbidden. Check your API Scopes.")
                        return None
                    
                    if response.status == 404:
                        logger.error(f"[API] Channel '{slug}' not found.")
                        return None
                    
                    if response.status == 200:
                        result = await response.json()
                        # API returns { "data": [...], "message": "OK" }
                        data = result.get("data", [])
                        if isinstance(data, list) and len(data) > 0:
                            return data[0]
                        return data
                    
                    error_text = await response.text()
                    logger.error(f"[API] Error {response.status}: {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"[API] Connection Error: {e}")
            return None
    
    async def _fetch_chatroom_from_internal_api(self, slug: str) -> Optional[int]:
        """
        Fetch chatroom ID from Kick's internal API using curl_cffi.
        
        Uses: https://kick.com/api/v2/channels/{slug}/chatroom
        curl_cffi impersonates Chrome to bypass Cloudflare protection.
        """
        url = f"https://kick.com/api/v2/channels/{slug}/chatroom"
        
        logger.info(f"[API] Fetching chatroom ID from internal API: {url}")
        
        try:
            from curl_cffi import requests as curl_requests
            
            # Use curl_cffi to bypass Cloudflare by impersonating Chrome
            response = curl_requests.get(url, impersonate="chrome", timeout=10)
            
            logger.info(f"[API] Internal API response: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract chatroom ID from response
                if isinstance(data, dict) and "id" in data:
                    chatroom_id = data["id"]
                    logger.info(f"[API] Found chatroom ID: {chatroom_id}")
                    
                    # Cache it for future use
                    self.chatroom_cache[slug.lower()] = chatroom_id
                    self._save_chatroom_cache()
                    
                    return chatroom_id
            else:
                logger.warning(f"[API] Internal API returned {response.status_code}")
                
        except ImportError:
            logger.warning("[API] curl_cffi not installed, cannot bypass Cloudflare")
        except Exception as e:
            logger.error(f"[API] Error fetching from internal API: {e}")
        
        return None
    
    async def _scrape_chatroom_id(self, slug: str) -> Optional[int]:
        """
        Scrape the chatroom ID from the channel's HTML page.
        
        This is a fallback when the Public API doesn't return chatroom info.
        The chatroom ID is embedded in the page's JavaScript data.
        """
        import re
        
        logger.info(f"[API] Scraping chatroom ID from kick.com/{slug}...")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://kick.com/{slug}", headers=headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        
                        # Look for chatroom ID in various patterns
                        patterns = [
                            r'"chatroom":\s*\{\s*"id":\s*(\d+)',
                            r'"chatroomId":\s*(\d+)',
                            r'"chatroom_id":\s*(\d+)',
                        ]
                        
                        for pattern in patterns:
                            match = re.search(pattern, html)
                            if match:
                                chatroom_id = int(match.group(1))
                                logger.info(f"[API] Found chatroom ID via HTML: {chatroom_id}")
                                return chatroom_id
                        
                        logger.warning("[API] Could not find chatroom ID in page HTML")
                    elif response.status == 403:
                        logger.warning("[API] Cloudflare blocked page access")
                    else:
                        logger.warning(f"[API] Page fetch returned {response.status}")
                        
        except Exception as e:
            logger.error(f"[API] Error scraping page: {e}")
        
        return None
    
    async def get_chatroom_id(self, slug: str) -> Optional[int]:
        """
        Get the chatroom ID for a channel.
        
        Tries multiple methods in order:
        1. Cache lookup (fastest)
        2. Internal API: /api/v2/channels/{slug}/chatroom (via curl_cffi)
        3. Public API: /channels?slug=xxx (unlikely to have chatroom)
        4. HTML scraping fallback
        
        IMPORTANT: The Public API does NOT return chatroom info!
        It only returns broadcaster_user_id which is different.
        
        Args:
            slug: Channel username/slug
            
        Returns:
            Chatroom ID or None on error
        """
        slug_lower = slug.lower()
        
        # Method 0: Check cache first
        if slug_lower in self.chatroom_cache:
            chatroom_id = self.chatroom_cache[slug_lower]
            logger.info(f"[API] Using cached chatroom ID for {slug}: {chatroom_id}")
            return chatroom_id
        
        # Method 1: Try internal API with curl_cffi (bypasses Cloudflare)
        chatroom_id = await self._fetch_chatroom_from_internal_api(slug)
        if chatroom_id:
            return chatroom_id
        
        # Method 2: Try the public API (might have chatroom in some cases)
        data = await self.get_channel_data(slug)
        
        if data:
            logger.debug(f"[API] Channel data keys: {data.keys() if isinstance(data, dict) else type(data)}")
            
            # Check if API returned chatroom info (unlikely for public API)
            if "chatroom" in data and "id" in data["chatroom"]:
                chat_id = data["chatroom"]["id"]
                logger.info(f"[API] Found Chatroom ID from API: {chat_id}")
                return chat_id
            
            if "chatroom_id" in data:
                logger.info(f"[API] Found chatroom_id: {data['chatroom_id']}")
                return data["chatroom_id"]
            
            # Public API doesn't have chatroom - need to scrape
            logger.info("[API] Public API doesn't include chatroom info, trying HTML scrape...")
        
        # Method 3: Fallback - scrape from HTML page
        chatroom_id = await self._scrape_chatroom_id(slug)
        if chatroom_id:
            return chatroom_id
        
        logger.error(f"[API] Could not get chatroom ID for '{slug}'")
        return None
    
    async def get_user_info(self) -> Optional[Dict[str, Any]]:
        """
        Get authenticated user info using USER ACCESS TOKEN.
        
        Returns:
            User data dict or None on error
        """
        if not self.user_token:
            logger.error("No user token available. Run 'python kick_auth.py' to authenticate.")
            return None
        
        url = f"{self.BASE_URL}/users"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self._get_headers(self.user_token)) as response:
                    if response.status == 200:
                        result = await response.json()
                        data = result.get("data", result)
                        if isinstance(data, list) and len(data) > 0:
                            return data[0]
                        return dict(data) if isinstance(data, dict) else None
                    return None
        except Exception as e:
            logger.error(f"Error getting user info: {e}")
            return None


# Synchronous wrapper for non-async code
class KickAPISync:
    """Synchronous wrapper for KickAPI"""
    
    def __init__(self, token: Optional[str] = None, client_id: Optional[str] = None, client_secret: Optional[str] = None):
        self._async_api = KickAPI(token, client_id, client_secret)
    
    def get_channel_data(self, slug: str) -> Optional[Dict[str, Any]]:
        import asyncio
        return asyncio.run(self._async_api.get_channel_data(slug))
    
    def get_chatroom_id(self, slug: str) -> Optional[int]:
        import asyncio
        return asyncio.run(self._async_api.get_chatroom_id(slug))
