"""
Twitch chat monitor using twitchio (Official Implementation)
"""
import asyncio
import logging
import time
import os
from typing import Optional
import random

try:
    from dotenv import load_dotenv # type: ignore
except ImportError:
    def load_dotenv(): pass

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except Exception:
    aiohttp = None  # type: ignore
    AIOHTTP_AVAILABLE = False

try:
    from .chat_base import BaseChatMonitor, ChatMessage
except Exception:
    # If the module is executed as a script (python src/monitors/chat_twitch.py)
    # the relative import will fail because there's no parent package. Fall
    # back to the absolute package import so the file can be run directly.
    from src.monitors.chat_base import BaseChatMonitor, ChatMessage

logger = logging.getLogger(__name__)

# Try to import twitchio
try:
    import twitchio
    from twitchio import eventsub
    TWITCHIO_AVAILABLE = True
except ImportError:
    TWITCHIO_AVAILABLE = False
    logger.warning("twitchio not installed - Twitch chat monitoring disabled")


class TwitchChatMonitor(BaseChatMonitor):
    """
    Monitors Twitch chat using the official twitchio library (v3.x).
    Requires TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET in environment.
    """
    
    def __init__(self, channel: str, oauth_token: Optional[str] = None, use_irc: Optional[bool] = None):
        """Create a TwitchChatMonitor.

        Args:
            channel: channel name (without #)
            oauth_token: optional OAuth token to use
            use_irc: if True, force using IRC fallback; if False, force EventSub; if None, respect `TWITCH_USE_IRC` env var.
        """
        # allow forcing IRC fallback via constructor or env
        use_irc_env = os.getenv("TWITCH_USE_IRC", "").lower() in ("1", "true", "yes")
        # explicit constructor param overrides environment
        if use_irc is None:
            self.use_irc = use_irc_env
        else:
            self.use_irc = bool(use_irc)
        super().__init__(channel.lower().lstrip('#'))
        
        load_dotenv()
        self.client_id = os.getenv("TWITCH_CLIENT_ID")
        self.client_secret = os.getenv("TWITCH_CLIENT_SECRET")
        self.bot_id = os.getenv("TWITCH_BOT_ID")
        self.refresh_token = os.getenv("TWITCH_REFRESH_TOKEN")
        
        # Prefer passed token, then env var
        self.oauth_token = oauth_token or os.getenv("TWITCH_OAUTH_TOKEN")
        
        if not self.client_id or not self.client_secret:
            logger.warning("TWITCH_CLIENT_ID or TWITCH_CLIENT_SECRET missing. Twitch monitor may fail.")

        self._client = None
        self._connected_event = asyncio.Event()
        self._irc_task = None
        self.bot_login = None

    async def _ensure_token_valid(self) -> bool:
        """Validate the current `self.oauth_token` and refresh it if expired.

        Returns True if a valid token is available after this call.
        """
        # Reload .env in case the user updated tokens at runtime
        try:
            load_dotenv()
        except Exception:
            pass

        # Allow .env to override in-memory values (helps when user updates .env)
        env_token = os.getenv("TWITCH_OAUTH_TOKEN") or os.getenv("TWITCH_ACCESS_TOKEN")
        env_refresh = os.getenv("TWITCH_REFRESH_TOKEN")
        env_client_id = os.getenv("TWITCH_CLIENT_ID")
        env_client_secret = os.getenv("TWITCH_CLIENT_SECRET")
        env_bot_id = os.getenv("TWITCH_BOT_ID")

        if env_token and env_token != self.oauth_token:
            self.oauth_token = env_token
        if env_refresh and env_refresh != self.refresh_token:
            self.refresh_token = env_refresh
        if env_client_id:
            self.client_id = env_client_id
        if env_client_secret:
            self.client_secret = env_client_secret
        if env_bot_id:
            self.bot_id = env_bot_id

        if not self.oauth_token:
            logger.debug("No oauth token available to validate/refresh")
            return False

        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available; cannot validate/refresh Twitch token")
            return True  # assume token is ok; user can set env manually

        # Twitch validate expects the raw access token (not prefixed with "oauth:")
        token_to_check = self.oauth_token
        if token_to_check and token_to_check.startswith("oauth:"):
            token_to_check = token_to_check.split(":", 1)[1]

        validate_url = "https://id.twitch.tv/oauth2/validate"
        headers = {"Authorization": f"Bearer {token_to_check}"}

        try:
            async with aiohttp.ClientSession() as sess:  # type: ignore
                async with sess.get(validate_url, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        # token valid — log details to help diagnose EventSub auth issues
                        logger.info(
                            "Twitch token valid: client_id=%s, login=%s, user_id=%s, scopes=%s",
                            data.get("client_id"), data.get("login"), data.get("user_id"), data.get("scopes"),
                        )
                        # If TWITCH_BOT_ID or login were not provided, adopt the validated values
                        try:
                            if not self.bot_id and data.get("user_id"):
                                self.bot_id = data.get("user_id")
                            if not getattr(self, 'bot_login', None) and data.get("login"):
                                self.bot_login = data.get("login")
                        except Exception:
                            pass
                        return True
                    else:
                        logger.info("Twitch token invalid or expired (status %s). Attempting refresh...", resp.status)
        except Exception as e:
            logger.error("Error validating Twitch token: %s", e)

        # Attempt refresh
        if not (self.client_id and self.client_secret and self.refresh_token):
            logger.warning("Missing client_id/client_secret/refresh_token; cannot refresh Twitch token")
            return False

        token_url = "https://id.twitch.tv/oauth2/token"
        data = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        try:
            async with aiohttp.ClientSession() as sess:  # type: ignore
                async with sess.post(token_url, data=data) as r:
                    if r.status != 200:
                        text = await r.text()
                        logger.error("Failed to refresh Twitch token (%s): %s", r.status, text)
                        # Attempt full re-authentication if refresh failed
                        return await self._attempt_full_reauth()
                    body = await r.json()
        except Exception as e:
            logger.error("Exception refreshing Twitch token: %s", e)
            return await self._attempt_full_reauth()

        # Update tokens
        new_access = body.get("access_token")
        new_refresh = body.get("refresh_token")
        if not new_access:
            logger.error("Refresh response did not include access_token: %s", body)
            return False

        self.oauth_token = new_access
        if new_refresh:
            self.refresh_token = new_refresh

        # Persist into .env if present so future runs pick up new token
        try:
            env_path = ".env"
            if os.path.exists(env_path):
                # read and replace keys
                with open(env_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                def set_key(lines, key, value):
                    key_eq = key + "="
                    for i, L in enumerate(lines):
                        if L.startswith(key_eq):
                            lines[i] = f"{key}={value}\n"
                            return lines
                    lines.append(f"{key}={value}\n")
                    return lines

                lines = set_key(lines, "TWITCH_OAUTH_TOKEN", self.oauth_token)
                if self.refresh_token:
                    lines = set_key(lines, "TWITCH_REFRESH_TOKEN", self.refresh_token)

                with open(env_path, "w", encoding="utf-8") as f:
                    f.writelines(lines)
                logger.info("Persisted refreshed Twitch tokens into .env")
        except Exception:
            # ignore persistence errors
            logger.debug("Could not persist tokens to .env; continuing without persistence")

        return True
    
    async def _attempt_full_reauth(self) -> bool:
        """
        Attempt full re-authentication when refresh token is also expired.
        Uses the TwitchAuth class to open browser for user authorization.
        """
        if not (self.client_id and self.client_secret):
            logger.error("Cannot re-authenticate: missing client_id or client_secret")
            return False
        
        try:
            from ..auth.twitch_auth import TwitchAuth
            
            logger.warning("Refresh token expired. Starting full re-authentication...")
            print("\n" + "="*60)
            print("⚠️  Twitch token expired! Re-authentication required.")
            print("="*60)
            
            auth = TwitchAuth(self.client_id, self.client_secret)
            tokens = await auth.authenticate(open_browser=True)
            
            if tokens:
                self.oauth_token = tokens.access_token
                self.refresh_token = tokens.refresh_token
                
                # Update .env file
                self._update_env_file(tokens.access_token, tokens.refresh_token)
                
                logger.info("Re-authentication successful! Tokens updated.")
                return True
            else:
                logger.error("Re-authentication failed")
                return False
                
        except Exception as e:
            logger.error(f"Full re-authentication failed: {e}")
            return False
    
    def _update_env_file(self, access_token: str, refresh_token: str):
        """Update the .env file with new tokens"""
        try:
            env_path = ".env"
            lines = []
            
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            
            def set_key(lines, key, value):
                key_eq = key + "="
                for i, L in enumerate(lines):
                    if L.startswith(key_eq):
                        lines[i] = f"{key}={value}\n"
                        return lines
                lines.append(f"{key}={value}\n")
                return lines
            
            lines = set_key(lines, "TWITCH_OAUTH_TOKEN", access_token)
            lines = set_key(lines, "TWITCH_ACCESS_TOKEN", access_token)
            lines = set_key(lines, "TWITCH_REFRESH_TOKEN", refresh_token)
            
            with open(env_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            
            # Also update environment variables for current session
            os.environ["TWITCH_OAUTH_TOKEN"] = access_token
            os.environ["TWITCH_ACCESS_TOKEN"] = access_token
            os.environ["TWITCH_REFRESH_TOKEN"] = refresh_token
            
            logger.info("Updated .env file with new Twitch tokens")
            
        except Exception as e:
            logger.error(f"Failed to update .env file: {e}")
    
    @property
    def platform_name(self) -> str:
        return "Twitch"
    
    async def _connect(self):
        """Initialize the Twitch Client"""
        if not TWITCHIO_AVAILABLE:
            raise ImportError("twitchio is required for Twitch chat monitoring")
        
        monitor = self

        # If we already have an OAuth/access token, expose it as TWITCH_ACCESS_TOKEN
        # before instantiating the twitchio Client so twitchio will load it into
        # its ManagedHTTPClient on startup. Prefer raw access tokens (no oauth: prefix).
        try:
            env_token = os.getenv("TWITCH_ACCESS_TOKEN") or os.getenv("TWITCH_OAUTH_TOKEN") or monitor.oauth_token
            if env_token:
                raw = env_token
                if raw.startswith("oauth:"):
                    raw = raw.split(":", 1)[1]
                # Only set if not already present to avoid overwriting explicit env config
                if not os.getenv("TWITCH_ACCESS_TOKEN"):
                    os.environ["TWITCH_ACCESS_TOKEN"] = raw
                if monitor.refresh_token and not os.getenv("TWITCH_REFRESH_TOKEN"):
                    os.environ["TWITCH_REFRESH_TOKEN"] = monitor.refresh_token
        except Exception:
            pass
        
        # Define the Client class inline to capture 'monitor' context
        class MonitorBot(twitchio.Client):
            def __init__(self):
                super().__init__(
                    client_id=monitor.client_id or "missing",
                    client_secret=monitor.client_secret or "missing",
                    bot_id=monitor.bot_id
                )

            async def event_ready(self):
                logger.info(f"Twitch Client Ready. Logged in as {self.bot_id or 'Unknown'}")

                # If the monitor was configured to skip EventSub, start IRC and return early
                if monitor.use_irc or os.getenv("TWITCH_USE_IRC", "").lower() in ("1", "true", "yes"):
                    logger.info("Skipping EventSub subscriptions because IRC fallback is forced; starting IRC instead.")
                    try:
                        nick = monitor.bot_login or getattr(self, 'bot_login', None) or getattr(self, 'user', {}).get('login', None) or f"justinfan{random.randint(1000,9999)}"
                    except Exception:
                        nick = monitor.bot_login or f"justinfan{random.randint(1000,9999)}"
                    asyncio.create_task(monitor._start_irc(nick, monitor.oauth_token, monitor.channel))
                    monitor._connected_event.set()
                    return

                # Normal EventSub setup
                monitor._connected_event.set()
                try:
                    # Resolve channel name to ID
                    users = await self.fetch_users(logins=[monitor.channel])
                    if not users:
                        logger.error(f"Could not find user for channel: {monitor.channel}")
                        return

                    broadcaster = users[0]

                    # Determine which token to use: prefer TWITCH_ACCESS_TOKEN (raw) then monitor.oauth_token
                    env_access = os.getenv("TWITCH_ACCESS_TOKEN") or os.getenv("TWITCH_OAUTH_TOKEN")
                    token_candidate = env_access or monitor.oauth_token

                    # We need our own user ID to subscribe
                    user_id = self.bot_id

                    if not user_id and token_candidate:
                        # Ensure token is valid / refreshed and normalized
                        try:
                            await monitor._ensure_token_valid()
                        except Exception as e:
                            logger.debug("Token ensure step raised: %s", e)

                        # Use raw token for validate_token (strip oauth: prefix)
                        token_for_validate = token_candidate
                        if token_for_validate and token_for_validate.startswith("oauth:"):
                            token_for_validate = token_for_validate.split(":", 1)[1]

                        try:
                            logger.debug("Using token for validation from env or oauth: %s", "TWITCH_ACCESS_TOKEN" if os.getenv("TWITCH_ACCESS_TOKEN") else "TWITCH_OAUTH_TOKEN")
                            val = await self._http.validate_token(token_for_validate)
                            user_id = val.user_id
                            # store discovered bot id into client internal state
                            try:
                                self._bot_id = user_id
                            except Exception:
                                pass

                            # If the token is not a User Access Token or doesn't include the
                            # required chat scope, avoid subscribing to EventSub and
                            # fall back to IRC.
                            token_scopes = getattr(val, 'scopes', None) or []
                            if isinstance(token_scopes, str):
                                token_scopes = token_scopes.split()

                            if not user_id or 'user:read:chat' not in token_scopes:
                                logger.error("Validated token is not a user access token with 'user:read:chat' scope — skipping EventSub and starting IRC fallback.")
                                try:
                                    nick = monitor.bot_login or getattr(self, 'user', {}).get('login', None) or f"justinfan{random.randint(1000,9999)}"
                                    asyncio.create_task(monitor._start_irc(nick, monitor.oauth_token, monitor.channel))
                                except Exception as e2:
                                    logger.error("Failed to start IRC fallback after token check: %s", e2)
                                return

                        except Exception as e:
                            logger.error(f"Failed to validate token to get user ID: {e}")

                    # Persist the chosen token back to monitor.oauth_token so later code (add_token) uses it
                    if token_candidate and token_candidate != monitor.oauth_token:
                        monitor.oauth_token = token_candidate

                    if not user_id:
                        logger.error("Cannot subscribe to chat: Bot User ID unknown (set TWITCH_BOT_ID)")
                        return

                    # Ensure the client's ManagedHTTPClient knows about our user token
                    try:
                        if monitor.oauth_token:
                            token_for_add = monitor.oauth_token
                            if token_for_add.startswith("oauth:"):
                                token_for_add = token_for_add.split(":", 1)[1]
                            refresh_for_add = monitor.refresh_token or os.getenv("TWITCH_REFRESH_TOKEN") or ""
                            try:
                                await self._http.add_token(token_for_add, refresh_for_add)
                                logger.info("Added user token to ManagedHTTPClient for user_id=%s", user_id)
                            except Exception as e:
                                logger.debug("Could not add token to ManagedHTTPClient: %s", e)
                    except Exception:
                        pass

                    # Subscribe to chat messages via EventSub
                    # Note: This requires user:read:chat scope on the token
                    # Allow disabling EventSub calls via env var for environments
                    # where Helix EventSub cannot/should not be used. Default: enabled.
                    enable_eventsub = os.getenv("TWITCH_ENABLE_EVENTSUB", "1").lower() in ("1", "true", "yes")
                    if not enable_eventsub:
                        logger.info("TWITCH_ENABLE_EVENTSUB disabled; skipping EventSub subscription and starting IRC fallback.")
                        try:
                            nick = monitor.bot_login or getattr(broadcaster, 'login', None) or f"justinfan{random.randint(1000,9999)}"
                            asyncio.create_task(monitor._start_irc(nick, monitor.oauth_token, monitor.channel))
                        except Exception as e2:
                            logger.error("Failed to start IRC fallback: %s", e2)
                        return

                    logger.info(f"Subscribing to chat for {broadcaster.name} (ID: {broadcaster.id})")
                    sub = eventsub.ChatMessageSubscription(
                        broadcaster_user_id=broadcaster.id,
                        user_id=user_id,
                    )

                    # Try subscribing to EventSub; if ManagedHTTPClient hasn't picked up
                    # the token yet, retry a few times before falling back to IRC.
                    subscribed = False
                    last_exc = None
                    for attempt in range(3):
                        try:
                            await self.subscribe_websocket(payload=sub, as_bot=True)
                            logger.info("Subscribed to chat messages")
                            subscribed = True
                            break
                        except Exception as e:
                            last_exc = e
                            msg = str(e)
                            logger.debug("EventSub subscribe attempt %s failed: %s", attempt + 1, msg)
                            if 'valid User Access Token' in msg or 'must be passed' in msg:
                                await asyncio.sleep(0.5)
                                continue
                            break

                    if not subscribed:
                        logger.error(f"Error subscribing to EventSub: {last_exc}")
                        if monitor.use_irc or os.getenv("TWITCH_USE_IRC", "").lower() in ("1","true","yes"):
                            logger.info("Falling back to IRC listener because EventSub subscription failed or IRC forced.")
                            try:
                                nick = monitor.bot_login or getattr(broadcaster, 'login', None) or f"justinfan{random.randint(1000,9999)}"
                                asyncio.create_task(monitor._start_irc(nick, monitor.oauth_token, monitor.channel))
                            except Exception as e2:
                                logger.error("Failed to start IRC fallback: %s", e2)
                        else:
                            if last_exc:
                                raise last_exc

                except Exception as e:
                    logger.error(f"Error during setup: {e}")

            async def event_message(self, message):
                # Handle legacy IRC messages if they still exist or if mapped
                # But for EventSub, we might need event_channel_chat_message
                await self._process_message(message)

            async def event_channel_chat_message(self, event):
                # EventSub chat message event
                # event is twitchio.models.eventsub_.ChatMessage
                await self._process_message(event)

            async def _process_message(self, message):
                # Unified handler
                if not message:
                    return

                # Normalize content from several possible message shapes
                content = ""
                # EventSub ChatMessage: .message.text
                if hasattr(message, 'message') and getattr(message, 'message') is not None:
                    inner = message.message
                    if hasattr(inner, 'text'):
                        content = inner.text
                    elif isinstance(inner, str):
                        content = inner
                # twitchio Message or similar: .content or .text
                if not content and hasattr(message, 'content'):
                    content = getattr(message, 'content') or ""
                if not content and hasattr(message, 'text'):
                    content = getattr(message, 'text') or ""
                # Internal ChatMessage dataclass (IRC fallback): .message
                if not content and hasattr(message, 'message') and isinstance(getattr(message, 'message'), str):
                    content = getattr(message, 'message')

                # Normalize author/username from several shapes
                author_name = "unknown"
                if hasattr(message, 'sender') and getattr(message.sender, 'name', None):
                    author_name = message.sender.name
                elif hasattr(message, 'author') and getattr(message.author, 'name', None):
                    author_name = message.author.name
                elif hasattr(message, 'username') and getattr(message, 'username', None):
                    author_name = message.username
                elif hasattr(message, 'user') and getattr(message.user, 'name', None):
                    author_name = message.user.name

                # Extract badges/flags if available
                badges = []
                is_subscriber = False
                is_moderator = False

                # Check for tag dict commonly present on IRC messages
                tags = None
                if hasattr(message, 'tags'):
                    tags = message.tags
                elif hasattr(message, 'message') and hasattr(message.message, 'tags'):
                    tags = message.message.tags

                if isinstance(tags, dict):
                    # badges often in tags['badges'] as comma-separated 'badge/version'
                    bstr = tags.get('badges') or tags.get('badge-info') or tags.get('badges-raw')
                    if isinstance(bstr, str):
                        for part in bstr.split(','):
                            if not part:
                                continue
                            name = part.split('/')[0]
                            badges.append(name)
                    # mod/sub flags
                    if str(tags.get('mod', '')).lower() in ('1', 'true', 'yes') or str(tags.get('user-type', '')).lower() == 'mod':
                        is_moderator = True
                    if str(tags.get('subscriber', '')).lower() in ('1', 'true', 'yes'):
                        is_subscriber = True

                # Some message objects expose badges as a list
                if not badges and hasattr(message, 'badges'):
                    try:
                        raw = getattr(message, 'badges') or []
                        if isinstance(raw, (list, tuple)):
                            badges = list(raw)
                    except Exception:
                        pass

                # Check author object flags as a fallback
                if hasattr(message, 'author'):
                    author_obj = message.author
                    if getattr(author_obj, 'is_mod', False) or getattr(author_obj, 'isModerator', False):
                        is_moderator = True
                    if getattr(author_obj, 'is_subscriber', False) or getattr(author_obj, 'isSubscriber', False):
                        is_subscriber = True

                # Internal ChatMessage dataclass may already have username/message
                if hasattr(message, 'username') and hasattr(message, 'message') and isinstance(message.message, str):
                    author_name = getattr(message, 'username')
                    content = getattr(message, 'message')

                # Convert to internal ChatMessage format
                chat_msg = ChatMessage(
                    timestamp=time.time(),
                    username=author_name,
                    message=content,
                    platform="twitch",
                    is_subscriber=is_subscriber,
                    is_moderator=is_moderator,
                    badges=badges,
                )

                try:
                    monitor._emit_message(chat_msg)
                except Exception:
                    pass

        self._client = MonitorBot()

    async def _listen(self):
        """Start the client"""
        if self._client:
            # Ensure token is valid / refresh if necessary before starting
            try:
                valid = await self._ensure_token_valid()
                if not valid:
                    logger.warning("Proceeding without a validated token; EventSub subscriptions may fail")
            except Exception as e:
                logger.debug("Token validation step raised: %s", e)

            # start() runs the loop and will call login/start internals
            # If user forced IRC via env or constructor, skip EventSub client start and start IRC directly
            if self.use_irc or os.getenv("TWITCH_USE_IRC", "").lower() in ("1","true","yes"):
                # derive nick from validated login if available
                nick = getattr(self, 'bot_login', None) or f"justinfan{random.randint(1000,9999)}"
                asyncio.create_task(self._start_irc(nick, self.oauth_token, self.channel))
            else:
                await self._client.start(token=self.oauth_token)

    async def _start_irc(self, nick: str, token: Optional[str], channel: str):
        """Start the IRC fallback task (idempotent)."""
        if self._irc_task and not self._irc_task.done():
            return
        self._irc_task = asyncio.create_task(self._irc_worker(nick, token, channel))
        return self._irc_task

    async def _irc_worker(self, nick: str, token: Optional[str], channel: str):
        """Connect to Twitch IRC and emit messages to the monitor."""
        server = os.getenv("TWITCH_IRC_HOST", "irc.chat.twitch.tv")
        port = int(os.getenv("TWITCH_IRC_PORT", "6667"))

        # Normalize token for PASS; Twitch expects either 'oauth:...' or the raw token prefixed
        pass_token = token or os.getenv("TWITCH_OAUTH_TOKEN") or ""
        if not pass_token.startswith("oauth:") and pass_token:
            pass_token = f"oauth:{pass_token}"

        try:
            reader, writer = await asyncio.open_connection(server, port)
            # Request tags/membership/commands capabilities
            writer.write(b"CAP REQ :twitch.tv/tags twitch.tv/commands twitch.tv/membership\r\n")
            if pass_token:
                writer.write(f"PASS {pass_token}\r\n".encode("utf-8"))
            writer.write(f"NICK {nick}\r\n".encode("utf-8"))
            writer.write(f"JOIN #{channel}\r\n".encode("utf-8"))
            await writer.drain()

            logger.info("IRC fallback connected to %s:%s as %s, joined #%s", server, port, nick, channel)

            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    text = line.decode("utf-8", errors="ignore").strip()
                except Exception:
                    continue

                if not text:
                    continue

                # Respond to PINGs
                if text.startswith("PING"):
                    try:
                        pong = text.split(" ", 1)[1] if " " in text else ":tmi.twitch.tv"
                        writer.write(f"PONG {pong}\r\n".encode("utf-8"))
                        await writer.drain()
                    except Exception:
                        pass
                    continue

                # Parse PRIVMSG lines (basic)
                # Example: @tags :username!username@username.tmi.twitch.tv PRIVMSG #channel :the message here
                if " PRIVMSG " in text:
                    tags = None
                    rest = text
                    if text.startswith("@"):
                        try:
                            tags, rest = text.split(" ", 1)
                        except ValueError:
                            rest = text

                    # extract author
                    author = "unknown"
                    if rest.startswith(":"):
                        try:
                            prefix, msgpart = rest.split(" ", 1)
                            if "!" in prefix:
                                author = prefix.split("!", 1)[0].lstrip(":")
                            else:
                                author = prefix.lstrip(":")
                        except Exception:
                            msgpart = rest
                    else:
                        msgpart = rest

                    # extract message after the second ':' that prefixes the text
                    message_text = ""
                    if " :" in msgpart:
                        try:
                            # split once on ' :' to get the message body
                            message_text = msgpart.split(" :", 1)[1]
                        except Exception:
                            message_text = ""

                    chat_msg = ChatMessage(
                        timestamp=time.time(),
                        username=author,
                        message=message_text,
                        platform="twitch",
                        is_subscriber=False,
                        is_moderator=False,
                        badges=[],
                    )

                    try:
                        self._emit_message(chat_msg)
                    except Exception:
                        pass

        except Exception as e:
            logger.error("IRC fallback failed: %s", e)
        finally:
            try:
                if 'writer' in locals() and writer is not None:
                    try:
                        writer.close()
                    except Exception:
                        pass
                    try:
                        # Python 3.7+ has wait_closed on StreamWriter
                        if hasattr(writer, 'wait_closed'):
                            await writer.wait_closed()
                    except Exception:
                        pass
            except Exception:
                pass

    async def _disconnect(self):
        if self._client:
            try:
                await self._client.close()
            except Exception:
                pass

        # Cancel IRC fallback task if running
        if self._irc_task:
            try:
                self._irc_task.cancel()
                await self._irc_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.debug("Error cancelling IRC task: %s", e)


# Alias for compatibility
TwitchChat = TwitchChatMonitor


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    channel = sys.argv[1] if len(sys.argv) > 1 else "shroud"
    
    print(f"Monitoring {channel}...")
    
    # Simple test runner
    async def run_test():
        monitor = TwitchChatMonitor(channel)
        monitor.start()
        
        # Run for 10 seconds
        start = time.time()
        while time.time() - start < 10:
            msgs = monitor.get_all_messages()
            #for m in msgs:
                #print(f"[{m.username}]: {m.message}")
            await asyncio.sleep(0.1)
            
        monitor.stop()

    try:
        asyncio.run(run_test())
    except KeyboardInterrupt:
        pass
