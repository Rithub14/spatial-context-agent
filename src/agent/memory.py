"""Session memory — persists conversation turns and visited landmarks in PostgreSQL."""

import logging
import uuid

from sqlalchemy.orm import Session

from src.db.models import ConversationTurn, UserSession

logger = logging.getLogger(__name__)


class SessionMemory:
    """Manages conversation history and visited landmark tracking for a session.

    Backed by PostgreSQL so memory persists across API restarts and can be
    queried for analytics.
    """

    def __init__(self, db: Session) -> None:
        """Initialise with a database session.

        Args:
            db: SQLAlchemy session.
        """
        self.db = db

    def create_session(self, persona: str = "historian", language: str = "en") -> str:
        """Create a new session and return its UUID string.

        Args:
            persona: Tour guide persona for the session.
            language: Language code for responses.

        Returns:
            Session UUID as string.
        """
        session = UserSession(persona=persona, language=language)
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        logger.info("Created session %s (persona=%r)", session.id, persona)
        return str(session.id)

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        landmark_name: str | None = None,
    ) -> None:
        """Append a conversation turn to the session.

        Args:
            session_id: UUID string of the session.
            role: "user" or "assistant".
            content: Message content.
            landmark_name: Optional landmark associated with this turn.
        """
        try:
            turn = ConversationTurn(
                session_id=uuid.UUID(session_id),
                role=role,
                content=content,
                landmark_name=landmark_name,
            )
            self.db.add(turn)
            self.db.commit()
        except Exception as exc:
            logger.error("Failed to save conversation turn: %s", exc)
            self.db.rollback()

    def get_history(self, session_id: str, last_n: int = 10) -> list[dict]:
        """Return the last N turns for a session, formatted for LangChain.

        Args:
            session_id: UUID string of the session.
            last_n: Maximum number of turns to return.

        Returns:
            List of dicts with 'role' and 'content' keys.
        """
        try:
            turns = (
                self.db.query(ConversationTurn)
                .filter(ConversationTurn.session_id == uuid.UUID(session_id))
                .order_by(ConversationTurn.created_at.desc())
                .limit(last_n)
                .all()
            )
            return [{"role": t.role, "content": t.content} for t in reversed(turns)]
        except Exception as exc:
            logger.error("Failed to load conversation history: %s", exc)
            return []

    def get_visited_landmarks(self, session_id: str) -> list[str]:
        """Return names of landmarks visited in this session.

        Args:
            session_id: UUID string of the session.

        Returns:
            Unique list of landmark names seen by the assistant in this session.
        """
        try:
            turns = (
                self.db.query(ConversationTurn)
                .filter(
                    ConversationTurn.session_id == uuid.UUID(session_id),
                    ConversationTurn.role == "assistant",
                    ConversationTurn.landmark_name.isnot(None),
                )
                .all()
            )
            seen = []
            for t in turns:
                if t.landmark_name and t.landmark_name not in seen:
                    seen.append(t.landmark_name)
            return seen
        except Exception as exc:
            logger.error("Failed to load visited landmarks: %s", exc)
            return []

    def has_visited(self, session_id: str, landmark_name: str) -> bool:
        """Return True if this landmark was already narrated in the session.

        Args:
            session_id: UUID string of the session.
            landmark_name: Landmark name to check.

        Returns:
            True if already visited, False otherwise.
        """
        return landmark_name in self.get_visited_landmarks(session_id)

    def get_session(self, session_id: str) -> UserSession | None:
        """Return the UserSession ORM object, or None if not found.

        Args:
            session_id: UUID string of the session.
        """
        try:
            return self.db.get(UserSession, uuid.UUID(session_id))
        except Exception:
            return None
