from typing import Optional

from core.llm_handler import LLMHandler
from core.query_analyzer import QueryAnalyzer, QueryAnalysis
from core.answer_generator import AnswerGenerator
from index.hybrid_searcher import HybridSearcher
from db.database import Database
from config_loader import cfg



_GREETINGS = {"hi", "hello", "hey", "salam", "assalam", "assalamualaikum"}
_FAREWELLS  = {"bye", "goodbye", "thanks", "thank you", "shukriya", "shukria"}
_META       = {"what can you do", "help", "what do you know", "?"}

_GREETING_RESPONSE = (
    "Hello! I'm the NED University Admissions Assistant. "
    "I can answer questions about programs, eligibility, fees, scholarships, "
    "deadlines, facilities, and more. What would you like to know?"
)
_FAREWELL_RESPONSE = (
    "You're welcome! Feel free to return if you have more questions about admissions."
)
_META_RESPONSE = (
    "I can help with:\n"
    "• Degree programs and departments\n"
    "• Eligibility criteria (BE, MS, PhD)\n"
    "• Fee structures and payment information\n"
    "• Scholarships and financial aid\n"
    "• Admission deadlines and schedules\n"
    "• Required documents\n"
    "• Hostel facilities\n"
    "• PhD supervisor lookup\n\n"
    "Just ask your question!"
)

# Keywords that strongly indicate a non-university / non-admissions query.
# Kept intentionally broad — the LLM fallback handles edge cases.

_OFF_TOPIC_KEYWORDS = {
    # Sports
    "fifa", "world cup", "cricket", "ipl", "psl", "football", "soccer",
    "basketball", "nba", "tennis", "olympics",
    # Entertainment
    "movie", "film", "drama", "series", "netflix", "youtube", "tiktok",
    "song", "music", "concert", "actor", "actress",
    # Politics / news (unrelated to university)
    "election", "prime minister", "president", "government policy",
    "war", "military", "army",
    # Tech / general
    "chatgpt", "openai", "artificial general intelligence",
    "stock", "bitcoin", "crypto", "forex",
    # Food / lifestyle
    "recipe", "restaurant", "weather", "horoscope",
}

_OFF_TOPIC_RESPONSE = (
    "I'm the NED University Admissions Assistant and can only help with "
    "questions related to NED University — such as programs, eligibility, "
    "fees, scholarships, deadlines, facilities, and campus services.\n\n"
    "For anything else, please use a general search engine. "
    "Is there anything about NED admissions I can help you with?"
)


def _is_off_topic(query: str) -> bool:
    """
    Lightweight keyword check for clearly off-topic queries.
    Returns True only when confident the query has nothing to do with
    university admissions. Errs on the side of letting queries through to RAG.
    """
    q_lower = query.lower()
    # Must match a keyword AND the query should NOT contain any university signal
    university_signals = {
        "ned", "neduet", "university", "admission", "program", "programme",
        "degree", "fee", "scholarship", "hostel", "department", "faculty",
        "eligib", "ms ", "phd", "be ", "bsc", "semester", "campus",
        "supervisor", "thesis", "enroll", "enrol", "apply", "test", "exam",
    }
    has_off_topic  = any(kw in q_lower for kw in _OFF_TOPIC_KEYWORDS)
    has_uni_signal = any(sig in q_lower for sig in university_signals)

    return has_off_topic and not has_uni_signal


class RAGPipeline:

    def __init__(
        self,
        searcher:  HybridSearcher,
        llm:       LLMHandler,
        db:        Database,
    ):
        self.searcher  = searcher
        self.llm       = llm
        self.db        = db
        self.analyzer  = QueryAnalyzer(llm)
        self.generator = AnswerGenerator(llm)


    def _check_conversational(self, query: str) -> Optional[str]:
        """Return a canned response if the query is purely conversational."""
        q = query.lower().strip().rstrip("!.,?")
        if q in _GREETINGS:
            return _GREETING_RESPONSE
        if q in _FAREWELLS:
            return _FAREWELL_RESPONSE
        if q in _META:
            return _META_RESPONSE
        return None


    def _get_context(self, session_id: str) -> str:
        """Fetch last N QA pairs from DB and format as context string."""
        history = self.db.get_recent_history(
            session_id, n_pairs=cfg.get("context_pairs", 2)
        )
        if not history:
            return ""
        lines = []
        for role, message in history:
            label = "User" if role == "user" else "Assistant"
            # Truncate long messages
            msg = message[:300] + "…" if len(message) > 300 else message
            lines.append(f"{label}: {msg}")
        return "\n".join(lines)

    def process(self, query: str, session_id: str) -> str:
        """
        Process a user query and return a response.

        Args:
            query:      Raw user input string.
            session_id: Session identifier for conversation memory.

        Returns:
            Answer string.
        """
        query = query.strip()

        canned = self._check_conversational(query)
        if canned:
            self.db.save_message(session_id, "user",      query,  intent="CONVERSATIONAL")
            self.db.save_message(session_id, "assistant", canned, intent="CONVERSATIONAL")
            return canned

        if _is_off_topic(query):
            self.db.save_message(session_id, "user",      query,              intent="OFF_TOPIC")
            self.db.save_message(session_id, "assistant", _OFF_TOPIC_RESPONSE, intent="OFF_TOPIC")
            return _OFF_TOPIC_RESPONSE

        context = self._get_context(session_id)
        analysis: QueryAnalysis = self.analyzer.analyze(query, context)

        initial_k = cfg.get("initial_top_k", 20)
        if analysis.is_comparison:
            initial_k = max(initial_k, 30) 
        results = self.searcher.search(
            query         = analysis.get_search_query(),
            key_terms     = analysis.key_terms,
            initial_k     = initial_k,
            final_k       = analysis.get_final_k(),
            prefer_tables = (analysis.needs_table_data or analysis.is_numerical),
        )

        answer = self.generator.generate(
            original_query       = query,
            analysis             = analysis,
            retrieved_chunks     = results,
            conversation_context = context,
        )

        sources = list({r["source_file"] for r in results})
        self.db.save_message(session_id, "user",      query,  sources=sources)
        self.db.save_message(session_id, "assistant", answer, sources=sources)

        return answer
