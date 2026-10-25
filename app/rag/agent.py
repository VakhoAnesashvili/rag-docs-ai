import logging

from app.rag import prompts
from app.rag.llm import LLMClient, LLMError
from app.rag.pipeline import RagAnswer
from app.rag.tools import TOOL_SPECS, ToolBox
from app.search.hybrid import HybridSearcher
from app.rag.prompts import _looks_like_refusal
log = logging.getLogger(__name__)


class AgentPipeline:
    def __init__(self, searcher: HybridSearcher, llm: LLMClient, max_steps: int = 5) -> None:
        self.searcher, self.llm, self.max_steps = searcher, llm, max_steps

    def ask(self, question: str) -> RagAnswer:
        box = ToolBox(self.searcher)
        system = prompts.AGENT_SYSTEM.format(tools=TOOL_SPECS, max_steps=self.max_steps)
        result = RagAnswer(question, None, True)
        transcript = f"QUESTION: {question}\n"

        for step in range(1, self.max_steps + 1):
            force = step == self.max_steps
            prompt = transcript + ("\nThis is your LAST step: you must give the final answer now.\n"
                                   if force else "\nYour next JSON action:")
            try:
                data = self.llm.generate_json(prompt, system=system)
            except LLMError as e:
                result.error = str(e)
                break

            if data.get("action") == "final" or force and "answer" in data:
                self._finalize(result, data, box)
                break

            tool, args = data.get("tool", ""), data.get("args", {})
            observation = box.call(tool, args)
            result.trace.append({"step": step, "tool": tool, "args": args,
                                 "observation_preview": observation[:200]})
            transcript += f"\nACTION: {tool} {args}\nOBSERVATION:\n{observation}\n"

        result.retrieved = list(box.passages.values())
        return result

    @staticmethod
    def _finalize(result: RagAnswer, data: dict, box: ToolBox) -> None:
        cited = [n for n in data.get("citations", []) if isinstance(n, int)]
        valid = [n for n in dict.fromkeys(cited) if n in box.passages]   # VERIFY
        sufficient = bool(data.get("sufficient")) and bool(data.get("answer"))
        if sufficient and _looks_like_refusal(data.get("answer", "")):
            sufficient = False
        result.abstained = not sufficient
        result.answer = data.get("answer") if sufficient else None
        result.conflict = bool(data.get("conflict"))
        result.conflict_note = data.get("conflict_note")
        result.sources = [box.passages[n] for n in valid]