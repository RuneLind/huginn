import re


class SessionMarkdownSplitter:
    """Splits Claude Code session markdown into multi-turn chunks.

    Groups consecutive user+assistant turn pairs into chunks of ~target_chars,
    with 1-turn overlap for context continuity. Strips noise (thinking blocks,
    tool-use lines) to maximize embedding quality.

    Returns the same format as MarkdownHeadingSplitter: list[{"text": str, "heading": str | None}]
    """

    _TURN_HEADING_RE = re.compile(r'^## (User|Assistant)\s*$', re.MULTILINE)
    _THINKING_BLOCK_RE = re.compile(
        r'<details>\s*<summary>\s*Thinking\s*</summary>.*?</details>',
        re.DOTALL | re.IGNORECASE,
    )
    _TOOL_LINE_RE = re.compile(r'^- \[Tool:.*$', re.MULTILINE)
    _CONSECUTIVE_BLANKS_RE = re.compile(r'\n{3,}')

    def __init__(self, target_chars=2500, min_chars=400):
        self.target_chars = target_chars
        self.min_chars = min_chars

    def split(self, text):
        """Split session markdown into multi-turn chunks.

        Returns list of {"text": str, "heading": str | None}.
        """
        turns = self._parse_turns(text)
        if not turns:
            cleaned = self._clean_turn_text(text)
            if cleaned.strip():
                return [{"text": cleaned.strip(), "heading": None}]
            return []

        # Group turns into exchanges (user + following assistant turns)
        exchanges = self._group_exchanges(turns)
        if not exchanges:
            return []

        # Merge exchanges into chunks targeting target_chars
        chunks = self._merge_exchanges(exchanges)
        return chunks

    def _parse_turns(self, text):
        """Parse markdown into list of {"role": str, "text": str}."""
        matches = list(self._TURN_HEADING_RE.finditer(text))
        if not matches:
            return []

        turns = []
        for i, match in enumerate(matches):
            role = match.group(1).lower()
            body_start = match.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[body_start:body_end].strip()
            if body:
                turns.append({"role": role, "text": body})
        return turns

    def _group_exchanges(self, turns):
        """Group turns into exchanges: each starts with a user turn and includes
        following assistant turns until the next user turn.

        Returns list of {"user": str, "assistant": str, "turn_num": int}.
        """
        exchanges = []
        current_user = None
        current_assistant_parts = []
        turn_num = 0

        for turn in turns:
            if turn["role"] == "user":
                # Save previous exchange
                if current_user is not None:
                    turn_num += 1
                    assistant_text = "\n\n".join(current_assistant_parts)
                    exchanges.append({
                        "user": current_user,
                        "assistant": self._clean_turn_text(assistant_text),
                        "turn_num": turn_num,
                    })
                current_user = turn["text"]
                current_assistant_parts = []
            elif current_user is not None:
                # Only collect assistant turns that follow a user turn
                current_assistant_parts.append(turn["text"])

        # Save last exchange
        if current_user is not None:
            turn_num += 1
            assistant_text = "\n\n".join(current_assistant_parts)
            exchanges.append({
                "user": current_user,
                "assistant": self._clean_turn_text(assistant_text),
                "turn_num": turn_num,
            })

        return exchanges

    def _clean_turn_text(self, text):
        """Strip noise from assistant text: thinking blocks, tool lines."""
        text = self._THINKING_BLOCK_RE.sub('', text)
        text = self._TOOL_LINE_RE.sub('', text)
        text = self._CONSECUTIVE_BLANKS_RE.sub('\n\n', text)
        return text.strip()

    def _format_exchange(self, exchange):
        """Format a single exchange as readable text."""
        parts = [f"**User:** {exchange['user']}"]
        if exchange["assistant"]:
            parts.append(f"**Assistant:** {exchange['assistant']}")
        return "\n\n".join(parts)

    def _merge_exchanges(self, exchanges):
        """Merge exchanges into chunks, targeting target_chars with 1-exchange overlap."""
        if not exchanges:
            return []

        # Pre-format all exchanges once (avoids formatting each exchange twice)
        formatted_cache = [self._format_exchange(ex) for ex in exchanges]

        chunks = []
        group_start = 0

        while group_start < len(exchanges):
            # Build up a group of exchanges until we hit target_chars
            group = []
            group_chars = 0

            for i in range(group_start, len(exchanges)):
                exchange_chars = len(formatted_cache[i])

                # Always include at least one exchange per chunk
                if not group:
                    group.append(i)
                    group_chars += exchange_chars
                    continue

                # Would adding this exchange exceed target?
                if group_chars + exchange_chars > self.target_chars:
                    break

                group.append(i)
                group_chars += exchange_chars

            # Format the chunk
            first_turn = exchanges[group[0]]["turn_num"]
            last_turn = exchanges[group[-1]]["turn_num"]

            chunk_text = "\n\n---\n\n".join(formatted_cache[i] for i in group)

            if first_turn == last_turn:
                heading = f"Turn {first_turn}"
            else:
                heading = f"Turns {first_turn}-{last_turn}"

            # Skip chunks that are too short (just noise remnants)
            if len(chunk_text) >= self.min_chars or not chunks:
                chunks.append({"text": chunk_text, "heading": heading})

            # Advance with 1-exchange overlap (but stop if overlap would only repeat the last exchange)
            if len(group) > 1 and group[-1] < len(exchanges) - 1:
                group_start = group[-1]  # overlap: last exchange of this group = first of next
            else:
                group_start = group[-1] + 1  # no overlap: single exchange or last exchange reached

        return chunks
