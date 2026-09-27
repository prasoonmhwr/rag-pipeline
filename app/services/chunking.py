import tiktoken

def chunk_recursive(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    separators = ["\n\n", "\n", ". ", " "]
    enc = tiktoken.get_encoding("cl100k_base")

    def _split(text: str, seps: list[str]) -> list[str]:
        if len(enc.encode(text)) <= chunk_size:
            return [text]
        if not seps:
            tokens = enc.encode(text)
            return [enc.decode(tokens[i:i+chunk_size]) for i in range(0, len(tokens), chunk_size)]
        sep, rest_seps = seps[0], seps[1:]
        parts = text.split(sep)
        results, buffer = [], ""
        for part in parts:
            candidate = buffer + sep + part if buffer else part
            if len(enc.encode(candidate)) <= chunk_size:
                buffer = candidate
            else:
                if buffer:
                    results.extend(_split(buffer, rest_seps))
                buffer = part
        if buffer:
            results.extend(_split(buffer, rest_seps))
        return results

    return _split(text, separators)