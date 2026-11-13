"""Pinned reference weights on MLX; no decoder modifications."""

from typing import Optional


class MLX:
    def __init__(self, snapshot, system, language: Optional[str] = "English"):
        import mlx.core as mx
        from mlx_audio.stt import load
        mx.set_memory_limit(16 * 2**30)
        mx.set_cache_limit(512 * 2**20)
        mx.random.seed(0)
        self.mx = mx
        self.system = system
        self.language = language
        self.detected_languages = []
        self.model = load(snapshot, strict=True)

    def generate(self, pcm, terms):
        if self.model is None:
            raise RuntimeError("model has been closed")
        options = {"max_tokens": 1024, "temperature": 0.0,
                   "verbose": False, "stream": False}
        if self.system.startswith("qwen"):
            options.update(language=self.language, system_prompt=", ".join(terms) if terms else None)
        else:
            prompt = "transcribe the speech to text."
            if terms:
                prompt += " Keywords: " + ", ".join(terms)
            options.update(prompt=prompt, task="asr")
        result = self.model.generate(pcm, **options)
        self.mx.synchronize()
        if result.generation_tokens >= options["max_tokens"]:
            raise RuntimeError("generation reached token limit; refusing a truncated transcript")
        languages = getattr(result, "language", None)
        self.detected_languages = languages if isinstance(languages, list) else [languages]
        return result.text

    def close(self):
        self.model = None
        self.mx.clear_cache()
