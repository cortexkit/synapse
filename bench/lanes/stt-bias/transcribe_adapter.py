"""Use the pinned reference binding and its existing C vocabulary extension."""

import ctypes as ct
import os

from safety import TARGET


class Transcribe:
    def __init__(self, model_path, boost):
        os.environ["TRANSCRIBE_LIBRARY"] = str(TARGET / "tc-build/src/libtranscribe.0.4.1.dylib")
        os.environ["TRANSCRIBE_BACKENDS"] = "cpu,metal"
        import transcribe_cpp as tc
        from transcribe_cpp import _generated, _lib

        class VocabularyStruct(ct.Structure):
            _fields_ = [("ext", _generated.transcribe_ext), ("boost", ct.c_float),
                        ("sounds_like", ct.POINTER(ct.c_char_p)),
                        ("sounds_like_term", ct.POINTER(ct.c_int32)),
                        ("n_sounds_like", ct.c_int32)]

        native_init = _lib.transcribe_parakeet_vocabulary_ext_init
        native_init.argtypes = [ct.POINTER(VocabularyStruct)]
        native_init.restype = None

        class Vocabulary(tc.FamilyExtension):
            _slot = "run"
            _kind = 0x42564B50
            _struct = VocabularyStruct
            _init = "transcribe_parakeet_vocabulary_ext_init"

            def _apply(self, ext):
                if ext.ext.size != ct.sizeof(VocabularyStruct) or ext.ext.kind != self._kind:
                    raise RuntimeError("Parakeet vocabulary C ABI mismatch")
                ext.boost = boost

        if tc.native_version() != "0.4.1" or not tc.native_commit().startswith("bcb639f"):
            raise RuntimeError("unexpected transcribe.cpp native runtime pin")
        self.model = tc.Model(str(model_path), backend="metal")
        if not self.model.supports("vocabulary"):
            raise RuntimeError("reference runtime does not support vocabulary")
        self.session = self.model.session(n_threads=4)
        self.vocabulary = Vocabulary()

    def generate(self, pcm, terms):
        result = self.session.run(pcm, language="en", vocabulary=terms or None,
                                  family=self.vocabulary if terms else None)
        return result.text

    def close(self):
        self.session.close()
        self.model.close()
