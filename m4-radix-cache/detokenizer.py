"""Detokenizer process: turns streamed token ids into streamed text.

SGLang equivalent: python/sglang/srt/managers/detokenizer_manager.py
  - DecodeStatus                  (~line 77)
  - DetokenizerManager.event_loop (~line 184)
  - _decode_batch_token_id_output (~line 310)

Why not just decode each new token on its own?
  1. Byte-level BPE can split one UTF-8 character (é, emoji, Cyrillic, CJK)
     across several tokens. Decoding half of it gives "�".
  2. Some tokenizers decide spacing from neighbouring tokens.
So we decode a window, ids[surr:read] and ids[surr:], and emit the difference.

While a character is incomplete, the clean text before it still goes out, the way
SGLang emits the printable prefix. naive=True (M1's --naive-detokenizer) turns all
of this off and decodes each batch of new ids on its own, to show what goes wrong.
"""

import dataclasses
from typing import Dict, List

import zmq
from loguru import logger
from transformers import AutoTokenizer

from common import BatchStrOut, BatchTokenIDOut, PortArgs, get_socket, setup_logging


@dataclasses.dataclass
class DecodeStatus:
    decode_ids: List[int]
    # decode_ids[surr_offset:read_offset] is text we already emitted. It is kept
    # only as context so the next token decodes the same way it would in a full decode.
    surr_offset: int
    read_offset: int
    # How much of the text after read_offset was already emitted as a printable
    # prefix, while the window waited for the rest of a character.
    sent_len: int = 0


class Detokenizer:
    def __init__(self, model_path: str, port_args: PortArgs, naive: bool = False):
        context = zmq.Context(2)
        self.recv_from_scheduler = get_socket(
            context, zmq.PULL, port_args.detokenizer_input, bind=True
        )
        self.send_to_tokenizer = get_socket(
            context, zmq.PUSH, port_args.tokenizer_input, bind=False
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.decode_status: Dict[str, DecodeStatus] = {}
        self.naive = naive

    def event_loop(self):
        while True:
            recv_obj = self.recv_from_scheduler.recv_pyobj()
            self.send_to_tokenizer.send_pyobj(self.handle_batch_token_id_out(recv_obj))

    def _decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def handle_batch_token_id_out(self, recv_obj: BatchTokenIDOut) -> BatchStrOut:
        if self.naive:
            return self._handle_naive(recv_obj)
        text_deltas = []
        for rid, new_ids, prompt_tail, finish_reason in zip(
            recv_obj.rids,
            recv_obj.new_token_ids,
            recv_obj.prompt_tails,
            recv_obj.finish_reasons,
        ):
            s = self.decode_status.get(rid)
            if s is None:
                # First output for this request. Seed the window with the prompt
                # tail, and mark it as already read so we never emit it.
                tail = list(prompt_tail or [])
                s = DecodeStatus(decode_ids=tail, surr_offset=0, read_offset=len(tail))
                self.decode_status[rid] = s
            s.decode_ids.extend(new_ids)

            surr_text = self._decode(s.decode_ids[s.surr_offset : s.read_offset])
            read_text = self._decode(s.decode_ids[s.surr_offset :])
            new_text = read_text[len(surr_text) :]

            if finish_reason is not None:
                # Last message: flush whatever is left and free the state.
                text_deltas.append(new_text[s.sent_len :])
                del self.decode_status[rid]
            elif new_text and not new_text.endswith("�"):
                # Clean text: emit it and slide the window forward.
                text_deltas.append(new_text[s.sent_len :])
                s.surr_offset = s.read_offset
                s.read_offset = len(s.decode_ids)
                s.sent_len = 0
            else:
                # Incomplete UTF-8 character: keep the offsets, so the next call
                # retries with more tokens, but emit the clean text before it.
                # SGLang: find_printable_text in _decode_batch_token_id_output.
                printable = printable_prefix(new_text)
                text_deltas.append(printable[s.sent_len :])
                s.sent_len = max(s.sent_len, len(printable))

        return BatchStrOut(
            rids=recv_obj.rids,
            text_deltas=text_deltas,
            finish_reasons=recv_obj.finish_reasons,
            completion_tokens=recv_obj.completion_tokens,
        )

    def _handle_naive(self, recv_obj: BatchTokenIDOut) -> BatchStrOut:
        """Decode only the new ids, with no window. Split characters come out as
        "�", and tokenizers that drop a leading space lose spacing too."""
        return BatchStrOut(
            rids=recv_obj.rids,
            text_deltas=[self._decode(ids) for ids in recv_obj.new_token_ids],
            finish_reasons=recv_obj.finish_reasons,
            completion_tokens=recv_obj.completion_tokens,
        )


def printable_prefix(text: str) -> str:
    """The part of text before its first "�", cut back to the last space.

    Cutting at a space keeps the prefix stable: a tokenizer may still change how
    the end of a word decodes once more tokens arrive. CJK text has no spaces, so
    a newline or a CJK character ends a safe prefix too. (SGLang: find_printable_text.)
    """
    clean = text.split("�", 1)[0]
    if clean.endswith("\n") or (clean and ord(clean[-1]) >= 0x3000):
        return clean
    return clean[: clean.rfind(" ") + 1]


def run_detokenizer_process(model_path: str, port_args: PortArgs, naive: bool = False):
    setup_logging("detokenizer")
    detokenizer = Detokenizer(model_path, port_args, naive)
    logger.info("Ready" + (" (naive: no decoding window)" if naive else ""))
    detokenizer.event_loop()
