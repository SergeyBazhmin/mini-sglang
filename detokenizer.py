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
"""

import dataclasses
import logging
from typing import Dict, List

import zmq
from transformers import AutoTokenizer

from common import BatchStrOut, BatchTokenIDOut, PortArgs, get_socket

logger = logging.getLogger("detokenizer")


@dataclasses.dataclass
class DecodeStatus:
    decode_ids: List[int]
    # decode_ids[surr_offset:read_offset] is text we already emitted. It is kept
    # only as context so the next token decodes the same way it would in a full decode.
    surr_offset: int
    read_offset: int


class Detokenizer:
    def __init__(self, model_path: str, port_args: PortArgs):
        context = zmq.Context(2)
        self.recv_from_scheduler = get_socket(
            context, zmq.PULL, port_args.detokenizer_input, bind=True
        )
        self.send_to_tokenizer = get_socket(
            context, zmq.PUSH, port_args.tokenizer_input, bind=False
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.decode_status: Dict[str, DecodeStatus] = {}

    def event_loop(self):
        while True:
            recv_obj = self.recv_from_scheduler.recv_pyobj()
            self.send_to_tokenizer.send_pyobj(self.handle_batch_token_id_out(recv_obj))

    def _decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def handle_batch_token_id_out(self, recv_obj: BatchTokenIDOut) -> BatchStrOut:
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
                text_deltas.append(new_text)
                del self.decode_status[rid]
            elif new_text and not new_text.endswith("�"):
                # Clean text: emit it and slide the window forward.
                text_deltas.append(new_text)
                s.surr_offset = s.read_offset
                s.read_offset = len(s.decode_ids)
            else:
                # Incomplete UTF-8 character: emit nothing and keep the offsets,
                # so the next call retries with more tokens. (SGLang also emits
                # the printable prefix here. Try adding that yourself.)
                text_deltas.append("")

        return BatchStrOut(
            rids=recv_obj.rids,
            text_deltas=text_deltas,
            finish_reasons=recv_obj.finish_reasons,
            completion_tokens=recv_obj.completion_tokens,
        )


def run_detokenizer_process(model_path: str, port_args: PortArgs):
    logging.basicConfig(level=logging.INFO, format="[detokenizer] %(message)s")
    Detokenizer(model_path, port_args).event_loop()
