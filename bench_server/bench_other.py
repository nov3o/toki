"""Train or encode with a non-BPE-family tokenizer, for speed/memory context only (outputs differ from BPE).

  python3 bench_other.py <method> <train|encode> <corpus> <vocab> <threads> <out_dir>
  method: sp_bpe | sp_unigram (SentencePiece), hf_wordpiece | hf_unigram (HF tokenizers)
Prints train_s= / encode_s= and tokens= (read by other.sh, timed under /usr/bin/time like run.sh).
"""
import os
import sys
import time

import common

method, step, corpus, vocab, threads, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
os.makedirs(out, exist_ok=True)
model = os.path.join(out, "model")

if method.startswith("sp_"):
    import sentencepiece as spm
    if step == "train":
        t = time.time()
        spm.SentencePieceTrainer.train(input=corpus, model_prefix=model, vocab_size=vocab, model_type=method[3:],
                                       num_threads=threads, character_coverage=1.0, max_sentence_length=1 << 16,
                                       byte_fallback=True, minloglevel=1)
        print(f"train_s={time.time() - t:.3f}")
    else:
        sp = spm.SentencePieceProcessor(model_file=model.replace("encode_", "train_") + ".model")
        t, n, batch = time.time(), 0, []
        with open(corpus, encoding="utf-8") as f:
            for line in f:
                batch.append(line)
                if len(batch) == 20000:
                    n += sum(map(len, sp.encode(batch, num_threads=threads))); batch = []
        n += sum(map(len, sp.encode(batch, num_threads=threads)))
        print(f"encode_s={time.time() - t:.3f} tokens={n}")
else:
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    pretok = "bert" if method == "hf_wordpiece" else "metaspace"
    if step == "train":
        if method == "hf_wordpiece":
            tk = Tokenizer(models.WordPiece(unk_token="[UNK]"))
            tk.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
            tr = trainers.WordPieceTrainer(vocab_size=vocab, special_tokens=["[UNK]"], show_progress=False)
        else:
            tk = Tokenizer(models.Unigram())
            tk.pre_tokenizer = pre_tokenizers.Metaspace()
            tr = trainers.UnigramTrainer(vocab_size=vocab, special_tokens=["<unk>"], unk_token="<unk>", show_progress=False)
        t = time.time()
        tk.train([corpus], tr)
        print(f"train_s={time.time() - t:.3f}")
        tk.save(model + ".json")
    else:
        tk = Tokenizer.from_file(model.replace("encode_", "train_") + ".json")
        t, n, batch = time.time(), 0, []
        for p in common.pieces(corpus, pretok):
            batch.append(p)
            if len(batch) == 32:
                n += sum(len(e.ids) for e in tk.encode_batch_fast(batch, add_special_tokens=False)); batch = []
        n += sum(len(e.ids) for e in tk.encode_batch_fast(batch, add_special_tokens=False))
        print(f"encode_s={time.time() - t:.3f} tokens={n}")
