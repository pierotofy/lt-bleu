#!/usr/bin/env python
import ctranslate2
import sentencepiece as spm
import os
import argparse
from sacrebleu import corpus_bleu
import logging
sacrelogger = logging.getLogger('sacrebleu')
sacrelogger.setLevel(logging.CRITICAL)

nllb_langs = {
    "af":"afr_Latn",
    "ak":"aka_Latn",
    "am":"amh_Ethi",
    "ar":"arb_Arab",
    "as":"asm_Beng",
    "ay":"ayr_Latn",
    "az":"azj_Latn",
    "bm":"bam_Latn",
    "be":"bel_Cyrl",
    "bn":"ben_Beng",
    "bho":"bho_Deva",
    "bs":"bos_Latn",
    "bg":"bul_Cyrl",
    "ca":"cat_Latn",
    "ceb":"ceb_Latn",
    "cs":"ces_Latn",
    "ckb":"ckb_Arab",
    "tt":"crh_Latn",
    "cy":"cym_Latn",
    "da":"dan_Latn",
    "de":"deu_Latn",
    "el":"ell_Grek",
    "en":"eng_Latn",
    "eo":"epo_Latn",
    "et":"est_Latn",
    "eu":"eus_Latn",
    "ee":"ewe_Latn",
    "fa":"pes_Arab",
    "fi":"fin_Latn",
    "fr":"fra_Latn",
    "gd":"gla_Latn",
    "ga":"gle_Latn",
    "gl":"glg_Latn",
    "gn":"grn_Latn",
    "gu":"guj_Gujr",
    "ht":"hat_Latn",
    "ha":"hau_Latn",
    "he":"heb_Hebr",
    "hi":"hin_Deva",
    "hr":"hrv_Latn",
    "hu":"hun_Latn",
    "hy":"hye_Armn",
    "nl":"nld_Latn",
    "ig":"ibo_Latn",
    "ilo":"ilo_Latn",
    "id":"ind_Latn",
    "is":"isl_Latn",
    "it":"ita_Latn",
    "jv":"jav_Latn",
    "ja":"jpn_Jpan",
    "kn":"kan_Knda",
    "ka":"kat_Geor",
    "kk":"kaz_Cyrl",
    "km":"khm_Khmr",
    "rw":"kin_Latn",
    "ko":"kor_Hang",
    "ku":"kmr_Latn",
    "lo":"lao_Laoo",
    "lv":"lvs_Latn",
    "ln":"lin_Latn",
    "lt":"lit_Latn",
    "lb":"ltz_Latn",
    "lg":"lug_Latn",
    "lus":"lus_Latn",
    "mai":"mai_Deva",
    "ml":"mal_Mlym",
    "mr":"mar_Deva",
    "mk":"mkd_Cyrl",
    "mg":"plt_Latn",
    "mt":"mlt_Latn",
    "mni-Mtei":"mni_Beng",
    "mni":"mni_Beng",
    "mn":"khk_Cyrl",
    "mi":"mri_Latn",
    "ms":"zsm_Latn",
    "my":"mya_Mymr",
    "no":"nno_Latn",
    "ne":"npi_Deva",
    "ny":"nya_Latn",
    "om":"gaz_Latn",
    "or":"ory_Orya",
    "pl":"pol_Latn",
    "pt":"por_Latn",
    "ps":"pbt_Arab",
    "qu":"quy_Latn",
    "ro":"ron_Latn",
    "ru":"rus_Cyrl",
    "sa":"san_Deva",
    "si":"sin_Sinh",
    "sk":"slk_Latn",
    "sl":"slv_Latn",
    "sm":"smo_Latn",
    "sn":"sna_Latn",
    "sd":"snd_Arab",
    "so":"som_Latn",
    "es":"spa_Latn",
    "sq":"als_Latn",
    "sr":"srp_Cyrl",
    "su":"sun_Latn",
    "sv":"swe_Latn",
    "sw":"swh_Latn",
    "ta":"tam_Taml",
    "te":"tel_Telu",
    "tg":"tgk_Cyrl",
    "tl":"tgl_Latn",
    "th":"tha_Thai",
    "ti":"tir_Ethi",
    "ts":"tso_Latn",
    "tk":"tuk_Latn",
    "tr":"tur_Latn",
    "ug":"uig_Arab",
    "uk":"ukr_Cyrl",
    "ur":"urd_Arab",
    "uz":"uzn_Latn",
    "vi":"vie_Latn",
    "xh":"xho_Latn",
    "yi":"ydd_Hebr",
    "yo":"yor_Latn",
    "zh-CN":"zho_Hans",
    "zh":"zho_Hans",
    "zh-TW":"zho_Hant",
    "zu":"zul_Latn",
    "pa":"pan_Guru"
}

parser = argparse.ArgumentParser(description='Evaluate BLEU scores for LibreTranslate and argos-translate models')
parser.add_argument('--model',
    type=str,
    default="en-it",
    help='Language model to evaluate')
parser.add_argument('--sentence',
    type=int,
    default=None,
    help='Sentence index to evaluate. Default: all')

args = parser.parse_args()

from_code, to_code = args.model.split("-")
tgt_lang = nllb_langs[to_code]
src_lang = nllb_langs[from_code]

#ct_model_path = os.path.join("datasets/nllb/nllb-200-distilled-600M-int8")
#ct_model_path = os.path.join("datasets/nllb/ct2-nllb-200-distilled-1.2B-int8")
ct_model_path = os.path.join("datasets/nllb/nllb-200-3.3B-int8")
sp_model_path = os.path.join("datasets/nllb/flores200_sacrebleu_tokenizer_spm.model")

device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"

sp = spm.SentencePieceProcessor()
sp.load(sp_model_path)

translator = ctranslate2.Translator(ct_model_path, device)

datasets_path = os.path.join(os.path.dirname(__file__), "datasets")
flores_dataset = os.path.join(datasets_path, "flores200_dataset", "dev")
src_f = os.path.join(flores_dataset, nllb_langs[from_code] + ".dev")
tgt_f = os.path.join(flores_dataset, nllb_langs[to_code] + ".dev")

src_text = [line.rstrip('\n') for line in open(src_f, encoding="utf-8")]
tgt_text = [line.rstrip('\n') for line in open(tgt_f, encoding="utf-8")]
if args.sentence is not None:
    src_text = [src_text[args.sentence]]
    tgt_text = [tgt_text[args.sentence]]

src_text = [sent.strip() for sent in src_text]
tgt_prefix = [[tgt_lang]] * len(src_text)

# Subword the source sentences
src_subworded = sp.encode_as_pieces(src_text)
src_subworded = [[src_lang] + sent + ["</s>"] for sent in src_subworded]
print("First subworded source sentence:", src_subworded[0], sep="\n")

# Translate the source sentences
translator = ctranslate2.Translator(ct_model_path, device=device)
translations_subworded = translator.translate_batch(src_subworded, batch_type="tokens", max_batch_size=2024, beam_size=4, target_prefix=tgt_prefix)
translations_subworded = [translation[0]['tokens'] for translation in translations_subworded]
for translation in translations_subworded:
  if tgt_lang in translation:
    translation.remove(tgt_lang)

# Desubword the target sentences
translations = sp.decode(translations_subworded)

print("", src_text[0], translations[0], sep="\n• ")

bleu_scores = {}
bleu_scores[f"{from_code}-{to_code}"] = round(corpus_bleu(
        [translations[0]], [[x] for x in tgt_text], tokenize="flores200"
    ).score, 5)
print(f"{from_code}-{to_code}: {bleu_scores[f'{from_code}-{to_code}']}")
