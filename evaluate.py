#!/usr/bin/env python
import time
import os
from argostranslate import package as packageManager
from sacrebleu import corpus_bleu
import sentencepiece
import ctranslate2
import threading
import urllib.request
import argparse
import logging

sacrelogger = logging.getLogger('sacrebleu')
sacrelogger.setLevel(logging.CRITICAL)

parser = argparse.ArgumentParser(description='Evaluate BLEU scores for LibreTranslate and argos-translate models')
parser.add_argument('--models',
    type=str,
    default="all",
    help='Language models to evaluate (comma separated). Default: all')
parser.add_argument('--sentence',
    type=int,
    default=None,
    help='Sentence index to evaluate. Default: all')
parser.add_argument('--text',
    type=str,
    default=None,
    help='Text to evaluate. Default: none')

args = parser.parse_args()

datasets_path = os.path.join(os.path.dirname(__file__), "datasets")
if not os.path.isdir(datasets_path):
    os.mkdir(datasets_path)

flores_dataset = os.path.join(datasets_path, "flores200_dataset", "dev")
if not os.path.isdir(flores_dataset):
    # Download first
    print("Downloading flores200 dataset...")
    fname = os.path.join(datasets_path, "flores200.tar.gz")
    flores_url = "https://tinyurl.com/flores200dataset"
    urllib.request.urlretrieve(flores_url, fname)

    import tarfile
    with tarfile.open(fname) as f:
        f.extractall(datasets_path)
    
    if os.path.isfile(fname):
        os.unlink(fname)

    if not os.path.isdir(flores_dataset):
        print(f"Cannot download flores200. Please manually download it from {flores_url} and place it in {flores_dataset}")
        exit(1)

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
bleu_scores = {}

def translator(package_path):
    device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    model = ctranslate2.Translator(f"{package_path}/model", device=device, compute_type="auto")
    tokenizer = sentencepiece.SentencePieceProcessor(
        f"{package_path}/sentencepiece.model"
    )
    return {"model": model, "tokenizer": tokenizer}

def encode(text, tokenizer):
    return tokenizer.Encode(text, out_type=str)

def decode(tokens, tokenizer):
    return tokenizer.Decode(tokens)

def process_flores(pkg):
    data = translator(pkg.package_path)

    src_f = os.path.join(flores_dataset, nllb_langs[pkg.from_code] + ".dev")
    tgt_f = os.path.join(flores_dataset, nllb_langs[pkg.to_code] + ".dev")

    src_text = [line.rstrip('\n') for line in open(src_f, encoding="utf-8")]
    tgt_text = [line.rstrip('\n') for line in open(tgt_f, encoding="utf-8")]
    
    if args.sentence is not None:
        src_text = [src_text[args.sentence]]
        tgt_text = [tgt_text[args.sentence]]
    
    if args.text is not None:
        src_text = [args.text]
        tgt_text = [""]
        

    translation_obj = data["model"].translate_batch(
        encode(src_text, data["tokenizer"]),
        beam_size=4, # same as argos
        return_scores=False, # speed up
    )

    translated_text = [
        decode(tokens.hypotheses[0], data["tokenizer"])
        for tokens in translation_obj
    ]
    
    print(f"• {src_text[0]}\n• {tgt_text[0]}\n• {' '.join(translated_text)}")

    bleu_scores[f"{pkg.from_code}-{pkg.to_code}"] = round(corpus_bleu(
        translated_text, [[x] for x in tgt_text], tokenize="flores200"
    ).score, 5)

    print(f"{pkg.from_code}-{pkg.to_code}: {bleu_scores[f'{pkg.from_code}-{pkg.to_code}']}")

packages = packageManager.get_installed_packages()

if args.models != "all":
    models = [{'from': m.split("-")[0], 'to': m.split("-")[1]} for m in args.models.split(",")]
    
    eval_packages = []
    for p in packages:
        for m in models:
            if p.from_code == m['from'] and p.to_code == m['to']:
                eval_packages.append(p)
                break
    print(f"Found {len(eval_packages)} matching models (out of {len(packages)})")
    packages = eval_packages

print("Evaluating:")
print(packages)

promises = []
for package in packages:

    THREAD = threading.Thread(target=process_flores, args=[package,])
    promises.append(THREAD)

executing = 0
max_concurrency = os.cpu_count()

for x in promises:
    executing = sum(1 for x in promises if x.is_alive())
    while executing >= max_concurrency:
        executing = sum(1 for x in promises if x.is_alive())
        time.sleep(1)
    if executing <= max_concurrency:
        x.start()

