from importlib import import_module
import spacy
from errant.annotator import Annotator
import requests
import shutil
from pathlib import Path
# ERRANT version
__version__ = '2.2.3'
# python -m spacy download lt_core_news_lg
# Load an ERRANT Annotator object for a given language


def load(lang, nlp=None):
    # Make sure the language is supported
    supported = {"lt"}
    if lang not in supported:
        raise Exception("%s is an unsupported or unknown language" % lang)
    # Load spacy
    try:
        nlp = spacy.load('lt_core_news_lg')
    except OSError:
        print('Downloading language model for the spaCy POS tagger\n'
              "(don't worry, this will only happen once)")
        from spacy.cli import download
        download('lt_core_news_lg')
        nlp = spacy.load('lt_core_news_lg')    # Load language edit merger
    merger = import_module("errant.%s.merger" % lang)

    folder = Path(__file__).parent / "lt" / 'resources' / 'hunspell'
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder / 'lt_LT_DML6.aff').exists():
        # dowload hunspell files
        url = 'https://clarin.vdu.lt/xmlui/bitstream/handle/20.500.11821/36/DML6_vs_JCL.zip?sequence=3&isAllowed=y'
        with requests.get(url, stream=True) as r:
            zipfile = folder / f"{lang}.zip"
            zipfile.touch(exist_ok=True)
            with zipfile.open(mode='wb') as f:
                shutil.copyfileobj(r.raw, f)
        # unpack a zip file
        shutil.unpack_archive(zipfile, zipfile.parents[0], "zip")
        shutil.copy(folder / 'DML6_vs_JCL' / 'lt_LT_DML6.aff', folder)
        shutil.copy(folder / 'DML6_vs_JCL' / 'lt_LT_DML6.dic', folder)
        zipfile.unlink()
        shutil.rmtree(folder / 'DML6_vs_JCL')

    # Load language edit classifier
    classifier = import_module("errant.%s.classifier" % lang)
    # The English classifier needs spacy
    if lang == "en": classifier.nlp = nlp

    # Return a configured ERRANT annotator
    return Annotator(lang, nlp, merger, classifier)