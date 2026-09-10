"""Trilingual bundles: alignment, RTL handling and terminology modes.

The package is multilingual by architecture (blueprint section 25), so a key
that exists in English and not in Arabic or French is a defect, not a task for
later.
"""

from __future__ import annotations

import json
from importlib import resources

import pytest

from visualmetrics.i18n.glossary import glossary_entries, lookup, normalize, search_terms
from visualmetrics.i18n.rtl import (
    axis_layout,
    contains_arabic,
    direction,
    flip_side,
    is_rtl,
    protect_latin,
    text_align,
)
from visualmetrics.i18n.translator import (
    LANGUAGES,
    RESOURCE_FILES,
    available_languages,
    get_translator,
    set_language,
)

BUNDLES = [(lang, name) for lang in LANGUAGES for name in RESOURCE_FILES]


def load(language: str, name: str) -> dict:
    path = resources.files("visualmetrics.i18n") / language / f"{name}.json"
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def leaves(data: dict, prefix: str = "") -> set[str]:
    out: set[str] = set()
    for key, value in data.items():
        full = f"{prefix}{key}"
        out |= leaves(value, full + ".") if isinstance(value, dict) else {full}
    return out


class TestBundleFiles:
    @pytest.mark.parametrize(("language", "name"), BUNDLES)
    def test_every_bundle_is_valid_json(self, language, name):
        assert isinstance(load(language, name), dict)

    @pytest.mark.parametrize("name", RESOURCE_FILES)
    def test_keys_are_aligned_across_all_three_languages(self, name):
        # The glossary keeps its language-neutral metadata (acronym, aliases,
        # concept link) in the English file only; the translated files carry the
        # term and its definition, and those must line up exactly.
        translatable = {"term", "definition"}
        keep = (lambda k: k.rsplit(".", 1)[-1] in translatable) if name == "glossary" else bool
        english = {k for k in leaves(load("en", name)) if keep(k)}
        for language in ("ar", "fr"):
            other = {k for k in leaves(load(language, name)) if keep(k)}
            missing = sorted(english - other)
            extra = sorted(other - english)
            assert not missing, f"{language}/{name}.json is missing {missing[:8]}"
            assert not extra, f"{language}/{name}.json has orphan keys {extra[:8]}"

    @pytest.mark.parametrize(("language", "name"), BUNDLES)
    def test_no_value_is_empty(self, language, name):
        def walk(data, prefix=""):
            for key, value in data.items():
                full = f"{prefix}{key}"
                if isinstance(value, dict):
                    walk(value, full + ".")
                elif not full.endswith((".acronym", ".aliases")):
                    # An empty acronym means "this term has none" - a real value.
                    assert str(value).strip(), f"{language}/{name}.json: {full} is empty"

        walk(load(language, name))

    def test_arabic_bundles_actually_contain_arabic(self):
        for name in RESOURCE_FILES:
            data = load("ar", name)
            values = [v for v in _values(data) if isinstance(v, str)]
            arabic = [v for v in values if contains_arabic(v)]
            assert len(arabic) > len(values) * 0.5, (
                f"ar/{name}.json looks mostly untranslated "
                f"({len(arabic)}/{len(values)} strings contain Arabic)"
            )

    def test_french_is_not_a_copy_of_english(self):
        for name in RESOURCE_FILES:
            en = load("en", name)
            fr = load("fr", name)
            en_flat = dict(_flat(en))
            fr_flat = dict(_flat(fr))
            shared = [k for k in en_flat if k in fr_flat and len(str(en_flat[k])) > 25]
            if not shared:
                continue
            identical = [k for k in shared if en_flat[k] == fr_flat[k]]
            assert len(identical) < len(shared) * 0.2, (
                f"fr/{name}.json repeats the English text for {len(identical)} of "
                f"{len(shared)} substantial keys"
            )


def _values(data):
    for value in data.values():
        if isinstance(value, dict):
            yield from _values(value)
        else:
            yield value


def _flat(data, prefix=""):
    for key, value in data.items():
        full = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from _flat(value, full + ".")
        else:
            yield full, value


class TestTranslator:
    def test_all_three_languages_are_available(self):
        assert set(available_languages()) == set(LANGUAGES)

    def test_a_missing_key_is_recorded_not_hidden(self):
        tr = get_translator("en")
        tr.missing.clear()
        tr.t("no.such.key.at.all", "fallback text")
        assert "no.such.key.at.all" in tr.missing

    def test_explicit_none_default_reports_the_gap(self):
        assert get_translator("en").t("no.such.key.at.all", None) is None

    def test_falls_back_to_english_rather_than_showing_a_raw_key(self):
        tr = get_translator("ar")
        text = tr.t("evidence.formal_proof.label")
        assert text and "evidence.formal_proof" not in text

    def test_parameters_are_interpolated(self):
        tr = get_translator("en")
        out = tr.t("proofs.ui.step_of", "Step {current} of {total}", current=2, total=9)
        assert "2" in out and "9" in out

    @pytest.mark.parametrize("language", LANGUAGES)
    def test_switching_language_is_global_and_reversible(self, language):
        set_language(language)
        assert get_translator().language == language
        set_language("en")
        assert get_translator().language == "en"

    def test_terminology_modes(self):
        translated = get_translator("ar", terminology="translated")
        bilingual = get_translator("ar", terminology="bilingual")
        english = get_translator("ar", terminology="english_technical")
        key = "glossary.heteroskedasticity.term"
        a, b, c = translated.term(key), bilingual.term(key), english.term(key)
        assert a and b and c
        assert contains_arabic(a)
        assert not contains_arabic(c), "english_technical must keep the Latin term"
        assert len(b) >= len(a), "bilingual shows both scripts"


class TestRTL:
    def test_only_arabic_is_right_to_left(self):
        assert is_rtl("ar")
        assert not is_rtl("en")
        assert not is_rtl("fr")

    def test_direction_and_alignment(self):
        assert direction("ar") == "rtl"
        assert direction("fr") == "ltr"
        assert text_align("ar") == "right"
        assert text_align("en") == "left"

    def test_sides_flip_only_for_arabic(self):
        assert flip_side("left", "ar") == "right"
        assert flip_side("left", "en") == "left"

    def test_latin_identifiers_are_isolated_inside_arabic_prose(self):
        arabic = "قيمة R-squared هي"
        protected = protect_latin(arabic, "ar")
        assert "⁦" in protected and "⁩" in protected, "LRI/PDI isolates expected"
        assert "R-squared" in protected

    def test_english_text_is_left_untouched(self):
        text = "R-squared is 0.87"
        assert protect_latin(text, "en") == text

    def test_axis_layout_mirrors_for_arabic(self):
        layout = axis_layout("ar")
        assert layout
        assert axis_layout("en") != layout


class TestGlossary:
    def test_every_entry_is_trilingual(self):
        entries = glossary_entries()
        assert len(entries) > 40
        for key, entry in entries.items():
            for language in ("en", "ar", "fr"):
                assert entry.terms.get(language), f"{key} has no {language} term"
                assert entry.definitions.get(language), f"{key} has no {language} definition"
            assert contains_arabic(entry.terms["ar"]), f"{key}: the Arabic term is not Arabic"

    def test_normalization_folds_arabic_orthography(self):
        assert normalize("الإحصاء") == normalize("الاحصاء")  # alef variants
        assert normalize("تحليـــل") == normalize("تحليل")  # tatweel
        assert normalize("عَيِّنَة") == normalize("عينة")  # diacritics

    def test_lookup_works_from_any_language(self):
        for probe in ("heteroskedasticity", "hétéroscédasticité", "عدم تجانس التباين"):
            entry = lookup(probe)
            assert entry is not None, f"{probe!r} did not resolve"

    def test_search_is_insensitive_to_arabic_orthography(self):
        entry = glossary_entries()["heteroskedasticity"]
        arabic = entry.terms["ar"]
        stretched = arabic.replace(" ", "ـ ")  # tatweel, as users often type it
        assert [e.key for e in search_terms(arabic)] == ["heteroskedasticity"]
        assert "heteroskedasticity" in [e.key for e in search_terms(stretched)]

    def test_search_works_from_latin_script_too(self):
        assert "regression" in " ".join(e.key for e in search_terms("regression"))
