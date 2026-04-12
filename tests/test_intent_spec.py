from prompt_refinery.core import (
    PromptCandidate,
    SlotSupport,
    build_intent_spec_from_retrieval,
    parse_massive_annot_utt,
)


def test_parse_massive_annot_utt_extracts_slot_pairs() -> None:
    parsed = parse_massive_annot_utt(
        "book a flight to [destination : Istanbul] on [date : Friday]"
    )

    assert parsed == [
        {"slot": "destination", "value": "Istanbul"},
        {"slot": "date", "value": "Friday"},
    ]


def test_intent_spec_uses_weighted_retrieval_data() -> None:
    prompt_candidates = [
        PromptCandidate(
            row_id=11,
            act="Write technical migration memo",
            prompt="...",
            for_devs=1,
            record_type="instruction",
            contributor="alice",
            score=0.90,
        ),
        PromptCandidate(
            row_id=22,
            act="Write marketing ad",
            prompt="...",
            for_devs=0,
            record_type="chat",
            contributor="bob",
            score=0.20,
        ),
    ]
    slot_support = [
        SlotSupport(
            row_id=1,
            source="MASSIVE",
            locale="tr-TR",
            intent="weather",
            utt="hava nasil",
            slots=[{"slot": "city", "value": "Istanbul"}],
            score=0.9,
        ),
        SlotSupport(
            row_id=2,
            source="MASSIVE",
            locale="en-US",
            intent="weather",
            utt="weather tomorrow",
            slots=[{"slot": "date", "value": "tomorrow"}],
            score=0.2,
        ),
    ]

    spec = build_intent_spec_from_retrieval(
        user_text="Write migration plan",
        prompt_candidates=prompt_candidates,
        slot_support=slot_support,
        quality_targets=["Fully specified output"],
    )

    assert spec.objective == "Write technical migration memo"
    assert spec.deliverable_type == "instruction"
    assert spec.audience == "developers"
    assert spec.language == "tr-TR"
    assert spec.dominant_intents == ["weather"]
    assert spec.relevant_slots == ["city", "date"]
    assert spec.quality_targets == ["Fully specified output"]
