"""A sampled reply must be served the way /mimic serves it, or the A/B judges the wrong thing."""

import sample_persona as sp

ADAPTERS = ["mimic_user2", "mimic_user1"]


def test_a_persona_is_its_live_adapter_on_the_multi_model():
    model, lora = sp.resolve("mimic_user1", ADAPTERS, 0.85)
    assert model == "mimic-multi"
    assert lora == [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 0.85}]


def test_base_is_the_multi_model_with_nothing_applied():
    model, lora = sp.resolve("base", ADAPTERS, 0.85)
    assert model == "mimic-multi" and all(s["scale"] == 0.0 for s in lora)


def test_candidate_and_other_presets():
    assert sp.resolve("candidate", ADAPTERS, 0.85) == ("mimic-candidate", [{"id": 0, "scale": 0.85}])
    assert sp.resolve("chat-liberal", ADAPTERS, 0.85) == ("chat-liberal", None)


def test_each_model_gets_the_prompt_shape_it_was_trained_on():
    old = [{"role": "system", "content": "old"}, {"role": "user", "content": "[erin]: hi"}]
    new = [{"role": "user", "content": "[erin]: hi"}]
    for msgs in (old, new):  # whichever way eval.jsonl was built
        assert sp.shaped(msgs, None) == new
        assert sp.shaped(msgs, "You are mimic_user1") == [{"role": "system", "content": "You are mimic_user1"}] + new
    assert old[0]["content"] == "old"  # the shared prompt is left alone for the other model


def test_whose_manifest_describes_a_model():
    assert sp.manifest_name("mimic_user1", ADAPTERS) == "mimic_user1"
    assert sp.manifest_name("candidate", ADAPTERS) == "candidate"
    assert sp.manifest_name("base", ADAPTERS) is None
    assert sp.manifest_name("chat-liberal", ADAPTERS) is None
