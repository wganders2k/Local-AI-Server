"""
render_as_served's split of an example into the served prompt and the trained
completion. A wrong split is silent: the run trains, on replies missing their
first characters or on a prompt the server never sends.

Run:  .venv-test/bin/python -m pytest tests -q
"""

import pytest

import train_worker as tw


class GemmaLikeTokenizer:
    """Renders like Gemma 4 with thinking off: the generation prompt carries an
    empty thought channel that a rendered assistant turn does not."""

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        out = "".join(f"<|turn>{m['role']}\n{m['content']}<turn|>\n" for m in msgs)
        if add_generation_prompt:
            out += "<|turn>model\n<|channel>thought\n<channel|>"
        return out.replace("<|turn>assistant", "<|turn>model")


PROMPT = [{"role": "user", "content": "[bob]: nice one"}]


def test_completion_is_the_reply_and_end_of_turn():
    split = tw.split_as_served(GemmaLikeTokenizer(), PROMPT, [{"role": "assistant", "content": "cheers"}], {})
    assert split["prompt"].endswith("<|channel>thought\n<channel|>")
    assert split["completion"] == "cheers<turn|>\n"


def test_reply_sharing_a_prefix_with_the_thought_marker_keeps_every_character():
    split = tw.split_as_served(GemmaLikeTokenizer(), PROMPT, [{"role": "assistant", "content": "<3 love it"}], {})
    assert split["completion"] == "<3 love it<turn|>\n"


def test_unsplittable_template_fails_loudly():
    class Dropping(GemmaLikeTokenizer):
        def apply_chat_template(self, msgs, **kw):
            return super().apply_chat_template([m for m in msgs if m["role"] != "assistant"], **kw)

    with pytest.raises(ValueError):
        tw.split_as_served(Dropping(), PROMPT, [{"role": "assistant", "content": "hi"}], {})


# ---- extends -----------------------------------------------------------------

def _yaml(path, text):
    path.write_text(text)
    return str(path)


def test_extends_layers_a_persona_over_the_recipe(tmp_path):
    from train_worker import load_config

    _yaml(tmp_path / "base.yaml", "model: m\nlora: {r: 32, alpha: 64}\ntraining: {epochs: 1, eval_steps: 200}\n")
    cfg = load_config(_yaml(tmp_path / "p.yaml", (
        "extends: base.yaml\n"
        "dataset: {path: /d/train.jsonl}\n"
        "output_dir: /o\n"
        "training: {epochs: 3}\n"
    )))

    assert cfg["model"] == "m" and cfg["lora"] == {"r": 32, "alpha": 64}
    assert cfg["training"] == {"epochs": 3, "eval_steps": 200}  # merged, not replaced
    assert "extends" not in cfg


def test_extends_is_relative_to_the_extending_file(tmp_path):
    from train_worker import load_config

    (tmp_path / "sub").mkdir()
    _yaml(tmp_path / "base.yaml", "model: m\n")
    cfg = load_config(_yaml(tmp_path / "sub" / "p.yaml", "extends: ../base.yaml\ndataset: d\noutput_dir: o\n"))
    assert cfg["model"] == "m"


def test_a_base_alone_is_not_a_runnable_config(tmp_path):
    from train_worker import load_config

    with pytest.raises(ValueError, match="dataset"):
        load_config(_yaml(tmp_path / "base.yaml", "model: m\n"))


def test_an_extends_cycle_is_an_error(tmp_path):
    from train_worker import load_config

    _yaml(tmp_path / "a.yaml", "extends: b.yaml\n")
    _yaml(tmp_path / "b.yaml", "extends: a.yaml\n")
    with pytest.raises(ValueError, match="extends itself"):
        load_config(str(tmp_path / "a.yaml"))


def test_every_committed_config_loads():
    """Including each persona's, through its extends."""
    import glob
    import os

    from train_worker import load_config

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = [p for p in glob.glob(os.path.join(here, "configs", "*.yaml"))
             if os.path.basename(p) != "mimic_gemma4_12b.yaml"]
    assert paths
    for path in paths:
        assert load_config(path)["model"]


def test_a_run_keeps_the_dataset_stats_it_started_with(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "stats.json").write_text('{"args": {"no_system": true}}')
    run = tmp_path / "run"
    run.mkdir()
    cfg = {"dataset": {"path": str(data / "train.jsonl")}}

    tw.snapshot_dataset_stats(cfg, str(run))
    (data / "stats.json").write_text('{"args": {"no_system": false}}')  # rebuilt mid-run
    tw.snapshot_dataset_stats(cfg, str(run))  # resumed

    assert (run / "dataset_stats.json").read_text() == '{"args": {"no_system": true}}'


def test_a_dataset_without_stats_records_nothing(tmp_path):
    tw.snapshot_dataset_stats({"dataset": str(tmp_path / "train.jsonl")}, str(tmp_path))
    assert not (tmp_path / "dataset_stats.json").exists()
