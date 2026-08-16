from pathlib import Path
import sys
import textwrap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from character.compiler import CharacterCompiler
from character.dsl_parser import CharacterDSLParser
from character.generator import CharacterGenerator, ProfileBuilder, RawCharacterDescription
from character.loader import CharacterLoader
from character.repository import CharacterRepository
from character.validator import CharacterValidator
from persistence.npc_state_store import NPCStateStore
from runtime.cogni_runtime import CogniRuntime
from simulation.scripted_runner import ScriptedRunner
from schemas.character_schema import MemorySeed, PsychologicalProfile


class FakeProvider:
    def __init__(self):
        self.calls = []

    def generate(self, prompt: str) -> RawCharacterDescription:
        self.calls.append(prompt)
        return RawCharacterDescription(
            name="Ada Vane",
            identity={"core_beliefs": ["I keep my word."], "values": ["order", "care"], "identity_claims": ["I protect the vulnerable."]},
            beliefs=["Silence is often a form of mercy."],
            values=["order", "care"],
            core_motivations=["protect the vulnerable", "keep secrets"],
            goals=["protect the vulnerable", "preserve a hidden truth"],
            personality={"traits": ["careful", "private"], "archetype": "guarded_caretaker"},
            speech_style={"tone": "measured", "pace": "slow"},
            relationships=["guardian of a younger sibling"],
            traumas=["a sibling vanished in a storm"],
            secrets=["I hid the truth about the disappearance"],
            defense_mechanisms=["deflect", "minimize"],
            identity_claims=["I protect the vulnerable."],
            threat_rules=[{"trigger_concepts": ["exposure"], "threatened_claim_idx": 0, "threat_type": "exposure", "defense_mechanism": "deflect"}],
            life_themes=["responsibility", "guilt"],
            memory_seeds=[
                MemorySeed(
                    seed_id="seed_001",
                    summary="I found my sibling's coat after the storm.",
                    theme="loss",
                    emotional_weight=7.0,
                    associated_concepts=["storm", "sibling", "loss"],
                    sensory_tags=["wet_coat", "cold_air"],
                    people_involved=["sibling"],
                    location="harbor",
                    event_type="trauma",
                    valence=-0.8,
                    suppression_level=0.4,
                )
            ],
        )


def test_yaml_parser_compiles_psychological_profile(tmp_path):
    yaml_path = tmp_path / "detective.yaml"
    yaml_path.write_text(
        textwrap.dedent(
            """
            name: Detective Vale
            identity:
              core_beliefs:
                - I protect the vulnerable.
              values:
                - justice
                - restraint
              identity_claims:
                - I keep people alive.
            beliefs:
              - Silence can save a life.
            values:
              - justice
              - restraint
            core_motivations:
              - protect the vulnerable
            goals:
              - keep the truth buried
            personality:
              archetype: guarded_detective
              traits:
                - careful
                - private
            speech_style:
              tone: measured
              pace: slow
            relationships:
              - former partner
            traumas:
              - a case went wrong
            secrets:
              - I hid a witness
            defense_mechanisms:
              - deflect
              - minimize
            threat_rules:
              - trigger_concepts:
                  - exposure
                threatened_claim_idx: 0
                threat_type: exposure
                defense_mechanism: deflect
            life_themes:
              - guilt
              - duty
            memory_seeds:
              - seed_id: seed_001
                summary: I found the witness in the rain.
                theme: loss
                emotional_weight: 7.0
                associated_concepts:
                  - witness
                  - rain
                  - guilt
                sensory_tags:
                  - wet_coat
                  - cold_air
                people_involved:
                  - witness
                location: dock
                event_type: trauma
                valence: -0.8
                suppression_level: 0.4
            """
        ).strip(),
        encoding="utf-8",
    )

    parser = CharacterDSLParser()
    profile = parser.parse(yaml_path)

    assert isinstance(profile, PsychologicalProfile)
    assert profile.identity["core_beliefs"][0] == "I protect the vulnerable."
    assert profile.memory_seeds[0].seed_id == "seed_001"


def test_generator_interface_builds_psychological_profile():
    provider = FakeProvider()
    generator = CharacterGenerator(provider=provider, builder=ProfileBuilder())

    profile = generator.generate("Generate a guarded caretaker")

    assert isinstance(profile, PsychologicalProfile)
    assert profile.identity["core_beliefs"][0] == "I keep my word."
    assert profile.memory_seeds[0].seed_id == "seed_001"
    assert provider.calls == ["Generate a guarded caretaker"]


def test_compiler_and_validator_create_runtime_character(tmp_path):
    profile = PsychologicalProfile(
        identity={"core_beliefs": ["I protect the vulnerable."], "values": ["order"], "identity_claims": ["I protect the vulnerable."]},
        values=["order"],
        beliefs=["Silence can save a life."],
        core_motivations=["protect the vulnerable"],
        goals=["protect the vulnerable"],
        personality={"traits": ["careful"], "archetype": "guarded_caretaker"},
        speech_style={"tone": "measured"},
        relationships=["sibling"],
        traumas=["a sibling vanished"],
        secrets=["I hid the truth"],
        defense_mechanisms=["deflect"],
        identity_claims=["I protect the vulnerable."],
        threat_rules=[{"trigger_concepts": ["exposure"], "threatened_claim_idx": 0, "threat_type": "exposure", "defense_mechanism": "deflect"}],
        life_themes=["responsibility"],
        memory_seeds=[MemorySeed(seed_id="seed_001", summary="Found the coat in the rain.", theme="loss", associated_concepts=["rain", "loss"], event_type="trauma")],
    )

    compiler = CharacterCompiler()
    validator = CharacterValidator()

    errors = validator.validate(profile)
    assert errors == []

    character = compiler.compile(profile)

    assert character.character_id == "psychological-profile"
    assert character.memories[0].memory_id == "seed_001"
    assert character.goals[0].goal_id == "g_0"
    assert character.threat_rules[0].threatened_claim_idx == 0
    assert character.provenance["compiler_version"]
    assert character.memories[0].provenance["source_seed"] == "seed_001"


def test_validator_returns_structural_errors():
    profile = PsychologicalProfile(
        identity={"core_beliefs": [], "values": [], "identity_claims": []},
        values=[],
        beliefs=[],
        core_motivations=[],
        goals=[],
        personality={},
        speech_style={},
        relationships=[],
        traumas=[],
        secrets=[],
        defense_mechanisms=[],
        identity_claims=[],
        threat_rules=[{"trigger_concepts": ["exposure"], "threatened_claim_idx": 0, "threat_type": "exposure", "defense_mechanism": "deflect"}],
        life_themes=[],
        memory_seeds=[],
    )

    errors = CharacterValidator().validate(profile)

    assert any("identity_claims" in error for error in errors)
    assert any("goals" in error for error in errors)
    assert any("memories" in error for error in errors)


def test_repository_can_load_yaml_and_compile(tmp_path):
    character_dir = tmp_path / "characters"
    character_dir.mkdir(parents=True)
    (character_dir / "detective.yaml").write_text(
        textwrap.dedent(
            """
            name: Detective Vale
            identity:
              core_beliefs:
                - I protect the vulnerable.
              values:
                - justice
              identity_claims:
                - I keep people alive.
            beliefs:
              - Silence can save a life.
            values:
              - justice
            core_motivations:
              - protect the vulnerable
            goals:
              - keep the truth buried
            personality:
              archetype: guarded_detective
              traits:
                - careful
            speech_style:
              tone: measured
            relationships:
              - former partner
            traumas:
              - a case went wrong
            secrets:
              - I hid a witness
            defense_mechanisms:
              - deflect
            threat_rules:
              - trigger_concepts:
                  - exposure
                threatened_claim_idx: 0
                threat_type: exposure
                defense_mechanism: deflect
            life_themes:
              - guilt
            memory_seeds:
              - seed_id: seed_001
                summary: I found the witness in the rain.
                theme: loss
                associated_concepts:
                  - witness
                  - rain
                sensory_tags:
                  - wet_coat
                people_involved:
                  - witness
                location: dock
                event_type: trauma
                valence: -0.8
                suppression_level: 0.4
            """
        ).strip(),
        encoding="utf-8",
    )

    repo = CharacterRepository(source_dir=character_dir)
    character = repo.get("detective")

    assert character.character_id == "detective"
    assert character.memories[0].memory_id == "seed_001"


def test_simulation_runs_through_runtime(tmp_path):
    character_dir = tmp_path / "characters"
    character_dir.mkdir(parents=True)
    (character_dir / "detective.yaml").write_text(
        textwrap.dedent(
            """
            name: Detective Vale
            identity:
              core_beliefs:
                - I protect the vulnerable.
              values:
                - justice
              identity_claims:
                - I keep people alive.
            beliefs:
              - Silence can save a life.
            values:
              - justice
            core_motivations:
              - protect the vulnerable
            goals:
              - keep the truth buried
            personality:
              archetype: guarded_detective
              traits:
                - careful
            speech_style:
              tone: measured
            relationships:
              - former partner
            traumas:
              - a case went wrong
            secrets:
              - I hid a witness
            defense_mechanisms:
              - deflect
            threat_rules:
              - trigger_concepts:
                  - exposure
                threatened_claim_idx: 0
                threat_type: exposure
                defense_mechanism: deflect
            life_themes:
              - guilt
            memory_seeds:
              - seed_id: seed_001
                summary: I found the witness in the rain.
                theme: loss
                associated_concepts:
                  - witness
                  - rain
                sensory_tags:
                  - wet_coat
                people_involved:
                  - witness
                location: dock
                event_type: trauma
                valence: -0.8
                suppression_level: 0.4
            """
        ).strip(),
        encoding="utf-8",
    )

    state_dir = tmp_path / "states"
    runtime = CogniRuntime(
        state_store=NPCStateStore(state_dir),
        loader=CharacterLoader(source_dir=character_dir),
    )

    results = ScriptedRunner.simulate(
        character_id="detective",
        transcript=["Hello", "Where were you?"],
        runtime=runtime,
    )

    assert len(results) == 2
    assert results[0].cognitive_state.turn >= 1


def test_morgan_loads_from_yaml_when_available(tmp_path):
    morgan_source = PROJECT_ROOT / "characters" / "morgan.yaml"
    if not morgan_source.exists():
        return

    loader = CharacterLoader(source_dir=PROJECT_ROOT / "characters")
    character = loader.load("morgan")

    assert character.character_id == "morgan"
    assert character.memories
