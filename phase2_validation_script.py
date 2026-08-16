import json
import sys
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path

PROJECT_ROOT = Path('D:/synthetic_npc').resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from character.generator import CharacterGenerator, ProfileBuilder, RawCharacterDescription
from character.validator import CharacterValidator
from character.compiler import CharacterCompiler
from character.repository import CharacterRepository
from runtime.cogni_runtime import CogniRuntime
from api.routes import TurnRequest
from schemas.character_schema import MemorySeed

class DemoProvider:
    def generate(self, prompt: str) -> RawCharacterDescription:
        return RawCharacterDescription(
            name='Mara Voss',
            identity={
                'core_beliefs': ['I protect the innocent.'],
                'values': ['justice', 'control'],
                'identity_claims': ['I am a detective who keeps order.', 'I can be ruthless when necessary.'],
            },
            beliefs=['Silence can shield the guilty.', 'The truth is often a weapon.'],
            values=['justice', 'control', 'secrecy'],
            core_motivations=['protect the innocent', 'hide the past', 'preserve authority'],
            goals=['protect the innocent', 'hide the truth of the framed man'],
            personality={'traits': ['retired', 'guarded', 'calculating'], 'archetype': 'retired_detective'},
            speech_style={'tone': 'measured', 'pace': 'slow'},
            relationships=['a former partner', 'an innocent man once framed'],
            traumas=['I once concealed a wrongful arrest'],
            secrets=['I secretly framed an innocent man to protect my own reputation'],
            defense_mechanisms=['deflect', 'minimize', 'externalize_blame'],
            identity_claims=['I am a detective who keeps order.', 'I can be ruthless when necessary.'],
            threat_rules=[
                {'trigger_concepts': ['exposure', 'guilt'], 'threatened_claim_idx': 0, 'threat_type': 'exposure', 'defense_mechanism': 'deflect'},
                {'trigger_concepts': ['innocence', 'truth'], 'threatened_claim_idx': 1, 'threat_type': 'moral_failure', 'defense_mechanism': 'minimize'},
            ],
            life_themes=['guilt', 'reputation', 'justice'],
            memory_seeds=[
                MemorySeed(
                    seed_id='seed_001',
                    summary='I signed the arrest report that ruined an innocent man.',
                    theme='guilt',
                    emotional_weight=8.2,
                    valence=-0.9,
                    suppression_level=0.7,
                    associated_concepts=['arrest', 'innocence', 'guilt', 'report'],
                    sensory_tags=['ink', 'paper', 'cold_room'],
                    people_involved=['innocent_man'],
                    location='station_house',
                    event_type='trauma',
                    age_at_event=62,
                )
            ],
        )

class FakeLLMClient:
    def generate_response(self, prompt=None, acting_note=None, system_instruction=None, history=None):
        return 'I speak carefully and with deliberate restraint.'


def to_jsonable(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value

prompt = 'A retired detective who secretly framed an innocent man.'
provider = DemoProvider()
generator = CharacterGenerator(provider=provider, builder=ProfileBuilder())
profile = generator.generate(prompt)
validator = CharacterValidator()
validation_errors = validator.validate(profile)
compiler = CharacterCompiler()
character = compiler.compile(profile, character_id='mara_voss')

base_dir = PROJECT_ROOT / 'tmp_phase2'
base_dir.mkdir(parents=True, exist_ok=True)
repo = CharacterRepository(source_dir=base_dir)
repo.save(character, base_dir / 'mara_voss.json')

from character.loader import CharacterLoader
runtime = CogniRuntime(state_store=None, loader=CharacterLoader(source_dir=base_dir))
runtime.load_character('mara_voss')
runtime_state = runtime._runtime_state['mara_voss']
runtime_state.pipeline.llm_client = FakeLLMClient()

turn_inputs = [
    'You look tired, Mara.',
    'Tell me what happened to the man you ruined.',
    'Why did you keep silent for so long?',
    'Did you ever feel guilty?',
    'What do you want now?',
]

results = {
    'prompt': prompt,
    'raw_character_description': to_jsonable(provider.generate(prompt)),
    'psychological_profile': to_jsonable(profile),
    'validation_errors': validation_errors,
    'character_schema': to_jsonable(character),
    'runtime_state': to_jsonable(runtime.get_state('mara_voss')),
    'scripted_conversation': [],
}
for message in turn_inputs:
    output = runtime.process_turn('mara_voss', TurnRequest(player_message=message, player_trust_level=0.6, threat_level=0.2))
    results['scripted_conversation'].append({
        'player_message': message,
        'npc_response': output.npc_response,
        'turn': output.cognitive_state.turn,
        'surface_emotion': output.cognitive_summary.surface_emotion if output.cognitive_summary else None,
        'strategic_intent': output.cognitive_summary.strategic_intent if output.cognitive_summary else None,
        'dominant_goal': output.cognitive_state.goal_state.dominant_goal.goal_id if output.cognitive_state.goal_state.dominant_goal else None,
        'chosen_strategy': output.cognitive_state.predictive_state.chosen_strategy,
        'coherence': output.cognitive_state.self_concept_state.coherence,
    })

print(json.dumps(results, indent=2))
