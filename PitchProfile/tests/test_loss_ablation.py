"""The partial-label ablation runs both arms the same way and never touches the adopted fine-tune."""
import subprocess
import sys
from pathlib import Path

from scripts.run_loss_ablation import ARMS, commands, recipe, summarise

ROOT = Path(__file__).resolve().parents[1]
ADOPTED_DIR = 'tdeed_team_bas_ft'


def test_recipe_copies_the_adopted_training_settings_only():
    log = {'args': {'steps': 3000, 'lr': 2e-4, 'no_mixup': False, 'freeze': 'stem,s1,s2', 'rare': '',
                    'out_dir': ADOPTED_DIR, 'start': 'published', 'workers': 8}}
    assert recipe(log) == ['--steps', '3000', '--lr', '0.0002', '--freeze', 'stem,s1,s2']
    assert recipe({}) == ['--steps', '3000']                      # no log: the documented run


def test_only_the_hard_arm_changes_labels_and_each_arm_has_its_own_folder():
    plan = commands(['--steps', '10'], ['partial', 'hard'], Path('W'))
    (train_p, scores_p), (train_h, scores_h) = plan['partial'], plan['hard']
    assert '--hard-labels' not in train_p and '--hard-labels' in train_h
    assert [c for c in train_p if c != '--hard-labels'] != train_h       # different output folders
    for arm, train, scores in (('partial', train_p, scores_p), ('hard', train_h, scores_h)):
        out_dir = train[train.index('--out-dir') + 1]
        assert out_dir == ARMS[arm] != ADOPTED_DIR
        assert [s[s.index('--dataset') + 1] for s in scores] == ['footpass', 'bas']
        for s in scores:
            assert s[s.index('--checkpoint') + 1] == str(Path('W') / out_dir / 'checkpoint_best.pt')
            assert s[s.index('--tag') + 1] == out_dir


def test_summary_puts_models_side_by_side():
    report = lambda m, ap: {'mean_average_precision': m, 'classes': {'Pass': {'average_precision': ap}}}
    table = summarise({'footpass_val': {'published': report(.478, .80), ARMS['hard']: report(.55, .88)}})
    assert table['footpass_val']['mean_average_precision'] == {'published': .478, 'hard': .55}
    assert table['footpass_val']['average_precision']['Pass'] == {'published': .80, 'hard': .88}


def test_hard_labels_refuse_the_adopted_checkpoint_folder():
    out = subprocess.run([sys.executable, 'scripts/finetune_action_spotter.py', '--hard-labels'], cwd=ROOT,
                         capture_output=True, text=True)
    assert out.returncode != 0 and 'own --out-dir' in out.stderr
