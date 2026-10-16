"""Dependency-free checks for research training control flow."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TrainingRegressions(unittest.TestCase):
    def test_validation_never_reads_test_split(self):
        for name in ('CLIP', 'ResNet50'):
            tree = ast.parse((ROOT / 'train' / f'train_with_{name}.py').read_text(encoding="utf-8"))
            validation = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'val')
            names = {n.id for n in ast.walk(validation) if isinstance(n, ast.Name)}
            self.assertIn('val_loader', names)
            self.assertNotIn('test_loader', names)

    def test_checkpoint_paths_format_successfully(self):
        for name in ('CLIP', 'ResNet50'):
            tree = ast.parse((ROOT / 'train' / f'train_with_{name}.py').read_text(encoding="utf-8"))
            paths = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute) and n.func.attr == 'format'
                     and isinstance(n.func.value, ast.Constant)
                     and str(n.func.value.value).endswith('.pt')]
            self.assertTrue(paths)
            for path in paths:
                path.func.value.value.format(*([1] * len(path.args)))

    def test_checkpoint_does_not_move_live_model(self):
        for name in ('CLIP', 'ResNet50'):
            tree = ast.parse((ROOT / 'train' / f'train_with_{name}.py').read_text(encoding="utf-8"))
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
            self.assertFalse(any(isinstance(n.func, ast.Attribute) and n.func.attr == 'cpu'
                                 and isinstance(n.func.value, ast.Name) and n.func.value.id == 'model'
                                 for n in calls))

    def test_active_ave_classifier_has_28_classes(self):
        tree = ast.parse((ROOT / 'train/train_with_ResNet50.py').read_text(encoding="utf-8"))
        constructors = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Name) and n.func.id == 'Temporal_Attention']
        self.assertEqual(len(constructors), 1)
        self.assertEqual(next(k.value.value for k in constructors[0].keywords if k.arg == 'num_classes'), 28)


if __name__ == '__main__':
    unittest.main()

