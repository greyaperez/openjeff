import tempfile
import unittest

try:
    import torch
    import peft
    from transformers import Gemma4UnifiedConfig, Gemma4UnifiedTextConfig, Gemma4UnifiedForConditionalGeneration
    HAS_RUNTIME = True
except ImportError:
    HAS_RUNTIME = False

@unittest.skipUnless(HAS_RUNTIME, 'optional pinned GPU/CPU training runtime')
class TrainingTests(unittest.TestCase):
    def test_masking_training_reload_and_padding(self):
        from openjeff.training import attach_lora, collate, candidate_logits, candidate_loss, adapter_digest
        from peft import PeftModel
        torch.manual_seed(23)
        cfg = Gemma4UnifiedTextConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            layer_types=['sliding_attention', 'full_attention'] * 2, sliding_window=32,
            max_position_embeddings=128, attention_k_eq_v=False, num_kv_shared_layers=2)
        def base():
            torch.manual_seed(23)
            return Gemma4UnifiedForConditionalGeneration(Gemma4UnifiedConfig(text_config=cfg))
        model = attach_lora(base(), rank=2, alpha=4, dropout=0)
        examples = [{'input_ids': [7, 8, 9], 'code_ids': [20, 21]},
                    {'input_ids': [11, 12, 13, 14, 15], 'code_ids': [20, 21, 22]}]
        batch = collate(examples, 0, 'cpu')
        model.eval()
        together = candidate_logits(model, batch)
        single = candidate_logits(model, collate(examples[:1], 0, 'cpu'))
        torch.testing.assert_close(together[0, :2], single[0], rtol=1e-4, atol=1e-5)
        self.assertTrue(torch.isneginf(together[0, 2]))
        model.train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        loss = candidate_loss(candidate_logits(model, batch), torch.tensor([1, 2]), batch['valid'])
        loss.backward()
        trainable = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
        self.assertTrue(all('lora_' in n for n,p in trainable))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for n,p in trainable))
        optimizer.step()
        model.eval()
        expected = candidate_logits(model, batch).detach()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, safe_serialization=True)
            self.assertEqual(len(adapter_digest(directory)), 64)
            reloaded = PeftModel.from_pretrained(base(), directory, local_files_only=True).eval()
            torch.testing.assert_close(candidate_logits(reloaded, batch), expected)
        with self.assertRaises(ValueError):
            candidate_loss(together, torch.tensor([2, 0]), batch['valid'])

if __name__ == '__main__': unittest.main()
