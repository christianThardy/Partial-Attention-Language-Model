import unittest

import torch

from palm.config import PALMConfig
from palm.model import PALMModel
from palm.data import load_and_preprocess_data
from palm.training import PALMTrainer
from palm.evaluation import generate_text, evaluate_generations, evaluate_information_extraction
from datasets import load_dataset
from transformers import AutoTokenizer


class TestPALMLibrary(unittest.TestCase):
    '''This class contains unit tests for the PALM model and its associated 
        components, using Python's unittest framework.
        

       setUpClass: This method runs once before all tests and is used to set
       up common objects that are shared across the test cases. It initializes 
       the PALM model, tokenizer, and a test dataset.
       
       test_model_initialization: Tests that the model is properly initialized 
       and that the configuration parameters are correctly set.
       
       test_model_forward_pass: Verifies that the model's forward pass works 
       correctly, producing the expected number of outputs and shapes.
       
       test_attention_mask_creation: Tests the creation of the attention mask, 
       ensuring it has the correct shape for the input sequence.
       
       test_generate_method: Verifies the text generation method, ensuring the 
       generated output is a tensor of the correct shape.
       
       test_save_and_load_pretrained: Tests the model's ability to save and load 
       itself from a directory, ensuring the model can be reloaded correctly.
       
       test_data_preprocessing: Tests the data preprocessing function to ensure it 
       correctly processes input examples into the required format for model input.
       
       test_trainer_initialization: Verifies that the trainer object is properly 
       initialized, using mock data loaders for training and evaluation.
       
       test_generate_text_function: Tests the generate_text function to ensure it 
       returns a string when generating text from a given prompt.
       
       test_evaluate_generations: Tests the evaluation function for text generations, 
       ensuring it returns the expected scores for different metrics
       
       test_evaluate_information_extraction: Verifies the information extraction evaluation 
       function, checking that it returns accuracy scores for the PALM and baseline outputs.
       
       test_dataset_structure: Ensures that the dataset is loaded correctly and contains the 
       expected columns (prompt and completion).
       
       test_tokenizer_model_consistency: Verifies that the tokenizer's vocabulary size and 
       padding token ID match the model's configuration.
       
       test_tokenization: Tests the tokenizer's ability to tokenize a sample sentence and ensures 
       special tokens like [PAD] are included in the tokenizer's vocabulary.
       
       test_oov_tokenization: Tests how the tokenizer handles out-of-vocabulary (OOV) words, 
       ensuring they are split into subtokens.
       
       test_encode_decode: Tests the tokenizer's encode-decode cycle, ensuring that the original 
       text is accurately recovered after encoding and decoding.
       
       test_model_tokenizer_vocab_size_match: Ensures that the tokenizer's vocabulary size matches 
       the model's configuration.'''

    @classmethod
    def setUpClass(cls):
        cls.config = PALMConfig(
            base_model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=512,
            max_position_embeddings=512
        )
        cls.model = PALMModel(cls.config)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.config.base_model_name)
        cls.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        cls.dataset = load_dataset("HuggingFaceH4/instruction-dataset", split="test")

    def test_model_initialization(self):
        self.assertIsInstance(self.model, PALMModel)
        self.assertEqual(self.model.config.vocab_size, 1000)
        self.assertEqual(self.model.config.hidden_size, 128)

    def test_model_forward_pass(self):
        input_ids = torch.randint(0, 1000, (1, 10))
        attention_mask = torch.ones_like(input_ids)
        labels = torch.randint(0, 1000, (1, 10))
        source_len = torch.tensor([5])

        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels, source_len=source_len)
        self.assertEqual(len(outputs), 4)  # lm_logits, combined_loss, loss, sae_loss
        self.assertEqual(outputs[0].shape, (1, 10, 1000))  # lm_logits shape

    def test_attention_mask_creation(self):
        input_ids = torch.randint(0, 1000, (1, 20))
        attention_mask = self.model.create_bidirectional_attention_mask(input_ids)
        self.assertEqual(attention_mask.shape, (1, 1, 20, 20))

    def test_generate_method(self):
        input_ids = torch.randint(0, 1000, (1, 10))
        generated = self.model.generate(input_ids, max_length=20)
        self.assertIsInstance(generated, torch.Tensor)
        self.assertEqual(generated.shape[1], 20)

    def test_save_and_load_pretrained(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdirname:
            self.model.save_pretrained(tmpdirname)
            loaded_model = PALMModel.from_pretrained(tmpdirname)
            self.assertIsInstance(loaded_model, PALMModel)

    def test_data_preprocessing(self):
        # Need to create a small test dataset
        dataset = {"prompt": ["Test prompt 1", "Test prompt 2"], "completion": ["Test completion 1", "Test completion 2"]}
        processed_data = load_and_preprocess_data(dataset, self.tokenizer, max_seq_length=512)
        self.assertIn("input_ids", processed_data)
        self.assertIn("attention_mask", processed_data)
        self.assertIn("labels", processed_data)
        self.assertIn("source_len", processed_data)

    def test_trainer_initialization(self):
        # Mock data loaders
        train_dataloader = torch.utils.data.DataLoader([torch.randn(10) for _ in range(10)])
        eval_dataloader = torch.utils.data.DataLoader([torch.randn(10) for _ in range(5)])
        
        trainer = PALMTrainer(self.model, train_dataloader, eval_dataloader, self.config)
        self.assertIsInstance(trainer, PALMTrainer)

    def test_generate_text_function(self):
        prompt = "Test prompt"
        generated_text = generate_text(self.model, self.tokenizer, prompt, max_length=20)
        self.assertIsInstance(generated_text, str)

    def test_evaluate_generations(self):
        reference = "This is a reference text."
        palm_output = "This is a PALM output."
        baseline_output = "This is a baseline output."
        scores = evaluate_generations(reference, palm_output, baseline_output)
        self.assertIn("palm_rouge", scores)
        self.assertIn("baseline_rouge", scores)
        self.assertIn("palm_bert_score", scores)
        self.assertIn("baseline_bert_score", scores)

    def test_evaluate_information_extraction(self):
        true_info = ["1995", "Jane Doe", "Electronics"]
        palm_output = "1. Year: 1995\n2. Name: Jane Doe\n3. Industry: Electronics"
        baseline_output = "1. Year: 1996\n2. Name: John Doe\n3. Industry: Software"
        results = evaluate_information_extraction(true_info, palm_output, baseline_output)
        self.assertIn("palm_accuracy", results)
        self.assertIn("baseline_accuracy", results)

    def test_dataset_structure(self):
        self.assertIsNotNone(self.dataset)
        self.assertIn('prompt', self.dataset.column_names)
        self.assertIn('completion', self.dataset.column_names)

    def test_tokenizer_model_consistency(self):
        self.assertEqual(len(self.tokenizer), self.config.vocab_size)
        self.assertEqual(self.tokenizer.pad_token_id, self.config.pad_token_id)

    def test_tokenization(self):
        sample_text = "Hello, world! This is a test sentence."
        tokens = self.tokenizer.tokenize(sample_text)
        self.assertIsInstance(tokens, list)
        self.assertTrue(len(tokens) > 0)

        special_tokens = self.tokenizer.all_special_tokens
        self.assertIn('[PAD]', special_tokens)

    def test_oov_tokenization(self):
        oov_word = "supercalifragilisticexpialidocious"
        tokens = self.tokenizer.tokenize(oov_word)
        self.assertGreater(len(tokens), 1)  # OOV word should be split into subtokens

    def test_encode_decode(self):
        original_text = "Your test sentence here."
        encoded = self.tokenizer.encode(original_text)
        decoded = self.tokenizer.decode(encoded)
        self.assertEqual(original_text, decoded)

    def test_model_tokenizer_vocab_size_match(self):
        self.assertEqual(len(self.tokenizer), self.model.config.vocab_size)


if __name__ == '__main__':
    unittest.main()