namespace BertOnnx
{
    public class Settings
    {
        public string ModelPath { get; set; } = "./Assets/model-token-class.onnx"; // "./Assets/model-optimized-quantized.onnx";

        public string ConfigPath { get; set; } = "./Assets/config.json";

        public string VocabPath { get; set; } = "./Assets/vocab.txt";

        public string[] ModelInput => new[] { "input_ids", "attention_mask","token_type_ids" };

        public string[] ModelOutput => new[] { "logits" };

        public int SequenceLength => 512;
    }
}
