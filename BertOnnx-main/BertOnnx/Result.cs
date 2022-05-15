using Microsoft.ML.Data;

namespace BertOnnx
{
    public class Result
    {
        [VectorType(1, 512, 8)]
        [ColumnName("logits")]
        public float[] Logits { get; set; }
    }
}
