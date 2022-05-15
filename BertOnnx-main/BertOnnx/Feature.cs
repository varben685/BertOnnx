using Microsoft.ML.Data;

namespace BertOnnx
{
    public class Feature
    {
        [VectorType(1, 512)]
        [ColumnName("input_ids")]
        public long[] InputIds { get; set; }

        [VectorType(1, 512)]
        [ColumnName("attention_mask")]
        public long[] AttentionMask { get; set; }

        [VectorType(1, 512)]
        [ColumnName("token_type_ids")]
        public long[] TokenTypeIds { get; set; }
    }
}
