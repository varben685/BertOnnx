using Microsoft.ML.Data;

namespace BertOnnx
{
    // A modell kimenetét reprezentáló osztály.
    public class Result
    {
        /*
        A kimenet a modell alapján "logits" nevű, 3 dimenziós vektor.
        A méretei 1, a batch száma (1 futás során 1 kimenet "adag"), utána 512, a szekvencia hossza, illetve 8, a NER kategóriák száma.
        */
        [VectorType(1, 512, 8)]
        [ColumnName("logits")]
        public float[] Logits { get; set; }
    }
}
