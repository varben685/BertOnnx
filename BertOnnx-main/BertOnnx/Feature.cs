using Microsoft.ML.Data;

namespace BertOnnx
{
    /*
    A modell bemenetét reprezentáló osztály.
    Mindegyik bemenet méretét a batch ("adag") érték, és a szekvencia hossz határozza meg, ez alapján mindegyik 2 dimenziós vektor lesz.
    */
    public class Feature
    {
        // Az input_ids a tokenek azonosítóval való összekötésére szolgál.
        [VectorType(1, 512)]
        [ColumnName("input_ids")]
        public long[] InputIds { get; set; }

        // Az attention_mask értéke dönti el, hogy a token a modell szempontjából lényeges-e, a nulla értékű padding tokeneket a modell nem veszi figyelembe.
        [VectorType(1, 512)]
        [ColumnName("attention_mask")]
        public long[] AttentionMask { get; set; }

        // A TokenTypeIds a mondatok közti összefüggést jósolja.
        [VectorType(1, 512)]
        [ColumnName("token_type_ids")]
        public long[] TokenTypeIds { get; set; }
    }
}
