namespace BertOnnx
{
    public class Settings
    {
        // A felhasznált modell relatív elérési útvonala.
        public string ModelPath { get; set; } = "./Assets/model-token-class.onnx";

        // A konfigurációs file elérésének megadása.
        public string ConfigPath { get; set; } = "./Assets/config.json";

        // A szótár szöveges file elérési útvonala.
        public string VocabPath { get; set; } = "./Assets/vocab.txt";

        // A modell bemenetét reprezentáló tömb, az egyes bemenetek neveivel.
        public string[] ModelInput => new[] { "input_ids", "attention_mask", "token_type_ids" };

        // A modell kimenete, névvel jelölve.
        public string[] ModelOutput => new[] { "logits" };

        // A szekvencia hossz, a modell (felépítése alapján) ennyi bemenetet tud egyszerre kezelni.
        public int SequenceLength => 512;
    }
}
