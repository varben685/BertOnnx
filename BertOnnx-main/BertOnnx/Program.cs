using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using MoreLinq.Extensions;

namespace BertOnnx
{
    class Program
    {
        private static readonly Stopwatch Stopwatch = new Stopwatch();

        static void Main(string[] args)
        {
            var settings = new Settings();

            Console.Write("Reading Hugging Face Config...");

            // A konfigurációs file-ok beolvasása a Settings.cs-ben tárolt értékek szerint.
            var config = HuggingFace.Config.FromFile(settings.ConfigPath);

            Console.WriteLine("Done");

            Console.Write("Reading Vocabulary...");

            // Tokenizáló létrehozása egy szótár fileból, így az már tartalmazni fogja a szótár szavait.
            var tokenizer = WordPieceTokenizer.FromVocabularyFile(settings.VocabPath);

            Console.WriteLine("Done");

            Console.WriteLine("Tokenizing...");

            // Bemenet mondatokra bontása, mondatvégi írásjelek alapján.
            var sentences = args[0].Split('.','?','!');

            // Tokenizáló meghívása a mondatok tokenekké alakításához, amit egy tömbbé alakítunk.
            var tokens = tokenizer.Tokenize(sentences).ToArray();

            // Tokenek paddelése nullásokkal, hogy a szekvencia méretnek megfelelő legyen a tokenek száma is.
            var padded = tokens.Select(t => (long)t.VocabularyIndex).Concat(Enumerable.Repeat(0L, settings.SequenceLength - tokens.Length)).ToArray();

            Console.WriteLine($"[{string.Join(',', tokens)}]");

            Console.WriteLine("...Done");

            // AttentionMask kitöltése 1-esekkel, ahol nem padding token szerepel, a maradéknál pedig nullásokkal.
            var attentionMask = Enumerable.Repeat(1L, tokens.Length).Concat(Enumerable.Repeat(0L,settings.SequenceLength-tokens.Length)).ToArray();

            // A modellhez tartozó bemeneti osztály létrehozása, a TokenTypeIds-t 0-kkal feltöltve.
            var feature = new Feature { InputIds = padded, AttentionMask = attentionMask, TokenTypeIds = Enumerable.Repeat(0L, settings.SequenceLength ).ToArray() };

            Console.Write("Creating Prediction Engine...");

            var loadingTime = Stopwatch.StartNew();

            // Az eredmény meghatározásához használt motor létrehozása, a modellhez tartozó be- és kimeneti típus, a konfiguráció, és a szekvencia hossz megadásával.
            var engine = Prediction.Engine<Feature, Result>.Create(settings, padded.Length);
            loadingTime.Stop();

            Console.WriteLine($"Done ({loadingTime.ElapsedMilliseconds} milliseconds)");

            Console.Write("Performing inference...");

            var inferendTime = Stopwatch.StartNew();

            // A motor Predict függvényének meghívása, így készítjük el a modell használatával a bemenetekből a megfelelő kimenetet.
            var result = engine.Predict(feature);
            inferendTime.Stop();

            Console.WriteLine($"Done ({inferendTime.ElapsedMilliseconds} milliseconds)");

            /*
            A tokenekből 8-asával tömböket kreálunk, majd a tokent, és a 8 szám értéket egy tuple-be rakjuk.
            A tokenek és a hozzájuk tartozó pontszámok mellett eltároljuk még a tokenről a szövegbeli
            kezdő-, és végpozícióját, ezt már a tokenizálás során kaptuk meg.
            Következő lépésként a legtöbb pontot kapó csoport alapján felcímkézzük a tokent.
            Ezek után már minden szükséges információ rendelkezésre áll, így ugyanolyan formátumú
            kimenetet kapunk, mint az eredeti, Pythonos megoldásban.
            */
            tokens
                .Zip(result.Logits.Batch(8).ToArray(), (token, values) => (Token: token, Values: values, StartIndex: token.StartIndex, EndIndex: token.EndIndex))
                .GroupBy(tuple => (WordIndex: tuple.Token.WordIndex, Word: tuple.Token.Word))
                .Select(group => GetWordCategory(config, group.Key.WordIndex, group.Key.Word,  group.SelectMany(g => g.Values)))
                .Where(tuple => tuple.Category > 0)
                .ForEach(tuple => Console.WriteLine($"entity: {tuple.Label}, score: {tuple.Prob}, word: {tuple.Word},start:{tokens.Where(w => w.Word == tuple.Word && w.WordIndex == tuple.WordIndex).Select(st => st.StartIndex).FirstOrDefault()},end:{tokens.Where(w => w.Word == tuple.Word && w.WordIndex == tuple.WordIndex).Select(st => st.EndIndex).FirstOrDefault()}"));

        }

        private static (int WordIndex, string Word, int Category, string Label, float Score,float Prob) GetWordCategory(HuggingFace.Config config, int wordIndex, string word, IEnumerable<float> values)
        {
            // Az értékekhez tarozó valószínűségek tárolása.
            var prob = GetProbability(values);
            
            /*
            Az értékek egész számmá alakítása, a megfelelő (legjobbnak vélt) NER kategóriába történő besoroláshoz.
            A kategória (szám) alapján pedig a megfelelő címkét rendeljük a tokenhez.
            Eltároljuk a kategóriához tartozó valószínűséget is.
            */
            return values
                .Select((v, i) => (Value: v, Index: i))
                .GroupBy(values => values.Index % 8)
                .Select((group, index) => (Category: index, Value: group.Average(g => g.Value)))
                .Where(tuple => tuple.Value > 0.1)
                .OrderByDescending(tuple => tuple.Value)
                .Select(tuple => (wordIndex, word, tuple.Category, config.id2label[tuple.Category.ToString()], tuple.Value,prob.Probability))
                .FirstOrDefault();
        }

        private static (int Category, float Probability) GetProbability(IEnumerable<float> values)
        {
            // A valószínűségek meghatározása a 8 NER kategóriára, a softmax függvény segítségével.
            return values
                .Select((v, i) => (Value: v, Index: i))
                .GroupBy(values => values.Index % 8)
                .Select((group, index) => (Category: index, Value: group.Softmax(g => g.Value))).FirstOrDefault();
        }
        
    }

    public static class SoftmaxEnumerableExtension
    {
        // A softmax normalizált exponenciális függvény megvalósítása, aktiváló függvényként.
        public static float Softmax<T>(
                                            this IEnumerable<T> collection,
                                            Func<T, float> scoreSelector)
        {
            var maxScore = collection.Max(scoreSelector);
            var sum = collection.Sum(r => Math.Exp(scoreSelector(r) - maxScore));

            return collection.Select(r => (float)(Math.Exp(scoreSelector(r) - maxScore) / sum)).First();
        }
    }
}
