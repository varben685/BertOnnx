﻿using System;
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

            var config = HuggingFace.Config.FromFile(settings.ConfigPath);

            Console.WriteLine("Done");

            Console.Write("Reading Vocabulary...");

            // Tokenizáló létrehozása egy szótár fileból, így az már tartalmazni fogja a szótár szavait.
            var tokenizer = WordPieceTokenizer.FromVocabularyFile(settings.VocabPath);

            Console.WriteLine("Done");

            Console.WriteLine("Tokenizing...");

            // Bemenet mondatokra bontása, mondatvégi írásjelek alapján.
            var sentences = args[0].Split('.','?','!');

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
            var engine = Prediction.Engine<Feature, Result>.Create(settings, padded.Length);
            loadingTime.Stop();

            Console.WriteLine($"Done ({loadingTime.ElapsedMilliseconds} milliseconds)");

            Console.Write("Performing inference...");

            var inferendTime = Stopwatch.StartNew();
            var result = engine.Predict(feature);
            inferendTime.Stop();

            Console.WriteLine($"Done ({inferendTime.ElapsedMilliseconds} milliseconds)");

            tokens
                .Zip(result.Logits.Batch(8).ToArray(), (token, values) => (Token: token, Values: values, StartIndex: token.StartIndex, EndIndex: token.EndIndex))
                .GroupBy(tuple => (WordIndex: tuple.Token.WordIndex, Word: tuple.Token.Word))
                .Select(group => GetWordCategory(config, group.Key.WordIndex, group.Key.Word,  group.SelectMany(g => g.Values)))
                .Where(tuple => tuple.Category > 0)
                .ForEach(tuple => Console.WriteLine($"entity: {tuple.Label}, score: {tuple.Prob}, word: {tuple.Word},start:{tokens.Where(w => w.Word == tuple.Word && w.WordIndex == tuple.WordIndex).Select(st => st.StartIndex).FirstOrDefault()},end:{tokens.Where(w => w.Word == tuple.Word && w.WordIndex == tuple.WordIndex).Select(st => st.EndIndex).FirstOrDefault()}"));

        }

        private static (int WordIndex, string Word, int Category, string Label, float Score,float Prob) GetWordCategory(HuggingFace.Config config, int wordIndex, string word, IEnumerable<float> values)
        {
            var prob = GetProbability(values);
            

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
            return values
                .Select((v, i) => (Value: v, Index: i))
                .GroupBy(values => values.Index % 8)
                .Select((group, index) => (Category: index, Value: group.Softmax(g => g.Value))).FirstOrDefault();
        }
        
    }

    public static class SoftmaxEnumerableExtension
    {
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
