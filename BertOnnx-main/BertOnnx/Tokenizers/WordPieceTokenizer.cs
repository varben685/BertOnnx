using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace BertOnnx
{
    // A tokenizálásért felelős osztály.
    public class WordPieceTokenizer
    {
        // A feldolgozás jelenlegi karakterindexe, kezdés előtt -1.
        public static int CurrentCharIndex = -1;

        /*
        A szótár nem üres sorainak beolvasása, ennek felhasználásával hozunk létre új tokenizálót.
        Mivel statikus, példány nélkül hívhatjuk, ez viszont egy új példányt ad vissza, a szótárat már tartalmazva.
        */
        public static WordPieceTokenizer FromVocabularyFile(string path)
        {
            var vocabulary = new List<string>();

            using (var reader = new StreamReader(path))
            {
                string line;

                while ((line = reader.ReadLine()) != null)
                {
                    if (!string.IsNullOrWhiteSpace(line))
                    {
                        vocabulary.Add(line);
                    }
                }
            }

            return new WordPieceTokenizer(vocabulary);
        }

        // A BERT alap tokenei.
        public class DefaultTokens
        {
            public const string Padding = "";
            public const string Unknown = "[UNK]";
            public const string Classification = "[CLS]";
            public const string Separation = "[SEP]";
            public const string Mask = "[MASK]";
        }

        private readonly List<string> _vocabulary;

        // Az osztály egy konstruktora, ami paraméterként a szótárat várja, egy string lista formájában.
        public WordPieceTokenizer(List<string> vocabulary)
        {
            _vocabulary = vocabulary;
        }

        /*
        A tokenizálásért felelős függvény.
        Ez szúrja be első tokenként a CLS alap tokent is, és hívja tovább a mondatok, illetve szórészek tokenizálásáért felelős függvényeket.
        */
        public List<(string Token, int VocabularyIndex, int WordIndex, string Word,int StartIndex,int EndIndex)> Tokenize(params string[] texts)
        {

            IEnumerable<(string Word, int Index)> tokens = new [] { (Word: DefaultTokens.Classification, Index: -1) };

            foreach (var text in texts)
            {
                tokens = tokens.Concat(TokenizeSentence(text));
                // Minden mondat végénél beszúrunk egy szeparátort is a tokenek közé.
                tokens = tokens.Append((DefaultTokens.Separation, -1));
                
            }

            return tokens
                .SelectMany(tuple => TokenizeSubwords(tuple.Word, tuple.Index))
                .ToList();
        }

        // A szavak/szórészek tokenizálásáért felelős függvény.
        private IEnumerable<(string Token, int VocabularyIndex, int WordIndex, string word,int StartIndex,int EndIndex)> TokenizeSubwords(string word, int index)
        {

            /*
            Ha a szótár tartalmazza az adott szót:
            visszaadjuk a szót, a szótárbeli indexét, a kezdő-, és végindexeket, és szóhossz alapján növeljük a jelenlegi karakterindexet.
            */
            if (_vocabulary.Contains(word))
            {
                if (word.Equals(DefaultTokens.Classification))
                {
                    CurrentCharIndex++;
                    
                    return new[] { (word, _vocabulary.IndexOf(word), index, word, CurrentCharIndex - 1, CurrentCharIndex) };
                }
                else if (word.Equals(DefaultTokens.Separation))
                {
                    CurrentCharIndex+=2;

                    return new[] { (word, _vocabulary.IndexOf(word), index, word, CurrentCharIndex - 2, CurrentCharIndex-1) };
                }
                else
                {
                    CurrentCharIndex += word.Length;
                    return new[] { (word, _vocabulary.IndexOf(word), index, word, CurrentCharIndex - word.Length-1, CurrentCharIndex) };
                }
                
                
            }

            var tokens = new List<(string Token, int VocabularyIndex, int WordIndex, string Word,int StartIndex,int EndIndex)>();
            var remaining = word;

            // A szót/szórészt addig bontjuk, és dolgozzuk fel tokenekre, amíg az nem üres, és a hossza nagyobb 2 karakternél.
            while (!string.IsNullOrEmpty(remaining) && remaining.Length > 2)
            {
                // Összetett szavak esetén, a maradék szóhoz olyat keres, ami a legjobban illeszkedik a szótár egy elemére.
                var prefix = _vocabulary.Where(remaining.StartsWith)
                    .OrderByDescending(o => o.Count())
                    .FirstOrDefault();

                // Ha a prefix null, növeljük a jelenlegi karakterindexet az eredeti szó hosszával, viszont "ismeretlen" alap tokent adunk vissza.
                if (prefix == null)
                {
                    CurrentCharIndex += word.Length;
                    tokens.Add((DefaultTokens.Unknown, _vocabulary.IndexOf(DefaultTokens.Unknown), index, word,CurrentCharIndex-word.Length,CurrentCharIndex));
                    CurrentCharIndex++;
                    return tokens;
                }

                // Résszavaknál a maradék első részét kicseréljük "##"-re, ezzel jelezve, hogy a token egy előző szó folytatása.
                remaining = remaining.Replace(prefix, "##");
                if (prefix.StartsWith("##"))
                {
                    CurrentCharIndex += prefix.Length-2;
                    tokens.Add((prefix, _vocabulary.IndexOf(prefix), index, word, CurrentCharIndex -2, CurrentCharIndex+1));
                    CurrentCharIndex += 1;
                }
                else
                {
                    CurrentCharIndex += prefix.Length;
                    tokens.Add((prefix, _vocabulary.IndexOf(prefix), index, word, CurrentCharIndex - prefix.Length, CurrentCharIndex));
                }
                
                
            }

            /*
            Ez fedi le azokat az eseteket, amikor a szónak van tartalma, de nem hosszabb 2 karakternél, és nincs a tokenek között.
            Ilyenkor szintén egy "ismeretlen" alap tokent adunk vissza.
            */
            if (!string.IsNullOrWhiteSpace(word) && !tokens.Any())
            {
                CurrentCharIndex += word.Length ;
                
                tokens.Add((DefaultTokens.Unknown, _vocabulary.IndexOf(DefaultTokens.Unknown), index, word, CurrentCharIndex - word.Length, CurrentCharIndex));
            }

            return tokens;
        }

        // Ez a függvény megadott határjelölő karakterek alapján részekre bontja a kapott string-et.
        private static IEnumerable<string> SplitAndKeep(string s, params char[] delimiters)
        {
            int start = 0, index;

            while ((index = s.IndexOfAny(delimiters, start)) != -1)
            {
                if (index - start > 0)
                    yield return s.Substring(start, index - start);

                yield return s.Substring(index, 1);

                start = index + 1;
            }

            if (start < s.Length)
            {
                yield return s.Substring(start);
            }
        }

        // Egyes mondatok tokenizálásáért felelős függvény, ez használja a fenti részekre bontást.
        private IEnumerable<(string Word, int Index)> TokenizeSentence(string text)
        {
           
            return text.Split(new string[] { " ", "   ", "\r\n" }, StringSplitOptions.None)
                .SelectMany(o => SplitAndKeep(o, ".,;:\\/?!#$%()=+-*\"'–_`<>&^@{}[]|~'".ToArray()))
                .Select((o, i) => (Word: o, Index: i));
        }
    }
}
