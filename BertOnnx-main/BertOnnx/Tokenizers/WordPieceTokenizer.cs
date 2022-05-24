using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace BertOnnx
{
    public class WordPieceTokenizer
    {
        public static int CurrentCharIndex = -1;
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

        public class DefaultTokens
        {
            public const string Padding = "";
            public const string Unknown = "[UNK]";
            public const string Classification = "[CLS]";
            public const string Separation = "[SEP]";
            public const string Mask = "[MASK]";
        }

        private readonly List<string> _vocabulary;

        public WordPieceTokenizer(List<string> vocabulary)
        {
            _vocabulary = vocabulary;
        }

        public List<(string Token, int VocabularyIndex, int WordIndex, string Word,int StartIndex,int EndIndex)> Tokenize(params string[] texts)
        {

            IEnumerable<(string Word, int Index)> tokens = new [] { (Word: DefaultTokens.Classification, Index: -1) };

            foreach (var text in texts)
            {
                tokens = tokens.Concat(TokenizeSentence(text));
                tokens = tokens.Append((DefaultTokens.Separation, -1));
                
            }

            return tokens
                .SelectMany(tuple => TokenizeSubwords(tuple.Word, tuple.Index))
                .ToList();
        }

        
        private IEnumerable<(string Token, int VocabularyIndex, int WordIndex, string word,int StartIndex,int EndIndex)> TokenizeSubwords(string word, int index)
        {

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

            while (!string.IsNullOrEmpty(remaining) && remaining.Length > 2)
            {
                var prefix = _vocabulary.Where(remaining.StartsWith)
                    .OrderByDescending(o => o.Count())
                    .FirstOrDefault();

                if (prefix == null)
                {
                    CurrentCharIndex += word.Length;
                    tokens.Add((DefaultTokens.Unknown, _vocabulary.IndexOf(DefaultTokens.Unknown), index, word,CurrentCharIndex-word.Length,CurrentCharIndex));
                    CurrentCharIndex++;
                    return tokens;
                }

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

            if (!string.IsNullOrWhiteSpace(word) && !tokens.Any())
            {
                CurrentCharIndex += word.Length ;
                
                tokens.Add((DefaultTokens.Unknown, _vocabulary.IndexOf(DefaultTokens.Unknown), index, word, CurrentCharIndex - word.Length, CurrentCharIndex));
            }

            return tokens;
        }

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

        private IEnumerable<(string Word, int Index)> TokenizeSentence(string text)
        {
           
            return text.Split(new string[] { " ", "   ", "\r\n" }, StringSplitOptions.None)
                .SelectMany(o => SplitAndKeep(o, ".,;:\\/?!#$%()=+-*\"'–_`<>&^@{}[]|~'".ToArray()))
                .Select((o, i) => (Word: o, Index: i));
        }
    }
}
