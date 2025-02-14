import marimo

__generated_with = "0.9.15"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(r"""###Imports""")
    return


@app.cell
def __():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import Counter
    import re
    return Counter, mo, np, plt, re


@app.cell
def __(mo):
    mo.md(r"""###Nettoyage de text""")
    return


@app.cell
def __(re):
    def clean_text(text):
        # Remove emojis
        text = re.sub(r'[^\w\s,]', '', text)  # Keeps only word characters and spaces
        
        # Remove user mentions (e.g., @AmadeusOfficiel)
        text = re.sub(r'@\w+', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove hashtags (e.g., #Jëli)
        text = re.sub(r'#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Convert text to lowercase
        text = text.lower()
        
        return text

    #text to clean
    mon_texte = "🔥 Trop cool le show d’hier avec @AmadeusOfficiel et @officielwallyseck😍 ! J’ai trop hâte de te voir 💯 à ziguinchor!!! #Jëli #Mbëggeeldafaneex # https://t.co/abc123efg "
    cleaned_text = clean_text(mon_texte)
    print(cleaned_text)
    return clean_text, cleaned_text, mon_texte


@app.cell
def __(mo):
    mo.md(r"""###N-grams""")
    return


@app.cell
def __():
    text_gram="Je suis là"
    return (text_gram,)


@app.cell
def __(text_gram):
    def n_grams(text, n):
        n_grams= list(zip(text_gram.split(), text_gram.split()[n-1:]))
        return n_grams
    return (n_grams,)


@app.cell
def __(n_grams, text_gram):
    print(n_grams(text_gram, 2))
    return


@app.cell
def __(mo):
    mo.md(r"""###Illustration de la loi de puissance de zipf""")
    return


@app.cell
def __(Counter, np, plt, re):
    # Exemple de texte en français
    texte = """
    Le soleil brillait fort sur la mer bleue. Les vagues venaient lécher le sable doré,
    et les mouettes planaient dans le ciel. Le vent soufflait doucement, apportant avec lui
    la odeur salée de le océan. Les enfants jouaient en riant, construisant des châteaux de sable
    tandis que les adultes profitaient de le soleil.
    """

    # Nettoyage et tokenisation du texte
    mots = re.findall(r'\b\w+\b', texte.lower())

    # Compter la fréquence des mots
    frequence_mots = Counter(mots)

    # Trier les mots par fréquence décroissante
    mots_tries = frequence_mots.most_common()

    # Récupérer les fréquences et les mots pour l'affichage
    frequences = [freq for mot, freq in mots_tries]
    rangs = np.arange(1, len(frequences) + 1)
    mots_texte = [mot for mot, freq in mots_tries]

    # Tracer le graphique en échelle logarithmique
    plt.figure(figsize=(12, 8))
    plt.loglog(rangs, frequences, marker="o", linestyle="none", color='b')
    plt.xlabel("Rang du mot")
    plt.ylabel("Fréquence du mot")
    plt.title("Illustration de la loi de puissance")
    plt.grid(True, which="both", linestyle="--", lw=0.5)

    # Ajouter les mots les plus fréquents au graphique, en vertical
    nb_mots_affiches = 20  # Nombre de mots à afficher
    for i in range(nb_mots_affiches):
        plt.text(rangs[i], frequences[i], mots_texte[i], fontsize=10, ha='right', rotation=90)

    plt.savefig("Illustration de la loi de puissance")
    plt.show()
    return (
        frequence_mots,
        frequences,
        i,
        mots,
        mots_texte,
        mots_tries,
        nb_mots_affiches,
        rangs,
        texte,
    )


@app.cell
def __(mo):
    mo.md(r"""###Illustration de la loi de Heaps""")
    return


@app.cell
def __(plt, re):
    # Sample text corpus (for a larger corpus, you could use a text file or larger sample)
    corpus = """
    Le soleil brillait fort sur la mer bleue. Les vagues venaient lécher le sable doré,
    et les mouettes planaient dans le ciel. Le vent soufflait doucement, apportant avec lui
    la odeur salée de le océan. Les enfants jouaient en riant, construisant des châteaux de sable
    tandis que les adultes profitaient du soleil.
    """ * 100  # Repeat to simulate a larger corpus

    # Tokenize the corpus into words
    words = re.findall(r'\b\w+\b', corpus.lower())

    # Initialize variables to store tokens and types counts
    tokens = []
    types_set = set()  # A set to keep track of unique types
    types_counts = []  # A list to store the number of unique types over time
    tokens_counts = []  # A list to store the number of tokens over time

    # Simulate the process of reading the corpus word-by-word
    for word in words:
        tokens.append(word)
        types_set.add(word)
        tokens_counts.append(len(tokens))  # Total tokens at this point
        types_counts.append(len(types_set))  # Unique types at this point

    # Plot Heaps' Law graph
    plt.figure(figsize=(12, 8))
    plt.plot(tokens_counts, types_counts, label="l'évolution du vocabulaire", color="b")
    plt.xlabel("Nombre de tokens")
    plt.ylabel("Nombre de types")
    plt.title("Illustration de la loi de Heaps")
    plt.grid(True, which="both", linestyle="--", lw=0.5)

    # Heaps' Law theoretical curve for comparison
    k = 10  # Heaps' Law constant (depends on language and corpus)
    beta = 0.5  # Typically between 0.4 and 0.6
    heaps_curve = [k * (n ** beta) for n in tokens_counts]
    plt.plot(tokens_counts, heaps_curve, label="La loi de Heaps (théorique)", color="red", linestyle="--")

    plt.legend()
    plt.savefig('illustration de la loi de heaps')
    plt.show()
    return (
        beta,
        corpus,
        heaps_curve,
        k,
        tokens,
        tokens_counts,
        types_counts,
        types_set,
        word,
        words,
    )


@app.cell
def __(mo):
    mo.md(r"""###Illustration de la loi de brièveté""")
    return


@app.cell
def __(Counter, plt, texte):
    # Tokenizing the text and cleaning up
    words_2 = texte.lower().split()

    # Count the frequency of each word
    word_counts = Counter(words_2)

    # Create lists for plotting
    word_lengths = []
    frequencies = []
    word_labels = []

    # Populate the lists
    for word_1, count in word_counts.items():
        word_lengths.append(len(word_1))
        frequencies.append(count)
        word_labels.append(word_1)

    # Plotting the data
    plt.figure(figsize=(10, 6))
    plt.scatter(word_lengths, frequencies)

    # Adding labels
    for i_, word_ in enumerate(word_labels):
        plt.text(word_lengths[i_] + 0.1, frequencies[i_], word_, fontsize=9, rotation=90)

    # Labels for axes
    plt.xlabel('Longeur de mot')
    plt.ylabel('Frequence de mot')

    # Show plot
    plt.title('Longeur de mot vs Frequence')
    plt.grid(True)
    plt.savefig("Illustration de la loi de brièveté")
    plt.show()
    return (
        count,
        frequencies,
        i_,
        word_,
        word_1,
        word_counts,
        word_labels,
        word_lengths,
        words_2,
    )


@app.cell
def __(mo):
    mo.md(r"""###Byte-Pair Encoding""")
    return


@app.cell
def __(Counter):
    from collections import defaultdict

    class BPETokenizer:
        def __init__(self, vocab_size=100):
            self.vocab_size = vocab_size
            self.vocab = {}

        # Token Learner: Trains the BPE vocabulary from a raw corpus
        def train(self, corpus):
            # Step 1: Preprocess corpus - separate words and add end-of-word token "</w>"
            corpus_words = [' '.join(word) + ' </w>' for word in corpus.split()]
            
            # Initialize the vocabulary dictionary
            self.vocab = defaultdict(int)
            
            # Step 2: Learn vocabulary through BPE merges
            for _ in range(self.vocab_size):
                # Calculate the frequency of each adjacent pair of characters
                pairs = self._get_pair_frequencies(corpus_words)
                if not pairs:
                    break
                
                # Find the most frequent pair
                most_frequent_pair = max(pairs, key=pairs.get)
                
                # Merge the most frequent pair in all corpus words
                corpus_words = self._merge_pair(most_frequent_pair, corpus_words)
                
                # Add merged token to vocabulary
                merged_token = ''.join(most_frequent_pair)
                self.vocab[merged_token] += 1

            print("Final Vocabulary:", list(self.vocab.keys()))

        # Token Segmenter: Segments a test sentence based on the learned vocabulary
        def segment(self, sentence):
            words = sentence.split()
            segmented_sentence = []

            # Segment each word individually
            for word in words:
                segmented_word = self._segment_word(word)
                segmented_sentence.extend(segmented_word)
            
            return segmented_sentence

        # Helper function to compute pair frequencies
        def _get_pair_frequencies(self, corpus_words):
            pairs = Counter()
            for word in corpus_words:
                tokens = word.split()
                for i in range(len(tokens) - 1):
                    pairs[(tokens[i], tokens[i + 1])] += 1
            return pairs

        # Helper function to merge the most frequent pair in all words
        def _merge_pair(self, pair, corpus_words):
            bigram = ' '.join(pair)
            replacement = ''.join(pair)
            new_corpus_words = [word.replace(bigram, replacement) for word in corpus_words]
            return new_corpus_words

        # Helper function to segment a word based on the vocabulary
        def _segment_word(self, word):
            word = ' '.join(word) + ' </w>'
            tokens = word.split()

            # Apply the BPE merges to segment the word
            segmented_tokens = []
            while tokens:
                match_found = False
                for i in range(len(tokens), 0, -1):
                    token = ''.join(tokens[:i])
                    if token in self.vocab:
                        segmented_tokens.append(token)
                        tokens = tokens[i:]
                        match_found = True
                        break
                if not match_found:  # If no match is found, take the first token
                    segmented_tokens.append(tokens[0])
                    tokens = tokens[1:]
            
            return segmented_tokens

    # Test the BPE Tokenizer
    corpus_1 = "this is a test corpus to learn byte pair encoding. I want to undo the untitled slot. I unwanted the inhabitants. He tests it"
    tokenizer = BPETokenizer(vocab_size=10)
    tokenizer.train(corpus_1)

    test_sentence = "test the tokenizer on unseen sentence"
    segmented_sentence = tokenizer.segment(test_sentence)

    print("\nSegmented Sentence:")
    print(segmented_sentence)


    return (
        BPETokenizer,
        corpus_1,
        defaultdict,
        segmented_sentence,
        test_sentence,
        tokenizer,
    )


@app.cell
def __(mo):
    mo.md(r"""###Calculating similarity between two texts using EDIT distance""")
    return


@app.cell
def __():
    # from nltk.metrics import edit_distance


    # # Sample texts
    # text1 = "The quick brown fox jumps over the lazy dog."
    # text2 = "The quick brown fox jumps over the lazy dog."

    # # Preprocess by converting texts to lowercase and splitting into words
    # words1 = text1.lower().split()
    # words2 = text2.lower().split()

    # # Initialize a list to store normalized edit distances
    # normalized_edit_distances = []

    # # Calculate normalized edit distances between each pair of words from text1 and text2
    # for word1 in words1:
    #     for word2 in words2:
    #         # Calculate the edit distance for each word pair
    #         distance = edit_distance(word1, word2)
    #         # Normalize by dividing by the maximum length of the two words
    #         max_len = max(len(word1), len(word2))
    #         normalized_distance = distance / max_len  # Normalized between 0 and 1
    #         normalized_edit_distances.append(normalized_distance)

    # # Calculate the average normalized edit distance
    # average_normalized_distance = np.mean(normalized_edit_distances)

    # # Convert to similarity score (1 - average distance)
    # similarity_score = 1 - average_normalized_distance
    # similarity_score

    return


@app.cell
def __(np):
    from nltk.metrics import edit_distance

    # Sample texts
    text1 = "The quick brown fox jumps over the lazy dog."
    text2 = "The quick brown fox jumped over some lazy dogs."

    # Preprocess by converting texts to lowercase and splitting into words
    words1 = text1.lower().split()
    words2 = text2.lower().split()

    # Calculate minimum normalized edit distance for each word in text1 against text2
    min_normalized_distances = []

    for word1 in words1:
        # Find the minimum normalized edit distance for `word1` across all words in `words2`
        min_distance = min(edit_distance(word1, word2) / max(len(word1), len(word2)) for word2 in words2)
        min_normalized_distances.append(min_distance)

    # Calculate the average of these minimum distances
    average_min_distance = np.mean(min_normalized_distances)

    # Convert to similarity score: 1 for identical texts, 0 for completely different
    similarity_score = 1 - average_min_distance
    similarity_score

    return (
        average_min_distance,
        edit_distance,
        min_distance,
        min_normalized_distances,
        similarity_score,
        text1,
        text2,
        word1,
        words1,
        words2,
    )


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
